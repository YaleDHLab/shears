import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from sklearn.mixture import GaussianMixture
from skimage.measure import label, regionprops
from multiprocessing import Pool
from skimage.color import rgb2gray
from skimage import segmentation, morphology
from collections import Counter
from copy import deepcopy
import numpy as np
import operator
import skimage
import glob2
import six
import os
import warnings

# clear internal warnings
warnings.filterwarnings('ignore')

##
# Main
##

def clip(im,
    # color removal params
    n_colors_to_remove=2,
    use_mask=True,
    mask_size=0.05,
    # gaussian blur params
    blur_sigma=2,
    # text removal params
    filter_min_size=None,
    filter_threshold=0.7,
    filter_connectivity=20,
    # cropping params
    crop_threshold=0.975,
    # returned values params
    padding=0,
    return_coords=False,
    plot=False):
  '''
  Main method for clipping images from a page scan.

  Parameters
  ----------
  im : str or numpy.ndarray
    An image to process, identified either by the path to an image file
    or a numpy array.
  n_colors_to_remove : int
    The number of colors to remove from the image. Colors are quantized into
    a ten unit color space and counted, and the `n_colors_to_remove` with highest
    frequency counts are zeroed out.
  use_mask : bool
    Indicates whether to use just the border pixels from the input image when
    identifying the most dominant colors in an image. Useful for identifying the
    color of the paper on which an illustration is printed (as opposed to the
    colors of the actual image itself).
  mask_size : float
    A decimal value that determines the percent of the image width and height
    to use when identifying the page mask. A `mask_size` of .1 means 10% of
    the image width and height will be used to create each edge of the mask.
    When processing smaller images, a larger mask may be used. Conversely,
    if an image occupies more of the page space, a smaller mask should be used.
  blur_sigma : float
    The standard deviation to use when performing a Gaussian blur on the input
    image. 0 = no blur, while larger positive values create larger blurs
  filter_min_size : int
    Minimum number of pixels == 1 in a neighborhood to retain. The filter size
    is used to remove small groups of pixels (like dust or letters on printed
    pages) from images. Increase this value to remove larger groups of pixels.
  filter_connectivity : int
    Determines the number of bordering pixels to use when identifying a pixel's
    neighborhood. Setting a higher value makes small pixel groups less
    likely to be retained.
  filter_threshold : float {0:1}
    The cutoff value for binarizing images when performing a filter operation.
    Lower filters include more pixels in the filtering process.
  crop_threshold : float {0:1}
    The maximum normalized mean pixel values a single row/column of pixels
    can have to be included in the cropped region of the input image. Because
    white pixels have value 1, and all filtering operations in this workflow
    attempt to set non-pictoral pixels to white, one usually wants this value
    to be close to but less than 1.
  padding : int {>0}
    Determines how many pixels of padding to add on each side of
    the cropped picture.
  return_coords : bool
    If True, returns bounding box coordinates with the order
    (y_min, y_max, x_min, x_max) instead of the cropped image.
  plot : bool
    Whether to show plots as an image passes through the pipeline.
  '''

  # load the image
  im = load_image(im)
  _im = deepcopy(im)

  # if the input image is binarized, only remove one color
  if len(im.shape) == 2: n_colors_to_remove = 1

  # remove background page color
  im = remove_dominant_colors(im,
    use_mask=use_mask,
    mask_size=mask_size,
    n_colors_to_remove=n_colors_to_remove)
  if plot: plot_image(im, 'Colors Removed')

  # remove text letters and small components
  if filter_min_size: im = filter_img(im,
    min_size=filter_min_size,
    threshold=filter_threshold,
    connectivity=filter_connectivity)
  if plot: plot_image(im, 'Filter Applied')

  # apply a gaussian filter to smooth out values
  im = gaussian_filter(im, sigma=blur_sigma)
  if plot: plot_image(im, 'Gaussian Blur Applied')

  # aggregate the pixel values for each axis
  y = np.mean(im, axis=1)
  x = np.mean(im, axis=0)
  if len(im.shape) == 3:
    y = np.mean(y, axis=-1)
    x = np.mean(x, axis=-1)
  if plot: plot_1d(y, 'Y Axis Pixel Values')
  if plot: plot_1d(x, 'X Axis Pixel Values')

  # determine the crop boundaries of the image
  y0, y1 = get_longest_match(y, threshold=crop_threshold, op=operator.lt)
  x0, x1 = get_longest_match(x, threshold=crop_threshold, op=operator.lt)

  # apply the padding to the returned coordinates
  padding = int(padding) # disallow decimal pads
  y0 = max(0, y0-padding)
  y1 = min(im.shape[0], y1+padding)
  x0 = max(0, x0-padding)
  x1 = min(im.shape[1], x1+padding)

  if return_coords: return [y0, y1, x0, x1]

  # crop out the image content and warn users if relevant
  cropped = _im[y0:y1, x0:x1]
  if _im.shape[:2] != im.shape[:2]:
    print('TODO - handle resize')
  if cropped.size < _im.size * 0.001:
    print('No image was found')
    return _im

  # return the cropped pictoral content
  return cropped


##
# Image Processing
##

def remove_dominant_colors(im,
  n_colors_to_remove=2,
  use_mask=False,
  mask_size=0.05):
  '''
  Given an image or the path to an image, find and remove the dominint colors.
  This is useful for zeroing out the paper on which an illustration is printed,
  for example.

  Parameters
  ----------
  im : str or numpy.ndarray
    An image to process, identified either by the path to an image file
    or a numpy array.
  n_colors_to_remove : int
    The number of colors to remove from the image. Colors are quantized into
    a ten unit color space and counted, and the `n_colors_to_remove` with highest
    frequency counts are zeroed out.
  use_mask : bool
    Indicates whether to use just the border pixels from the input image when
    identifying the most dominant colors in an image. Useful for identifying the
    color of the paper on which an illustration is printed (as opposed to the
    colors of the actual image itself).
  mask_size : float
    A decimal value that determines the percent of the image width and height
    to use when identifying the page mask. A `mask_size` of .1 means 10% of
    the image width and height will be used to create each edge of the mask.
    When processing smaller images, a larger mask may be used. Conversely,
    if an image occupies more of the page space, a smaller mask should be used.
  '''
  img = load_image(im) # get the user-provided image
  if np.max(img) > 1.0: img = img / 255.0 # scale image 0:1 if it isn't already
  img_orig = deepcopy(img) # handle case of rgb input that uses the grayscale analysis below

  # run additional pre-processing only on the copy of the image fed to Gaussian model
  if len(img.shape) > 2: img = rgb2gray(img) # ensure we're analyzing grayscale images in the analysis below
  if np.any(np.array(img.shape) >= 1000): img = skimage.transform.rescale(img, 0.05) # shrink big inputs to expedite

  # mask out non-frame pixels
  if use_mask:
    mask = np.ones(img.shape) # create a mask with the image's shape
    bw = mask_size # identify border width and height as fraction of image size
    bx = int(img.shape[1] * bw) # get the x dimension border width
    by = int(img.shape[0] * bw) # get the y dimension border height
    mask[bx:-bx,by:-by] = -1 # create a mask that has 1 for the border and -1 for the inside
    color_vals = img * mask # multiply the pixels by the mask to zero out non-border pixels
  else:
    color_vals = img

  # remove the n most common colors
  arr = np.around(color_vals, 1).flatten()
  arr = arr[np.where(arr >= 0)] # remove pixels that fall outside of mask
  pixel_counts = Counter(arr) # count instances of each pixel
  no_paper = np.around(img_orig, 1) # create a copy of the image we can mutate to white out the paper
  for i in range(n_colors_to_remove):
    paper_color = pixel_counts.most_common(n_colors_to_remove)[i][0]
    vals = np.where(no_paper == paper_color) # find quantized pixels that == quantized gaussian mean
    if len(vals) > 2: vals = vals[:2]
    xs, ys = vals
    if len(no_paper.shape) == 2: no_paper[xs, ys] = 1
    elif len(no_paper.shape) == 3: no_paper[xs, ys, :] = 1
    else: raise Exception('Input image size not recognized')

  # return the image with pixels of the dominant color removed
  return no_paper


def filter_img(im, min_size=250, connectivity=20, threshold=0.7):
  '''
  Remove small pixel clusters from an image. Useful for removing visual
  artifacts like dust or larger pixel groupings like text characters from
  a page scan before identifying image boundaries.

  Parameters
  ----------
  im : str or numpy.ndarray
    An image to process, identified either by the path to an image file
    or a numpy array.
  min_size : int
    Minimum number of pixels == 1 in a neighborhood to retain. The filter size
    is used to remove small groups of pixels (like dust or letters on printed
    pages) from images. Increase this value to remove larger groups of pixels.
  connectivity : int
    Determines the number of bordering pixels to use when identifying a pixel's
    neighborhood. Setting a higher value makes small pixel groups less
    likely to be retained.
  threshold : float {0:1}
    The cutoff value for binarizing images when performing a filter operation.
    Lower filters include more pixels in the filtering process.
  '''
  im = load_image(im)
  binarized = np.where(rgb2gray(im)>threshold, 0, 1).astype(bool)
  filtered = morphology.remove_small_objects(binarized,
    min_size=min_size,
    connectivity=connectivity)
  mask = np.array(1 - filtered).astype(int)
  return mask


def get_longest_match(arr, threshold=0.985, op=operator.gt):
  '''
  Find the longest contiguous region of an input region that is greater than
  a specified threshold if op == operator.gt, or less than that threshold
  if op == operator.lt.

  Parameters
  ----------
  arr : numpy.ndarray
    A one-dimensional numpy array.
  threshold : float {0:1}
    The maximum normalized mean pixel values a single row/column of pixels
    can have to be included in the cropped region of the input image if
    op == operator.gt. Because white pixels have value 1, and all filtering
    operations in this workflow attempt to set non-pictoral pixels to white,
    one usually wants this value to be close to but less than 1.
  op : function {operator.gt|operator.lt}
    The comparison function to use when comparing array members to the
    threshold. If pixel op threshold == True, that pixel will be part of
    a contiguous sequence of matches, the longest of which will be returned
    to the calling agent.
  '''
  if len(arr.shape) > 1: raise Exception('Input array contained too many axes')
  if not threshold: raise Exception('Please provide a threshold')
  if op not in [operator.gt, operator.lt]: raise Exception('Operator not supported')

  streak = []
  max_streak = []
  for idx, i in enumerate(arr):
    if op(i, threshold):
      streak.append(idx)
    else:
      if len(streak) > len(max_streak):
        max_streak = streak
      streak = []
  if streak and (len(streak) > len(max_streak)):
    max_streak = streak
  if not max_streak:
    print('No longest match was found for axis')
    return arr
  return np.array([max_streak[0], max_streak[-1]])


def crop_with_color_diffs(im, plot=False):
  '''
  Given an image with grayscale pictoral content and solid color paper,
  crop out the grayscale pictoral content and return that image array.

  Parameters
  ----------
  im : str or numpy.ndarray
    An image to process, identified either by the path to an image file
    or a numpy array.
  plot : bool
    Whether to plot the resulting diffs array. This can be useful to
    visually inspect the separation between bw and color regions of
    an image
  '''
  im = load_image(im)
  diffs = im_to_color_diffs(im, plot=plot)

  # find the color diffs along two axes
  y_sums = np.sum( np.sum(diffs, axis=1), axis=-1) / (diffs.shape[1] * 255)
  x_sums = np.sum( np.sum(diffs, axis=0), axis=-1) / (diffs.shape[0] * 255)

  # find the midpoint that separates the color and bw sections of each axis
  # plotting the axis_sums shows that they separate out the image from te background nicely.
  # model the axis_sums distribution as a mixture of two gaussians
  # the two means will represent the diffs values for the image and paper portions of the image
  clf = GaussianMixture(n_components=2)
  y_means = clf.fit(np.expand_dims(y_sums, -1)).means_
  x_means = clf.fit(np.expand_dims(x_sums, -1)).means_

  # use midpoints of means along each axis to divide color from bw regions
  y_midpoint = np.mean(y_means)
  x_midpoint = np.mean(x_means)

  # find the region of each axis that separates bw from color
  y0, y1 = get_longest_match(y_sums, threshold=y_midpoint, op=operator.lt)
  x0, x1 = get_longest_match(x_sums, threshold=x_midpoint, op=operator.lt)

  # crop out the image
  return im[y0:y1, x0:x1]


def im_to_color_diffs(im, plot=False):
  '''
  Given an image with shape (height, width, color), return a df with the
  same shape indicating the aggregate color channel deltas for each pixel.
  This is useful for identifying regions of paper that contain grayscale
  and color data.

  Parameters
  ----------
  im : str or numpy.ndarray
    An image to process, identified either by the path to an image file
    or a numpy array.
  plot : bool
    Whether to plot the resulting diffs array. This can be useful to
    visually inspect the separation between bw and color regions of
    an image
  '''
  im = load_image(im)
  if len(im.shape) < 3:
    print('Warning: images without color channels have no color diffs')
    diffs = np.zeros(im.shape[0], im.shape[1], 3)

  # find the differences between the color channels in each pixel
  else:
    diffs = np.zeros(im.shape)
    for y_idx, y in enumerate(im):
      for x_idx, x in enumerate(y):
        diffs[y_idx][x_idx] = (x[0] - x[1]) + (x[1] - x[2]) + (x[0] - x[2])

  # plot the pixel color diffs if requested
  if plot:
    plot_image(np.sum(diffs, axis=2), colorbar=True)

  return diffs

##
# Helpers
##

def load_image(im):
  '''
  Return a numpy array containing the raster data from an image.

  Parameters
  ----------
  im : str or numpy.ndarray
    An image to process, identified either by the path to an image file
    or a numpy array.
  '''
  if isinstance(im, np.ndarray):
    return im
  return plt.imread(im)


def save_image(*args):
  '''
  Save an image to disk.

  Parameters
  ----------
  im : str or numpy.ndarray
    An image to process, identified either by the path to an image file
    or a numpy array.
  path : str
    The location on disk where the file should be saved
  '''
  if len(args) == 1:
    im = args[0]
    path = 'cropped.jpg'
  elif len(args) == 2:
    im, path = args
  # handle case where user passes e.g. float as filename
  if not isinstance(path, six.string_types):
    try:
      path = str(path)
    except:
      raise Exception('Please provide a string-like filename')
  # handle case where output path contains dirs that don't exist
  out_dir, out_file = os.path.split(os.path.abspath(path))
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  # persist the image - cmap is ignored for RGB(A) data
  im = load_image(im)
  plt.imsave(path, im, cmap='gray')


def scale_1d_array(arr):
  '''
  Given a one dimensional numpy array, scale that array 0:1.

  Parameters
  ----------
  arr : numpy.ndarray
    A one-dimensional numpy array.
  '''
  _min = np.min(arr)
  _max = np.max(arr)
  arr -= _min
  arr /= _max-_min
  return arr


##
# Plotting
##

def plot_image(im, title='', colorbar=False):
  '''
  Simple helper to plot an image with an optional title attribute.

  Parameters
  ----------
  im : str or numpy.ndarray
    An image to process, identified either by the path to an image file
    or a numpy array.
  title : str
    The title to add to the image (if desired).
  colorbar : bool
    Whether to show a colorbar for the image.
  '''
  plt.imshow(load_image(im), cmap = 'gray')
  plt.title(title)
  if colorbar: plt.colorbar()
  plt.show()


def plot_1d(arr, title=''):
  '''
  Simple helper to plot an array with an optional title attribute.

  Parameters
  ----------
  im : str or numpy.ndarray
    An image to process, identified either by the path to an image file
    or a numpy array.
  title : str
    The title to add to the image (if desired).
  '''
  plt.plot(arr)
  plt.title(title)
  plt.show()


##
# Aliases
##

show_image = plot_image
