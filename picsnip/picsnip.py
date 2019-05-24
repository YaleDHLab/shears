import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from sklearn.mixture import GaussianMixture
from skimage.measure import label, regionprops
from multiprocessing import Pool
from skimage.color import rgb2gray
from skimage import segmentation, morphology
from scipy.misc import imsave
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

  # load the image and resize to expedite processing if image is large
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


##
# Color Analysis
##

def can_crop_by_color_channels(im, threshold=0.001, log=False):
  '''
  Determine whether an image can be cropped using only the deltas between
  the r,g,b color channels in the input image. This approach is useful
  if the input image contains grayscale pictoral content on grayscale
  backgrounds or grayscale pictoral content on colored backgrounds.

  Parameters
  ----------
  im : str or numpy.ndarray
    An image to process, identified either by the path to an image file
    or a numpy array.
  threshold : float
    The minimum standard deviation to use as a criterion that the pictoral
    content in the input image can be cropped by virtue of color channel
    information only. Lower thresholds require a stronger demarcation between
    rgb foreground and grayscale background or vice-versa.
  log : bool
    Whether to log the means and standard deviations of the Gaussian Mixture
    Model to stdout.
  '''
  im = skimage.transform.rescale(im, 0.05)
  diffs = im_to_color_diffs(im)
  # fit a two component mixed gaussian and measure the sds
  clf = GaussianMixture(n_components=2)
  results = clf.fit(np.expand_dims(diffs.flatten(), axis=-1))
  means = results.means_.flatten()
  stds = results.covariances_.flatten()
  if log: print('means:', means, ' ---  stds:', stds)
  # if the first standard deviation is sufficiently low, we can crop by color
  # channels, else there's too much color confusion to use this technique
  return stds[0] < threshold


def im_to_color_diffs(im):
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
  '''
  if len(im.shape) < 3:
    print('Warning: images without color channels have no color diffs')
    return np.zeros(im.shape[0], im.shape[1], 3)
  d = np.diff(im, axis=-1) # find color channel diffs
  s = np.sum(d, axis=-1) # sum color channel diffs by pixel
  s = (a - np.min(a)) / (np.max(a)-np.min(a)) # center values 0:1
  return 1-s # invert axis to make large deltas large


def crop_with_color_deltas(im):
  '''
  Given an image with grayscale pictoral content and solid color paper,
  crop out the grayscale pictoral content and return that image array.

  Parameters
  ----------
  im : str or numpy.ndarray
    An image to process, identified either by the path to an image file
    or a numpy array.
  '''
  resized = skimage.transform.rescale(im, 0.05) # resize for faster model fit
  diffs = im_to_color_diffs(resized)

  m = np.mean(diffs, axis=-1) # mean color channel deltas; shape = im.height, im.width
  x_diffs = scale_1d_array( np.sum(m, axis=0) )
  y_diffs = scale_1d_array( np.sum(m, axis=1) )

  clf = GaussianMixture(n_components=2) # two dominant values in color channel diffs
  results = clf.fit(np.expand_dims(m.flatten(), axis=-1))
  means = results.means_
  threshold = np.sum(means)/2

  c0 = crop_axis(x_diffs, threshold=threshold)
  c1 = crop_axis(y_diffs, threshold=threshold)

  s0 = np.multiply( np.divide(c0, resized.shape[:2]), im.shape[:2]) # scale
  s1 = np.multiply( np.divide(c1, resized.shape[:2]), im.shape[:2])

  cropped = im[ int(s1[0]) : int(s1[1]), int(s0[0]) : int(s0[1]) ] # crop
  return cropped


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
    return np.array([0,0])
  return np.array([max_streak[0], max_streak[-1]])


##
# Helpers
##

def load_image(im):
  '''
  Return a numpy array containing the raster data from an image. If the
  calling agent passes a numpy array, create a deepcopy to avoid array
  mutations.

  Parameters
  ----------
  im : str or numpy.ndarray
    An image to process, identified either by the path to an image file
    or a numpy array.
  '''
  if isinstance(im, np.ndarray):
    return deepcopy(im)
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

def plot_image(im, title=''):
  '''
  Simple helper to plot an image with an optional title attribute.

  Parameters
  ----------
  im : str or numpy.ndarray
    An image to process, identified either by the path to an image file
    or a numpy array.
  title : str
    The title to add to the image (if desired).
  '''
  plt.imshow(load_image(im), cmap = 'gray')
  plt.title(title)
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
