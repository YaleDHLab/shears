# Shears

> Extract pictures from historical book scans.

## Installation

```
pip install shears
```

## Basic Usage

Suppose you want to extract the image content within the following page scan:

![Sample book page scan](https://gist.githubusercontent.com/duhaime/2f70ac5c0b772f1f790c94302121faf8/raw/55afb0e5145cc64288410476f455c9a820617fcc/gb3010_papmc_Folder-05795_0003.jpg)

Assuming you have saved the page scan to your current working directory, you can extract the image content with the following:

```
import shears

# extract the image content
result = shears.clip('input.jpg')

# show the extracted image
shears.plot_image(result)

# save the extracted image
shears.save_image(result, 'result.jpg')
```

This returns and saves the following image:

![Sample cropped illustration](https://gist.githubusercontent.com/duhaime/2f70ac5c0b772f1f790c94302121faf8/raw/55afb0e5145cc64288410476f455c9a820617fcc/cropped.jpg)

That's all it takes! The examples below show how to process more complex input images.

## Processing Book Scans

Suppose you want to extract the illustration content from the page scan below:

![Sample book page scan](https://gist.githubusercontent.com/duhaime/2f70ac5c0b772f1f790c94302121faf8/raw/55afb0e5145cc64288410476f455c9a820617fcc/1812_Page_03.jpg)

To extract illustrations in pages like this, one can pass `filter` arguments to shears:

```
import shears

# use the filter parameters to pull out the illustration on a page
result = shears.clip(i,
                      filter_min_size=900,
                      filter_threshold=0.8,
                      filter_connectivity=1)

# show the extracted illustration
shears.plot_image(result, 'Extracted Image')
```

This returns the following image:

![Sample cropped illustration](https://gist.githubusercontent.com/duhaime/2f70ac5c0b772f1f790c94302121faf8/raw/55afb0e5145cc64288410476f455c9a820617fcc/telescope.png)

For additional examples, please see the [sample notebooks in this repository](./examples).

## Testing

To run the test suite, one can run:

```
pytest
```