# Images

Note: Functions taking `Tensor` arguments can also take anything accepted by
@{tf.convert_to_tensor}.

[TOC]

## Encoding and Decoding

TensorFlow provides Ops to decode and encode JPEG and PNG formats.  Encoded
images are represented by scalar string Tensors, decoded images by 3-D uint8
tensors of shape `[height, width, channels]`. (PNG also supports uint16.)

The encode and decode Ops apply to one image at a time.  Their input and output
are all of variable size.  If you need fixed size images, pass the output of
the decode Ops to one of the cropping and resizing Ops.

Note: The PNG encode and decode Ops support RGBA, but the conversions Ops
presently only support RGB, HSV, and GrayScale. Presently, the alpha channel has
to be stripped from the image and re-attached using slicing ops.

*   @{tf.image.decode_gif}
*   @{tf.image.decode_jpeg}
*   @{tf.image.encode_jpeg}
*   @{tf.image.decode_png}
*   @{tf.image.encode_png}
*   @{tf.image.decode_image}

## Resizing

The resizing Ops accept input images as tensors of several types.  They always
output resized images as float32 tensors.

The convenience function @{tf.image.resize_images} supports both 4-D
and 3-D tensors as input and output.  4-D tensors are for batches of images,
3-D tensors for individual images.

Other resizing Ops only support 4-D batches of images as input:
@{tf.image.resize_area}, @{tf.image.resize_bicubic},
@{tf.image.resize_bilinear},
@{tf.image.resize_nearest_neighbor}.

Example:

```python
# Decode a JPG image and resize it to 299 by 299 using default method.
image = tf.image.decode_jpeg(...)
resized_image = tf.image.resize_images(image, [299, 299])
```

*   @{tf.image.resize_images}
*   @{tf.image.resize_area}
*   @{tf.image.resize_bicubic}
*   @{tf.image.resize_bilinear}
*   @{tf.image.resize_nearest_neighbor}

## Cropping

*   @{tf.image.resize_image_with_crop_or_pad}
*   @{tf.image.central_crop}
*   @{tf.image.pad_to_bounding_box}
*   @{tf.image.crop_to_bounding_box}
*   @{tf.image.extract_glimpse}
*   @{tf.image.crop_and_resize}

## Flipping, Rotating and Transposing

*   @{tf.image.flip_up_down}
*   @{tf.image.random_flip_up_down}
*   @{tf.image.flip_left_right}
*   @{tf.image.random_flip_left_right}
*   @{tf.image.transpose_image}
*   @{tf.image.rot90}

## Converting Between Colorspaces

Image ops work either on individual images or on batches of images, depending on
the shape of their input Tensor.

If 3-D, the shape is `[height, width, channels]`, and the Tensor represents one
image. If 4-D, the shape is `[batch_size, height, width, channels]`, and the
Tensor represents `batch_size` images.

Currently, `channels` can usefully be 1, 2, 3, or 4. Single-channel images are
grayscale, images with 3 channels are encoded as either RGB or HSV. Images
with 2 or 4 channels include an alpha channel, which has to be stripped from the
image before passing the image to most image processing functions (and can be
re-attached later).

Internally, images are either stored in as one `float32` per channel per pixel
(implicitly, values are assumed to lie in `[0,1)`) or one `uint8` per channel
per pixel (values are assumed to lie in `[0,255]`).

TensorFlow can convert between images in RGB or HSV. The conversion functions
work only on float images, so you need to convert images in other formats using
@{tf.image.convert_image_dtype}.

Example:

```python
# Decode an image and convert it to HSV.
rgb_image = tf.image.decode_png(...,  channels=3)
rgb_image_float = tf.image.convert_image_dtype(rgb_image, tf.float32)
hsv_image = tf.image.rgb_to_hsv(rgb_image)
```

*   @{tf.image.rgb_to_grayscale}
*   @{tf.image.grayscale_to_rgb}
*   @{tf.image.hsv_to_rgb}
*   @{tf.image.rgb_to_hsv}
*   @{tf.image.convert_image_dtype}

## Image Adjustments

TensorFlow provides functions to adjust images in various ways: brightness,
contrast, hue, and saturation.  Each adjustment can be done with predefined
parameters or with random parameters picked from predefined intervals. Random
adjustments are often useful to expand a training set and reduce overfitting.

If several adjustments are chained it is advisable to minimize the number of
redundant conversions by first converting the images to the most natural data
type and representation (RGB or HSV).

*   @{tf.image.adjust_brightness}
*   @{tf.image.random_brightness}
*   @{tf.image.adjust_contrast}
*   @{tf.image.random_contrast}
*   @{tf.image.adjust_hue}
*   @{tf.image.random_hue}
*   @{tf.image.adjust_gamma}
*   @{tf.image.adjust_saturation}
*   @{tf.image.random_saturation}
*   @{tf.image.per_image_standardization}

## Working with Bounding Boxes

*   @{tf.image.draw_bounding_boxes}
*   @{tf.image.non_max_suppression}
*   @{tf.image.sample_distorted_bounding_box}

## Denoising

*   @{tf.image.total_variation}
