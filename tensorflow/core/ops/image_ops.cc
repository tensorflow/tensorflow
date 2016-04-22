/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"

namespace tensorflow {

// --------------------------------------------------------------------------
REGISTER_OP("ResizeArea")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: float")
    .Attr("T: {uint8, int8, int16, int32, int64, float, double}")
    .Attr("align_corners: bool = false")
    .Doc(R"doc(
Resize `images` to `size` using area interpolation.

Input images can be of different types but output images are always float.

images: 4-D with shape `[batch, height, width, channels]`.
size:= A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
  new size for the images.
align_corners: If true, rescale input by (new_height - 1) / (height - 1), which
  exactly aligns the 4 corners of images and resized images. If false, rescale
  by new_height / height. Treat similarly the width dimension.
resized_images: 4-D with shape
  `[batch, new_height, new_width, channels]`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("ResizeBicubic")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: float")
    .Attr("T: {uint8, int8, int16, int32, int64, float, double}")
    .Attr("align_corners: bool = false")
    .Doc(R"doc(
Resize `images` to `size` using bicubic interpolation.

Input images can be of different types but output images are always float.

images: 4-D with shape `[batch, height, width, channels]`.
size:= A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
  new size for the images.
align_corners: If true, rescale input by (new_height - 1) / (height - 1), which
  exactly aligns the 4 corners of images and resized images. If false, rescale
  by new_height / height. Treat similarly the width dimension.
resized_images: 4-D with shape
  `[batch, new_height, new_width, channels]`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("ResizeBilinear")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: float")
    .Attr("T: {uint8, int8, int16, int32, int64, float, double}")
    .Attr("align_corners: bool = false")
    .Doc(R"doc(
Resize `images` to `size` using bilinear interpolation.

Input images can be of different types but output images are always float.

images: 4-D with shape `[batch, height, width, channels]`.
size:= A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
  new size for the images.
align_corners: If true, rescale input by (new_height - 1) / (height - 1), which
  exactly aligns the 4 corners of images and resized images. If false, rescale
  by new_height / height. Treat similarly the width dimension.
resized_images: 4-D with shape
  `[batch, new_height, new_width, channels]`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("ResizeBilinearGrad")
    .Input("grads: float")
    .Input("original_image: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("align_corners: bool = false")
    .Doc(R"doc(
Computes the gradient of bilinear interpolation.

grads: 4-D with shape `[batch, height, width, channels]`.
original_image: 4-D with shape `[batch, orig_height, orig_width, channels]`,
  The image tensor that was resized.
align_corners: If true, rescale grads by (orig_height - 1) / (height - 1), which
  exactly aligns the 4 corners of grads and original_image. If false, rescale by
  orig_height / height. Treat similarly the width dimension.
output: 4-D with shape `[batch, orig_height, orig_width, channels]`.
  Gradients with respect to the input image. Input image must have been
  float or double.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("ResizeNearestNeighbor")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: T")
    .Attr("T: {uint8, int8, int16, int32, int64, float, double}")
    .Attr("align_corners: bool = false")
    .Doc(R"doc(
Resize `images` to `size` using nearest neighbor interpolation.

images: 4-D with shape `[batch, height, width, channels]`.
size:= A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
  new size for the images.
align_corners: If true, rescale input by (new_height - 1) / (height - 1), which
  exactly aligns the 4 corners of images and resized images. If false, rescale
  by new_height / height. Treat similarly the width dimension.
resized_images: 4-D with shape
  `[batch, new_height, new_width, channels]`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("ResizeNearestNeighborGrad")
    .Input("grads: T")
    .Input("size: int32")
    .Output("output: T")
    .Attr("T: {uint8, int8, int32, float, double}")
    .Attr("align_corners: bool = false")
    .Doc(R"doc(
Computes the gradient of nearest neighbor interpolation.

grads: 4-D with shape `[batch, height, width, channels]`.
size:= A 1-D int32 Tensor of 2 elements: `orig_height, orig_width`. The
  original input size.
align_corners: If true, rescale grads by (orig_height - 1) / (height - 1), which
  exactly aligns the 4 corners of grads and original_image. If false, rescale by
  orig_height / height. Treat similarly the width dimension.
output: 4-D with shape `[batch, orig_height, orig_width, channels]`. Gradients
  with respect to the input image.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("RandomCrop")
    .Input("image: T")
    .Input("size: int64")
    .Output("output: T")
    .Attr("T: {uint8, int8, int16, int32, int64, float, double}")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetIsStateful()
    .Deprecated(8, "Random crop is now pure Python")
    .Doc(R"doc(
Randomly crop `image`.

`size` is a 1-D int64 tensor with 2 elements representing the crop height and
width.  The values must be non negative.

This Op picks a random location in `image` and crops a `height` by `width`
rectangle from that location.  The random location is picked so the cropped
area will fit inside the original image.

image: 3-D of shape `[height, width, channels]`.
size: 1-D of length 2 containing: `crop_height`, `crop_width`..
seed: If either seed or seed2 are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: An second seed to avoid seed collision.
output: 3-D of shape `[crop_height, crop_width, channels].`
)doc");
// TODO(shlens): Support variable rank in RandomCrop.

// --------------------------------------------------------------------------
REGISTER_OP("DecodeJpeg")
    .Input("contents: string")
    .Attr("channels: int = 0")
    .Attr("ratio: int = 1")
    .Attr("fancy_upscaling: bool = true")
    .Attr("try_recover_truncated: bool = false")
    .Attr("acceptable_fraction: float = 1.0")
    .Output("image: uint8")
    .Doc(R"doc(
Decode a JPEG-encoded image to a uint8 tensor.

The attr `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

*   0: Use the number of channels in the JPEG-encoded image.
*   1: output a grayscale image.
*   3: output an RGB image.

If needed, the JPEG-encoded image is transformed to match the requested number
of color channels.

The attr `ratio` allows downscaling the image by an integer factor during
decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
downscaling the image later.

contents: 0-D.  The JPEG-encoded image.
channels: Number of color channels for the decoded image.
ratio: Downscaling ratio.
fancy_upscaling: If true use a slower but nicer upscaling of the
  chroma planes (yuv420/422 only).
try_recover_truncated:  If true try to recover an image from truncated input.
acceptable_fraction: The minimum required fraction of lines before a truncated
  input is accepted.
image: 3-D with shape `[height, width, channels]`..
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("EncodeJpeg")
    .Input("image: uint8")
    .Attr("format: {'', 'grayscale', 'rgb'} = ''")
    .Attr("quality: int = 95")
    .Attr("progressive: bool = false")
    .Attr("optimize_size: bool = false")
    .Attr("chroma_downsampling: bool = true")
    .Attr("density_unit: {'in', 'cm'} = 'in'")
    .Attr("x_density: int = 300")
    .Attr("y_density: int = 300")
    .Attr("xmp_metadata: string = ''")
    .Output("contents: string")
    .Doc(R"doc(
JPEG-encode an image.

`image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.

The attr `format` can be used to override the color format of the encoded
output.  Values can be:

*   `''`: Use a default format based on the number of channels in the image.
*   `grayscale`: Output a grayscale JPEG image.  The `channels` dimension
    of `image` must be 1.
*   `rgb`: Output an RGB JPEG image. The `channels` dimension
    of `image` must be 3.

If `format` is not specified or is the empty string, a default format is picked
in function of the number of channels in `image`:

*   1: Output a grayscale image.
*   3: Output an RGB image.

image: 3-D with shape `[height, width, channels]`.
format: Per pixel image format.
quality: Quality of the compression from 0 to 100 (higher is better and slower).
progressive: If True, create a JPEG that loads progressively (coarse to fine).
optimize_size: If True, spend CPU/RAM to reduce size with no quality change.
chroma_downsampling: See http://en.wikipedia.org/wiki/Chroma_subsampling.
density_unit: Unit used to specify `x_density` and `y_density`:
   pixels per inch (`'in'`) or centimeter (`'cm'`).
x_density: Horizontal pixels per density unit.
y_density: Vertical pixels per density unit.
xmp_metadata: If not empty, embed this XMP metadata in the image header.
contents: 0-D. JPEG-encoded image.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("AdjustContrast")
    .Input("images: T")
    .Input("contrast_factor: float")
    .Input("min_value: float")
    .Input("max_value: float")
    .Output("output: float")
    .Attr("T: {uint8, int8, int16, int32, int64, float, double}")
    .Deprecated(2, "Use AdjustContrastv2 instead")
    .Doc(R"Doc(
Deprecated. Disallowed in GraphDef version >= 2.
)Doc");

// --------------------------------------------------------------------------
REGISTER_OP("AdjustContrastv2")
    .Input("images: float")
    .Input("contrast_factor: float")
    .Output("output: float")
    .Doc(R"Doc(
Adjust the contrast of one or more images.

`images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
interpreted as `[height, width, channels]`.  The other dimensions only
represent a collection of images, such as `[batch, height, width, channels].`

Contrast is adjusted independently for each channel of each image.

For each channel, the Op first computes the mean of the image pixels in the
channel and then adjusts each component of each pixel to
`(x - mean) * contrast_factor + mean`.

images: Images to adjust.  At least 3-D.
contrast_factor: A float multiplier for adjusting contrast.
output: The contrast-adjusted image or images.
)Doc");

// --------------------------------------------------------------------------
REGISTER_OP("DecodePng")
    .Input("contents: string")
    .Attr("channels: int = 0")
    .Attr("dtype: {uint8, uint16} = DT_UINT8")
    .Output("image: dtype")
    .Doc(R"doc(
Decode a PNG-encoded image to a uint8 or uint16 tensor.

The attr `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

*   0: Use the number of channels in the PNG-encoded image.
*   1: output a grayscale image.
*   3: output an RGB image.
*   4: output an RGBA image.

If needed, the PNG-encoded image is transformed to match the requested number
of color channels.

contents: 0-D.  The PNG-encoded image.
channels: Number of color channels for the decoded image.
image: 3-D with shape `[height, width, channels]`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("EncodePng")
    .Attr("compression: int = -1")
    .Attr("T: {uint8, uint16} = DT_UINT8")
    .Input("image: T")
    .Output("contents: string")
    .Doc(R"doc(
PNG-encode an image.

`image` is a 3-D uint8 or uint16 Tensor of shape `[height, width, channels]`
where `channels` is:

*   1: for grayscale.
*   2: for grayscale + alpha.
*   3: for RGB.
*   4: for RGBA.

The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
default or a value from 0 to 9.  9 is the highest compression level, generating
the smallest output, but is slower.

image: 3-D with shape `[height, width, channels]`.
compression: Compression level.
contents: 0-D. PNG-encoded image.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("RGBToHSV")
    .Input("images: float")
    .Output("output: float")
    .Doc(R"doc(
Converts one or more images from RGB to HSV.

Outputs a tensor of the same shape as the `images` tensor, containing the HSV
value of the pixels. The output is only well defined if the value in `images`
are in `[0,1]`.

`output[..., 0]` contains hue, `output[..., 1]` contains saturation, and
`output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0
corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue.

images: 1-D or higher rank. RGB data to convert. Last dimension must be size 3.
output: `images` converted to HSV.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("HSVToRGB")
    .Input("images: float")
    .Output("output: float")
    .Doc(R"doc(
Convert one or more images from HSV to RGB.

Outputs a tensor of the same shape as the `images` tensor, containing the RGB
value of the pixels. The output is only well defined if the value in `images`
are in `[0,1]`.

See `rgb_to_hsv` for a description of the HSV encoding.

images: 1-D or higher rank. HSV data to convert. Last dimension must be size 3.
output: `images` converted to RGB.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("DrawBoundingBoxes")
    .Input("images: float")
    .Input("boxes: float")
    .Output("output: float")
    .Doc(R"doc(
Draw bounding boxes on a batch of images.

Outputs a copy of `images` but draws on top of the pixels zero or more bounding
boxes specified by the locations in `boxes`. The coordinates of the each
bounding box in `boxes are encoded as `[y_min, x_min, y_max, x_max]`. The
bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
height of the underlying image.

For example, if an image is 100 x 200 pixels and the bounding box is
`[0.1, 0.5, 0.2, 0.9]`, the bottom-left and upper-right coordinates of the
bounding box will be `(10, 40)` to `(50, 180)`.

Parts of the bounding box may fall outside the image.

images: 4-D with shape `[batch, height, width, depth]`. A batch of images.
boxes: 3-D with shape `[batch, num_bounding_boxes, 4]` containing bounding
  boxes.
output: 4-D with the same shape as `images`. The batch of input images with
  bounding boxes drawn on the images.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("SampleDistortedBoundingBox")
    .Input("image_size: T")
    .Input("bounding_boxes: float")
    .Output("begin: T")
    .Output("size: T")
    .Output("bboxes: float")
    .Attr("T: {uint8, int8, int16, int32, int64}")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("min_object_covered: float = 0.1")
    .Attr("aspect_ratio_range: list(float) = [0.75, 1.33]")
    .Attr("area_range: list(float) = [0.05, 1.0]")
    .Attr("max_attempts: int = 100")
    .Attr("use_image_if_no_bounding_boxes: bool = false")
    .SetIsStateful()
    .Doc(R"doc(
Generate a single randomly distorted bounding box for an image.

Bounding box annotations are often supplied in addition to ground-truth labels
in image recognition or object localization tasks. A common technique for
training such a system is to randomly distort an image while preserving
its content, i.e. *data augmentation*. This Op outputs a randomly distorted
localization of an object, i.e. bounding box, given an `image_size`,
`bounding_boxes` and a series of constraints.

The output of this Op is a single bounding box that may be used to crop the
original image. The output is returned as 3 tensors: `begin`, `size` and
`bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
image. The latter may be supplied to `tf.image.draw_bounding_box` to visualize
what the bounding box looks like.

Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
height of the underlying image.

For example,

    # Generate a single distorted bounding box.
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bounding_boxes)

    # Draw the bounding box in an image summary.
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                  bbox_for_draw)
    tf.image_summary('images_with_box', image_with_box)

    # Employ the bounding box to distort the image.
    distorted_image = tf.slice(image, begin, size)

Note that if no bounding box information is available, setting
`use_image_if_no_bounding_boxes = true` will assume there is a single implicit
bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
false and no bounding boxes are supplied, an error is raised.

image_size: 1-D, containing `[height, width, channels]`.
bounding_boxes: 3-D with shape `[batch, N, 4]` describing the N bounding boxes
  associated with the image.
begin: 1-D, containing `[offset_height, offset_width, 0]`. Provide as input to
  `tf.slice`.
size: 1-D, containing `[target_height, target_width, -1]`. Provide as input to
  `tf.slice`.
bboxes: 3-D with shape `[1, 1, 4]` containing the distorted bounding box.
  Provide as input to `tf.image.draw_bounding_boxes`.
seed: If either `seed` or `seed2` are set to non-zero, the random number
  generator is seeded by the given `seed`.  Otherwise, it is seeded by a random
  seed.
seed2: A second seed to avoid seed collision.
min_object_covered: The cropped area of the image must contain at least this
  fraction of any bounding box supplied.
aspect_ratio_range: The cropped area of the image must have an aspect ratio =
  width / height within this range.
area_range: The cropped area of the image must contain a fraction of the
  supplied image within in this range.
max_attempts: Number of attempts at generating a cropped region of the image
  of the specified constraints. After `max_attempts` failures, return the entire
  image.
use_image_if_no_bounding_boxes: Controls behavior if no bounding boxes supplied.
  If true, assume an implicit bounding box covering the whole input. If false,
  raise an error.
)doc");


// --------------------------------------------------------------------------

// glimpse = extract_glimpse(input, size, offsets) extract the glimpse
// of size `size` centered at location `offsets` from the input tensor
// `input`.
//
// REQUIRES: input.dims() == 4
//
REGISTER_OP("ExtractGlimpse")
    .Input("input: float")
    .Input("size: int32")
    .Input("offsets: float")
    .Output("glimpse: float")
    .Attr("centered: bool = true")
    .Attr("normalized: bool = true")
    .Attr("uniform_noise: bool = true")
    .Doc(R"doc(
Extracts a glimpse from the input tensor.

Returns a set of windows called glimpses extracted at location
`offsets` from the input tensor. If the windows only partially
overlaps the inputs, the non overlapping areas will be filled with
random noise.

The result is a 4-D tensor of shape `[batch_size, glimpse_height,
glimpse_width, channels]`. The channels and batch dimensions are the
same as that of the input tensor. The height and width of the output
windows are specified in the `size` parameter.

The argument `normalized` and `centered` controls how the windows are
built:

* If the coordinates are normalized but not centered, 0.0 and 1.0
  correspond to the minimum and maximum of each height and width
  dimension.
* If the coordinates are both normalized and centered, they range from
  -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper
  left corner, the lower right corner is located at (1.0, 1.0) and the
  center is at (0, 0).
* If the coordinates are not normalized they are interpreted as
  numbers of pixels.

input: A 4-D float tensor of shape `[batch_size, height, width, channels]`.
size: A 1-D tensor of 2 elements containing the size of the glimpses
  to extract.  The glimpse height must be specified first, following
  by the glimpse width.
offsets: A 2-D integer tensor of shape `[batch_size, 2]` containing
  the x, y locations of the center of each window.
glimpse: A tensor representing the glimpses `[batch_size,
  glimpse_height, glimpse_width, channels]`.
centered: indicates if the offset coordinates are centered relative to
  the image, in which case the (0, 0) offset is relative to the center
  of the input images. If false, the (0,0) offset corresponds to the
  upper left corner of the input images.
normalized: indicates if the offset coordinates are normalized.
uniform_noise: indicates if the noise should be generated using a
  uniform distribution or a gaussian distribution.
)doc");

}  // namespace tensorflow
