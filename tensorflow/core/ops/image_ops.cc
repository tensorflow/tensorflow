#include "tensorflow/core/framework/op.h"

namespace tensorflow {

// --------------------------------------------------------------------------
REGISTER_OP("ResizeArea")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: float")
    .Attr("T: {uint8, int8, int32, float, double}")
    .Doc(R"doc(
Resize `images` to `size` using area interpolation.

Input images can be of different types but output images are always float.

images: 4-D with shape `[batch, height, width, channels]`.
size:= A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
  new size for the images.
resized_images:  4-D with shape
  `[batch, new_height, new_width, channels]`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("ResizeBicubic")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: float")
    .Attr("T: {uint8, int8, int32, float, double}")
    .Doc(R"doc(
Resize `images` to `size` using bicubic interpolation.

Input images can be of different types but output images are always float.

images: 4-D with shape `[batch, height, width, channels]`.
size:= A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
  new size for the images.
resized_images:  4-D with shape
  `[batch, new_height, new_width, channels]`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("ResizeBilinear")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: float")
    .Attr("T: {uint8, int8, int32, float, double}")
    .Doc(R"doc(
Resize `images` to `size` using bilinear interpolation.

Input images can be of different types but output images are always float.

images: 4-D with shape `[batch, height, width, channels]`.
size:= A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
  new size for the images.
resized_images:  4-D with shape
  `[batch, new_height, new_width, channels]`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("ResizeNearestNeighbor")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: T")
    .Attr("T: {uint8, int8, int32, float, double}")
    .Doc(R"doc(
Resize `images` to `size` using nearest neighbor interpolation.

Input images can be of different types but output images are always float.

images: 4-D with shape `[batch, height, width, channels]`.
size:= A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
  new size for the images.
resized_images:  4-D with shape
  `[batch, new_height, new_width, channels]`.
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
    .Doc(R"Doc(
Adjust the contrast of one or more images.

`images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
interpreted as `[height, width, channels]`.  The other dimensions only
represent a collection of images, such as `[batch, height, width, channels].`

Contrast is adjusted independently for each channel of each image.

For each channel, the Op first computes the mean of the image pixels in the
channel and then adjusts each component of each pixel to
`(x - mean) * contrast_factor + mean`.

These adjusted values are then clipped to fit in the `[min_value, max_value]`
interval.

`images: Images to adjust.  At least 3-D.
contrast_factor: A float multiplier for adjusting contrast.
min_value: Minimum value for clipping the adjusted pixels.
max_value: Maximum value for clipping the adjusted pixels.
output: The constrast-adjusted image or images.
)Doc");

// --------------------------------------------------------------------------
REGISTER_OP("DecodePng")
    .Input("contents: string")
    .Attr("channels: int = 0")
    .Output("image: uint8")
    .Doc(R"doc(
Decode a PNG-encoded image to a uint8 tensor.

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
    .Input("image: uint8")
    .Attr("compression: int = -1")
    .Output("contents: string")
    .Doc(R"doc(
PNG-encode an image.

`image` is a 3-D uint8 Tensor of shape `[height, width, channels]` where
`channels` is:

*   1: for grayscale.
*   3: for RGB.
*   4: for RGBA.

The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
default or a value from 0 to 9.  9 is the highest compression level, generating
the smallest output, but is slower.

image: 3-D with shape `[height, width, channels]`.
compression: Compression level.
contents: 0-D. PNG-encoded image.
)doc");

}  // namespace tensorflow
