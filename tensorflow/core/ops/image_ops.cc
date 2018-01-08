/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

const char kDecodeJpegCommonDocStr[] = R"doc(
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

)doc";

const char kDecodeJpegCommonParamsDocStr[] = R"doc(
channels: Number of color channels for the decoded image.
ratio: Downscaling ratio.
fancy_upscaling: If true use a slower but nicer upscaling of the
  chroma planes (yuv420/422 only).
try_recover_truncated:  If true try to recover an image from truncated input.
acceptable_fraction: The minimum required fraction of lines before a truncated
  input is accepted.
dct_method: string specifying a hint about the algorithm used for
  decompression.  Defaults to "" which maps to a system-specific
  default.  Currently valid values are ["INTEGER_FAST",
  "INTEGER_ACCURATE"].  The hint may be ignored (e.g., the internal
  jpeg library changes to a version that does not have that specific
  option.)
image: 3-D with shape `[height, width, channels]`..
)doc";

// Sets output[0] to shape [batch_dim,height,width,channel_dim], where
// height and width come from the size_tensor.
Status SetOutputToSizedImage(InferenceContext* c, DimensionHandle batch_dim,
                             int size_input_idx, DimensionHandle channel_dim) {
  // Verify shape of size input.
  ShapeHandle size;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(size_input_idx), 1, &size));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(size, 0), 2, &unused));

  // Get size values from the size tensor.
  const Tensor* size_tensor = c->input_tensor(size_input_idx);
  DimensionHandle width;
  DimensionHandle height;
  if (size_tensor == nullptr) {
    width = c->UnknownDim();
    height = c->UnknownDim();
  } else {
    // TODO(petewarden) - Remove once we have constant evaluation in C++ only.
    if (size_tensor->dtype() != DT_INT32) {
      return errors::InvalidArgument(
          "Bad size input type for SetOutputToSizedImage: Expected DT_INT32 "
          "but got ",
          DataTypeString(size_tensor->dtype()), " for input #", size_input_idx,
          " in ", c->DebugString());
    }
    auto vec = size_tensor->vec<int32>();
    height = c->MakeDim(vec(0));
    width = c->MakeDim(vec(1));
  }
  c->set_output(0, c->MakeShape({batch_dim, height, width, channel_dim}));
  return Status::OK();
}

Status ResizeShapeFn(InferenceContext* c) {
  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
  return SetOutputToSizedImage(c, c->Dim(input, 0), 1 /* size_input_idx */,
                               c->Dim(input, 3));
}

Status DecodeImageShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
  DimensionHandle channels_dim;
  int32 channels;
  TF_RETURN_IF_ERROR(c->GetAttr("channels", &channels));
  if (channels == 0) {
    channels_dim = c->UnknownDim();
  } else {
    if (channels < 0) {
      return errors::InvalidArgument("channels must be non-negative, got ",
                                     channels);
    }
    channels_dim = c->MakeDim(channels);
  }

  c->set_output(0, c->MakeShape({InferenceContext::kUnknownDim,
                                 InferenceContext::kUnknownDim, channels_dim}));
  return Status::OK();
}

Status EncodeImageShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &unused));
  c->set_output(0, c->Scalar());
  return Status::OK();
}

Status ColorspaceShapeFn(InferenceContext* c) {
  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &input));

  // The last dimension value is always 3.
  DimensionHandle last_dim;
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(input, -1), 3, &last_dim));
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->ReplaceDim(input, -1, last_dim, &out));
  c->set_output(0, out);

  return Status::OK();
}

}  // namespace

// --------------------------------------------------------------------------
REGISTER_OP("ResizeArea")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: float")
    .Attr("T: {int8, uint8, int16, uint16, int32, int64, half, float, double}")
    .Attr("align_corners: bool = false")
    .SetShapeFn(ResizeShapeFn)
    .Doc(R"doc(
Resize `images` to `size` using area interpolation.

Input images can be of different types but output images are always float.

Each output pixel is computed by first transforming the pixel's footprint into
the input tensor and then averaging the pixels that intersect the footprint. An
input pixel's contribution to the average is weighted by the fraction of its
area that intersects the footprint.  This is the same as OpenCV's INTER_AREA.

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
    .Attr("T: {int8, uint8, int16, uint16, int32, int64, half, float, double}")
    .Attr("align_corners: bool = false")
    .SetShapeFn(ResizeShapeFn)
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
REGISTER_OP("ResizeBicubicGrad")
    .Input("grads: float")
    .Input("original_image: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("align_corners: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    })
    .Doc(R"doc(
Computes the gradient of bicubic interpolation.

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
REGISTER_OP("ResizeBilinear")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: float")
    .Attr("T: {int8, uint8, int16, uint16, int32, int64, half, float, double}")
    .Attr("align_corners: bool = false")
    .SetShapeFn(ResizeShapeFn)
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
REGISTER_OP("QuantizedResizeBilinear")
    .Input("images: T")
    .Input("size: int32")
    .Input("min: float")
    .Input("max: float")
    .Output("resized_images: T")
    .Output("out_min: float")
    .Output("out_max: float")
    .Attr("T: {quint8, qint32, float}")
    .Attr("align_corners: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(ResizeShapeFn(c));
      ShapeHandle min_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &min_shape));
      ShapeHandle max_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &max_shape));
      c->set_output(1, c->MakeShape({}));
      c->set_output(2, c->MakeShape({}));
      return Status::OK();
    })
    .Doc(R"doc(
Resize quantized `images` to `size` using quantized bilinear interpolation.

Input images and output images must be quantized types.

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
    .Attr("T: {float, half, double}")
    .Attr("align_corners: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    })
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
    .Attr("T: {int8, uint8, int16, uint16, int32, int64, half, float, double}")
    .Attr("align_corners: bool = false")
    .SetShapeFn(ResizeShapeFn)
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
    .Attr("T: {uint8, int8, int32, half, float, double}")
    .Attr("align_corners: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
      ShapeHandle unused;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(unused, 0), 2, &unused_dim));
      const Tensor* size = c->input_tensor(1);
      if (size == nullptr) {
        TF_RETURN_IF_ERROR(c->ReplaceDim(input, 1, c->UnknownDim(), &input));
        TF_RETURN_IF_ERROR(c->ReplaceDim(input, 2, c->UnknownDim(), &input));
      } else {
        auto size_vec = size->vec<int32>();
        TF_RETURN_IF_ERROR(
            c->ReplaceDim(input, 1, c->MakeDim(size_vec(0)), &input));
        TF_RETURN_IF_ERROR(
            c->ReplaceDim(input, 2, c->MakeDim(size_vec(1)), &input));
      }
      c->set_output(0, input);
      return Status::OK();
    })
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
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle image;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &image));
      DimensionHandle channels = c->Dim(image, -1);

      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->Merge(c->input(1), c->Vector(2), &unused));

      const Tensor* size = c->input_tensor(1);
      DimensionHandle h;
      DimensionHandle w;
      if (size == nullptr) {
        h = c->UnknownDim();
        w = c->UnknownDim();
      } else {
        auto size_vec = size->vec<int64>();
        h = c->MakeDim(size_vec(0));
        w = c->MakeDim(size_vec(1));
      }
      c->set_output(0, c->MakeShape({h, w, channels}));
      return Status::OK();
    })
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
    .Attr("dct_method: string = ''")
    .Output("image: uint8")
    .SetShapeFn(DecodeImageShapeFn)
    .Doc(strings::StrCat(R"doc(
Decode a JPEG-encoded image to a uint8 tensor.
)doc",
                         kDecodeJpegCommonDocStr, R"doc(
This op also supports decoding PNGs and non-animated GIFs since the interface is
the same, though it is cleaner to use `tf.image.decode_image`.

contents: 0-D.  The JPEG-encoded image.
)doc",
                         kDecodeJpegCommonParamsDocStr));

// --------------------------------------------------------------------------
REGISTER_OP("DecodeAndCropJpeg")
    .Input("contents: string")
    .Input("crop_window: int32")
    .Attr("channels: int = 0")
    .Attr("ratio: int = 1")
    .Attr("fancy_upscaling: bool = true")
    .Attr("try_recover_truncated: bool = false")
    .Attr("acceptable_fraction: float = 1.0")
    .Attr("dct_method: string = ''")
    .Output("image: uint8")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      DimensionHandle channels_dim = c->UnknownDim();
      DimensionHandle h = c->UnknownDim();
      DimensionHandle w = c->UnknownDim();

      int32 channels;
      TF_RETURN_IF_ERROR(c->GetAttr("channels", &channels));
      if (channels != 0) {
        if (channels < 0) {
          return errors::InvalidArgument("channels must be non-negative, got ",
                                         channels);
        }
        channels_dim = c->MakeDim(channels);
      }

      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(unused, 0), 4, &unused_dim));

      const Tensor* crop_window = c->input_tensor(1);
      if (crop_window != nullptr) {
        auto crop_window_vec = crop_window->vec<int32>();
        h = c->MakeDim(crop_window_vec(2));
        w = c->MakeDim(crop_window_vec(3));
      }
      c->set_output(0, c->MakeShape({h, w, channels_dim}));
      return Status::OK();
    })
    .Doc(strings::StrCat(R"doc(
Decode and Crop a JPEG-encoded image to a uint8 tensor.
)doc",
                         kDecodeJpegCommonDocStr, R"doc(
It is equivalent to a combination of decode and crop, but much faster by only
decoding partial jpeg image.

contents: 0-D.  The JPEG-encoded image.
crop_window: 1-D.  The crop window: [crop_y, crop_x, crop_height, crop_width].
)doc",
                         kDecodeJpegCommonParamsDocStr));

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
    .SetShapeFn(EncodeImageShapeFn)
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
REGISTER_OP("ExtractJpegShape")
    .Input("contents: string")
    .Output("image_shape: output_type")
    .Attr("output_type: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(0, c->Vector(3));
      return Status::OK();
    })
    .Doc(R"doc(
Extract the shape information of a JPEG-encoded image.

This op only parses the image header, so it is much faster than DecodeJpeg.

contents: 0-D. The JPEG-encoded image.
image_shape: 1-D. The image shape with format [height, width, channels].
output_type: (Optional) The output type of the operation (int32 or int64).
    Defaults to int32.
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
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    })
    .Doc(R"Doc(
Deprecated. Disallowed in GraphDef version >= 2.
)Doc");

// --------------------------------------------------------------------------
REGISTER_OP("AdjustContrastv2")
    .Input("images: float")
    .Input("contrast_factor: float")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    })
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
REGISTER_OP("AdjustHue")
    .Input("images: float")
    .Input("delta: float")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    })
    .Doc(R"Doc(
Adjust the hue of one or more images.

`images` is a tensor of at least 3 dimensions.  The last dimension is
interpretted as channels, and must be three.

The input image is considered in the RGB colorspace. Conceptually, the RGB
colors are first mapped into HSV. A delta is then applied all the hue values,
and then remapped back to RGB colorspace.

images: Images to adjust.  At least 3-D.
delta: A float delta to add to the hue.
output: The hue-adjusted image or images.
)Doc");

// --------------------------------------------------------------------------
REGISTER_OP("AdjustSaturation")
    .Input("images: float")
    .Input("scale: float")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    })
    .Doc(R"Doc(
Adjust the saturation of one or more images.

`images` is a tensor of at least 3 dimensions.  The last dimension is
interpretted as channels, and must be three.

The input image is considered in the RGB colorspace. Conceptually, the RGB
colors are first mapped into HSV. A scale is then applied all the saturation
values, and then remapped back to RGB colorspace.

images: Images to adjust.  At least 3-D.
scale: A float scale to add to the saturation.
output: The hue-adjusted image or images.
)Doc");

// --------------------------------------------------------------------------
REGISTER_OP("DecodePng")
    .Input("contents: string")
    .Attr("channels: int = 0")
    .Attr("dtype: {uint8, uint16} = DT_UINT8")
    .Output("image: dtype")
    .SetShapeFn(DecodeImageShapeFn)
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

This op also supports decoding JPEGs and non-animated GIFs since the interface
is the same, though it is cleaner to use `tf.image.decode_image`.

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
    .SetShapeFn(EncodeImageShapeFn)
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
REGISTER_OP("DecodeBmp")
    .Input("contents: string")
    .Output("image: uint8")
    .Attr("channels: int = 0")
    .SetShapeFn(DecodeImageShapeFn)
    .Doc(R"doc(
Decode the first frame of a BMP-encoded image to a uint8 tensor.

The attr `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

*   0: Use the number of channels in the BMP-encoded image.
*   3: output an RGB image.
*   4: output an RGBA image.

contents: 0-D.  The BMP-encoded image.
image: 3-D with shape `[height, width, channels]`. RGB order
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("DecodeGif")
    .Input("contents: string")
    .Output("image: uint8")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(0, c->MakeShape({InferenceContext::kUnknownDim,
                                     InferenceContext::kUnknownDim,
                                     InferenceContext::kUnknownDim, 3}));
      return Status::OK();
    })
    .Doc(R"doc(
Decode the first frame of a GIF-encoded image to a uint8 tensor.

GIF with frame or transparency compression are not supported
convert animated GIF from compressed to uncompressed by:

    convert $src.gif -coalesce $dst.gif

This op also supports decoding JPEGs and PNGs, though it is cleaner to use
`tf.image.decode_image`.

contents: 0-D.  The GIF-encoded image.
image: 4-D with shape `[num_frames, height, width, 3]`. RGB order
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("RGBToHSV")
    .Input("images: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double} = DT_FLOAT")
    .SetShapeFn(ColorspaceShapeFn)
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
    .Input("images: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double} = DT_FLOAT")
    .SetShapeFn(ColorspaceShapeFn)
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
    .Input("images: T")
    .Input("boxes: float")
    .Output("output: T")
    .Attr("T: {float, half} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    })
    .Doc(R"doc(
Draw bounding boxes on a batch of images.

Outputs a copy of `images` but draws on top of the pixels zero or more bounding
boxes specified by the locations in `boxes`. The coordinates of the each
bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`. The
bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
height of the underlying image.

For example, if an image is 100 x 200 pixels (height x width) and the bounding
box is `[0.1, 0.2, 0.5, 0.9]`, the upper-left and bottom-right coordinates of
the bounding box will be `(40, 10)` to `(100, 50)` (in (x,y) coordinates).

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
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(3));
      c->set_output(1, c->Vector(3));
      c->set_output(2, c->MakeShape({1, 1, 4}));
      return Status::OK();
    })
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
image. The latter may be supplied to `tf.image.draw_bounding_boxes` to visualize
what the bounding box looks like.

Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
height of the underlying image.

For example,

```python
    # Generate a single distorted bounding box.
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bounding_boxes)

    # Draw the bounding box in an image summary.
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                  bbox_for_draw)
    tf.summary.image('images_with_box', image_with_box)

    # Employ the bounding box to distort the image.
    distorted_image = tf.slice(image, begin, size)
```

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
  fraction of any bounding box supplied. The value of this parameter should be
  non-negative. In the case of 0, the cropped area does not need to overlap
  any of the bounding boxes supplied.
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

REGISTER_OP("SampleDistortedBoundingBoxV2")
    .Input("image_size: T")
    .Input("bounding_boxes: float")
    .Input("min_object_covered: float")
    .Output("begin: T")
    .Output("size: T")
    .Output("bboxes: float")
    .Attr("T: {uint8, int8, int16, int32, int64}")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("aspect_ratio_range: list(float) = [0.75, 1.33]")
    .Attr("area_range: list(float) = [0.05, 1.0]")
    .Attr("max_attempts: int = 100")
    .Attr("use_image_if_no_bounding_boxes: bool = false")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(3));
      c->set_output(1, c->Vector(3));
      c->set_output(2, c->MakeShape({1, 1, 4}));
      return Status::OK();
    })
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
image. The latter may be supplied to `tf.image.draw_bounding_boxes` to visualize
what the bounding box looks like.

Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
height of the underlying image.

For example,

```python
    # Generate a single distorted bounding box.
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bounding_boxes)

    # Draw the bounding box in an image summary.
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                  bbox_for_draw)
    tf.summary.image('images_with_box', image_with_box)

    # Employ the bounding box to distort the image.
    distorted_image = tf.slice(image, begin, size)
```

Note that if no bounding box information is available, setting
`use_image_if_no_bounding_boxes = true` will assume there is a single implicit
bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
false and no bounding boxes are supplied, an error is raised.

image_size: 1-D, containing `[height, width, channels]`.
bounding_boxes: 3-D with shape `[batch, N, 4]` describing the N bounding boxes
  associated with the image.
min_object_covered: The cropped area of the image must contain at least this
  fraction of any bounding box supplied. The value of this parameter should be
  non-negative. In the case of 0, the cropped area does not need to overlap
  any of the bounding boxes supplied.
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
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
      ShapeHandle offsets;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &offsets));

      DimensionHandle batch_dim;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(input, 0), c->Dim(offsets, 0), &batch_dim));
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(offsets, 1), 2, &unused));

      return SetOutputToSizedImage(c, batch_dim, 1 /* size_input_idx */,
                                   c->Dim(input, 3));
    })
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

The argument `normalized` and `centered` controls how the windows are built:

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
  the y, x locations of the center of each window.
glimpse: A tensor representing the glimpses `[batch_size,
  glimpse_height, glimpse_width, channels]`.
centered: indicates if the offset coordinates are centered relative to
  the image, in which case the (0, 0) offset is relative to the center
  of the input images. If false, the (0,0) offset corresponds to the
  upper left corner of the input images.
normalized: indicates if the offset coordinates are normalized.
uniform_noise: indicates if the noise should be generated using a
  uniform distribution or a Gaussian distribution.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("CropAndResize")
    .Input("image: T")
    .Input("boxes: float")
    .Input("box_ind: int32")
    .Input("crop_size: int32")
    .Output("crops: float")
    .Attr("T: {uint8, uint16, int8, int16, int32, int64, half, float, double}")
    .Attr("method: {'bilinear'} = 'bilinear'")
    .Attr("extrapolation_value: float = 0")
    .SetShapeFn([](InferenceContext* c) {
      // Get inputs and validate ranks.
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
      ShapeHandle boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &boxes));
      ShapeHandle box_ind;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &box_ind));

      // boxes[0] and box_ind[0] are both num_boxes.
      DimensionHandle num_boxes_dim;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(boxes, 0), c->Dim(box_ind, 0), &num_boxes_dim));

      // boxes.dim(1) is 4.
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 1), 4, &unused));

      return SetOutputToSizedImage(c, num_boxes_dim, 3 /* size_input_idx */,
                                   c->Dim(input, 3));
    })
    .Doc(R"doc(
Extracts crops from the input image tensor and bilinearly resizes them (possibly
with aspect ratio change) to a common output size specified by `crop_size`. This
is more general than the `crop_to_bounding_box` op which extracts a fixed size
slice from the input image and does not allow resizing or aspect ratio change.

Returns a tensor with `crops` from the input `image` at positions defined at the
bounding box locations in `boxes`. The cropped boxes are all resized (with
bilinear interpolation) to a fixed `size = [crop_height, crop_width]`. The
result is a 4-D tensor `[num_boxes, crop_height, crop_width, depth]`. The
resizing is corner aligned. In particular, if `boxes = [[0, 0, 1, 1]]`, the
method will give identical results to using `tf.image.resize_bilinear()`
with `align_corners=True`.

image: A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
  Both `image_height` and `image_width` need to be positive.
boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
  specifies the coordinates of a box in the `box_ind[i]` image and is specified
  in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of
  `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the
  `[0, 1]` interval of normalized image height is mapped to
  `[0, image_height - 1]` in image height coordinates. We do allow `y1` > `y2`, in
  which case the sampled crop is an up-down flipped version of the original
  image. The width dimension is treated similarly. Normalized coordinates
  outside the `[0, 1]` range are allowed, in which case we use
  `extrapolation_value` to extrapolate the input image values.
box_ind: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
  The value of `box_ind[i]` specifies the image that the `i`-th box refers to.
crop_size: A 1-D tensor of 2 elements, `size = [crop_height, crop_width]`. All
  cropped image patches are resized to this size. The aspect ratio of the image
  content is not preserved. Both `crop_height` and `crop_width` need to be
  positive.
crops: A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.
method: A string specifying the interpolation method. Only 'bilinear' is
  supported for now.
extrapolation_value: Value used for extrapolation, when applicable.
)doc");

REGISTER_OP("CropAndResizeGradImage")
    .Input("grads: float")
    .Input("boxes: float")
    .Input("box_ind: int32")
    .Input("image_size: int32")
    .Output("output: T")
    .Attr("T: {float, half, double}")
    .Attr("method: {'bilinear'} = 'bilinear'")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(3, &out));
      TF_RETURN_IF_ERROR(c->WithRank(out, 4, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Computes the gradient of the crop_and_resize op wrt the input image tensor.

grads: A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.
boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
  specifies the coordinates of a box in the `box_ind[i]` image and is specified
  in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of
  `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the
  `[0, 1]` interval of normalized image height is mapped to
  `[0, image_height - 1] in image height coordinates. We do allow y1 > y2, in
  which case the sampled crop is an up-down flipped version of the original
  image. The width dimension is treated similarly. Normalized coordinates
  outside the `[0, 1]` range are allowed, in which case we use
  `extrapolation_value` to extrapolate the input image values.
box_ind: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
  The value of `box_ind[i]` specifies the image that the `i`-th box refers to.
image_size: A 1-D tensor with value `[batch, image_height, image_width, depth]`
  containing the original image size. Both `image_height` and `image_width` need
  to be positive.
output: A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
method: A string specifying the interpolation method. Only 'bilinear' is
  supported for now.
)doc");

REGISTER_OP("CropAndResizeGradBoxes")
    .Input("grads: float")
    .Input("image: T")
    .Input("boxes: float")
    .Input("box_ind: int32")
    .Output("output: float")
    .Attr("T: {uint8, uint16, int8, int16, int32, int64, half, float, double}")
    .Attr("method: {'bilinear'} = 'bilinear'")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(2));
      return Status::OK();
    })
    .Doc(R"doc(
Computes the gradient of the crop_and_resize op wrt the input boxes tensor.

grads: A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.
image: A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
  Both `image_height` and `image_width` need to be positive.
boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
  specifies the coordinates of a box in the `box_ind[i]` image and is specified
  in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of
  `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the
  `[0, 1]` interval of normalized image height is mapped to
  `[0, image_height - 1] in image height coordinates. We do allow y1 > y2, in
  which case the sampled crop is an up-down flipped version of the original
  image. The width dimension is treated similarly. Normalized coordinates
  outside the `[0, 1]` range are allowed, in which case we use
  `extrapolation_value` to extrapolate the input image values.
box_ind: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
  The value of `box_ind[i]` specifies the image that the `i`-th box refers to.
output: A 2-D tensor of shape `[num_boxes, 4]`.
method: A string specifying the interpolation method. Only 'bilinear' is
  supported for now.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("NonMaxSuppression")
    .Input("boxes: float")
    .Input("scores: float")
    .Input("max_output_size: int32")
    .Output("selected_indices: int32")
    .Attr("iou_threshold: float = 0.5")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(c->UnknownDim()));
      return Status::OK();
    })
    .Doc(R"doc(
Greedily selects a subset of bounding boxes in descending order of score,
pruning away boxes that have high intersection-over-union (IOU) overlap
with previously selected boxes.  Bounding boxes are supplied as
[y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
diagonal pair of box corners and the coordinates can be provided as normalized
(i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
is agnostic to where the origin is in the coordinate system.  Note that this
algorithm is invariant to orthogonal transformations and translations
of the coordinate system; thus translating or reflections of the coordinate
system result in the same boxes being selected by the algorithm.
The output of this operation is a set of integers indexing into the input
collection of bounding boxes representing the selected boxes.  The bounding
box coordinates corresponding to the selected indices can then be obtained
using the `tf.gather operation`.  For example:
  selected_indices = tf.image.non_max_suppression(
      boxes, scores, max_output_size, iou_threshold)
  selected_boxes = tf.gather(boxes, selected_indices)
boxes: A 2-D float tensor of shape `[num_boxes, 4]`.
scores: A 1-D float tensor of shape `[num_boxes]` representing a single
  score corresponding to each box (each row of boxes).
max_output_size: A scalar integer tensor representing the maximum number of
  boxes to be selected by non max suppression.
iou_threshold: A float representing the threshold for deciding whether boxes
  overlap too much with respect to IOU.
selected_indices: A 1-D integer tensor of shape `[M]` representing the selected
  indices from the boxes tensor, where `M <= max_output_size`.
)doc");

REGISTER_OP("NonMaxSuppressionV2")
    .Input("boxes: float")
    .Input("scores: float")
    .Input("max_output_size: int32")
    .Input("iou_threshold: float")
    .Output("selected_indices: int32")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(c->UnknownDim()));
      return Status::OK();
    })
    .Doc(R"doc(
Greedily selects a subset of bounding boxes in descending order of score,
pruning away boxes that have high intersection-over-union (IOU) overlap
with previously selected boxes.  Bounding boxes are supplied as
[y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
diagonal pair of box corners and the coordinates can be provided as normalized
(i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
is agnostic to where the origin is in the coordinate system.  Note that this
algorithm is invariant to orthogonal transformations and translations
of the coordinate system; thus translating or reflections of the coordinate
system result in the same boxes being selected by the algorithm.

The output of this operation is a set of integers indexing into the input
collection of bounding boxes representing the selected boxes.  The bounding
box coordinates corresponding to the selected indices can then be obtained
using the `tf.gather operation`.  For example:

  selected_indices = tf.image.non_max_suppression_v2(
      boxes, scores, max_output_size, iou_threshold)
  selected_boxes = tf.gather(boxes, selected_indices)

boxes: A 2-D float tensor of shape `[num_boxes, 4]`.
scores: A 1-D float tensor of shape `[num_boxes]` representing a single
  score corresponding to each box (each row of boxes).
max_output_size: A scalar integer tensor representing the maximum number of
  boxes to be selected by non max suppression.
iou_threshold: A 0-D float tensor representing the threshold for deciding whether
  boxes overlap too much with respect to IOU.
selected_indices: A 1-D integer tensor of shape `[M]` representing the selected
  indices from the boxes tensor, where `M <= max_output_size`.
)doc");

}  // namespace tensorflow
