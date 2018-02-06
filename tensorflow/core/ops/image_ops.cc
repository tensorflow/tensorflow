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
    .SetShapeFn(ResizeShapeFn);

// --------------------------------------------------------------------------
REGISTER_OP("ResizeBicubic")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: float")
    .Attr("T: {int8, uint8, int16, uint16, int32, int64, half, float, double}")
    .Attr("align_corners: bool = false")
    .SetShapeFn(ResizeShapeFn);

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
    });

// --------------------------------------------------------------------------
REGISTER_OP("ResizeBilinear")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: float")
    .Attr(
        "T: {int8, uint8, int16, uint16, int32, int64, bfloat16, half, "
        "float, double}")
    .Attr("align_corners: bool = false")
    .SetShapeFn(ResizeShapeFn);

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
    });

// --------------------------------------------------------------------------
REGISTER_OP("ResizeBilinearGrad")
    .Input("grads: float")
    .Input("original_image: T")
    .Output("output: T")
    .Attr("T: {float, bfloat16, half, double}")
    .Attr("align_corners: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

// --------------------------------------------------------------------------
REGISTER_OP("ResizeNearestNeighbor")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: T")
    .Attr("T: {int8, uint8, int16, uint16, int32, int64, half, float, double}")
    .Attr("align_corners: bool = false")
    .SetShapeFn(ResizeShapeFn);

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
    });

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
    });
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
    .SetShapeFn(DecodeImageShapeFn);

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
    });

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
    .SetShapeFn(EncodeImageShapeFn);

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
    });

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
    });

// --------------------------------------------------------------------------
REGISTER_OP("AdjustContrastv2")
    .Input("images: float")
    .Input("contrast_factor: float")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    });

// --------------------------------------------------------------------------
REGISTER_OP("AdjustHue")
    .Input("images: float")
    .Input("delta: float")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    });

// --------------------------------------------------------------------------
REGISTER_OP("AdjustSaturation")
    .Input("images: float")
    .Input("scale: float")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    });

// --------------------------------------------------------------------------
REGISTER_OP("DecodePng")
    .Input("contents: string")
    .Attr("channels: int = 0")
    .Attr("dtype: {uint8, uint16} = DT_UINT8")
    .Output("image: dtype")
    .SetShapeFn(DecodeImageShapeFn);

// --------------------------------------------------------------------------
REGISTER_OP("EncodePng")
    .Attr("compression: int = -1")
    .Attr("T: {uint8, uint16} = DT_UINT8")
    .Input("image: T")
    .Output("contents: string")
    .SetShapeFn(EncodeImageShapeFn);

// --------------------------------------------------------------------------
REGISTER_OP("DecodeBmp")
    .Input("contents: string")
    .Output("image: uint8")
    .Attr("channels: int = 0")
    .SetShapeFn(DecodeImageShapeFn);

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
    });

// --------------------------------------------------------------------------
REGISTER_OP("RGBToHSV")
    .Input("images: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double} = DT_FLOAT")
    .SetShapeFn(ColorspaceShapeFn);

// --------------------------------------------------------------------------
REGISTER_OP("HSVToRGB")
    .Input("images: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double} = DT_FLOAT")
    .SetShapeFn(ColorspaceShapeFn);

// --------------------------------------------------------------------------
REGISTER_OP("DrawBoundingBoxes")
    .Input("images: T")
    .Input("boxes: float")
    .Output("output: T")
    .Attr("T: {float, half} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    });

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
    });

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
    });

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
    });

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
    });

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
    });

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
    });

// --------------------------------------------------------------------------

REGISTER_OP("NonMaxSuppression")
    .Input("boxes: float")
    .Input("scores: float")
    .Input("max_output_size: int32")
    .Output("selected_indices: int32")
    .Attr("iou_threshold: float = 0.5")
    .SetShapeFn([](InferenceContext* c) {
      // Get inputs and validate ranks.
      ShapeHandle boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &boxes));
      ShapeHandle scores;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &scores));
      ShapeHandle max_output_size;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &max_output_size));
      // The boxes is a 2-D float Tensor of shape [num_boxes, 4].
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 1), 4, &unused));

      c->set_output(0, c->Vector(c->UnknownDim()));
      return Status::OK();
    });

REGISTER_OP("NonMaxSuppressionV2")
    .Input("boxes: float")
    .Input("scores: float")
    .Input("max_output_size: int32")
    .Input("iou_threshold: float")
    .Output("selected_indices: int32")
    .SetShapeFn([](InferenceContext* c) {
      // Get inputs and validate ranks.
      ShapeHandle boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &boxes));
      ShapeHandle scores;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &scores));
      ShapeHandle max_output_size;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &max_output_size));
      ShapeHandle iou_threshold;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &iou_threshold));
      // The boxes is a 2-D float Tensor of shape [num_boxes, 4].
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 1), 4, &unused));

      c->set_output(0, c->Vector(c->UnknownDim()));
      return Status::OK();
    });

}  // namespace tensorflow
