/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// This is a helper struct to package up the input and output
// parameters of an image resizer (the height, widths, etc.).  To
// reduce code duplication and ensure consistency across the different
// resizers, it performs the input validation.

#ifndef TENSORFLOW_CORE_UTIL_IMAGE_RESIZER_STATE_H_
#define TENSORFLOW_CORE_UTIL_IMAGE_RESIZER_STATE_H_

#define EIGEN_USE_THREADS
#include <math.h>

#include <algorithm>
#include <array>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

// CalculateResizeScale determines the float scaling factor.
inline float CalculateResizeScale(int64_t in_size, int64_t out_size,
                                  bool align_corners) {
  return (align_corners && in_size > 1 && out_size > 1)
             ? (in_size - 1) / static_cast<float>(out_size - 1)
             : in_size / static_cast<float>(out_size);
}

// Half pixel scaler scales assuming that the pixel centers are at 0.5, i.e. the
// floating point coordinates of the top,left pixel is 0.5,0.5.
struct HalfPixelScaler {
  HalfPixelScaler(){};
  inline float operator()(const int x, const float scale) const {
    // Note that we subtract 0.5 from the return value, as the existing bilinear
    // sampling code etc assumes pixels are in the old coordinate system.
    return (static_cast<float>(x) + 0.5f) * scale - 0.5f;
  }
};

// Older incorrect scaling method that causes all resizes to have a slight
// translation leading to inconsistent results. For example, a flip then a
// resize gives different results then a resize then a flip.
struct LegacyScaler {
  LegacyScaler(){};
  inline float operator()(const int x, const float scale) const {
    return static_cast<float>(x) * scale;
  }
};

struct ImageResizerState {
  explicit ImageResizerState(bool align_corners, bool half_pixel_centers)
      : align_corners_(align_corners),
        half_pixel_centers_(half_pixel_centers) {}

  // ValidateAndCalculateOutputSize checks the bounds on the input tensors
  // and requested size, sets up some of the resizing state such as the
  // height_scale and width_scale, and calculates the output size.
  // If any of these operations fails, it sets an error status in
  // the context, which the caller must check.
  void ValidateAndCalculateOutputSize(OpKernelContext* context) {
    OP_REQUIRES(
        context,
        !half_pixel_centers_ || (half_pixel_centers_ && !align_corners_),
        errors::InvalidArgument("If half_pixel_centers is True, "
                                "align_corners must be False."));

    const TensorShape& input_shape = context->input(0).shape();
    OP_REQUIRES(context, input_shape.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input_shape.DebugString()));
    batch_size = input_shape.dim_size(0);
    channels = input_shape.dim_size(3);
    OP_REQUIRES(
        context, channels > 0,
        errors::InvalidArgument("image must have at least one channel"));

    // Verify and assign `in_height` and `in_width`.
    OP_REQUIRES(
        context, input_shape.dim_size(1) > 0 && input_shape.dim_size(2) > 0,
        errors::InvalidArgument("input image must be of non-zero size"));
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_shape.dim_size(1),
                        std::numeric_limits<int32>::max()) &&
            FastBoundsCheck(input_shape.dim_size(2),
                            std::numeric_limits<int32>::max()),
        errors::InvalidArgument("input sizes must be between 0 and max int32"));
    in_height = static_cast<int32>(input_shape.dim_size(1));
    in_width = static_cast<int32>(input_shape.dim_size(2));

    // Verify the output tensor's shape.
    const Tensor& shape_t = context->input(1);
    OP_REQUIRES(context, shape_t.dims() == 1,
                errors::InvalidArgument("shape_t must be 1-dimensional",
                                        shape_t.shape().DebugString()));
    OP_REQUIRES(context, shape_t.NumElements() == 2,
                errors::InvalidArgument("shape_t must have two elements",
                                        shape_t.shape().DebugString()));

    // Verify and assign `out_height` and `out_width`.
    auto Svec = shape_t.vec<int32>();
    out_height = internal::SubtleMustCopy(Svec(0));
    out_width = internal::SubtleMustCopy(Svec(1));
    OP_REQUIRES(context, out_height > 0 && out_width > 0,
                errors::InvalidArgument("output dimensions must be positive"));

    height_scale = CalculateResizeScale(in_height, out_height, align_corners_);
    width_scale = CalculateResizeScale(in_width, out_width, align_corners_);

    // Guard against overflows
    OP_REQUIRES(context,
                ceilf((out_height - 1) * height_scale) <=
                    static_cast<float>(std::numeric_limits<int64_t>::max()),
                errors::InvalidArgument(
                    "input image height scale would cause an overflow"));
    OP_REQUIRES(
        context,
        ceilf((out_width - 1) * width_scale) <= static_cast<float>(INT_MAX),
        errors::InvalidArgument(
            "input image width scale would cause an overflow"));
  }

  // Calculates all the required variables, and allocates the output.
  void ValidateAndCreateOutput(OpKernelContext* context) {
    ValidateAndCalculateOutputSize(context);
    if (!context->status().ok()) return;

    TensorShape shape;
    // Guard against shape overflow
    OP_REQUIRES_OK(context, shape.AddDimWithStatus(batch_size));
    OP_REQUIRES_OK(context, shape.AddDimWithStatus(out_height));
    OP_REQUIRES_OK(context, shape.AddDimWithStatus(out_width));
    OP_REQUIRES_OK(context, shape.AddDimWithStatus(channels));

    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));
  }

  int64_t batch_size;
  int64_t out_height;
  int64_t out_width;
  int64_t in_height;
  int64_t in_width;
  int64_t channels;
  float height_scale;
  float width_scale;
  Tensor* output = nullptr;

 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

struct ImageResizerGradientState {
  explicit ImageResizerGradientState(bool align_corners,
                                     bool half_pixel_centers)
      : align_corners_(align_corners),
        half_pixel_centers_(half_pixel_centers) {}

  void ValidateAndCreateOutput(OpKernelContext* context) {
    OP_REQUIRES(
        context,
        !half_pixel_centers_ || (half_pixel_centers_ && !align_corners_),
        errors::InvalidArgument("If half_pixel_centers is True, "
                                "align_corners must be False."));

    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input_grad must be 4-dimensional",
                                        input.shape().DebugString()));

    // Resizers always produce float images, so input gradient must
    // always be a float.
    OP_REQUIRES(context, input.dtype() == DT_FLOAT,
                errors::InvalidArgument("input_grad must be of type float",
                                        DataTypeString(input.dtype())));

    batch_size = input.dim_size(0);
    channels = input.dim_size(3);

    resized_height = input.dim_size(1);
    resized_width = input.dim_size(2);

    // The following check is also carried out for the forward op. It is added
    // here to prevent a divide-by-zero exception when either height_scale or
    // width_scale is being calculated.
    OP_REQUIRES(context, resized_height > 0 && resized_width > 0,
                errors::InvalidArgument("resized dimensions must be positive"));

    const TensorShape& output_shape = context->input(1).shape();
    OP_REQUIRES(context, output_shape.dims() == 4,
                errors::InvalidArgument("original_image must be 4-dimensional",
                                        output_shape.DebugString()));
    original_height = output_shape.dim_size(1);
    original_width = output_shape.dim_size(2);

    // The following check is also carried out for the forward op. It is added
    // here to prevent either height_scale or width_scale from being set to
    // zero, which would cause a divide-by-zero exception in the deterministic
    // back-prop path.
    OP_REQUIRES(
        context, original_height > 0 && original_width > 0,
        errors::InvalidArgument("original dimensions must be positive"));

    OP_REQUIRES(
        context,
        FastBoundsCheck(original_height, std::numeric_limits<int32>::max()) &&
            FastBoundsCheck(original_width, std::numeric_limits<int32>::max()),
        errors::InvalidArgument(
            "original sizes must be between 0 and max int32"));

    height_scale =
        CalculateResizeScale(original_height, resized_height, align_corners_);
    width_scale =
        CalculateResizeScale(original_width, resized_width, align_corners_);

    OP_REQUIRES_OK(context, context->allocate_output(
                                0,
                                TensorShape({batch_size, original_height,
                                             original_width, channels}),
                                &output));
  }

  int64_t batch_size;
  int64_t channels;
  int64_t resized_height;
  int64_t resized_width;
  int64_t original_height;
  int64_t original_width;
  float height_scale;
  float width_scale;
  Tensor* output = nullptr;

 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_IMAGE_RESIZER_STATE_H_
