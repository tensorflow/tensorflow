/* Copyright 2016 Google Inc. All Rights Reserved.

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

// This is a helper struct to package up the input and ouput
// parameters of an image resizer (the height, widths, etc.).  To
// reduce code duplication and ensure consistency across the different
// resizers, it performs the input validation.

#ifndef TENSORFLOW_KERNELS_IMAGE_RESIZER_STATE_H_
#define TENSORFLOW_KERNELS_IMAGE_RESIZER_STATE_H_

#define EIGEN_USE_THREADS

#include <math.h>
#include <algorithm>
#include <array>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"

namespace tensorflow {

// CalculateResizeScale determines the float scaling factor.
inline float CalculateResizeScale(int64 in_size, int64 out_size,
                                  bool align_corners) {
  return (align_corners && out_size > 1)
             ? (in_size - 1) / static_cast<float>(out_size - 1)
             : in_size / static_cast<float>(out_size);
}

struct ImageResizerState {
  explicit ImageResizerState(bool align_corners)
      : align_corners_(align_corners) {}

  // ValidateAndCreateOutput checks the bounds on the input tensors
  // and requested size, sets up some of the resizing state such as the
  // height_scale and width_scale, and allocates the output.
  // If any of these operations fails, it sets an error status in
  // the context, which the caller must check.
  void ValidateAndCreateOutput(OpKernelContext* context, const Tensor& input) {
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    const Tensor& shape_t = context->input(1);
    OP_REQUIRES(context, shape_t.dims() == 1,
                errors::InvalidArgument("shape_t must be 1-dimensional",
                                        shape_t.shape().DebugString()));
    OP_REQUIRES(context, shape_t.NumElements() == 2,
                errors::InvalidArgument("shape_t must have two elements",
                                        shape_t.shape().DebugString()));
    auto Svec = shape_t.vec<int32>();
    batch_size = input.dim_size(0);
    out_height = internal::SubtleMustCopy(Svec(0));
    out_width = internal::SubtleMustCopy(Svec(1));
    OP_REQUIRES(
        context,
        FastBoundsCheck(input.dim_size(1), std::numeric_limits<int32>::max()) &&
            FastBoundsCheck(input.dim_size(2),
                            std::numeric_limits<int32>::max()),
        errors::InvalidArgument("input sizes must be between 0 and max int32"));

    in_height = static_cast<int32>(input.dim_size(1));
    in_width = static_cast<int32>(input.dim_size(2));
    channels = input.dim_size(3);
    OP_REQUIRES(context, out_height > 0 && out_width > 0,
                errors::InvalidArgument("output dimensions must be positive"));
    OP_REQUIRES(
        context, channels > 0,
        errors::InvalidArgument("image must have at least one channel"));
    OP_REQUIRES(
        context, input.dim_size(1) > 0 && input.dim_size(2) > 0,
        errors::InvalidArgument("input image must be of non-zero size"));
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({input.dim_size(0), out_height,
                                                out_width, input.dim_size(3)}),
                                &output));
    height_scale = CalculateResizeScale(in_height, out_height, align_corners_);
    width_scale = CalculateResizeScale(in_width, out_width, align_corners_);
  }

  int64 batch_size;
  int64 out_height;
  int64 out_width;
  int64 in_height;
  int64 in_width;
  int64 channels;
  float height_scale;
  float width_scale;
  Tensor* output;

 private:
  bool align_corners_;
};

struct ImageResizerGradientState {
  explicit ImageResizerGradientState(bool align_corners)
      : align_corners_(align_corners) {}

  void ValidateAndCreateOutput(OpKernelContext* context, const Tensor& input,
                               const Tensor& original_image) {
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input_grad must be 4-dimensional",
                                        input.shape().DebugString()));
    // Resizers always produce float images, so input gradient must
    // always be a float.
    OP_REQUIRES(context, input.dtype() == DT_FLOAT,
                errors::InvalidArgument("input_grad must be of type float",
                                        input.dtype()));

    OP_REQUIRES(context, original_image.dims() == 4,
                errors::InvalidArgument("original_image must be 4-dimensional",
                                        original_image.shape().DebugString()));

    // Allocate output and initialize to zeros.
    batch_size = input.dim_size(0);
    channels = input.dim_size(3);
    resized_height = input.dim_size(1);
    resized_width = input.dim_size(2);
    original_height = original_image.dim_size(1);
    original_width = original_image.dim_size(2);

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
    output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({batch_size, original_height,
                                                original_width, channels}),
                                &output));
  }

  int64 batch_size;
  int64 channels;
  int64 resized_height;
  int64 resized_width;
  int64 original_height;
  int64 original_width;
  float height_scale;
  float width_scale;
  Tensor* output;

 private:
  bool align_corners_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_IMAGE_RESIZER_STATE_H_
