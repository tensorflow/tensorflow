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

// See docs in ../ops/image_ops.cc
#define EIGEN_USE_THREADS

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/image_resizer_state.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class ResizeBilinearOp : public OpKernel {
 public:
  explicit ResizeBilinearOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    ImageResizerState st(align_corners_);
    st.ValidateAndCreateOutput(context, input);

    if (!context->status().ok()) return;

    typename TTypes<T, 4>::ConstTensor input_data = input.tensor<T, 4>();
    typename TTypes<float, 4>::Tensor output_data =
        st.output->tensor<float, 4>();

    for (int b = 0; b < st.batch_size; ++b) {
      for (int y = 0; y < st.out_height; ++y) {
        const float in_y = y * st.height_scale;
        const int top_y_index = static_cast<int>(floorf(in_y));
        const int bottom_y_index =
            std::min(static_cast<int64>(ceilf(in_y)), (st.in_height - 1));
        const float y_lerp = in_y - top_y_index;
        for (int x = 0; x < st.out_width; ++x) {
          const float in_x = x * st.width_scale;
          const int left_x_index = static_cast<int>(floorf(in_x));
          const int right_x_index =
              std::min(static_cast<int64>(ceilf(in_x)), (st.in_width - 1));
          const float x_lerp = in_x - left_x_index;
          for (int c = 0; c < st.channels; ++c) {
            const float top_left(input_data(b, top_y_index, left_x_index, c));
            const float top_right(input_data(b, top_y_index, right_x_index, c));
            const float bottom_left(
                input_data(b, bottom_y_index, left_x_index, c));
            const float bottom_right(
                input_data(b, bottom_y_index, right_x_index, c));
            const float top = top_left + (top_right - top_left) * x_lerp;
            const float bottom =
                bottom_left + (bottom_right - bottom_left) * x_lerp;
            output_data(b, y, x, c) = top + (bottom - top) * y_lerp;
          }
        }
      }
    }
  }

 private:
  bool align_corners_;
};

template <typename Device, typename T>
class ResizeBilinearOpGrad : public OpKernel {
 public:
  explicit ResizeBilinearOpGrad(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
  }

  void Compute(OpKernelContext* context) override {
    // Validate input.
    // First argument is gradient with respect to resized image.
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input_grad must be 4-dimensional",
                                        input.shape().DebugString()));
    // ResizeBilinear always produces float images, so the input gradient is
    // always a float.
    OP_REQUIRES(context, input.dtype() == DT_FLOAT,
                errors::InvalidArgument("input_grad must be of type float",
                                        input.dtype()));

    // The second argument is the original input to resize_bilinear.
    const Tensor& original_image = context->input(1);
    OP_REQUIRES(context, original_image.dims() == 4,
                errors::InvalidArgument("original_image must be 4-dimensional",
                                        original_image.shape().DebugString()));

    // Allocate output and initialize to zeros.
    const int64 batch_size = input.dim_size(0);
    const int64 channels = input.dim_size(3);
    const int64 resized_height = input.dim_size(1);
    const int64 resized_width = input.dim_size(2);
    const int64 original_height = original_image.dim_size(1);
    const int64 original_width = original_image.dim_size(2);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({batch_size, original_height,
                                                original_width, channels}),
                                &output));

    typename TTypes<float, 4>::ConstTensor input_grad =
        input.tensor<float, 4>();
    typename TTypes<T, 4>::Tensor output_grad = output->tensor<T, 4>();

    for (int c = 0; c < channels; ++c) {
      for (int y = 0; y < original_height; ++y) {
        for (int x = 0; x < original_width; ++x) {
          for (int b = 0; b < batch_size; ++b) {
            output_grad(b, y, x, c) = 0;
          }
        }
      }
    }

    const float height_scale =
        (align_corners_ && resized_height > 1)
            ? (original_height - 1) / static_cast<float>(resized_height - 1)
            : original_height / static_cast<float>(resized_height);
    const float width_scale =
        (align_corners_ && resized_width > 1)
            ? (original_width - 1) / static_cast<float>(resized_width - 1)
            : original_width / static_cast<float>(resized_width);

    // Each resized pixel was computed as a weighted average of four input
    // pixels. Here we find the pixels that contributed to each output pixel
    // and add the corresponding coefficient to the gradient.
    // resized(b, y, x, c) = top_left * (1 - y) * (1 - x)
    //                       +  top_right * (1 - y) * x
    //                       +  bottom_left * y * (1 - x)
    //                       +  bottom_right * y * x
    for (int b = 0; b < batch_size; ++b) {
      for (int y = 0; y < resized_height; ++y) {
        const float in_y = y * height_scale;
        const int top_y_index = static_cast<int>(floorf(in_y));
        const int bottom_y_index =
            std::min(static_cast<int64>(ceilf(in_y)), (original_height - 1));
        const float y_lerp = in_y - top_y_index;
        const float inverse_y_lerp = (1.0f - y_lerp);
        for (int x = 0; x < resized_width; ++x) {
          const float in_x = x * width_scale;
          const int left_x_index = static_cast<int>(floorf(in_x));
          const int right_x_index =
              std::min(static_cast<int64>(ceilf(in_x)), (original_width - 1));
          const float x_lerp = in_x - left_x_index;
          const float inverse_x_lerp = (1.0f - x_lerp);
          for (int c = 0; c < channels; ++c) {
            output_grad(b, top_y_index, left_x_index, c) +=
                input_grad(b, y, x, c) * inverse_y_lerp * inverse_x_lerp;
            output_grad(b, top_y_index, right_x_index, c) +=
                input_grad(b, y, x, c) * inverse_y_lerp * x_lerp;
            output_grad(b, bottom_y_index, left_x_index, c) +=
                input_grad(b, y, x, c) * y_lerp * inverse_x_lerp;
            output_grad(b, bottom_y_index, right_x_index, c) +=
                input_grad(b, y, x, c) * y_lerp * x_lerp;
          }
        }
      }
    }
  }

 private:
  bool align_corners_;
};

#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("ResizeBilinear")      \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("size"),    \
                          ResizeBilinearOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

REGISTER_KERNEL_BUILDER(Name("ResizeBilinearGrad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        ResizeBilinearOpGrad<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("ResizeBilinearGrad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        ResizeBilinearOpGrad<CPUDevice, double>);
}  // namespace tensorflow
