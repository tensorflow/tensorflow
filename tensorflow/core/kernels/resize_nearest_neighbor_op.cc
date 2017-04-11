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

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/resize_nearest_neighbor_op_gpu.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class ResizeNearestNeighborOp : public OpKernel {
 public:
  explicit ResizeNearestNeighborOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    ImageResizerState st(align_corners_);
    st.ValidateAndCreateOutput(context, input);

    if (!context->status().ok()) return;

    OP_REQUIRES(context, st.in_height < (1 << 24) && st.in_width < (1 << 24),
                errors::InvalidArgument("nearest neighbor requires max height "
                                        "& width of 2^24"));

    typename TTypes<T, 4>::ConstTensor input_data = input.tensor<T, 4>();
    typename TTypes<T, 4>::Tensor output_data = st.output->tensor<T, 4>();

    for (int b = 0; b < st.batch_size; ++b) {
      for (int y = 0; y < st.out_height; ++y) {
        const int64 in_y =
            std::min(static_cast<int64>(floorf(y * st.height_scale)),
                     (st.in_height - 1));
        for (int x = 0; x < st.out_width; ++x) {
          const int64 in_x =
              std::min(static_cast<int64>(floorf(x * st.width_scale)),
                       (st.in_width - 1));
          std::copy_n(&input_data(b, in_y, in_x, 0), st.channels,
                      &output_data(b, y, x, 0));
        }
      }
    }
  }

 private:
  bool align_corners_;
};

template <typename Device, typename T>
class ResizeNearestNeighborOpGrad : public OpKernel {
 public:
  explicit ResizeNearestNeighborOpGrad(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab and validate the input:
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));

    // Grab and validate the output shape:
    const Tensor& shape_t = context->input(1);
    OP_REQUIRES(context, shape_t.dims() == 1,
                errors::InvalidArgument("shape_t must be 1-dimensional",
                                        shape_t.shape().DebugString()));
    OP_REQUIRES(context, shape_t.NumElements() == 2,
                errors::InvalidArgument("shape_t must have two elements",
                                        shape_t.shape().DebugString()));

    auto sizes = shape_t.vec<int32>();
    OP_REQUIRES(context, sizes(0) > 0 && sizes(1) > 0,
                errors::InvalidArgument("shape_t's elements must be positive"));

    // Initialize shape to the batch size of the input, then add
    // the rest of the dimensions
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0,
                                TensorShape({input.dim_size(0), sizes(0),
                                             sizes(1), input.dim_size(3)}),
                                &output));

    const int64 batch_size = input.dim_size(0);
    const int64 in_height = input.dim_size(1);
    const int64 in_width = input.dim_size(2);
    const int64 channels = input.dim_size(3);

    const int64 out_height = output->dim_size(1);
    const int64 out_width = output->dim_size(2);

    typename TTypes<T, 4>::ConstTensor input_data = input.tensor<T, 4>();
    typename TTypes<T, 4>::Tensor output_data = output->tensor<T, 4>();

    const float height_scale =
        CalculateResizeScale(out_height, in_height, align_corners_);
    const float width_scale =
        CalculateResizeScale(out_width, in_width, align_corners_);
    output_data.setZero();

    for (int c = 0; c < channels; ++c) {
      for (int y = 0; y < in_height; ++y) {
        const int64 out_y = std::min(
            static_cast<int64>(floorf(y * height_scale)), (out_height - 1));

        for (int x = 0; x < in_width; ++x) {
          const int64 out_x = std::min(
              static_cast<int64>(floorf(x * width_scale)), (out_width - 1));

          for (int b = 0; b < batch_size; ++b) {
            output_data(b, out_y, out_x, c) += input_data(b, y, x, c);
          }
        }
      }
    }
  }

 private:
  bool align_corners_;
};

#define REGISTER_KERNEL(T)                                        \
  REGISTER_KERNEL_BUILDER(Name("ResizeNearestNeighbor")           \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<T>("T")             \
                              .HostMemory("size"),                \
                          ResizeNearestNeighborOp<CPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(Name("ResizeNearestNeighborGrad")       \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<T>("T")             \
                              .HostMemory("size"),                \
                          ResizeNearestNeighborOpGrad<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#if GOOGLE_CUDA

template <typename T>
class ResizeNearestNeighborGPUOp : public OpKernel {
 public:
  explicit ResizeNearestNeighborGPUOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    ImageResizerState st(align_corners_);
    st.ValidateAndCreateOutput(context, input);
    if (!context->status().ok()) return;

    bool status = ResizeNearestNeighbor<T>(
        input.flat<T>().data(), st.batch_size, st.in_height, st.in_width,
        st.channels, st.out_height, st.out_width, st.height_scale,
        st.width_scale, st.output->flat<T>().data(),
        context->eigen_gpu_device());

    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching ResizeNearestNeighbor"));
    }
  }

 private:
  bool align_corners_;
};

#define REGISTER_KERNEL(T)                              \
  REGISTER_KERNEL_BUILDER(Name("ResizeNearestNeighbor") \
                              .Device(DEVICE_GPU)       \
                              .TypeConstraint<T>("T")   \
                              .HostMemory("size"),      \
                          ResizeNearestNeighborGPUOp<T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

template <typename T>
class ResizeNearestNeighborGPUOpGrad : public OpKernel {
 public:
  explicit ResizeNearestNeighborGPUOpGrad(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab and validate the input:
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));

    // Grab and validate the output shape:
    const Tensor& shape_t = context->input(1);
    OP_REQUIRES(context, shape_t.dims() == 1,
                errors::InvalidArgument("shape_t must be 1-dimensional",
                                        shape_t.shape().DebugString()));
    OP_REQUIRES(context, shape_t.NumElements() == 2,
                errors::InvalidArgument("shape_t must have two elements",
                                        shape_t.shape().DebugString()));

    auto sizes = shape_t.vec<int32>();
    OP_REQUIRES(context, sizes(0) > 0 && sizes(1) > 0,
                errors::InvalidArgument("shape_t's elements must be positive"));

    // Initialize shape to the batch size of the input, then add
    // the rest of the dimensions
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0,
                                TensorShape({input.dim_size(0), sizes(0),
                                             sizes(1), input.dim_size(3)}),
                                &output));

    const int64 batch_size = input.dim_size(0);
    const int64 in_height = input.dim_size(1);
    const int64 in_width = input.dim_size(2);
    const int64 channels = input.dim_size(3);

    const int64 out_height = output->dim_size(1);
    const int64 out_width = output->dim_size(2);

    const float height_scale =
        CalculateResizeScale(out_height, in_height, align_corners_);
    const float width_scale =
        CalculateResizeScale(out_width, in_width, align_corners_);

    bool status = ResizeNearestNeighborBackward(
        input.flat<T>().data(), batch_size, in_height, in_width, channels,
        out_height, out_width, height_scale, width_scale,
        output->flat<T>().data(), context->eigen_gpu_device());

    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching ResizeNearestNeighborGrad"));
    }
  }
  bool align_corners_;
};

#define REGISTER_KERNEL(T)                                  \
  REGISTER_KERNEL_BUILDER(Name("ResizeNearestNeighborGrad") \
                              .Device(DEVICE_GPU)           \
                              .TypeConstraint<T>("T")       \
                              .HostMemory("size"),          \
                          ResizeNearestNeighborGPUOpGrad<T>);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
