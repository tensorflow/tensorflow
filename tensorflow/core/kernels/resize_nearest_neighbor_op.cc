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

#include "tensorflow/core/kernels/resize_nearest_neighbor_op.h"

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
typedef Eigen::GpuDevice GPUDevice;

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

    // Return if the output is empty.
    if (st.output->NumElements() == 0) return;

    typename TTypes<T, 4>::ConstTensor input_data = input.tensor<T, 4>();
    typename TTypes<T, 4>::Tensor output_data = st.output->tensor<T, 4>();

    bool status;
    if (align_corners_) {
      status =
          functor::ResizeNearestNeighbor<Device, T, /*align_corners=*/true>()(
              context->eigen_device<Device>(), input_data, st.height_scale,
              st.width_scale, output_data);
    } else {
      status =
          functor::ResizeNearestNeighbor<Device, T, /*align_corners=*/false>()(
              context->eigen_device<Device>(), input_data, st.height_scale,
              st.width_scale, output_data);
    }
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching ResizeNearestNeighbor"));
    }
  }

 private:
  bool align_corners_;
};

// Partial specialization of ResizeNearestNeighbor functor for a CPUDevice.
namespace functor {
template <typename T, bool align_corners>
struct ResizeNearestNeighbor<CPUDevice, T, align_corners> {
  bool operator()(const CPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  const float height_scale, const float width_scale,
                  typename TTypes<T, 4>::Tensor output) {
    const int batch_size = input.dimension(0);
    const int64 in_height = input.dimension(1);
    const int64 in_width = input.dimension(2);
    const int channels = input.dimension(3);

    const int64 out_height = output.dimension(1);
    const int64 out_width = output.dimension(2);

    for (int b = 0; b < batch_size; ++b) {
      for (int y = 0; y < out_height; ++y) {
        const int64 in_y = std::min(
            (align_corners) ? static_cast<int64>(roundf(y * height_scale))
                            : static_cast<int64>(floorf(y * height_scale)),
            in_height - 1);
        for (int x = 0; x < out_width; ++x) {
          const int64 in_x = std::min(
              (align_corners) ? static_cast<int64>(roundf(x * width_scale))
                              : static_cast<int64>(floorf(x * width_scale)),
              in_width - 1);
          std::copy_n(&input(b, in_y, in_x, 0), channels, &output(b, y, x, 0));
        }
      }
    }
    return true;
  }
};
}  // namespace functor

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

    const int64 batch_size = input.dim_size(0);
    const int64 in_height = input.dim_size(1);
    const int64 in_width = input.dim_size(2);
    const int64 channels = input.dim_size(3);

    const int64 out_height = sizes(0);
    const int64 out_width = sizes(1);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({batch_size, out_height, out_width, channels}),
            &output));

    // Return if the output is empty.
    if (output->NumElements() == 0) return;

    typename TTypes<T, 4>::ConstTensor input_data = input.tensor<T, 4>();
    typename TTypes<T, 4>::Tensor output_data = output->tensor<T, 4>();

    const float height_scale =
        CalculateResizeScale(out_height, in_height, align_corners_);
    const float width_scale =
        CalculateResizeScale(out_width, in_width, align_corners_);

    bool status;
    if (align_corners_) {
      status = functor::ResizeNearestNeighborGrad<Device, T,
                                                  /*align_corners=*/true>()(
          context->eigen_device<Device>(), input_data, height_scale,
          width_scale, output_data);
    } else {
      status = functor::ResizeNearestNeighborGrad<Device, T,
                                                  /*align_corners=*/false>()(
          context->eigen_device<Device>(), input_data, height_scale,
          width_scale, output_data);
    }
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching ResizeNearestNeighborGrad"));
    }
  }

 private:
  bool align_corners_;
};

// Partial specialization of ResizeNearestNeighborGrad functor for a CPUDevice.
namespace functor {
template <typename T, bool align_corners>
struct ResizeNearestNeighborGrad<CPUDevice, T, align_corners> {
  bool operator()(const CPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  const float height_scale, const float width_scale,
                  typename TTypes<T, 4>::Tensor output) {
    const int batch_size = input.dimension(0);
    const int64 in_height = input.dimension(1);
    const int64 in_width = input.dimension(2);
    const int channels = input.dimension(3);

    const int64 out_height = output.dimension(1);
    const int64 out_width = output.dimension(2);

    output.setZero();

    for (int y = 0; y < in_height; ++y) {
      const int64 out_y = std::min(
          (align_corners) ? static_cast<int64>(roundf(y * height_scale))
                          : static_cast<int64>(floorf(y * height_scale)),
          out_height - 1);
      for (int x = 0; x < in_width; ++x) {
        const int64 out_x = std::min(
            (align_corners) ? static_cast<int64>(roundf(x * width_scale))
                            : static_cast<int64>(floorf(x * width_scale)),
            out_width - 1);
        for (int b = 0; b < batch_size; ++b) {
          for (int c = 0; c < channels; ++c) {
            output(b, out_y, out_x, c) += input(b, y, x, c);
          }
        }
      }
    }
    return true;
  }
};
}  // namespace functor

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

#define REGISTER_KERNEL(T)                                        \
  REGISTER_KERNEL_BUILDER(Name("ResizeNearestNeighbor")           \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<T>("T")             \
                              .HostMemory("size"),                \
                          ResizeNearestNeighborOp<GPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(Name("ResizeNearestNeighborGrad")       \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<T>("T")             \
                              .HostMemory("size"),                \
                          ResizeNearestNeighborOpGrad<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
