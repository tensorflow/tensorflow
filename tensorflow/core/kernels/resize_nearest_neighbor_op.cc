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
    OP_REQUIRES_OK(
        context, context->GetAttr("half_pixel_centers", &half_pixel_centers_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    ImageResizerState st(align_corners_, half_pixel_centers_);
    st.ValidateAndCreateOutput(context, input);

    if (!context->status().ok()) return;

    OP_REQUIRES(context, st.in_height < (1 << 24) && st.in_width < (1 << 24),
                errors::InvalidArgument("nearest neighbor requires max height "
                                        "& width of 2^24"));

    // Return if the output is empty.
    if (st.output->NumElements() == 0) return;

    typename TTypes<T, 4>::ConstTensor input_data(input.tensor<T, 4>());
    typename TTypes<T, 4>::Tensor output_data(st.output->tensor<T, 4>());

    bool status;
    if (half_pixel_centers_) {
      if (align_corners_) {
        status = functor::ResizeNearestNeighbor<Device, T,
                                                /*half_pixe_centers=*/true,
                                                /*align_corners=*/true>()(
            context->eigen_device<Device>(), input_data, st.height_scale,
            st.width_scale, output_data);
      } else {
        status = functor::ResizeNearestNeighbor<Device, T,
                                                /*half_pixe_centers=*/true,
                                                /*align_corners=*/false>()(
            context->eigen_device<Device>(), input_data, st.height_scale,
            st.width_scale, output_data);
      }
    } else {
      if (align_corners_) {
        status = functor::ResizeNearestNeighbor<Device, T,
                                                /*half_pixe_centers=*/false,
                                                /*align_corners=*/true>()(
            context->eigen_device<Device>(), input_data, st.height_scale,
            st.width_scale, output_data);
      } else {
        status = functor::ResizeNearestNeighbor<Device, T,
                                                /*half_pixe_centers=*/false,
                                                /*align_corners=*/false>()(
            context->eigen_device<Device>(), input_data, st.height_scale,
            st.width_scale, output_data);
      }
    }
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching ResizeNearestNeighbor"));
    }
  }

 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

// Helper struct to convert a bool to the correct scaler type.
template <bool half_pixel_centers>
struct BoolToScaler {};

struct HalfPixelScalerForNN {
  inline float operator()(const int x, const float scale) const {
    // All of the nearest neigbor code below immediately follows a call to this
    // function with a std::floor(), so instead of subtracting the 0.5 as we
    // do in HalfPixelScale, we leave it as is, as the std::floor does the
    // correct thing.
    return (static_cast<float>(x) + 0.5f) * scale;
  }
};

template <>
struct BoolToScaler<true> {
  typedef HalfPixelScalerForNN Scaler;
};

template <>
struct BoolToScaler<false> {
  typedef LegacyScaler Scaler;
};

// Partial specialization of ResizeNearestNeighbor functor for a CPUDevice.
namespace functor {
template <typename T, bool half_pixel_centers, bool align_corners>
struct ResizeNearestNeighbor<CPUDevice, T, half_pixel_centers, align_corners> {
  bool operator()(const CPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  const float height_scale, const float width_scale,
                  typename TTypes<T, 4>::Tensor output) {
    typename BoolToScaler<half_pixel_centers>::Scaler scaler;
    const Eigen::Index batch_size = input.dimension(0);
    const Eigen::Index in_height = input.dimension(1);
    const Eigen::Index in_width = input.dimension(2);
    const Eigen::Index channels = input.dimension(3);
    const Eigen::Index out_height = output.dimension(1);
    const Eigen::Index out_width = output.dimension(2);

#ifdef PLATFORM_GOOGLE
    // The parallel version is significantly slower than the serial version
    // internally. Only call the serial version for now.
    // TODO(b/145019377): Make the parallel version work for PLATFORM_GOOGLE.
    for (Eigen::Index b = 0; b < batch_size; ++b) {
      for (Eigen::Index y = 0; y < out_height; ++y) {
        Eigen::Index in_y = std::min(
            (align_corners)
                ? static_cast<Eigen::Index>(roundf(scaler(y, height_scale)))
                : static_cast<Eigen::Index>(floorf(scaler(y, height_scale))),
            in_height - 1);
        if (half_pixel_centers) {
          in_y = std::max(static_cast<Eigen::Index>(0), in_y);
        }
        for (Eigen::Index x = 0; x < out_width; ++x) {
          Eigen::Index in_x = std::min(
              (align_corners)
                  ? static_cast<Eigen::Index>(roundf(scaler(x, width_scale)))
                  : static_cast<Eigen::Index>(floorf(scaler(x, width_scale))),
              in_width - 1);
          if (half_pixel_centers) {
            in_x = std::max(static_cast<Eigen::Index>(0), in_x);
          }
          std::copy_n(&input(b, in_y, in_x, 0), channels, &output(b, y, x, 0));
        }
      }
    }
#else
    auto ParallelResize = [&](Eigen::Index start, Eigen::Index end) {
      for (Eigen::Index b = start; b < end; ++b) {
        Eigen::Index x = b % out_width;
        Eigen::Index y = (b / out_width) % out_height;
        Eigen::Index bs = (b / out_width) / out_height;
        Eigen::Index in_y = std::min(
            (align_corners)
                ? static_cast<Eigen::Index>(roundf(scaler(y, height_scale)))
                : static_cast<Eigen::Index>(floorf(scaler(y, height_scale))),
            in_height - 1);
        if (half_pixel_centers) {
          in_y = std::max(static_cast<Eigen::Index>(0), in_y);
        }
        Eigen::Index in_x = std::min(
            (align_corners)
                ? static_cast<Eigen::Index>(roundf(scaler(x, width_scale)))
                : static_cast<Eigen::Index>(floorf(scaler(x, width_scale))),
            in_width - 1);
        if (half_pixel_centers) {
          in_x = std::max(static_cast<Eigen::Index>(0), in_x);
        }
        std::copy_n(&input(bs, in_y, in_x, 0), channels, &output(bs, y, x, 0));
      }
    };
    Eigen::Index N = batch_size * out_height * out_width;
    const int input_bytes = channels * sizeof(T);
    const int output_bytes = channels * sizeof(T);
    const int compute_cycles = (Eigen::TensorOpCost::ModCost<T>() * 2 +
                                Eigen::TensorOpCost::DivCost<T>() * 3 +
                                Eigen::TensorOpCost::AddCost<T>() * 2 +
                                Eigen::TensorOpCost::MulCost<T>() * 2);
    const Eigen::TensorOpCost cost(input_bytes, output_bytes, compute_cycles);
    d.parallelFor(N, cost, ParallelResize);
#endif  // PLATFORM_GOOGLE
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
    OP_REQUIRES_OK(
        context, context->GetAttr("half_pixel_centers", &half_pixel_centers_));
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

    typename TTypes<T, 4>::ConstTensor input_data(input.tensor<T, 4>());
    typename TTypes<T, 4>::Tensor output_data(output->tensor<T, 4>());

    const float height_scale =
        CalculateResizeScale(out_height, in_height, align_corners_);
    const float width_scale =
        CalculateResizeScale(out_width, in_width, align_corners_);

    bool status;
    if (half_pixel_centers_) {
      if (align_corners_) {
        status = functor::ResizeNearestNeighborGrad<Device, T,
                                                    /*half_pixel_centers=*/true,
                                                    /*align_corners=*/true>()(
            context->eigen_device<Device>(), input_data, height_scale,
            width_scale, output_data);
      } else {
        status = functor::ResizeNearestNeighborGrad<Device, T,
                                                    /*half_pixel_centers=*/true,
                                                    /*align_corners=*/false>()(
            context->eigen_device<Device>(), input_data, height_scale,
            width_scale, output_data);
      }
    } else {
      if (align_corners_) {
        status =
            functor::ResizeNearestNeighborGrad<Device, T,
                                               /*half_pixel_centers=*/false,
                                               /*align_corners=*/true>()(
                context->eigen_device<Device>(), input_data, height_scale,
                width_scale, output_data);
      } else {
        status =
            functor::ResizeNearestNeighborGrad<Device, T,
                                               /*half_pixel_centers=*/false,
                                               /*align_corners=*/false>()(
                context->eigen_device<Device>(), input_data, height_scale,
                width_scale, output_data);
      }
    }
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching ResizeNearestNeighborGrad"));
    }
  }

 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

// Partial specialization of ResizeNearestNeighborGrad functor for a CPUDevice.
namespace functor {
template <typename T, bool half_pixel_centers, bool align_corners>
struct ResizeNearestNeighborGrad<CPUDevice, T, half_pixel_centers,
                                 align_corners> {
  bool operator()(const CPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  const float height_scale, const float width_scale,
                  typename TTypes<T, 4>::Tensor output) {
    typename BoolToScaler<half_pixel_centers>::Scaler scaler;
    const Eigen::Index batch_size = input.dimension(0);
    const Eigen::Index in_height = input.dimension(1);
    const Eigen::Index in_width = input.dimension(2);
    const Eigen::Index channels = input.dimension(3);

    const Eigen::Index out_height = output.dimension(1);
    const Eigen::Index out_width = output.dimension(2);

    output.setZero();

    for (Eigen::Index y = 0; y < in_height; ++y) {
      const Eigen::Index out_y = std::min(
          (align_corners)
              ? static_cast<Eigen::Index>(roundf(scaler(y, height_scale)))
              : static_cast<Eigen::Index>(floorf(scaler(y, height_scale))),
          out_height - 1);
      for (Eigen::Index x = 0; x < in_width; ++x) {
        const Eigen::Index out_x = std::min(
            (align_corners)
                ? static_cast<Eigen::Index>(roundf(scaler(x, width_scale)))
                : static_cast<Eigen::Index>(floorf(scaler(x, width_scale))),
            out_width - 1);
        for (Eigen::Index b = 0; b < batch_size; ++b) {
          for (Eigen::Index c = 0; c < channels; ++c) {
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

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

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
