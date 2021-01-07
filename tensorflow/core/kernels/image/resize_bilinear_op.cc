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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/image/resize_bilinear_op.h"

#include <memory>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/cast_op.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/image_resizer_state.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class ResizeBilinearOp : public OpKernel {
 public:
  explicit ResizeBilinearOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
    OP_REQUIRES_OK(
        context, context->GetAttr("half_pixel_centers", &half_pixel_centers_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    ImageResizerState st(align_corners_, half_pixel_centers_);
    st.ValidateAndCreateOutput(context, input);

    if (!context->status().ok()) return;

    // Return if the output is empty.
    if (st.output->NumElements() == 0) return;

    typename TTypes<T, 4>::ConstTensor image_data(input.tensor<T, 4>());
    TTypes<float, 4>::Tensor output_data = st.output->tensor<float, 4>();

    functor::ResizeBilinear<Device, T>()(
        context->eigen_device<Device>(), image_data, st.height_scale,
        st.width_scale, half_pixel_centers_, output_data);
  }

 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

namespace {
// Compute the interpolation indices only once.
struct CachedInterpolation {
  int64 lower;  // Lower source index used in the interpolation
  int64 upper;  // Upper source index used in the interpolation
  // 1-D linear interpolation scale (see:
  // https://en.wikipedia.org/wiki/Bilinear_interpolation)
  float lerp;
};

template <typename Scaler>
inline void compute_interpolation_weights(const Scaler scaler,
                                          const int64 out_size,
                                          const int64 in_size,
                                          const float scale,
                                          CachedInterpolation* interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  for (int64 i = out_size - 1; i >= 0; --i) {
    const float in = scaler(i, scale);
    const float in_f = std::floor(in);
    interpolation[i].lower =
        std::max(static_cast<int64>(in_f), static_cast<int64>(0));
    interpolation[i].upper =
        std::min(static_cast<int64>(std::ceil(in)), in_size - 1);
    interpolation[i].lerp = in - in_f;
  }
}

/**
 * Computes the bilinear interpolation from the appropriate 4 float points
 * and the linear interpolation weights.
 */
inline float compute_lerp(const float top_left, const float top_right,
                          const float bottom_left, const float bottom_right,
                          const float x_lerp, const float y_lerp) {
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}

template <typename Device>
struct CastFloatToHalf {
  void operator()(const Device& d, typename TTypes<float>::ConstFlat input,
                  typename TTypes<Eigen::half>::Flat output) {
    output.device(d) = input.template cast<Eigen::half>();
  }
};

template <>
struct CastFloatToHalf<GPUDevice> {
  void operator()(const GPUDevice& d, typename TTypes<float>::ConstFlat input,
                  typename TTypes<Eigen::half>::Flat output) {
    // Use existing cast functor instead of directly casting Eigen tensor, as
    // otherwise we need to instantiate the cast function in a .cu.cc file
    functor::CastFunctor<GPUDevice, Eigen::half, float> cast;
    cast(d, output, input);
  }
};

}  // namespace

namespace generator {
template <typename T>
class ResizeBilinearGenerator {
 public:
  EIGEN_ALWAYS_INLINE ResizeBilinearGenerator(
      typename TTypes<T, 4>::ConstTensor input,
      const std::vector<CachedInterpolation>& ys,
      const std::vector<CachedInterpolation>& xs)
      : input_(input), ys_(ys), xs_(xs) {}

  EIGEN_ALWAYS_INLINE float operator()(
      const Eigen::array<Eigen::Index, 4>& coords) const {
    const Eigen::Index b = coords[0];
    const Eigen::Index y = coords[1];
    const Eigen::Index x = coords[2];
    const Eigen::Index c = coords[3];

    const float top_left = input_(b, ys_[y].lower, xs_[x].lower, c);
    const float top_right = input_(b, ys_[y].lower, xs_[x].upper, c);
    const float bottom_left = input_(b, ys_[y].upper, xs_[x].lower, c);
    const float bottom_right = input_(b, ys_[y].upper, xs_[x].upper, c);
    const float ys_lerp = ys_[y].lerp;
    const float xs_lerp = xs_[x].lerp;
    return compute_lerp(top_left, top_right, bottom_left, bottom_right, xs_lerp,
                        ys_lerp);
  }

 private:
  typename TTypes<T, 4>::ConstTensor input_;
  const std::vector<CachedInterpolation> ys_;
  const std::vector<CachedInterpolation> xs_;
};
}  // namespace generator

// Partial specialization of ResizeBilinear functor for a CPUDevice.
namespace functor {
template <typename T>
struct ResizeBilinear<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T, 4>::ConstTensor images,
                  const float height_scale, const float width_scale,
                  bool half_pixel_centers,
                  typename TTypes<float, 4>::Tensor output) {
    const int64 in_height = images.dimension(1);
    const int64 in_width = images.dimension(2);

    const int64 out_height = output.dimension(1);
    const int64 out_width = output.dimension(2);

    // Handle no-op resizes efficiently.
    if (out_height == in_height && out_width == in_width) {
      output = images.template cast<float>();
      return;
    }

    std::vector<CachedInterpolation> ys(out_height + 1);
    std::vector<CachedInterpolation> xs(out_width + 1);

    // Compute the cached interpolation weights on the x and y dimensions.
    if (half_pixel_centers) {
      compute_interpolation_weights(HalfPixelScaler(), out_height, in_height,
                                    height_scale, ys.data());
      compute_interpolation_weights(HalfPixelScaler(), out_width, in_width,
                                    width_scale, xs.data());

    } else {
      compute_interpolation_weights(LegacyScaler(), out_height, in_height,
                                    height_scale, ys.data());
      compute_interpolation_weights(LegacyScaler(), out_width, in_width,
                                    width_scale, xs.data());
    }
    output.device(d) =
        output.generate(generator::ResizeBilinearGenerator<T>(images, ys, xs));
  }
};
}  // namespace functor

template <typename Device, typename T>
class ResizeBilinearOpGrad : public OpKernel {
 public:
  explicit ResizeBilinearOpGrad(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
    OP_REQUIRES_OK(
        context, context->GetAttr("half_pixel_centers", &half_pixel_centers_));
  }

  void Compute(OpKernelContext* context) override {
    // Validate input.
    // First argument is gradient with respect to resized image.
    const Tensor& input = context->input(0);
    const Tensor& original_image = context->input(1);

    ImageResizerGradientState st(align_corners_, half_pixel_centers_);
    st.ValidateAndCreateOutput(context, input, original_image);

    if (!context->status().ok()) return;

    TTypes<float, 4>::ConstTensor input_grad = input.tensor<float, 4>();

    if (!std::is_same<T, Eigen::half>::value) {
      typename TTypes<T, 4>::Tensor output_grad(st.output->tensor<T, 4>());
      functor::ResizeBilinearGrad<Device, T>()(
          context->eigen_device<Device>(), input_grad, st.height_scale,
          st.width_scale, half_pixel_centers_, output_grad);
    } else {
      // Accumulate output to float instead of half tensor, since float
      // accumulation is more numerically stable and GPU half implementation is
      // slow.
      // TODO(b/165759037): Create optimized and numerically stable half
      // implementation
      Tensor output_grad;
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DT_FLOAT, st.output->shape(), &output_grad));
      functor::ResizeBilinearGrad<Device, float>()(
          context->eigen_device<Device>(), input_grad, st.height_scale,
          st.width_scale, half_pixel_centers_, output_grad.tensor<float, 4>());
      const Tensor& output_grad_const = output_grad;
      CastFloatToHalf<Device>{}(context->template eigen_device<Device>(),
                                output_grad_const.template flat<float>(),
                                st.output->template flat<Eigen::half>());
    }
  }

 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

// Partial specialization of ResizeBilinearGrad functor for a CPUDevice.
namespace functor {

template <typename T>
struct ResizeBilinearGrad<CPUDevice, T> {
  template <typename Scaler>
  void ResizeGradCore(const Scaler& scaler,
                      typename TTypes<float, 4>::ConstTensor input_grad,
                      const float height_scale, const float width_scale,
                      typename TTypes<T, 4>::Tensor output_grad) {
    const Eigen::Index batch = output_grad.dimension(0);
    const Eigen::Index original_height = output_grad.dimension(1);
    const Eigen::Index original_width = output_grad.dimension(2);
    const Eigen::Index channels = output_grad.dimension(3);

    const Eigen::Index resized_height = input_grad.dimension(1);
    const Eigen::Index resized_width = input_grad.dimension(2);

    output_grad.setZero();

    // Each resized output pixel was computed as a weighted average of four
    // input pixels. Here we find the four input pixel locations that
    // contributed to each output pixel and propagate the gradient at the output
    // pixel location to each of those four input pixel locations in the same
    // proportions that they originally contributed to the output pixel.
    // Here is the forward-propagation pseudo-code, for reference:
    // resized(b, y, x, c) = top_left     * (1 - y) * (1 - x)
    //                     + top_right    * (1 - y) *      x
    //                     + bottom_left  *      y  * (1 - x)
    //                     + bottom_right *      y  *      x
    for (Eigen::Index b = 0; b < batch; ++b) {
      for (Eigen::Index y = 0; y < resized_height; ++y) {
        const float in_y = scaler(y, height_scale);
        const Eigen::Index top_y_index =
            std::max(static_cast<Eigen::Index>(floorf(in_y)),
                     static_cast<Eigen::Index>(0));
        const Eigen::Index bottom_y_index = std::min(
            static_cast<Eigen::Index>(ceilf(in_y)), original_height - 1);
        const float y_lerp = in_y - floorf(in_y);
        const float inverse_y_lerp = (1.0f - y_lerp);
        for (Eigen::Index x = 0; x < resized_width; ++x) {
          const float in_x = scaler(x, width_scale);
          const Eigen::Index left_x_index =
              std::max(static_cast<Eigen::Index>(floorf(in_x)),
                       static_cast<Eigen::Index>(0));
          const Eigen::Index right_x_index = std::min(
              static_cast<Eigen::Index>(ceilf(in_x)), original_width - 1);
          const float x_lerp = in_x - floorf(in_x);
          const float inverse_x_lerp = (1.0f - x_lerp);
          // TODO(b/158287314): Look into vectorizing this.
          for (Eigen::Index c = 0; c < channels; ++c) {
            output_grad(b, top_y_index, left_x_index, c) +=
                T(input_grad(b, y, x, c) * inverse_y_lerp * inverse_x_lerp);
            output_grad(b, top_y_index, right_x_index, c) +=
                T(input_grad(b, y, x, c) * inverse_y_lerp * x_lerp);
            output_grad(b, bottom_y_index, left_x_index, c) +=
                T(input_grad(b, y, x, c) * y_lerp * inverse_x_lerp);
            output_grad(b, bottom_y_index, right_x_index, c) +=
                T(input_grad(b, y, x, c) * y_lerp * x_lerp);
          }
        }
      }
    }
  }
  void operator()(const CPUDevice& d,
                  typename TTypes<float, 4>::ConstTensor input_grad,
                  const float height_scale, const float width_scale,
                  const bool half_pixel_centers,
                  typename TTypes<T, 4>::Tensor output_grad) {
    if (half_pixel_centers) {
      return ResizeGradCore(HalfPixelScaler(), input_grad, height_scale,
                            width_scale, output_grad);
    } else {
      return ResizeGradCore(LegacyScaler(), input_grad, height_scale,
                            width_scale, output_grad);
    }
  }
};

}  // namespace functor

#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("ResizeBilinear")      \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("size"),    \
                          ResizeBilinearOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#define REGISTER_GRAD_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("ResizeBilinearGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ResizeBilinearOpGrad<CPUDevice, T>);

TF_CALL_half(REGISTER_GRAD_KERNEL);
TF_CALL_float(REGISTER_GRAD_KERNEL);
TF_CALL_double(REGISTER_GRAD_KERNEL);

#undef REGISTER_GRAD_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("ResizeBilinear")      \
                              .Device(DEVICE_GPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("size"),    \
                          ResizeBilinearOp<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#define REGISTER_GRAD_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("ResizeBilinearGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ResizeBilinearOpGrad<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GRAD_KERNEL);

#undef REGISTER_GRAD_KERNEL

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
