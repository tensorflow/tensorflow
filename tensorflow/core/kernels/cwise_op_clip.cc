/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/cwise_op_clip.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Basic coefficient-wise tenary operations.
// This is the case for example of the clip_by_value.
//   Device: E.g., CPUDevice, GPUDevice.
//   Functor: defined above. E.g., functor::clip.
template <typename Device, typename T>
class ClipOp : public OpKernel {
 public:
  explicit ClipOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    const Tensor& in2 = ctx->input(2);
    OP_REQUIRES(ctx, (in0.shape() == in1.shape() ||
                      TensorShapeUtils::IsScalar(in1.shape())) &&
                     (in0.shape() == in2.shape() ||
                      TensorShapeUtils::IsScalar(in2.shape())),
                errors::InvalidArgument(
                    "clip_value_min and clip_value_max must be either of "
                    "the same shape as input, or a scalar. ",
                    "input shape: ", in0.shape().DebugString(),
                    "clip_value_min shape: ", in1.shape().DebugString(),
                    "clip_value_max shape: ", in2.shape().DebugString()));

    Tensor* out = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->forward_input_or_allocate_output({0}, 0, in0.shape(), &out));
    if (out->NumElements() == 0) return;  // Nothing to do for empty output

    auto in0_flat = in0.flat<T>();
    auto in1_flat = in1.flat<T>();
    auto in2_flat = in2.flat<T>();
    auto out_flat = out->flat<T>();
    const Device& d = ctx->eigen_device<Device>();

    if (in1.shape() == in2.shape()) {
      if (in0.shape() == in1.shape()) {
        functor::TernaryClipOp<Device, T>()(d, in0_flat, in1_flat, in2_flat,
                                            out_flat);
      } else {
        functor::UnaryClipOp<Device, T>()(d, in0_flat, in1_flat, in2_flat,
                                          out_flat);
      }
    } else {
      if (in0.shape() == in1.shape()) {
        functor::BinaryLeftClipOp<Device, T>()(d, in0_flat, in1_flat, in2_flat,
                                               out_flat);
      } else {
        functor::BinaryRightClipOp<Device, T>()(d, in0_flat, in1_flat, in2_flat,
                                                out_flat);
      }
    }
  }
};

namespace functor {
// Unary functor for clip [Tensor, Scalar, Scalar]
template <typename T, bool is_complex = Eigen::NumTraits<T>::IsComplex>
struct UnaryClipFunc {
  UnaryClipFunc(const T& value_min, const T& value_max)
      : value_min(value_min), value_max(value_max) {}
  T operator()(const T& value) const {
    return std::max(std::min(value, value_max), value_min);
  }
  T value_min;
  T value_max;
};

template <typename T>
struct UnaryClipFunc<T, /*is_complex=*/true> {
  UnaryClipFunc(const T& value_min, const T& value_max)
      : value_min(value_min), value_max(value_max) {}
  T operator()(const T& value) const {
    // Clip real and imaginary component separately, as if the clipping bounds
    // form a box in the imaginary plane.
    return T{std::max(std::min(Eigen::numext::real(value),
                               Eigen::numext::real(value_max)),
                      Eigen::numext::real(value_min)),
             std::max(std::min(Eigen::numext::imag(value),
                               Eigen::numext::imag(value_max)),
                      Eigen::numext::imag(value_min))};
  }
  T value_min;
  T value_max;
};

template <typename T>
struct UnaryClipOp<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstFlat& in0_flat,
                  typename TTypes<T>::ConstFlat& in1_flat,
                  typename TTypes<T>::ConstFlat& in2_flat,
                  typename TTypes<T>::Flat& out_flat) const {
    out_flat = in0_flat.unaryExpr(UnaryClipFunc<T>(in1_flat(0), in2_flat(0)));
  }
};

// Binary functor for clip [Tensor, Scalar, Tensor]
template <typename T, bool is_complex = Eigen::NumTraits<T>::IsComplex>
struct BinaryRightClipFunc {
  explicit BinaryRightClipFunc(const T& value_min) : value_min(value_min) {}
  T operator()(const T& value, const T& value_max) const {
    return std::max(std::min(value, value_max), value_min);
  }
  T value_min;
};
template <typename T>
struct BinaryRightClipFunc<T, /*is_complex=*/true> {
  explicit BinaryRightClipFunc(const T& value_min) : value_min(value_min) {}
  T operator()(const T& value, const T& value_max) const {
    // Clip real and imaginary component separately, as if the clipping bounds
    // form a box in the imaginary plane.
    return T{std::max(std::min(Eigen::numext::real(value),
                               Eigen::numext::real(value_max)),
                      Eigen::numext::real(value_min)),
             std::max(std::min(Eigen::numext::imag(value),
                               Eigen::numext::imag(value_max)),
                      Eigen::numext::imag(value_min))};
  }
  T value_min;
};
template <typename T>
struct BinaryRightClipOp<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstFlat& in0_flat,
                  typename TTypes<T>::ConstFlat& in1_flat,
                  typename TTypes<T>::ConstFlat& in2_flat,
                  typename TTypes<T>::Flat& out_flat) const {
    out_flat =
        in0_flat.binaryExpr(in2_flat, BinaryRightClipFunc<T>(in1_flat(0)));
  }
};

// Binary functor for clip [Tensor, Tensor, Scalar]
template <typename T, bool is_complex = Eigen::NumTraits<T>::IsComplex>
struct BinaryLeftClipFunc {
  explicit BinaryLeftClipFunc(const T& value_max) : value_max(value_max) {}
  T operator()(const T& value, const T& value_min) const {
    return std::max(std::min(value, value_max), value_min);
  }
  T value_max;
};
template <typename T>
struct BinaryLeftClipFunc<T, /*is_complex=*/true> {
  explicit BinaryLeftClipFunc(const T& value_max) : value_max(value_max) {}
  T operator()(const T& value, const T& value_min) const {
    // Clip real and imaginary component separately, as if the clipping bounds
    // form a box in the imaginary plane.
    return T{std::max(std::min(Eigen::numext::real(value),
                               Eigen::numext::real(value_max)),
                      Eigen::numext::real(value_min)),
             std::max(std::min(Eigen::numext::imag(value),
                               Eigen::numext::imag(value_max)),
                      Eigen::numext::imag(value_min))};
  }
  T value_max;
};
template <typename T>
struct BinaryLeftClipOp<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstFlat& in0_flat,
                  typename TTypes<T>::ConstFlat& in1_flat,
                  typename TTypes<T>::ConstFlat& in2_flat,
                  typename TTypes<T>::Flat& out_flat) const {
    out_flat =
        in0_flat.binaryExpr(in1_flat, BinaryLeftClipFunc<T>(in2_flat(0)));
  }
};

// Ternary functor for clip [Tensor, Tensor, Tensor]
template <typename T, bool is_complex = Eigen::NumTraits<T>::IsComplex>
struct BinaryClipAboveFunc {
  explicit BinaryClipAboveFunc() = default;
  T operator()(const T& value, const T& value_max) const {
    return std::min(value, value_max);
  }
};
template <typename T>
struct BinaryClipAboveFunc<T, /*is_complex=*/true> {
  explicit BinaryClipAboveFunc() = default;
  T operator()(const T& value, const T& value_max) const {
    // Clip real and imaginary component separately, as if the clipping bounds
    // form a box in the imaginary plane.
    return T{
        std::min(Eigen::numext::real(value), Eigen::numext::real(value_max)),
        std::min(Eigen::numext::imag(value), Eigen::numext::imag(value_max))};
  }
};
template <typename T, bool is_complex = Eigen::NumTraits<T>::IsComplex>
struct BinaryClipBelowFunc {
  explicit BinaryClipBelowFunc() = default;
  T operator()(const T& value, const T& value_min) const {
    return std::max(value, value_min);
  }
};
template <typename T>
struct BinaryClipBelowFunc<T, /*is_complex=*/true> {
  explicit BinaryClipBelowFunc() = default;
  T operator()(const T& value, const T& value_min) const {
    // Clip real and imaginary component separately, as if the clipping bounds
    // form a box in the imaginary plane.
    return T{
        std::max(Eigen::numext::real(value), Eigen::numext::real(value_min)),
        std::max(Eigen::numext::imag(value), Eigen::numext::imag(value_min))};
  }
};

template <typename T>
struct TernaryClipOp<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstFlat& in0_flat,
                  typename TTypes<T>::ConstFlat& in1_flat,
                  typename TTypes<T>::ConstFlat& in2_flat,
                  typename TTypes<T>::Flat& out_flat) const {
    if constexpr (Eigen::NumTraits<T>::IsComplex) {
      out_flat.device(d) =
          in0_flat.binaryExpr(in2_flat, BinaryClipAboveFunc<T>())
              .binaryExpr(in1_flat, BinaryClipBelowFunc<T>());
    } else {
      out_flat.device(d) = in0_flat.cwiseMin(in2_flat).cwiseMax(in1_flat);
    }
  }
};

#define INSTANTIATE_CPU(T)                         \
  template struct UnaryClipOp<CPUDevice, T>;       \
  template struct BinaryRightClipOp<CPUDevice, T>; \
  template struct BinaryLeftClipOp<CPUDevice, T>;  \
  template struct TernaryClipOp<CPUDevice, T>;
INSTANTIATE_CPU(Eigen::half);
INSTANTIATE_CPU(float);
INSTANTIATE_CPU(double);
INSTANTIATE_CPU(bfloat16);
INSTANTIATE_CPU(int8);
INSTANTIATE_CPU(int16);
INSTANTIATE_CPU(int32);
INSTANTIATE_CPU(int64_t);
INSTANTIATE_CPU(uint8);
INSTANTIATE_CPU(uint16);
INSTANTIATE_CPU(std::complex<float>);
INSTANTIATE_CPU(std::complex<double>);
#undef INSTANTIATE_CPU
}  // namespace functor

#define REGISTER_CPU_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ClipByValue").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ClipOp<CPUDevice, type>);

REGISTER_CPU_KERNEL(Eigen::half);
REGISTER_CPU_KERNEL(float);
REGISTER_CPU_KERNEL(double);
REGISTER_CPU_KERNEL(bfloat16);
REGISTER_CPU_KERNEL(int8);
REGISTER_CPU_KERNEL(int16);
REGISTER_CPU_KERNEL(int32);
REGISTER_CPU_KERNEL(int64_t);
REGISTER_CPU_KERNEL(uint8);
REGISTER_CPU_KERNEL(uint16);
REGISTER_CPU_KERNEL(std::complex<float>);
REGISTER_CPU_KERNEL(std::complex<double>);
#undef REGISTER_CPU_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ClipByValue").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      ClipOp<GPUDevice, type>);
REGISTER_GPU_KERNEL(Eigen::half);
REGISTER_GPU_KERNEL(bfloat16);
REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(double);
REGISTER_GPU_KERNEL(int8);
REGISTER_GPU_KERNEL(int16);
REGISTER_GPU_KERNEL(int64_t);
REGISTER_GPU_KERNEL(uint8);
REGISTER_GPU_KERNEL(uint16);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("ClipByValue")
                            .Device(DEVICE_GPU)
                            .HostMemory("t")
                            .HostMemory("clip_value_min")
                            .HostMemory("clip_value_max")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        ClipOp<CPUDevice, int32>);

#undef REGISTER_GPU_KERNEL
#endif

}  // namespace tensorflow
