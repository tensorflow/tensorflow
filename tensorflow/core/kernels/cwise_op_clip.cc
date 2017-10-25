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

#include "tensorflow/core/kernels/cwise_ops_common.h"

//#include "third_party/eigen3/Eigen/Core/CwiseTernaryOp.h"

namespace tensorflow {

// Unary functor for clip
template <typename T>
struct UnaryClipOp {
  UnaryClipOp(const T& value_min, const T& value_max)
      : value_min_(value_min), value_max_(value_max) {}
  const T operator()(const T& value) const {
    return std::max(std::min(value, value_max_), value_min_);
  }
  T value_min_;
  T value_max_;
};

// Binary functor for clip
template <typename T>
struct BinaryClipMinOp {
  BinaryClipMinOp(const T& value_min) : value_min_(value_min) {}
  const T operator()(const T& value, const T& value_max) const {
    return std::max(std::min(value, value_max), value_min_);
  }
  T value_min_;
};

// Binary functor for clip
template <typename T>
struct BinaryClipMaxOp {
  BinaryClipMaxOp(const T& value_max) : value_max_(value_max) {}
  const T operator()(const T& value, const T& value_min) const {
    return std::max(std::min(value, value_max_), value_min);
  }
  T value_max_;
};

// Basic coefficient-wise tenary operations.
// This is the case for example of the clip_by_value.
//   Device: E.g., CPUDevice, GPUDevice.
//   Functor: defined above. E.g., functor::clip.
template <typename Device, typename T>
class TenaryOp : public OpKernel {
 public:
  explicit TenaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    const Tensor& in2 = ctx->input(2);

    auto in0_flat = in0.flat<T>();
    auto in1_flat = in1.flat<T>();
    auto in2_flat = in2.flat<T>();
    const Device& d = ctx->eigen_device<Device>();

    Tensor* out = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->forward_input_or_allocate_output({0}, 0, in0.shape(), &out));
    auto out_flat = out->flat<T>();
    if (in1.shape() == in2.shape()) {
      if (in0.shape() == in1.shape()) {
        out_flat = in0_flat.cwiseMin(in2_flat).cwiseMax(in1_flat);
      } else {
        OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(in1.shape()),
                    errors::InvalidArgument(
                        "clip_value_min and clip_value_max must be either of "
                        "the same shape as input, or a scalar. ",
                        "input shape: ", in0.shape().DebugString(),
                        "clip_value_min shape: ", in1.shape().DebugString(),
                        "clip_value_max shape: ", in2.shape().DebugString()));
        out_flat = in0_flat.unaryExpr(UnaryClipOp<T>(in1_flat(0), in2_flat(0)));
      }
    } else {
      if (in0.shape() == in1.shape()) {
        OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(in2.shape()),
                    errors::InvalidArgument(
                        "clip_value_min and clip_value_max must be either of "
                        "the same shape as input, or a scalar. ",
                        "input shape: ", in0.shape().DebugString(),
                        "clip_value_min shape: ", in1.shape().DebugString(),
                        "clip_value_max shape: ", in2.shape().DebugString()));
        out_flat =
            in0_flat.binaryExpr(in1_flat, BinaryClipMaxOp<T>(in2_flat(0)));

      } else {
        OP_REQUIRES(ctx, (in0.shape() == in2.shape() &&
                          TensorShapeUtils::IsScalar(in1.shape())),
                    errors::InvalidArgument(
                        "clip_value_min and clip_value_max must be either of "
                        "the same shape as input, or a scalar. ",
                        "input shape: ", in0.shape().DebugString(),
                        "clip_value_min shape: ", in1.shape().DebugString(),
                        "clip_value_max shape: ", in2.shape().DebugString()));
        out_flat =
            in0_flat.binaryExpr(in2_flat, BinaryClipMinOp<T>(in1_flat(0)));
      }
    }
  }
};

#define REGISTER_CPU_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ClipByValue").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      TenaryOp<CPUDevice, type>);

REGISTER_CPU_KERNEL(Eigen::half);
REGISTER_CPU_KERNEL(float);
REGISTER_CPU_KERNEL(double);
REGISTER_CPU_KERNEL(int8);
REGISTER_CPU_KERNEL(int16);
REGISTER_CPU_KERNEL(int32);
REGISTER_CPU_KERNEL(int64);
REGISTER_CPU_KERNEL(uint8);
REGISTER_CPU_KERNEL(uint16);

#undef REGISTER_CPU_KERNEL

#if GOOGLE_CUDA
// REGISTER3(BinaryOp, GPU, "Add", functor::add, float, Eigen::half, double);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("ClipByValue")
                            .Device(DEVICE_GPU)
                            .HostMemory("t")
                            .HostMemory("clip_value_min")
                            .HostMemory("clip_value_min")
                            .TypeConstraint<int32>("T"),
                        TenaryOp<CPUDevice, int32>);
#endif

}  // namespace tensorflow
