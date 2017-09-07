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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/mirror_pad_op.h"

#include <string>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/mirror_pad_mode.h"

namespace tensorflow {

template <typename Device, typename T>
class MirrorPadOp : public OpKernel {
 public:
  explicit MirrorPadOp(OpKernelConstruction* context) : OpKernel(context) {
    MirrorPadMode mode;
    OP_REQUIRES_OK(context, context->GetAttr("mode", &mode));

    switch (mode) {
      case MirrorPadMode::SYMMETRIC: {
        offset_ = 0;
        break;
      }
      case MirrorPadMode::REFLECT: {
        offset_ = 1;
        break;
      }
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "mode must be either REFLECT or SYMMETRIC."));
    }
  }

  ~MirrorPadOp() override = default;

  void Compute(OpKernelContext* context) override {
    const Tensor& in0 = context->input(0);
    const Tensor& in1 = context->input(1);
    const int dims = in0.dims();
    constexpr int kMinDims = 0;
    constexpr int kMaxDims = 5;
    OP_REQUIRES(context, kMinDims <= dims && dims <= kMaxDims,
                errors::Unimplemented("inputs rank not in [", kMinDims, ",",
                                      kMaxDims, "]: ", dims));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsMatrix(in1.shape()) && in1.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                in1.shape().DebugString()));
    OP_REQUIRES(
        context, dims == in1.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of paddings must be the rank of inputs",
            in1.shape().DebugString(), ", ", in0.shape().DebugString()));

    // Compute the shape of the output tensor, and allocate it.
    TensorShape output_shape;
    TTypes<int32>::ConstMatrix paddings = in1.matrix<int32>();
    for (int d = 0; d < dims; ++d) {
      const int32 before = paddings(d, 0);  // Pad before existing elements.
      const int32 after = paddings(d, 1);   // Pad after existing elements.
      OP_REQUIRES(context, before >= 0 && after >= 0,
                  errors::InvalidArgument("paddings must be non-negative: ",
                                          before, " ", after));
      if (offset_ == 0) {  // SYMMETRIC mode.
        OP_REQUIRES(context,
                    before <= in0.dim_size(d) && after <= in0.dim_size(d),
                    errors::InvalidArgument("paddings must be no greater "
                                            "than the dimension size: ",
                                            before, ", ", after,
                                            " greater than ", in0.dim_size(d)));
      } else if (offset_ == 1) {  // REFLECT mode.
        OP_REQUIRES(
            context, before < in0.dim_size(d) && after < in0.dim_size(d),
            errors::InvalidArgument("paddings must be less than"
                                    " the dimension size: ",
                                    before, ", ", after, " not less than ",
                                    in0.dim_size(d)));
      }

      output_shape.AddDim(before + in0.dim_size(d) + after);
    }

    if (output_shape.num_elements() == in0.NumElements()) {
      // When num_elements == 0, shape may have changed.
      Tensor out;
      CHECK(out.CopyFrom(in0, output_shape));
      context->set_output(0, out);
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

#define MIRROR_PAD_CASE(i)                                                \
  case i: {                                                               \
    functor::MirrorPad<Device, T, i>()(                                   \
        context->eigen_device<Device>(), To32Bit(output->tensor<T, i>()), \
        To32Bit(in0.tensor<T, i>()), paddings, offset_);                  \
    break;                                                                \
  }

    // Invoke the dims-specific implementation.
    switch (dims) {
      MIRROR_PAD_CASE(1)
      MIRROR_PAD_CASE(2)
      MIRROR_PAD_CASE(3)
      MIRROR_PAD_CASE(4)
      MIRROR_PAD_CASE(5)
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Unsupported rank: ",
                                            in0.shape().DebugString()));
    }
#undef MIRROR_PAD_CASE
  }

 private:
  int offset_;
};

using CpuDevice = Eigen::ThreadPoolDevice;
using GpuDevice = Eigen::GpuDevice;

namespace functor {
// Forward declarations of the functor specializations defined in the sharded
// files.
#define DECLARE_CPU_SPEC(T, i)                                               \
  template <>                                                                \
  void MirrorPad<CpuDevice, T, i>::operator()(                               \
      const CpuDevice&, typename TTypes<T, i, int32>::Tensor,                \
      typename TTypes<T, i, int32>::ConstTensor, TTypes<int32>::ConstMatrix, \
      int);                                                                  \
  extern template struct MirrorPad<CpuDevice, T, i>;

#define DECLARE_CPU_SPECS(T) \
  DECLARE_CPU_SPEC(T, 1);    \
  DECLARE_CPU_SPEC(T, 2);    \
  DECLARE_CPU_SPEC(T, 3);    \
  DECLARE_CPU_SPEC(T, 4);    \
  DECLARE_CPU_SPEC(T, 5);

TF_CALL_POD_TYPES(DECLARE_CPU_SPECS);

#undef DECLARE_CPU_SPEC
#undef DECLARE_CPU_SPECS
}  // namespace functor

#define REGISTER_KERNEL(type)                                     \
  REGISTER_KERNEL_BUILDER(Name("MirrorPad")                       \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int32>("Tpaddings") \
                              .HostMemory("paddings"),            \
                          MirrorPadOp<CpuDevice, type>);

// Note that we do register for bool type, but not in the gradient op.
TF_CALL_POD_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
namespace functor {
// Forward declarations of the functor specializations for GPU.
#define DECLARE_GPU_SPEC(T, i)                                               \
  template <>                                                                \
  void MirrorPad<GpuDevice, T, i>::operator()(                               \
      const GpuDevice&, typename TTypes<T, i, int32>::Tensor,                \
      typename TTypes<T, i, int32>::ConstTensor, TTypes<int32>::ConstMatrix, \
      int);                                                                  \
  extern template struct MirrorPad<GpuDevice, T, i>;

#define DECLARE_GPU_SPECS(T) \
  DECLARE_GPU_SPEC(T, 1);    \
  DECLARE_GPU_SPEC(T, 2);    \
  DECLARE_GPU_SPEC(T, 3);    \
  DECLARE_GPU_SPEC(T, 4);    \
  DECLARE_GPU_SPEC(T, 5);

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("MirrorPad")                       \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<T>("T")             \
                              .TypeConstraint<int32>("Tpaddings") \
                              .HostMemory("paddings"),            \
                          MirrorPadOp<GpuDevice, T>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA

// Gradient op.
template <typename Device, typename T>
class MirrorPadGradOp : public OpKernel {
 public:
  explicit MirrorPadGradOp(OpKernelConstruction* context) : OpKernel(context) {
    MirrorPadMode mode;
    OP_REQUIRES_OK(context, context->GetAttr("mode", &mode));

    switch (mode) {
      case MirrorPadMode::SYMMETRIC: {
        offset_ = 0;
        break;
      }
      case MirrorPadMode::REFLECT: {
        offset_ = 1;
        break;
      }
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "mode must be either REFLECT or SYMMETRIC."));
    }
  }

  ~MirrorPadGradOp() override = default;

  void Compute(OpKernelContext* context) override {
    const Tensor& in0 = context->input(0);
    const Tensor& in1 = context->input(1);
    const int dims = in0.dims();
    constexpr int kMinDims = 0;
    constexpr int kMaxDims = 5;
    OP_REQUIRES(context, kMinDims <= dims && dims <= kMaxDims,
                errors::Unimplemented("inputs rank not in [", kMinDims, ",",
                                      kMaxDims, "]: ", dims));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsMatrix(in1.shape()) && in1.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                in1.shape().DebugString()));
    OP_REQUIRES(
        context, dims == in1.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of paddings must be the rank of inputs",
            in1.shape().DebugString(), " ", in0.shape().DebugString()));

    // Compute the shape of the output tensor, and allocate it.
    TensorShape output_shape;
    TTypes<int32>::ConstMatrix paddings = in1.matrix<int32>();
    for (int d = 0; d < dims; ++d) {
      const int32 before = paddings(d, 0);  // Pad before existing elements.
      const int32 after = paddings(d, 1);   // Pad after existing elements.
      OP_REQUIRES(context, before >= 0 && after >= 0,
                  errors::InvalidArgument("Paddings must be non-negative: ",
                                          before, ", ", after));

      const int64 out_size = in0.dim_size(d) - (before + after);
      if (offset_ == 0) {  // SYMMETRIC mode.
        OP_REQUIRES(context, before <= out_size && after <= out_size,
                    errors::InvalidArgument("paddings must be no greater "
                                            "than the output dimension size: ",
                                            before, ", ", after,
                                            " greater than ", out_size));
      } else if (offset_ == 1) {  // REFLECT mode.
        OP_REQUIRES(context, before < out_size && after < out_size,
                    errors::InvalidArgument("paddings must be less than"
                                            " the output dimension size: ",
                                            before, ", ", after,
                                            " not less than ", out_size));
      }
      output_shape.AddDim(out_size);
    }

    if (output_shape == in0.shape()) {
      context->set_output(0, in0);
      return;
    }

    Tensor scratch;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   in0.shape(), &scratch));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

#define MIRROR_PAD_GRAD_CASE(k)                                           \
  case k: {                                                               \
    functor::MirrorPadGrad<Device, T, k>()(                               \
        context->eigen_device<Device>(), To32Bit(output->tensor<T, k>()), \
        To32Bit(in0.tensor<T, k>()), paddings, offset_,                   \
        To32Bit(scratch.tensor<T, k>()));                                 \
    break;                                                                \
  }

    // Invoke the dims-specific implementation.
    switch (dims) {
      MIRROR_PAD_GRAD_CASE(1);
      MIRROR_PAD_GRAD_CASE(2);
      MIRROR_PAD_GRAD_CASE(3);
      MIRROR_PAD_GRAD_CASE(4);
      MIRROR_PAD_GRAD_CASE(5);
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Unsupported rank: ",
                                            in0.shape().DebugString()));
    }
#undef MIRROR_PAD_GRAD_CASE
  }

 private:
  int offset_;
};

namespace functor {
// Forward declarations of the functor specializations defined in the sharded
// files.
#define DECLARE_CPU_SPEC(T, k)                                               \
  template <>                                                                \
  void MirrorPadGrad<CpuDevice, T, k>::operator()(                           \
      const CpuDevice&, typename TTypes<T, k, int32>::Tensor,                \
      typename TTypes<T, k, int32>::ConstTensor, TTypes<int32>::ConstMatrix, \
      int, typename TTypes<T, k, int32>::Tensor);                            \
  extern template struct MirrorPadGrad<CpuDevice, T, k>;

#define DECLARE_CPU_SPECS(T) \
  DECLARE_CPU_SPEC(T, 1);    \
  DECLARE_CPU_SPEC(T, 2);    \
  DECLARE_CPU_SPEC(T, 3);    \
  DECLARE_CPU_SPEC(T, 4);    \
  DECLARE_CPU_SPEC(T, 5);

TF_CALL_NUMBER_TYPES(DECLARE_CPU_SPECS);
#undef DECLARE_CPU_SPECS
#undef DECLARE_CPU_SPEC
}  // namespace functor

#define REGISTER_KERNEL(type)                                     \
  REGISTER_KERNEL_BUILDER(Name("MirrorPadGrad")                   \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int32>("Tpaddings") \
                              .HostMemory("paddings"),            \
                          MirrorPadGradOp<CpuDevice, type>);

TF_CALL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
namespace functor {
// Forward declarations of the functor specializations for GPU.
#define DECLARE_GPU_SPEC(T, k)                                               \
  template <>                                                                \
  void MirrorPadGrad<GpuDevice, T, k>::operator()(                           \
      const GpuDevice&, typename TTypes<T, k, int32>::Tensor,                \
      typename TTypes<T, k, int32>::ConstTensor, TTypes<int32>::ConstMatrix, \
      int, typename TTypes<T, k, int32>::Tensor);                            \
  extern template struct MirrorPadGrad<GpuDevice, T, k>;

#define DECLARE_GPU_SPECS(T) \
  DECLARE_GPU_SPEC(T, 1);    \
  DECLARE_GPU_SPEC(T, 2);    \
  DECLARE_GPU_SPEC(T, 3);    \
  DECLARE_GPU_SPEC(T, 4);    \
  DECLARE_GPU_SPEC(T, 5);

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("MirrorPadGrad")                   \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<T>("T")             \
                              .TypeConstraint<int32>("Tpaddings") \
                              .HostMemory("paddings"),            \
                          MirrorPadGradOp<GpuDevice, T>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
