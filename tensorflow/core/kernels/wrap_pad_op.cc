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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/wrap_pad_op.h"

#include <string>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <typename Device, typename T, typename Tpaddings>
class WrapPadOp : public OpKernel {
  public:
    explicit WrapPadOp(OpKernelConstruction* context) : OpKernel(context) {}

    ~WrapPadOp() override = default;

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
      typename TTypes<Tpaddings>::ConstMatrix paddings = in1.matrix<Tpaddings>();
      for (int d = 0; d < dims; ++d) {
        const Tpaddings before = paddings(d, 0);  // Pad before existing elements.
        const Tpaddings after = paddings(d, 1);   // Pad after existing elements.
        OP_REQUIRES(context, before >= 0 && after >= 0,
                    errors::InvalidArgument(
                        "paddings must be non-negative: ", before, " ", after));
        OP_REQUIRES(
            context, before < in0.dim_size(d) && after < in0.dim_size(d),
            errors::InvalidArgument("paddings must be less than"
                                    " the dimension size: ",
                                    before, ", ", after, " not less than ",
                                    in0.dim_size(d)));
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

#define WRAP_PAD_CASE(i)                                                    \
  case i: {                                                                 \
    functor::WrapPad<Device, T, Tpaddings, i>()(                            \
        context->eigen_device<Device>(), To32Bit(output->tensor<T, i>()),   \
        To32Bit(in0.tensor<T, i>()), paddings);                             \
    break;                                                                  \
  }

      // Invoke the dims-specific implementation.
      switch (dims) {
        WRAP_PAD_CASE(1)
        WRAP_PAD_CASE(2)
        WRAP_PAD_CASE(3)
        WRAP_PAD_CASE(4)
        WRAP_PAD_CASE(5)
        default:
          OP_REQUIRES(context, false,
                      errors::InvalidArgument("Unsupported rank: ",
                                              in0.shape().DebugString()));
      }
#undef WRAP_PAD_CASE
    }
};

using CpuDevice = Eigen::ThreadPoolDevice;
using GpuDevice = Eigen::GpuDevice;

namespace functor {
// Forward declarations of the functor specializations defined in the sharded
// files.
#define DECLARE_CPU_SPEC(T, Tpaddings, i)                     \
  template<>                                                  \
  void WrapPad<CpuDevice, T, Tpaddings, i>::operator()(       \
      const CpuDevice&, typename TTypes<T, i, int32>::Tensor, \
      typename TTypes<T, i, int32>::ConstTensor,              \
      TTypes<Tpaddings>::ConstMatrix);                        \
  extern template struct WrapPad<CpuDevice, T, Tpaddings, i>;

#define DECLARE_CPU_SPECS(T)       \
  DECLARE_CPU_SPEC(T, int32, 1);   \
  DECLARE_CPU_SPEC(T, int32, 2);   \
  DECLARE_CPU_SPEC(T, int32, 3);   \
  DECLARE_CPU_SPEC(T, int32, 4);   \
  DECLARE_CPU_SPEC(T, int32, 5);   \
  DECLARE_CPU_SPEC(T, int64_t, 1); \
  DECLARE_CPU_SPEC(T, int64_t, 2); \
  DECLARE_CPU_SPEC(T, int64_t, 3); \
  DECLARE_CPU_SPEC(T, int64_t, 4); \
  DECLARE_CPU_SPEC(T, int64_t, 5);

TF_CALL_POD_TYPES(DECLARE_CPU_SPECS);
TF_CALL_QUANTIZED_TYPES(DECLARE_CPU_SPECS);
TF_CALL_tstring(DECLARE_CPU_SPECS);

#undef DECLARE_CPU_SPEC
#undef DECLARE_CPU_SPECS
}  // namespace functor

#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(Name("WrapPad")                                  \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<type>("T")                   \
                              .TypeConstraint<int32>("Tpaddings")          \
                              .HostMemory("paddings"),                     \
                          WrapPadOp<CpuDevice, type, int32>);              \
  REGISTER_KERNEL_BUILDER(Name("WrapPad")                                  \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<type>("T")                   \
                              .TypeConstraint<int64_t>("Tpaddings")        \
                              .HostMemory("paddings"),                     \
                          WrapPadOp<CpuDevice, type, int64>);

// Note that we do register for bool type.
TF_CALL_POD_TYPES(REGISTER_KERNEL);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNEL);
TF_CALL_tstring(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
namespace functor {
// Forward declarations of the functor specializations for GPU.
#define DECLARE_GPU_SPEC(T, Tpaddings, i)                     \
  template <>                                                 \
  void WrapPad<GpuDevice, T, Tpaddings, i>::operator()(       \
      const GpuDevice&, typename TTypes<T, i, int32>::Tensor, \
      typename TTypes<T, i, int32>::ConstTensor,              \
      TTypes<Tpaddings>::ConstMatrix);                        \
  extern template struct WrapPad<GpuDevice, T, Tpaddings, i>;

#define DECLARE_GPU_SPECS(T)       \
  DECLARE_GPU_SPEC(T, int32, 1);   \
  DECLARE_GPU_SPEC(T, int32, 2);   \
  DECLARE_GPU_SPEC(T, int32, 3);   \
  DECLARE_GPU_SPEC(T, int32, 4);   \
  DECLARE_GPU_SPEC(T, int32, 5);   \
  DECLARE_GPU_SPEC(T, int64_t, 1); \
  DECLARE_GPU_SPEC(T, int64_t, 2); \
  DECLARE_GPU_SPEC(T, int64_t, 3); \
  DECLARE_GPU_SPEC(T, int64_t, 4); \
  DECLARE_GPU_SPEC(T, int64_t, 5);

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                      \
  REGISTER_KERNEL_BUILDER(Name("WrapPad")                           \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<int32>("Tpaddings")   \
                              .HostMemory("paddings"),              \
                          WrapPadOp<GpuDevice, T, int32>);          \
  REGISTER_KERNEL_BUILDER(Name("WrapPad")                           \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<int64_t>("Tpaddings")  \
                              .HostMemory("paddings"),              \
                          WrapPadOp<GpuDevice, T, int64>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
