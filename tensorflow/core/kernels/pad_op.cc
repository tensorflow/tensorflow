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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/pad_op.h"

#include <memory>
#include <string>
#include <utility>

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

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif // TENSORFLOW_USE_SYCL

template <typename Device, typename T>
class PadOp : public OpKernel {
 public:
  explicit PadOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& in0 = context->input(0);
    const Tensor& in1 = context->input(1);
    const int dims = in0.dims();
    static const int kMinDims = 0;
    static const int kMaxDims = 6;
    OP_REQUIRES(context, kMinDims <= dims && dims <= kMaxDims,
                errors::Unimplemented("inputs rank not in [", kMinDims, ",",
                                      kMaxDims, "]: ", dims));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsMatrix(in1.shape()) && in1.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                in1.shape().DebugString()));
    const int fixed_dims =
        (allow_legacy_scalars() && dims == 0 && in1.dim_size(0) == 1) ? 1
                                                                      : dims;
    OP_REQUIRES(
        context, fixed_dims == in1.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of paddings must be the rank of inputs",
            in1.shape().DebugString(), " ", in0.shape().DebugString()));

    T pad_value(0);
    if (context->num_inputs() == 3) {
      const Tensor& constant_values = context->input(2);
      OP_REQUIRES(
          context, TensorShapeUtils::IsScalar(constant_values.shape()),
          errors::InvalidArgument("constant_values must be a scalar. Found: ",
                                  constant_values.shape().DebugString()));
      pad_value = context->input(2).scalar<T>()();
    }

    // Compute the shape of the output tensor, and allocate it.
    TensorShape output_shape;
    TTypes<int32>::ConstMatrix paddings = in1.matrix<int32>();
    for (int d = 0; d < fixed_dims; ++d) {
      const int32 before_d = paddings(d, 0);  // Pad before existing elements.
      const int32 after_d = paddings(d, 1);   // Pad after existing elements.
      OP_REQUIRES(context, before_d >= 0 && after_d >= 0,
                  errors::InvalidArgument("Paddings must be non-negative: ",
                                          before_d, " ", after_d));
      const int64 size_d =
          (allow_legacy_scalars() && d == in0.dims()) ? 1 : in0.dim_size(d);
      output_shape.AddDim(before_d + size_d + after_d);
    }

    // If there is no padding to be done, forward the input to output.
    if (output_shape.num_elements() == in0.NumElements()) {
      // When num_elements == 0, shape may have changed.
      Tensor out;
      CHECK(out.CopyFrom(in0, output_shape));
      context->set_output(0, out);
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    // Invoke the dims-specific implementation.
    switch (fixed_dims) {
      case 0:
        Operate<0>(context, in0.tensor<T, 0>(), paddings, pad_value, output);
        break;
      case 1:
        // TODO(irving): Once Pad doesn't need a scalar special case,
        // change flat to tensor.  That is, once !allow_legacy_scalars().
        Operate<1>(context, in0.flat<T>(), paddings, pad_value, output);
        break;
      case 2:
        Operate<2>(context, in0.tensor<T, 2>(), paddings, pad_value, output);
        break;
      case 3:
        Operate<3>(context, in0.tensor<T, 3>(), paddings, pad_value, output);
        break;
      case 4:
        Operate<4>(context, in0.tensor<T, 4>(), paddings, pad_value, output);
        break;
      case 5:
        Operate<5>(context, in0.tensor<T, 5>(), paddings, pad_value, output);
        break;
      case 6:
        Operate<6>(context, in0.tensor<T, 6>(), paddings, pad_value, output);
        break;
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Only ranks up to 6 supported: ",
                                            in0.shape().DebugString()));
    }
  }

 private:
  template <int Dims>
  void Operate(OpKernelContext* context,
               typename TTypes<T, Dims>::ConstTensor input,
               TTypes<int32>::ConstMatrix paddings, T pad_value,
               Tensor* output) {
    CHECK_EQ(Dims, paddings.dimension(0));
    CHECK_EQ(2, paddings.dimension(1));
    Eigen::array<std::pair<int32, int32>, Dims> paddings_array;
    for (int i = 0; i < Dims; ++i) {
      paddings_array[i] = std::make_pair(paddings(i, 0), paddings(i, 1));
    }
    functor::Pad<Device, T, Dims> functor;
    functor(context->eigen_device<Device>(), output->tensor<T, Dims>(), input,
            paddings_array, pad_value);
  }
};

#define REGISTER_KERNEL(type)                                 \
  REGISTER_KERNEL_BUILDER(Name("Pad")                         \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("T")      \
                              .HostMemory("paddings"),        \
                          PadOp<CPUDevice, type>);            \
  REGISTER_KERNEL_BUILDER(Name("PadV2")                       \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("T")      \
                              .HostMemory("paddings")         \
                              .HostMemory("constant_values"), \
                          PadOp<CPUDevice, type>);

TF_CALL_POD_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Dims)                                         \
  template <>                                                             \
  void Pad<GPUDevice, T, Dims>::operator()(                               \
      const GPUDevice& d, typename TTypes<T, Dims>::Tensor output,        \
      typename TTypes<T, Dims>::ConstTensor input,                        \
      Eigen::array<std::pair<int32, int32>, Dims> paddings, T pad_value); \
  extern template struct Pad<GPUDevice, T, Dims>;

#define DECLARE_GPU_SPECS(T) \
  DECLARE_GPU_SPEC(T, 0);    \
  DECLARE_GPU_SPEC(T, 1);    \
  DECLARE_GPU_SPEC(T, 2);    \
  DECLARE_GPU_SPEC(T, 3);    \
  DECLARE_GPU_SPEC(T, 4);    \
  DECLARE_GPU_SPEC(T, 5);    \
  DECLARE_GPU_SPEC(T, 6);

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("Pad")                             \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<T>("T")             \
                              .TypeConstraint<int32>("Tpaddings") \
                              .HostMemory("paddings"),            \
                          PadOp<GPUDevice, T>);                   \
  REGISTER_KERNEL_BUILDER(Name("PadV2")                           \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<T>("T")             \
                              .TypeConstraint<int32>("Tpaddings") \
                              .HostMemory("paddings")             \
                              .HostMemory("constant_values"),     \
                          PadOp<GPUDevice, T>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Pad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tpaddings")
                            .HostMemory("input")
                            .HostMemory("paddings")
                            .HostMemory("output"),
                        PadOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("PadV2")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tpaddings")
                            .HostMemory("input")
                            .HostMemory("paddings")
                            .HostMemory("constant_values")
                            .HostMemory("output"),
                        PadOp<CPUDevice, int32>);
#endif

#ifdef TENSORFLOW_USE_SYCL
// Registration of the GPU implementations.
#define REGISTER_SYCL_KERNEL(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("Pad")                             \
                              .Device(DEVICE_SYCL)                \
                              .TypeConstraint<T>("T")             \
                              .TypeConstraint<int32>("Tpaddings") \
                              .HostMemory("paddings"),            \
                          PadOp<SYCLDevice, T>);                  \
  REGISTER_KERNEL_BUILDER(Name("PadV2")                           \
                              .Device(DEVICE_SYCL)                \
                              .TypeConstraint<T>("T")             \
                              .TypeConstraint<int32>("Tpaddings") \
                              .HostMemory("paddings")             \
                              .HostMemory("constant_values"),     \
                          PadOp<SYCLDevice, T>)

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SYCL_KERNEL);
REGISTER_KERNEL_BUILDER(Name("Pad")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tpaddings")
                            .HostMemory("input")
                            .HostMemory("paddings")
                            .HostMemory("output"),
                        PadOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("PadV2")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tpaddings")
                            .HostMemory("input")
                            .HostMemory("paddings")
                            .HostMemory("constant_values")
                            .HostMemory("output"),
                        PadOp<CPUDevice, int32>);
#undef REGISTER_SYCL_KERNEL
#endif // TENSORFLOW_USE_SYCL

}  // end namespace tensorflow
