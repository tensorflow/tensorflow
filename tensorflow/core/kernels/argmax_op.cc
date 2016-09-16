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

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/argmax_op.h"

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename ArgFunctor>
class ArgOp : public OpKernel {
 public:
  explicit ArgOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& dimension = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(dimension.shape()),
                errors::InvalidArgument(
                    "dim must be a scalar, but received tensor of shape: ",
                    dimension.shape().DebugString()));

    const int32 dim = internal::SubtleMustCopy(dimension.scalar<int32>()());
    const int input_dims = input.dims();

    OP_REQUIRES(context, dim >= 0, errors::InvalidArgument("dim must be >= 0"));
    OP_REQUIRES(context, dim < input_dims,
                errors::InvalidArgument("Minimum tensor rank: ", dim + 1,
                                        " but got: ", input_dims));
    OP_REQUIRES(
        context, input.dim_size(dim) > 0,
        errors::InvalidArgument("Reduction axis ", dim, " is empty in shape ",
                                input.shape().DebugString()));

    TensorShape output_shape;
    const TensorShape& input_shape = input.shape();
    for (int d = 0; d < input_dims - 1; ++d) {
      output_shape.AddDim(input_shape.dim_size((d < dim) ? d : d + 1));
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

#define HANDLE_DIM(NDIM)                                         \
  case NDIM:                                                     \
    ArgFunctor::Reduce##NDIM(context->eigen_device<Device>(),    \
                             input.tensor<T, NDIM>(), dim,       \
                             output->tensor<int64, NDIM - 1>()); \
    break;

    switch (input_dims) {
      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);

      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "ArgOp : Unhandled input dimensions: ", input_dims));
    }
  }
#undef HANDLE_DIM

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ArgOp);
};

template <typename Device, typename T>
class ArgMaxOp : public ArgOp<Device, T, functor::ArgMax<Device, T> > {
 public:
  explicit ArgMaxOp(OpKernelConstruction* context)
      : ArgOp<Device, T, functor::ArgMax<Device, T> >(context) {}
};

template <typename Device, typename T>
class ArgMinOp : public ArgOp<Device, T, functor::ArgMin<Device, T> > {
 public:
  explicit ArgMinOp(OpKernelConstruction* context)
      : ArgOp<Device, T, functor::ArgMin<Device, T> >(context) {}
};

#define REGISTER_ARGMAX(type)                            \
  REGISTER_KERNEL_BUILDER(Name("ArgMax")                 \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("dimension"),  \
                          ArgMaxOp<CPUDevice, type>);    \
  REGISTER_KERNEL_BUILDER(Name("ArgMin")                 \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("dimension"),  \
                          ArgMinOp<CPUDevice, type>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_ARGMAX);

#if GOOGLE_CUDA

// Forward declarations of the functor specializations for GPU.
namespace functor {

#define DECLARE_GPU_SPEC(T, Dims)                                              \
  template <>                                                                  \
  void ArgMax<GPUDevice, T>::Reduce##Dims(                                     \
      const GPUDevice& d, typename TTypes<T, Dims>::ConstTensor input,         \
      const int32 dimension, typename TTypes<int64, Dims - 1>::Tensor output); \
  template <>                                                                  \
  void ArgMin<GPUDevice, T>::Reduce##Dims(                                     \
      const GPUDevice& d, typename TTypes<T, Dims>::ConstTensor input,         \
      const int32 dimension, typename TTypes<int64, Dims - 1>::Tensor output);

#define DECLARE_GPU_SPECS(T) \
  DECLARE_GPU_SPEC(T, 1);    \
  DECLARE_GPU_SPEC(T, 2);    \
  DECLARE_GPU_SPEC(T, 3);    \
  DECLARE_GPU_SPEC(T, 4);    \
  DECLARE_GPU_SPEC(T, 5);

#define DECLARE_GPU_CLASS(T)                   \
  extern template struct ArgMax<GPUDevice, T>; \
  extern template struct ArgMin<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_CLASS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_CLASS

}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_ARGMAX_GPU(type)                        \
  REGISTER_KERNEL_BUILDER(Name("ArgMax")                 \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("dimension"),  \
                          ArgMaxOp<GPUDevice, type>);    \
  REGISTER_KERNEL_BUILDER(Name("ArgMin")                 \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("dimension"),  \
                          ArgMinOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_ARGMAX_GPU);

#undef REGISTER_ARGMAX_GPU

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
