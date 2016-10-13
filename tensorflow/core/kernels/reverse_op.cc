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

// See docs in ../ops/array_ops.cc
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/reverse_op.h"
#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class ReverseOp : public OpKernel {
 public:
  explicit ReverseOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& dims = context->input(1);

    if (TensorShapeUtils::IsScalar(input.shape())) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, input.shape(), &output));
      output->scalar<T>() = input.scalar<T>();

    } else {
      const int input_dims = input.dims();
      OP_REQUIRES(context, TensorShapeUtils::IsVector(dims.shape()),
                  errors::InvalidArgument("'dims' must be 1-dimension, not ",
                                          dims.dims()));

      OP_REQUIRES(
          context, input_dims == dims.dim_size(0),
          errors::InvalidArgument(
              "'dims' must have the same number of values as 'input' has "
              "dimensions. 'input' has ",
              input_dims, "'dims' has ", dims.dim_size(0), " values"));
      OP_REQUIRES(context, input_dims <= 8,
                  errors::Unimplemented(
                      "reverse is not implemented for tensors of rank > 8."));

      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, input.shape(), &output));

#define HANDLE_REVERSE(NDIMS)                                      \
  case NDIMS:                                                      \
    functor::Reverse<Device, T, NDIMS>()(                          \
        context->eigen_device<Device>(), input.tensor<T, NDIMS>(), \
        dims.vec<bool>(), output->tensor<T, NDIMS>());             \
    return;

      switch (input_dims) {
        HANDLE_REVERSE(0);
        HANDLE_REVERSE(1);
        HANDLE_REVERSE(2);
        HANDLE_REVERSE(3);
        HANDLE_REVERSE(4);
        HANDLE_REVERSE(5);
        HANDLE_REVERSE(6);
        HANDLE_REVERSE(7);
        HANDLE_REVERSE(8);
      }
#undef HANDLE_REVERSE
    }
  }
};

#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("Reverse")             \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("dims"),    \
                          ReverseOp<CPUDevice, T>)

TF_CALL_POD_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA

// Forward declarations of the function specializations for GPU (to prevent
// building the GPU versions here, they will be built compiling _gpu.cu.cc).
namespace functor {
#define DECLARE_GPU_SPEC_DIM(T, DIM)                                  \
  template <>                                                         \
  void Reverse<GPUDevice, T, DIM>::operator()(                        \
      const GPUDevice& d, typename TTypes<T, DIM>::ConstTensor input, \
      typename TTypes<bool, 1>::ConstTensor dims,                     \
      typename TTypes<T, DIM>::Tensor output);                        \
  extern template struct Reverse<GPUDevice, T, DIM>;
#define DECLARE_GPU_SPEC(T)  \
  DECLARE_GPU_SPEC_DIM(T, 0) \
  DECLARE_GPU_SPEC_DIM(T, 1) \
  DECLARE_GPU_SPEC_DIM(T, 2) \
  DECLARE_GPU_SPEC_DIM(T, 3) \
  DECLARE_GPU_SPEC_DIM(T, 4) \
  DECLARE_GPU_SPEC_DIM(T, 5) \
  DECLARE_GPU_SPEC_DIM(T, 6) \
  DECLARE_GPU_SPEC_DIM(T, 7) \
  DECLARE_GPU_SPEC_DIM(T, 8)

TF_CALL_uint8(DECLARE_GPU_SPEC);
TF_CALL_int8(DECLARE_GPU_SPEC);
TF_CALL_bool(DECLARE_GPU_SPEC);
TF_CALL_half(DECLARE_GPU_SPEC);
TF_CALL_float(DECLARE_GPU_SPEC);
TF_CALL_double(DECLARE_GPU_SPEC);
TF_CALL_complex64(DECLARE_GPU_SPEC);
TF_CALL_complex128(DECLARE_GPU_SPEC);
#undef DECLARE_GPU_SPEC
#undef DECLARE_GPU_SPEC_DIM
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                        \
  REGISTER_KERNEL_BUILDER(Name("Reverse")             \
                              .Device(DEVICE_GPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("dims"),    \
                          ReverseOp<GPUDevice, T>)
TF_CALL_uint8(REGISTER_GPU_KERNEL);
TF_CALL_int8(REGISTER_GPU_KERNEL);
// TODO decide whether we want to enable the bool kernel.
// TF_CALL_bool(REGISTER_GPU_KERNEL);
TF_CALL_half(REGISTER_GPU_KERNEL);
TF_CALL_float(REGISTER_GPU_KERNEL);
TF_CALL_double(REGISTER_GPU_KERNEL);
TF_CALL_complex64(REGISTER_GPU_KERNEL);
TF_CALL_complex128(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Reverse")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("tensor")
                            .HostMemory("dims")
                            .HostMemory("output"),
                        ReverseOp<CPUDevice, int32>);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
