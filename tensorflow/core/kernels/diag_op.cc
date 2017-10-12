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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/diag_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/logging.h"
#include <algorithm>

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Generate the diagonal tensor with the diagonal set to the input tensor.
template <typename Device, typename T>
class DiagOp : public OpKernel {
 public:
  explicit DiagOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& diagonal = context->input(0);
    const int num_dims = diagonal.dims();
    OP_REQUIRES(context, 0 != num_dims, errors::InvalidArgument(
          "Input must be at least rank 1, got 0"));
    TensorShape out_shape;
    for (int i = 0; i < num_dims; ++i) {
      out_shape.AddDim(diagonal.dim_size(i));
    }
    for (int i = 0; i < num_dims; ++i) {
      out_shape.AddDim(diagonal.dim_size(i));
    }
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, out_shape, &output_tensor));
    functor::DiagFunctor<Device, T> diagFunc;
    diagFunc(context->eigen_device<Device>(),
             diagonal.NumElements(),
             diagonal.flat<T>().data(),
             output_tensor->flat<T>().data());
    return;
  }
};

// Extract the diagonal tensor with the diagonal set to the input tensor.
template <typename Device, typename T>
class DiagPartOp : public OpKernel {
 public:
  explicit DiagPartOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor = context->input(0);
    const int num_dims = tensor.dims();
    const int out_dims = num_dims / 2;
    OP_REQUIRES(context, 0 == num_dims % 2,
                errors::InvalidArgument("The rank of the tensor should be \
                                         even and positive, got shape ",
                                        tensor.shape().DebugString()));
    for (int i = 0; i < out_dims; i++){
      OP_REQUIRES(context, tensor.dim_size(i) == tensor.dim_size(i + out_dims),
                  errors::InvalidArgument(
                    "Invalid shape ", tensor.shape().DebugString(),
                    ": dimensions ", i, " and ", i + out_dims, " do not match.")
                  );
    }

    TensorShape out_shape;
    for (int i = 0; i < out_dims; ++i) {
      out_shape.AddDim(tensor.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, out_shape, &output));
    functor::DiagPartFunctor<Device, T> diagPartFunc;
    diagPartFunc(context->eigen_device<Device>(),
                 out_shape.num_elements(),
                 tensor.flat<T>().data(),
                 output->flat<T>().data());
    return;
  }
};

// Implementation of the functor specialization for CPU.
namespace functor {
template <typename T>
struct DiagFunctor<CPUDevice, T> {
  void operator() (const CPUDevice& device, const int64 size,
                   const T* in, T* out) {
    std::fill(out, out + size * size, T());
    for (int64 index = 0; index < size; index++) {
      out[(1 + size) * index] = in[index];
    }
  }
};

template <typename T>
struct DiagPartFunctor<CPUDevice, T> {
  void operator() (const CPUDevice& device, const int64 size,
                   const T* in, T* out) {
    for (int64 index = 0; index < size; index++) {
      out[index] = in[(1 + size) * index];
    }
  }
};
}  // namespace functor


// Register the CPU kernels.
#define REGISTER_DIAGOP(T)                                    \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Diag").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DiagOp<CPUDevice, T>)

REGISTER_DIAGOP(double);
REGISTER_DIAGOP(float);
REGISTER_DIAGOP(int32);
REGISTER_DIAGOP(int64);
REGISTER_DIAGOP(complex64);
REGISTER_DIAGOP(complex128);
#undef REGISTER_DIAGOP

#define REGISTER_DIAGPARTOP(T)                                    \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("DiagPart").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DiagPartOp<CPUDevice, T>)

REGISTER_DIAGPARTOP(double);
REGISTER_DIAGPARTOP(float);
REGISTER_DIAGPARTOP(int32);
REGISTER_DIAGPARTOP(int64);
REGISTER_DIAGPARTOP(complex64);
REGISTER_DIAGPARTOP(complex128);
#undef REGISTER_DIAGPARTOP

// Register the GPU kernels.
#ifdef GOOGLE_CUDA

// Forward declarations of the functor specializations for GPU.
namespace functor {
extern template struct DiagFunctor<GPUDevice, double>;
extern template struct DiagFunctor<GPUDevice, float>;
extern template struct DiagFunctor<GPUDevice, int32>;
extern template struct DiagFunctor<GPUDevice, int64>;
extern template struct DiagFunctor<GPUDevice, complex64>;
extern template struct DiagFunctor<GPUDevice, complex128>;
}  // namespace functor

#define REGISTER_DIAGOP_GPU(T)                                \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Diag").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DiagOp<GPUDevice, T>);

REGISTER_DIAGOP_GPU(double);
REGISTER_DIAGOP_GPU(float);
REGISTER_DIAGOP_GPU(int32);
REGISTER_DIAGOP_GPU(int64);
REGISTER_DIAGOP_GPU(complex64);
REGISTER_DIAGOP_GPU(complex128);

#undef REGISTER_DIAGOP_GPU

// Forward declarations of the functor specializations for GPU.
namespace functor {
extern template struct DiagPartFunctor<GPUDevice, double>;
extern template struct DiagPartFunctor<GPUDevice, float>;
extern template struct DiagPartFunctor<GPUDevice, int32>;
extern template struct DiagPartFunctor<GPUDevice, int64>;
extern template struct DiagPartFunctor<GPUDevice, complex64>;
extern template struct DiagPartFunctor<GPUDevice, complex128>;
}  // namespace functor

#define REGISTER_DIAGPARTOP_GPU(T)                                \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("DiagPart").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DiagPartOp<GPUDevice, T>);

REGISTER_DIAGPARTOP_GPU(double);
REGISTER_DIAGPARTOP_GPU(float);
REGISTER_DIAGPARTOP_GPU(int32);
REGISTER_DIAGPARTOP_GPU(int64);
REGISTER_DIAGPARTOP_GPU(complex64);
REGISTER_DIAGPARTOP_GPU(complex128);

#undef REGISTER_DIAGPARTOP_GPU

#endif  // GOOGLE_CUDA


}  // namespace tensorflow

