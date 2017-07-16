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

#include "tensorflow/core/kernels/softmax_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif // TENSORFLOW_USE_SYCL

// Partial specialization for a CPUDevice, that uses the Eigen implementation
// from SoftmaxEigenImpl.
namespace functor {
template <typename Device, typename T>
struct SoftmaxFunctorBase {
  void operator()(const Device& d, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<T>::Matrix softmax, const bool log) {
    SoftmaxEigenImpl<Device, T>::Compute(d, logits, softmax, log);
  }
};
template <typename T>
struct SoftmaxFunctor<CPUDevice, T> : SoftmaxFunctorBase<CPUDevice, T> {};

#ifdef TENSORFLOW_USE_SYCL
template <typename T>
struct SoftmaxFunctor<SYCLDevice, T> : SoftmaxFunctorBase<SYCLDevice, T> {};
#endif // TENSORFLOW_USE_SYCL
}  // namespace functor

#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Softmax").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SoftmaxOp<CPUDevice, T>);
TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

#undef REGISTER_CPU
#define REGISTER_CPU(T)                                             \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("LogSoftmax").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SoftmaxOp<CPUDevice, T>);
TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("Softmax").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    SoftmaxOp<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("Softmax").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    SoftmaxOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("Softmax").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    SoftmaxOp<GPUDevice, double>);
REGISTER_KERNEL_BUILDER(
    Name("LogSoftmax").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    SoftmaxOp<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("LogSoftmax").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    SoftmaxOp<GPUDevice, float>);
#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(
    Name("Softmax").Device(DEVICE_SYCL).TypeConstraint<float>("T"),
    SoftmaxOp<SYCLDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("Softmax").Device(DEVICE_SYCL).TypeConstraint<double>("T"),
    SoftmaxOp<SYCLDevice, double>);
#endif // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
