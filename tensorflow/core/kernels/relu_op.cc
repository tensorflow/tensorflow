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

#include "tensorflow/core/kernels/relu_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_RELU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu").Device(DEVICE_CPU).TypeConstraint<type>("T"),          \
      ReluOp<CPUDevice, type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("ReluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      ReluGradOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu6").Device(DEVICE_CPU).TypeConstraint<type>("T"),         \
      Relu6Op<CPUDevice, type>);                                          \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu6Grad").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      Relu6GradOp<CPUDevice, type>)                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LeakyReluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      LeakyReluGradOp<CPUDevice, type>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_RELU_KERNELS);
#undef REGISTER_RELU_KERNELS

// Register LeakyRelu here for all types except bfloat16
// bfloat16 is in cwise_op_leakyrelu_bf16.cc
#define REGISTER_LEAKYRELU_KERNELS(type)                              \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("LeakyRelu").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      LeakyReluOp<CPUDevice, type>);

TF_CALL_INTEGRAL_TYPES(REGISTER_LEAKYRELU_KERNELS)
TF_CALL_half(REGISTER_LEAKYRELU_KERNELS)
    TF_CALL_float(REGISTER_LEAKYRELU_KERNELS)
        TF_CALL_double(REGISTER_LEAKYRELU_KERNELS)
#undef REGISTER_LEAKYRELU_KERNELS

#define REGISTER_ELU_KERNELS(type)                                   \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Elu").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      EluOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("EluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),  \
      EluGradOp<CPUDevice, type>);                                   \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Selu").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      SeluOp<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("SeluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SeluGradOp<CPUDevice, type>)

    // Elu and Selu only make sense with float or double.
    TF_CALL_FLOAT_TYPES(REGISTER_ELU_KERNELS);
#undef REGISTER_ELU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace functor {
#define DECLARE_GPU_NO_MLIR_SPEC(T)                                            \
  template <>                                                                  \
  void Relu<GPUDevice, T>::operator()(                                         \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,            \
      typename TTypes<T>::Tensor activations);                                 \
  extern template struct Relu<GPUDevice, T>;                                   \
                                                                               \
  template <>                                                                  \
  void Elu<GPUDevice, T>::operator()(const GPUDevice& d,                       \
                                     typename TTypes<T>::ConstTensor features, \
                                     typename TTypes<T>::Tensor activations);  \
  extern template struct Elu<GPUDevice, T>;                                    \
                                                                               \
  template <>                                                                  \
  void Selu<GPUDevice, T>::operator()(                                         \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,            \
      typename TTypes<T>::Tensor activations);                                 \
  extern template struct Selu<GPUDevice, T>;

// TODO(trevor-m): Use TF_CALL_GPU_NUMBER_TYPES when MLIR-generated bfloat16 is
// enabled.
#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
TF_CALL_half(DECLARE_GPU_NO_MLIR_SPEC);
TF_CALL_float(DECLARE_GPU_NO_MLIR_SPEC);
TF_CALL_double(DECLARE_GPU_NO_MLIR_SPEC);
#endif
TF_CALL_bfloat16(DECLARE_GPU_NO_MLIR_SPEC);
#undef DECLARE_GPU_NO_MLIR_SPEC
}  // namespace functor

#define REGISTER_GPU_NO_MLIR_KERNELS(type)                       \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Relu").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      ReluOp<GPUDevice, type>);                                  \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Elu").Device(DEVICE_GPU).TypeConstraint<type>("T"),  \
      EluOp<GPUDevice, type>);                                   \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Selu").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SeluOp<GPUDevice, type>);

#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
TF_CALL_half(REGISTER_GPU_NO_MLIR_KERNELS);
TF_CALL_float(REGISTER_GPU_NO_MLIR_KERNELS);
TF_CALL_double(REGISTER_GPU_NO_MLIR_KERNELS);
#endif
TF_CALL_bfloat16(REGISTER_GPU_NO_MLIR_KERNELS);
#undef REGISTER_GPU_NO_MLIR_KERNELS

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                          \
  template <>                                                        \
  void ReluGrad<GPUDevice, T>::operator()(                           \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients, \
      typename TTypes<T>::ConstTensor features,                      \
      typename TTypes<T>::Tensor backprops);                         \
  extern template struct ReluGrad<GPUDevice, T>;                     \
                                                                     \
  template <>                                                        \
  void Relu6<GPUDevice, T>::operator()(                              \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,  \
      typename TTypes<T>::Tensor activations);                       \
  extern template struct Relu6<GPUDevice, T>;                        \
                                                                     \
  template <>                                                        \
  void Relu6Grad<GPUDevice, T>::operator()(                          \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients, \
      typename TTypes<T>::ConstTensor features,                      \
      typename TTypes<T>::Tensor backprops);                         \
  extern template struct Relu6Grad<GPUDevice, T>;                    \
                                                                     \
  template <>                                                        \
  void LeakyRelu<GPUDevice, T>::operator()(LeakyReluArgs args);      \
  extern template struct LeakyRelu<GPUDevice, T>;                    \
                                                                     \
  template <>                                                        \
  void LeakyReluGrad<GPUDevice, T>::operator()(                      \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients, \
      typename TTypes<T>::ConstTensor features, T alpha,             \
      typename TTypes<T>::Tensor backprops);                         \
  extern template struct LeakyReluGrad<GPUDevice, T>;                \
                                                                     \
  template <>                                                        \
  void EluGrad<GPUDevice, T>::operator()(                            \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients, \
      typename TTypes<T>::ConstTensor activations,                   \
      typename TTypes<T>::Tensor backprops);                         \
  extern template struct EluGrad<GPUDevice, T>;                      \
                                                                     \
  template <>                                                        \
  void SeluGrad<GPUDevice, T>::operator()(                           \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients, \
      typename TTypes<T>::ConstTensor activations,                   \
      typename TTypes<T>::Tensor backprops);                         \
  extern template struct SeluGrad<GPUDevice, T>;

template <>
void Relu<GPUDevice, qint8>::operator()(
    const GPUDevice& d, typename TTypes<qint8>::ConstTensor features,
    typename TTypes<qint8>::Tensor activations);
extern template struct Relu<GPUDevice, qint8>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("ReluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
      ReluGradOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu6").Device(DEVICE_GPU).TypeConstraint<type>("T"),         \
      Relu6Op<GPUDevice, type>);                                          \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu6Grad").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      Relu6GradOp<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LeakyRelu").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      LeakyReluOp<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LeakyReluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      LeakyReluGradOp<GPUDevice, type>);                                  \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("EluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),       \
      EluGradOp<GPUDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SeluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
      SeluGradOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS

template <typename Device>
class ReluOp<Device, qint8>
    : public UnaryElementWiseOp<qint8, ReluOp<Device, qint8>> {
 public:
  using UnaryElementWiseOp<qint8, ReluOp<Device, qint8>>::UnaryElementWiseOp;

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    auto flat_input = input.flat<qint8>();
    OP_REQUIRES(context, (flat_input.size() % 4) == 0,
                errors::InvalidArgument(
                    "Tensor size must be a multiple of 4 for Relu<qint8>. Got ",
                    flat_input.size()));
    functor::Relu<Device, qint8> func;
    func(context->eigen_device<Device>(), flat_input, output->flat<qint8>());
  }
};

REGISTER_KERNEL_BUILDER(
    Name("Relu").Device(DEVICE_GPU).TypeConstraint<qint8>("T"),
    ReluOp<GPUDevice, qint8>);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
