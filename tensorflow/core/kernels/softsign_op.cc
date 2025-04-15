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

#include "tensorflow/core/kernels/softsign_op.h"

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class SoftsignOp : public UnaryElementWiseOp<T, SoftsignOp<Device, T>> {
 public:
  explicit SoftsignOp(OpKernelConstruction* context)
      : UnaryElementWiseOp<T, SoftsignOp<Device, T>>(context) {}

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    functor::Softsign<Device, T> functor;
    functor(context->eigen_device<Device>(), input.flat<T>(),
            output->flat<T>());
  }
};

template <typename Device, typename T>
class SoftsignGradOp
    : public BinaryElementWiseOp<T, SoftsignGradOp<Device, T>> {
 public:
  explicit SoftsignGradOp(OpKernelConstruction* context)
      : BinaryElementWiseOp<T, SoftsignGradOp<Device, T>>(context) {}

  void OperateNoTemplate(OpKernelContext* context, const Tensor& g,
                         const Tensor& a, Tensor* output);

  // INPUTS:
  //   g (gradients): backpropagated gradients
  //   a (inputs): inputs that were passed to SoftsignOp()
  // OUTPUT:
  //   gradients to backprop
  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
    OperateNoTemplate(context, g, a, output);
  }
};

template <typename Device, typename T>
void SoftsignGradOp<Device, T>::OperateNoTemplate(OpKernelContext* context,
                                                  const Tensor& g,
                                                  const Tensor& a,
                                                  Tensor* output) {
  OP_REQUIRES(context, a.IsSameSize(g),
              errors::InvalidArgument("g and a must be the same size"));
  functor::SoftsignGrad<Device, T> functor;
  functor(context->eigen_device<Device>(), g.flat<T>(), a.flat<T>(),
          output->flat<T>());
}

#define REGISTER_KERNELS(type)                                           \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Softsign").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      SoftsignOp<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SoftsignGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SoftsignGradOp<CPUDevice, type>);

TF_CALL_FLOAT_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_SOFTSIGN_GPU_SPEC(T)                                \
  template <>                                                       \
  void Softsign<GPUDevice, T>::operator()(                          \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features, \
      typename TTypes<T>::Tensor activations);                      \
  extern template struct Softsign<GPUDevice, T>;

#define DECLARE_SOFTSIGN_GRAD_GPU_SPEC(T)                            \
  template <>                                                        \
  void SoftsignGrad<GPUDevice, T>::operator()(                       \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients, \
      typename TTypes<T>::ConstTensor features,                      \
      typename TTypes<T>::Tensor backprops);                         \
  extern template struct SoftsignGrad<GPUDevice, T>;

#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
TF_CALL_GPU_NUMBER_TYPES(DECLARE_SOFTSIGN_GPU_SPEC);
#endif
TF_CALL_GPU_NUMBER_TYPES(DECLARE_SOFTSIGN_GRAD_GPU_SPEC);
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_SOFTSIGN_GPU_KERNELS(type)                          \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Softsign").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SoftsignOp<GPUDevice, type>);

#define REGISTER_SOFTSIGN_GRAD_GPU_KERNELS(type)                         \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SoftsignGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SoftsignGradOp<GPUDevice, type>);

#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SOFTSIGN_GPU_KERNELS);
#endif
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SOFTSIGN_GRAD_GPU_KERNELS);
#undef REGISTER_SOFTSIGN_GPU_KERNELS
#undef REGISTER_SOFTSIGN_GRAD_GPU_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
