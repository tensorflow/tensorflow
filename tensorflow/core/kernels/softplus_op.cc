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

#include "tensorflow/core/kernels/softplus_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class SoftplusOp : public UnaryElementWiseOp<T, SoftplusOp<Device, T>> {
 public:
  explicit SoftplusOp(OpKernelConstruction* context)
      : UnaryElementWiseOp<T, SoftplusOp<Device, T>>(context) {}

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    functor::Softplus<Device, T> functor;
    functor(context->eigen_device<Device>(), input.flat<T>(),
            output->flat<T>());
  }
};

template <typename Device, typename T>
class SoftplusGradOp
    : public BinaryElementWiseOp<T, SoftplusGradOp<Device, T>> {
 public:
  explicit SoftplusGradOp(OpKernelConstruction* context)
      : BinaryElementWiseOp<T, SoftplusGradOp<Device, T>>(context) {}

  void OperateNoTemplate(OpKernelContext* context, const Tensor& g,
                         const Tensor& a, Tensor* output);

  // INPUTS:
  //   g (gradients): backpropagated gradients
  //   a (inputs): inputs that were passed to SoftplusOp()
  // OUTPUT:
  //   gradients to backprop
  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
    OperateNoTemplate(context, g, a, output);
  }
};
template <typename Device, typename T>
void SoftplusGradOp<Device, T>::OperateNoTemplate(OpKernelContext* context,
                                                  const Tensor& g,
                                                  const Tensor& a,
                                                  Tensor* output) {
  OP_REQUIRES(context, a.IsSameSize(g),
              errors::InvalidArgument("g and a must be the same size"));
  functor::SoftplusGrad<Device, T> functor;
  functor(context->eigen_device<Device>(), g.flat<T>(), a.flat<T>(),
          output->flat<T>());
}

#define REGISTER_KERNELS(type)                                           \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Softplus").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      SoftplusOp<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SoftplusGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SoftplusGradOp<CPUDevice, type>);

TF_CALL_FLOAT_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_SOFTPLUS_GPU_SPEC(T)                                \
  template <>                                                       \
  void Softplus<GPUDevice, T>::operator()(                          \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features, \
      typename TTypes<T>::Tensor activations);                      \
  extern template struct Softplus<GPUDevice, T>;

#define DECLARE_SOFTPLUS_GRAD_GPU_SPEC(T)                            \
  template <>                                                        \
  void SoftplusGrad<GPUDevice, T>::operator()(                       \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients, \
      typename TTypes<T>::ConstTensor features,                      \
      typename TTypes<T>::Tensor backprops);                         \
  extern template struct SoftplusGrad<GPUDevice, T>;

#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
TF_CALL_GPU_NUMBER_TYPES(DECLARE_SOFTPLUS_GPU_SPEC);
#endif

TF_CALL_GPU_NUMBER_TYPES(DECLARE_SOFTPLUS_GRAD_GPU_SPEC);
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_SOFTPLUS_GPU_KERNELS(type)                          \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Softplus").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SoftplusOp<GPUDevice, type>);

#define REGISTER_SOFTPLUS_GRAD_GPU_KERNELS(type)                         \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SoftplusGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SoftplusGradOp<GPUDevice, type>);

#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SOFTPLUS_GPU_KERNELS);
#endif

TF_CALL_GPU_NUMBER_TYPES(REGISTER_SOFTPLUS_GRAD_GPU_KERNELS);

#undef REGISTER_SOFTPLUS_GPU_KERNELS
#undef REGISTER_SOFTPLUS_GRAD_GPU_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
