// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/numeric_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/relu_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class ReluOp : public UnaryElementWiseOp<T, ReluOp<Device, T>> {
 public:
  using UnaryElementWiseOp<T, ReluOp<Device, T>>::UnaryElementWiseOp;

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    functor::Relu<Device, T> functor;
    functor(context->eigen_device<Device>(), input.flat<T>(),
            output->flat<T>());
  }
};

template <typename Device, typename T>
class Relu6Op : public UnaryElementWiseOp<T, Relu6Op<Device, T>> {
 public:
  using UnaryElementWiseOp<T, Relu6Op<Device, T>>::UnaryElementWiseOp;

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    functor::Relu6<Device, T> functor;
    functor(context->eigen_device<Device>(), input.flat<T>(),
            output->flat<T>());
  }
};

template <typename Device, typename T>
class ReluGradOp : public BinaryElementWiseOp<T, ReluGradOp<Device, T>> {
 public:
  using BinaryElementWiseOp<T, ReluGradOp<Device, T>>::BinaryElementWiseOp;

  // INPUTS:
  //   g (gradients): backpropagated gradients
  //   a (inputs): inputs that were passed to ReluOp()
  // OUTPUT:
  //   gradients to backprop
  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
    OP_REQUIRES(context, a.IsSameSize(g),
                errors::InvalidArgument("g and a must be the same size"));
    functor::ReluGrad<Device, T> functor;
    functor(context->eigen_device<Device>(), g.flat<T>(), a.flat<T>(),
            output->flat<T>());
  }
};

template <typename Device, typename T>
class Relu6GradOp : public BinaryElementWiseOp<T, Relu6GradOp<Device, T>> {
 public:
  using BinaryElementWiseOp<T, Relu6GradOp<Device, T>>::BinaryElementWiseOp;

  // INPUTS:
  //   g (gradients): backpropagated gradients
  //   a (inputs): inputs that were passed to Relu6Op()
  // OUTPUT:
  //   gradients to backprop
  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
    OP_REQUIRES(context, a.IsSameSize(g),
                errors::InvalidArgument("g and a must be the same size"));
    functor::Relu6Grad<Device, T> functor;
    functor(context->eigen_device<Device>(), g.flat<T>(), a.flat<T>(),
            output->flat<T>());
  }
};

#define REGISTER_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      ReluOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu6").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      Relu6Op<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("ReluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),  \
      ReluGradOp<CPUDevice, type>);                                   \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu6Grad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      Relu6GradOp<CPUDevice, type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                          \
  template <>                                                        \
  void Relu<GPUDevice, T>::operator()(                               \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,  \
      typename TTypes<T>::Tensor activations);                       \
  extern template struct Relu<GPUDevice, T>;                         \
                                                                     \
  template <>                                                        \
  void ReluGrad<GPUDevice, T>::operator()(                           \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients, \
      typename TTypes<T>::ConstTensor features,                      \
      typename TTypes<T>::Tensor backprops);                         \
                                                                     \
  extern template struct ReluGrad<GPUDevice, T>;                     \
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
  extern template struct Relu6Grad<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
      ReluOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu6").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      Relu6Op<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("ReluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),  \
      ReluGradOp<GPUDevice, type>);                                   \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu6Grad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      Relu6GradOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
