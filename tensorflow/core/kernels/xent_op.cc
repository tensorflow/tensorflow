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

#include "tensorflow/core/kernels/xent_op.h"
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
#endif  // TENSORFLOW_USE_SYCL

template <typename Device, typename T>
class SoftmaxXentWithLogitsOp : public OpKernel {
 public:
  explicit SoftmaxXentWithLogitsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in = context->input(0);
    const Tensor& labels_in = context->input(1);
    OP_REQUIRES(context, logits_in.IsSameSize(labels_in),
                errors::InvalidArgument(
                    "logits and labels must be same size: logits_size=",
                    logits_in.shape().DebugString(),
                    " labels_size=", labels_in.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits_in.shape()),
                errors::InvalidArgument("logits must be 2-dimensional"));
    // As we already tested that both inputs have the same shape no need to
    // check that "labels" is a matrix too.

    // loss is 1-D (one per example), and size is batch_size.

    Tensor scratch;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<T>::value,
                                        TensorShape({logits_in.dim_size(0), 1}),
                                        &scratch));

    Tensor* loss_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({logits_in.dim_size(0)}), &loss_out));
    Tensor* back_out = nullptr;
    // Try to reuse the logits_in buffer for the backprop output.
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 1, logits_in.shape(), &back_out));
    if (logits_in.dim_size(0) > 0) {
      functor::XentFunctor<Device, T> functor;
      functor(context->eigen_device<Device>(), logits_in.matrix<T>(),
              labels_in.matrix<T>(), scratch.matrix<T>(), loss_out->vec<T>(),
              back_out->matrix<T>());
    }
  }
};

// Partial specialization for a CPUDevice, that uses the Eigen implementation
// from XentEigenImpl.
namespace functor {
template <typename Device, typename T>
struct XentFunctorBase {
  void operator()(const Device& d, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<T>::ConstMatrix labels,
                  typename TTypes<T>::Matrix scratch,
                  typename TTypes<T>::Vec loss,
                  typename TTypes<T>::Matrix backprop) {
    XentEigenImpl<Device, T>::Compute(d, logits, labels, scratch, loss,
                                      backprop);
  }
};

template <typename T>
struct XentFunctor<CPUDevice, T> : XentFunctorBase<CPUDevice, T> {};

#ifdef TENSORFLOW_USE_SYCL
template <typename T>
struct XentFunctor<SYCLDevice, T> : XentFunctorBase<SYCLDevice, T> {};
#endif  // TENSORFLOW_USE_SYCL
}  // namespace functor

#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropyWithLogits") \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<T>("T"),          \
                          SoftmaxXentWithLogitsOp<CPUDevice, T>);
TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropyWithLogits")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T"),
                        SoftmaxXentWithLogitsOp<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropyWithLogits")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T"),
                        SoftmaxXentWithLogitsOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropyWithLogits")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T"),
                        SoftmaxXentWithLogitsOp<GPUDevice, double>);
#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropyWithLogits")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<float>("T"),
                        SoftmaxXentWithLogitsOp<SYCLDevice, float>);
#endif  // TENSORFLOW_USE_SYCL

}  // namespace tensorflow
