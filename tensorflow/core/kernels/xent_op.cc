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
#include "tensorflow/core/util/bcast.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class SoftmaxXentWithLogitsOp : public OpKernel {
 public:
  explicit SoftmaxXentWithLogitsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in = context->input(0);
    const Tensor& labels_in = context->input(1);

    TensorShape shape_in = logits_in.shape();

    BCast bcast(BCast::FromShape(logits_in.shape()),
                BCast::FromShape(labels_in.shape()),
                /*fewer_dims_optimization=*/false);
    if (!logits_in.IsSameSize(labels_in)) {
      OP_REQUIRES(context, bcast.IsValid(),
                  errors::InvalidArgument(
                      "logits and labels must be broadcastable: logits_size=",
                      logits_in.shape().DebugString(),
                      " labels_size=", labels_in.shape().DebugString()));
      shape_in = BCast::ToShape(bcast.output_shape());
    }
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(shape_in),
                errors::InvalidArgument("logits and labels must be either "
                                        "2-dimensional, or broadcasted to be "
                                        "2-dimensional"));

    if (std::is_same<Device, GPUDevice>::value) {
      OP_REQUIRES(context, !OpDeterminismRequired(),
                  errors::Unimplemented(
                      "The GPU implementation of SoftmaxCrossEntropyWithLogits"
                      " that would have been executed is not deterministic."
                      " Note that the Python API uses an alternative,"
                      " deterministic, GPU-accelerated path when determinism is"
                      " enabled."));
    }

    // loss is 1-D (one per example), and size is batch_size.

    Tensor scratch;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<T>::value,
                                        TensorShape({shape_in.dim_size(0), 1}),
                                        &scratch));

    Tensor* loss_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({shape_in.dim_size(0)}), &loss_out));
    Tensor* back_out = nullptr;
    // Try to reuse the logits_in buffer for the backprop output.
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 1, shape_in, &back_out));
    if (shape_in.dim_size(0) > 0) {
      functor::XentFunctor<Device, T> functor;
      functor(context->eigen_device<Device>(), shape_in.AsEigenDSizes<2>(),
              BCast::ToIndexArray<2>(bcast.x_bcast()),
              BCast::ToIndexArray<2>(bcast.y_bcast()),
              logits_in.template shaped<T, 2>(bcast.x_reshape()),
              labels_in.template shaped<T, 2>(bcast.y_reshape()),
              scratch.matrix<T>(), loss_out->vec<T>(), back_out->matrix<T>());
    }
  }
};

// Partial specialization for a CPUDevice, that uses the Eigen implementation
// from XentEigenImpl.
namespace functor {
template <typename Device, typename T>
struct XentFunctorBase {
  void operator()(const Device& d,
                  const Eigen::DSizes<Eigen::DenseIndex, 2>& shape,
                  const Eigen::array<Eigen::DenseIndex, 2>& logits_bcast,
                  const Eigen::array<Eigen::DenseIndex, 2>& labels_bcast,
                  typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<T>::ConstMatrix labels,
                  typename TTypes<T>::Matrix scratch,
                  typename TTypes<T>::Vec loss,
                  typename TTypes<T>::Matrix backprop) {
    XentEigenImpl<Device, T>::Compute(d, shape, logits_bcast, labels_bcast,
                                      logits, labels, scratch, loss, backprop);
  }
};

template <typename T>
struct XentFunctor<CPUDevice, T> : XentFunctorBase<CPUDevice, T> {};

}  // namespace functor

#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropyWithLogits") \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<T>("T"),          \
                          SoftmaxXentWithLogitsOp<CPUDevice, T>);
TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
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
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


}  // namespace tensorflow
