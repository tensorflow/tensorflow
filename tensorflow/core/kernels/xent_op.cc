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

#include <type_traits>

#include "tensorflow/core/kernels/xent_op.h"

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
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

namespace functor {

// Applies backend-independent corrections to a functor's output. Most types
// and devices do not require any correction.
template <typename Device, typename T>
struct XentGradientCorrector {
  static void Correct(
      const Device&, const Eigen::DSizes<Eigen::DenseIndex, 2>&,
      const Eigen::array<Eigen::DenseIndex, 2>&,
      typename TTypes<T>::ConstMatrix, typename TTypes<T>::Matrix,
      typename TTypes<T>::Matrix) {}
};

template <>
struct XentGradientCorrector<CPUDevice, double> {
  static void Correct(
      const CPUDevice& d, const Eigen::DSizes<Eigen::DenseIndex, 2>& shape,
      const Eigen::array<Eigen::DenseIndex, 2>& labels_bcast,
      TTypes<double>::ConstMatrix labels, TTypes<double>::Matrix scratch,
      TTypes<double>::Matrix backprop) {
    const int kClassDim = 1;
    const int batch_size = shape[0];
    const int num_classes = shape[1];
    Eigen::IndexList<Eigen::type2index<kClassDim>> along_class;
    Eigen::IndexList<int> batch_only;
    batch_only.set(0, batch_size);
    Eigen::IndexList<Eigen::type2index<1>, int> one_by_class;
    one_by_class.set(1, num_classes);

    auto probabilities = (backprop + labels.broadcast(labels_bcast)).eval();
    scratch.reshape(batch_only).device(d) = backprop.sum(along_class);
    // Remove the row-sum residual that can remain when a probability rounds
    // to one. Running after the selected functor keeps fallback and optimized
    // CPU implementations consistent.
    backprop.device(d) =
        backprop - scratch.broadcast(one_by_class) * probabilities;
  }
};

}  // namespace functor

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
                  absl::InvalidArgumentError(absl::StrCat(
                      "logits and labels must be broadcastable: logits_size=",
                      logits_in.shape().DebugString(),
                      " labels_size=", labels_in.shape().DebugString())));
      shape_in = BCast::ToShape(bcast.output_shape());
    }
    OP_REQUIRES(
        context, TensorShapeUtils::IsMatrix(shape_in),
        absl::InvalidArgumentError("logits and labels must be either "
                                   "2-dimensional, or broadcasted to be "
                                   "2-dimensional"));

    if (std::is_same<Device, GPUDevice>::value) {
      OP_REQUIRES(context, !OpDeterminismRequired(),
                  absl::UnimplementedError(
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
      const Device& d = context->eigen_device<Device>();
      functor::XentFunctor<Device, T> functor;
      functor(d, shape_in.AsEigenDSizes<2>(),
              BCast::ToIndexArray<2>(bcast.x_bcast()),
              BCast::ToIndexArray<2>(bcast.y_bcast()),
              logits_in.template shaped<T, 2>(bcast.x_reshape()),
              labels_in.template shaped<T, 2>(bcast.y_reshape()),
              scratch.matrix<T>(), loss_out->vec<T>(), back_out->matrix<T>());
      functor::XentGradientCorrector<Device, T>::Correct(
          d, shape_in.AsEigenDSizes<2>(),
          BCast::ToIndexArray<2>(bcast.y_bcast()),
          labels_in.template shaped<T, 2>(bcast.y_reshape()),
          scratch.matrix<T>(), back_out->matrix<T>());
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
    if (shape[0] > 0) {
      XentEigenImpl<Device, T>::Compute(d, shape, logits_bcast, labels_bcast,
                                        logits, labels, scratch, loss,
                                        backprop);
    }
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
TF_CALL_bfloat16(REGISTER_CPU);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

#define REGISTER_GPU(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropyWithLogits") \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<T>("T"),          \
                          SoftmaxXentWithLogitsOp<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
