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

#include "tensorflow/core/kernels/sparse_xent_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Index>
Status CheckInvalidLabelIndex(const Tensor& labels, int64 max_index) {
  if (labels.NumElements() == 0) return Status::OK();
  const auto label_values = labels.vec<Index>();
  int64 bad_index;
  auto min_max_dim_value = std::minmax_element(
      label_values.data(), label_values.data() + label_values.size());
  if (*min_max_dim_value.first < 0 || *min_max_dim_value.second >= max_index) {
    bad_index = (*min_max_dim_value.first < 0) ? *min_max_dim_value.first
                                               : *min_max_dim_value.second;
    return errors::InvalidArgument(
        "Received a label value of ", bad_index,
        " which is outside the valid range of [0, ", max_index,
        ").  Label values: ", labels.SummarizeValue(labels.NumElements()));
  }
  return Status::OK();
}

template <typename Device, typename T, typename Index>
class SparseSoftmaxXentWithLogitsOp : public OpKernel {
 public:
  explicit SparseSoftmaxXentWithLogitsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& logits = context->input(0);
    const Tensor& labels = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits.shape()),
                errors::InvalidArgument("logits must be 2-D, but got shape ",
                                        logits.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(labels.shape()),
                errors::InvalidArgument("labels must be 1-D, but got shape ",
                                        labels.shape().DebugString()));
    OP_REQUIRES(context, logits.dim_size(0) == labels.dim_size(0),
                errors::InvalidArgument(
                    "logits and labels must have the same first dimension, "
                    "got logits shape ",
                    logits.shape().DebugString(), " and labels shape ",
                    labels.shape().DebugString()));
    OP_REQUIRES(context, logits.dim_size(1) > 0,
                errors::InvalidArgument(
                    "Must have at least one class, but got logits shape ",
                    logits.shape().DebugString()));

    Tensor scratch;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   labels.shape(), &scratch));

    Tensor* loss_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {1}, 0, labels.shape(), &loss_out));
    Tensor* back_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 1, logits.shape(), &back_out));

    if (logits.dim_size(0) > 0) {
      if (std::is_same<Device, CPUDevice>::value) {
        OP_REQUIRES_OK(
            context, CheckInvalidLabelIndex<Index>(labels, logits.dim_size(1)));
      }
      functor::SparseXentFunctor<Device, T, Index> functor;
      functor(context->eigen_device<Device>(), logits.matrix<T>(),
              labels.vec<Index>(), scratch.vec<T>(), loss_out->vec<T>(),
              back_out->matrix<T>());
    }
  }
};

// Partial specialization for a CPUDevice, that uses the Eigen implementation
// from XentEigenImpl.
namespace functor {
template <typename T, typename Index>
struct SparseXentFunctor<CPUDevice, T, Index> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<Index>::ConstVec labels,
                  typename TTypes<T>::Vec scratch, typename TTypes<T>::Vec loss,
                  typename TTypes<T>::Matrix backprop) {
    SparseXentEigenImpl<CPUDevice, T, Index>::Compute(d, logits, labels,
                                                      scratch, loss, backprop);
  }
};
}  // namespace functor

#define REGISTER(Dev, T, Index)                   \
  REGISTER_KERNEL_BUILDER(                        \
      Name("SparseSoftmaxCrossEntropyWithLogits") \
          .Device(DEVICE_##Dev)                   \
          .TypeConstraint<T>("T")                 \
          .TypeConstraint<Index>("Tlabels"),      \
      SparseSoftmaxXentWithLogitsOp<Dev##Device, T, Index>);
REGISTER(CPU, float, int32)
REGISTER(CPU, float, int64)
REGISTER(CPU, double, int32)
REGISTER(CPU, double, int64)
REGISTER(CPU, Eigen::half, int32)
REGISTER(CPU, Eigen::half, int64)

#if GOOGLE_CUDA
REGISTER(GPU, float, int32)
REGISTER(GPU, float, int64)
REGISTER(GPU, Eigen::half, int32)
REGISTER(GPU, Eigen::half, int64)
#endif  // GOOGLE_CUDA

#undef REGISTER

}  // namespace tensorflow
