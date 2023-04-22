/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/sparse_tensor_dense_add_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
// NOTE: does not support GPU yet.

namespace {

template <typename Index>
Status ValidateInputs(const Tensor *a_indices, const Tensor *a_values,
                      const Tensor *a_shape, const Tensor *b) {
  if (!TensorShapeUtils::IsMatrix(a_indices->shape())) {
    return errors::InvalidArgument(
        "Input a_indices should be a matrix but received shape: ",
        a_indices->shape().DebugString());
  }
  if (!TensorShapeUtils::IsVector(a_values->shape()) ||
      !TensorShapeUtils::IsVector(a_shape->shape())) {
    return errors::InvalidArgument(
        "Inputs a_values and a_shape should be vectors "
        "but received shapes: ",
        a_values->shape().DebugString(), " and ",
        a_shape->shape().DebugString());
  }
  if (a_shape->NumElements() != b->dims()) {
    return errors::InvalidArgument(
        "Two operands have different ranks; received: ", a_shape->NumElements(),
        " and ", b->dims());
  }
  const auto a_shape_flat = a_shape->flat<Index>();
  for (int i = 0; i < b->dims(); ++i) {
    if (a_shape_flat(i) != b->dim_size(i)) {
      return errors::InvalidArgument(
          "Dimension ", i,
          " does not equal (no broadcasting is supported): sparse side ",
          a_shape_flat(i), " vs dense side ", b->dim_size(i));
    }
  }
  return Status::OK();
}

}  // namespace

template <typename Device, typename T, typename Index>
class SparseTensorDenseAddOp : public OpKernel {
 public:
  explicit SparseTensorDenseAddOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    const Tensor *a_indices_t, *a_values_t, *a_shape_t, *b;
    OP_REQUIRES_OK(ctx, ctx->input("a_indices", &a_indices_t));
    OP_REQUIRES_OK(ctx, ctx->input("a_values", &a_values_t));
    OP_REQUIRES_OK(ctx, ctx->input("a_shape", &a_shape_t));
    OP_REQUIRES_OK(ctx, ctx->input("b", &b));
    OP_REQUIRES_OK(
        ctx, ValidateInputs<Index>(a_indices_t, a_values_t, a_shape_t, b));

    Tensor *out_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, b->shape(), &out_t));

    const int ndims = static_cast<int>(a_indices_t->dim_size(1));
    const auto a_indices_mat = a_indices_t->flat_inner_dims<Index>();
    const auto a_values_flat = a_values_t->flat<T>();

    switch (ndims) {
#define NDIMS_CASE(N)                                                     \
  case N: {                                                               \
    auto out_tensor = out_t->tensor<T, N>();                              \
    out_tensor.device(ctx->eigen_device<Device>()) = b->tensor<T, N>();   \
    const Index result =                                                  \
        functor::ScatterNdFunctor<Device, T, Index, N,                    \
                                  scatter_op::UpdateOp::ADD>()(           \
            ctx->eigen_device<Device>(), a_indices_mat, a_values_flat,    \
            out_tensor);                                                  \
    OP_REQUIRES(                                                          \
        ctx, result == -1,                                                \
        errors::InvalidArgument(                                          \
            "Sparse tensor has some invalid index on dimension ", result, \
            "; dense tensor shape: ", b->shape().DebugString()));         \
  } break;

      NDIMS_CASE(1);
      NDIMS_CASE(2);
      NDIMS_CASE(3);
      NDIMS_CASE(4);
      NDIMS_CASE(5);
      default:
        OP_REQUIRES(
            ctx, false,
            errors::InvalidArgument("Only tensors with ranks between 1 and 5 "
                                    "are currently supported.  Tensor rank: ",
                                    ndims));
#undef NDIMS_CASE
    }
  }
};

namespace functor {
template <typename T, typename Index, int NDIMS>
struct ScatterNdFunctor<CPUDevice, T, Index, NDIMS, scatter_op::UpdateOp::ADD> {
  Index operator()(const CPUDevice &d,
                   typename TTypes<Index>::ConstMatrix indices,
                   typename TTypes<T>::ConstFlat updates,
                   typename TTypes<T, NDIMS>::Tensor out) {
    Eigen::array<Eigen::DenseIndex, NDIMS> idx;
    const int num_nnz = static_cast<int>(indices.dimension(0));
    for (int i = 0; i < num_nnz; ++i) {
      for (int d = 0; d < NDIMS; ++d) {
        idx[d] = internal::SubtleMustCopy(indices(i, d));
        if (!FastBoundsCheck(idx[d], out.dimension(d))) {
          return d;  // on failure: d nonnegative
        }
      }
      out(idx) += updates(i);
    }
    return -1;  // on success
  }
};
}  // namespace functor

#define REGISTER_KERNELS_CPU(TypeT, TypeIndex)                        \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorDenseAdd")                \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<TypeT>("T")             \
                              .TypeConstraint<TypeIndex>("Tindices"), \
                          SparseTensorDenseAddOp<CPUDevice, TypeT, TypeIndex>)

#define REGISTER_KERNELS(T)       \
  REGISTER_KERNELS_CPU(T, int64); \
  REGISTER_KERNELS_CPU(T, int32)

TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS
#undef REGISTER_KERNELS_CPU
}  // namespace tensorflow
