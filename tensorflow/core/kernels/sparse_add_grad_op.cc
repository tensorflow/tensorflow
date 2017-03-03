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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

template <typename T>
class SparseAddGradOp : public OpKernel {
 public:
  explicit SparseAddGradOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    // Gradient for op: SparseAdd(a, b) == sum.
    const Tensor *backprop_val_grad, *a_indices, *b_indices, *sum_indices;
    OP_REQUIRES_OK(ctx, ctx->input("backprop_val_grad", &backprop_val_grad));
    OP_REQUIRES_OK(ctx, ctx->input("a_indices", &a_indices));
    OP_REQUIRES_OK(ctx, ctx->input("b_indices", &b_indices));
    OP_REQUIRES_OK(ctx, ctx->input("sum_indices", &sum_indices));

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a_indices->shape()) &&
                         TensorShapeUtils::IsMatrix(b_indices->shape()) &&
                         TensorShapeUtils::IsMatrix(sum_indices->shape()),
                errors::InvalidArgument(
                    "Input indices should be matrices but received shapes: ",
                    a_indices->shape().DebugString(), " and ",
                    b_indices->shape().DebugString(), " and ",
                    sum_indices->shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(backprop_val_grad->shape()),
        errors::InvalidArgument(
            "Input backprop_val_grad should be a vector but received shape: ",
            backprop_val_grad->shape().DebugString()));
    OP_REQUIRES(
        ctx, a_indices->dim_size(1) == b_indices->dim_size(1) &&
                 b_indices->dim_size(1) == sum_indices->dim_size(1),
        errors::InvalidArgument("The densified operands should have the same "
                                "ndims; for A, B, sum got: ",
                                a_indices->dim_size(1), b_indices->dim_size(1),
                                sum_indices->dim_size(1)));
    OP_REQUIRES(
        ctx, backprop_val_grad->NumElements() == sum_indices->dim_size(0),
        errors::InvalidArgument("# elements of backprop_val_grad and # rows of "
                                "sum_indices should match (#nnz of sum): got ",
                                backprop_val_grad->NumElements(), " and ",
                                sum_indices->dim_size(0)));

    const int num_dims = a_indices->dim_size(1);
    const int64 a_nnz = a_indices->dim_size(0);
    const int64 b_nnz = b_indices->dim_size(0);
    const int64 sum_nnz = backprop_val_grad->NumElements();

    const auto a_indices_mat = a_indices->matrix<int64>();
    const auto b_indices_mat = b_indices->matrix<int64>();
    const auto sum_indices_mat = sum_indices->matrix<int64>();

    Tensor *a_val_grad, *b_val_grad;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({a_nnz}), &a_val_grad));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, TensorShape({b_nnz}), &b_val_grad));

    T *a_val_grad_flat = a_val_grad->flat<T>().data();
    T *b_val_grad_flat = b_val_grad->flat<T>().data();
    const T *backprop_val_grad_flat = backprop_val_grad->flat<T>().data();
    memset(a_val_grad_flat, 0, sizeof(T) * a_nnz);
    memset(b_val_grad_flat, 0, sizeof(T) * b_nnz);

#define COMPARE(a_or_b, idx)                                                \
  switch (sparse::DimComparator::cmp(a_or_b##_indices_mat, sum_indices_mat, \
                                     idx, k, num_dims)) {                   \
    case 0:                                                                 \
      a_or_b##_val_grad_flat[idx] = backprop_val_grad_flat[k];              \
      ++idx;                                                                \
      break;                                                                \
    case -1:                                                                \
      ++idx;                                                                \
      a_or_b##_idx_geq = false;                                             \
      break;                                                                \
    case 1:                                                                 \
      break;                                                                \
  }

    // Set-intersect the indices; fill in grads for positions in the
    // intersection.
    int64 i = 0, j = 0, k = 0;
    bool a_idx_geq, b_idx_geq;
    while (i < a_nnz && j < b_nnz && k < sum_nnz) {
      a_idx_geq = b_idx_geq = true;
      COMPARE(a, i);
      COMPARE(b, j);
      // increment pointer into sum_indices iff both the current A, B indices >=
      // the current sum index.
      if (a_idx_geq && b_idx_geq) ++k;
    }

    // at most one loop below will run
    while (i < a_nnz && k < sum_nnz) {
      a_idx_geq = true;
      COMPARE(a, i);
      if (a_idx_geq) ++k;
    }
    while (j < b_nnz && k < sum_nnz) {
      b_idx_geq = true;
      COMPARE(b, j);
      if (b_idx_geq) ++k;
    }
#undef COMPARE
  }
};

#define REGISTER_KERNELS(type)                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SparseAddGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseAddGradOp<type>)

// This op should work for any T that SparseAdd is registered with.
REGISTER_KERNELS(float);
REGISTER_KERNELS(double);
REGISTER_KERNELS(int64);
REGISTER_KERNELS(int32);
REGISTER_KERNELS(int16);
REGISTER_KERNELS(int8);
REGISTER_KERNELS(complex64);
REGISTER_KERNELS(complex128);
#undef REGISTER_KERNELS
}  // namespace tensorflow
