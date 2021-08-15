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
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

template <typename T, typename Treal>
class SparseAddOp : public OpKernel {
 public:
  explicit SparseAddOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    // (0) validations
    const Tensor *a_indices, *b_indices, *a_values_t, *b_values_t, *a_shape,
        *b_shape, *thresh_t;

    OP_REQUIRES_OK(ctx, ctx->input("a_indices", &a_indices));
    OP_REQUIRES_OK(ctx, ctx->input("b_indices", &b_indices));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsMatrix(a_indices->shape()) &&
                    TensorShapeUtils::IsMatrix(b_indices->shape()),
                errors::InvalidArgument(
                    "Input indices should be matrices but received shapes: ",
                    a_indices->shape().DebugString(), " and ",
                    b_indices->shape().DebugString()));
    const int64_t a_nnz = a_indices->dim_size(0);
    const int64_t b_nnz = b_indices->dim_size(0);
    const int num_dims = a_indices->dim_size(1);
    OP_REQUIRES(ctx, b_indices->dim_size(1) == num_dims,
                errors::InvalidArgument(
                    "Input indices must have the same dimension, got ",
                    num_dims, " and ", b_indices->dim_size(1)));

    OP_REQUIRES_OK(ctx, ctx->input("a_values", &a_values_t));
    OP_REQUIRES_OK(ctx, ctx->input("b_values", &b_values_t));

    OP_REQUIRES(ctx,
                TensorShapeUtils::IsVector(a_values_t->shape()) &&
                    TensorShapeUtils::IsVector(b_values_t->shape()),
                errors::InvalidArgument(
                    "Input values should be vectors but received shapes: ",
                    a_values_t->shape().DebugString(), " and ",
                    b_values_t->shape().DebugString()));
    auto a_values = ctx->input(1).vec<T>();
    auto b_values = ctx->input(4).vec<T>();
    OP_REQUIRES(
        ctx, a_values.size() == a_nnz && b_values.size() == b_nnz,
        errors::InvalidArgument("Expected ", a_nnz, " and ", b_nnz,
                                " non-empty input values, got ",
                                a_values.size(), " and ", b_values.size()));

    OP_REQUIRES_OK(ctx, ctx->input("a_shape", &a_shape));
    OP_REQUIRES_OK(ctx, ctx->input("b_shape", &b_shape));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsVector(a_shape->shape()) &&
                    TensorShapeUtils::IsVector(b_shape->shape()),
                errors::InvalidArgument(
                    "Input shapes should be a vector but received shapes ",
                    a_shape->shape().DebugString(), " and ",
                    b_shape->shape().DebugString()));
    OP_REQUIRES(
        ctx, a_shape->NumElements() == num_dims,
        errors::InvalidArgument("Second dimension of a_indices and length of "
                                "a_shape must match, got ",
                                num_dims, " and ", a_shape->NumElements()));
    OP_REQUIRES(ctx, num_dims > 0,
                errors::InvalidArgument("Tesors must not be empty"));
    OP_REQUIRES(
        ctx, a_shape->IsSameSize(*b_shape),
        errors::InvalidArgument(
            "Operands do not have the same ranks; got shapes: ",
            a_shape->SummarizeValue(10), " and ", b_shape->SummarizeValue(10)));
    const auto a_shape_flat = a_shape->flat<int64_t>();
    const auto b_shape_flat = b_shape->flat<int64_t>();
    for (int i = 0; i < a_shape->NumElements(); ++i) {
      OP_REQUIRES(ctx, a_shape_flat(i) == b_shape_flat(i),
                  errors::InvalidArgument(
                      "Operands' shapes do not match: got ", a_shape_flat(i),
                      " and ", b_shape_flat(i), " for dimension ", i));
    }

    OP_REQUIRES_OK(ctx, ctx->input("thresh", &thresh_t));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(thresh_t->shape()),
                errors::InvalidArgument(
                    "The magnitude threshold must be a scalar: got shape ",
                    thresh_t->shape().DebugString()));
    // std::abs() so that it works for complex{64,128} values as well
    const Treal thresh = thresh_t->scalar<Treal>()();

    // (1) do a pass over inputs, and append values and indices to vectors
    auto a_indices_mat = a_indices->matrix<int64_t>();
    auto b_indices_mat = b_indices->matrix<int64_t>();
    std::vector<std::pair<bool, int64>> entries_to_copy;  // from_a?, idx
    entries_to_copy.reserve(a_nnz + b_nnz);
    std::vector<T> out_values;

    // The input and output sparse tensors are assumed to be ordered along
    // increasing dimension number.
    int64_t i = 0, j = 0;
    T s;
    while (i < a_nnz && j < b_nnz) {
      switch (sparse::DimComparator::cmp(a_indices_mat, b_indices_mat, i, j,
                                         num_dims)) {
        case -1:
          entries_to_copy.emplace_back(true, i);
          out_values.push_back(a_values(i));
          ++i;
          break;
        case 0:
          s = a_values(i) + b_values(j);
          if (thresh <= std::abs(s)) {
            entries_to_copy.emplace_back(true, i);
            out_values.push_back(s);
          }
          ++i;
          ++j;
          break;
        case 1:
          entries_to_copy.emplace_back(false, j);
          out_values.push_back(b_values(j));
          ++j;
          break;
      }
    }

#define HANDLE_LEFTOVERS(A_OR_B, IDX, IS_A)     \
  while (IDX < A_OR_B##_nnz) {                  \
    entries_to_copy.emplace_back(IS_A, IDX);    \
    out_values.push_back(A_OR_B##_values(IDX)); \
    ++IDX;                                      \
  }

    // at most one of these calls appends new values
    HANDLE_LEFTOVERS(a, i, true);
    HANDLE_LEFTOVERS(b, j, false);
#undef HANDLE_LEFTOVERS

    // (2) allocate and fill output tensors
    const int64_t sum_nnz = out_values.size();
    Tensor *out_indices_t, *out_values_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({sum_nnz, num_dims}),
                                        &out_indices_t));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({sum_nnz}), &out_values_t));
    auto out_indices_mat = out_indices_t->matrix<int64_t>();
    auto out_values_flat = out_values_t->vec<T>();

    for (i = 0; i < sum_nnz; ++i) {
      const bool from_a = entries_to_copy[i].first;
      const int64_t idx = entries_to_copy[i].second;
      out_indices_mat.chip<0>(i) =
          from_a ? a_indices_mat.chip<0>(idx) : b_indices_mat.chip<0>(idx);
    }
    if (sum_nnz > 0) {
      std::copy_n(out_values.begin(), sum_nnz, &out_values_flat(0));
    }
    ctx->set_output(2, *a_shape);
  }
};

#define REGISTER_KERNELS(type, thresh_type)                           \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("SparseAdd").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseAddOp<type, thresh_type>)

// The list below is equivalent to TF_CALL_REAL_NUMBER_TYPES, minus uint8.  This
// is because std::abs() on uint8 does not compile.
REGISTER_KERNELS(float, float);
REGISTER_KERNELS(double, double);
REGISTER_KERNELS(int64_t, int64);
REGISTER_KERNELS(int32, int32);
REGISTER_KERNELS(int16, int16);
REGISTER_KERNELS(int8, int8);
REGISTER_KERNELS(complex64, float);
REGISTER_KERNELS(complex128, double);
#undef REGISTER_KERNELS
}  // namespace tensorflow
