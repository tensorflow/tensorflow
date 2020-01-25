/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename T>
class SparseFillEmptyRowsOp : public OpKernel {
 public:
  explicit SparseFillEmptyRowsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const int kIndicesInput = 0;
    const int kValuesInput = 1;
    const int kDenseShapeInput = 2;
    const int kDefaultValueInput = 3;

    const int kOutputIndicesOutput = 0;
    const int kOutputValuesOutput = 1;
    const int kEmptyRowIndicatorOutput = 2;
    const int kReverseIndexMapOutput = 3;

    const Tensor& indices_t = context->input(kIndicesInput);
    const Tensor& values_t = context->input(kValuesInput);
    const Tensor& dense_shape_t = context->input(kDenseShapeInput);
    const Tensor& default_value_t = context->input(kDefaultValueInput);

    const CPUDevice& d = context->eigen_device<CPUDevice>();

    OP_REQUIRES(context, TensorShapeUtils::IsVector(dense_shape_t.shape()),
                errors::InvalidArgument("dense_shape must be a vector, saw: ",
                                        dense_shape_t.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(indices_t.shape()),
                errors::InvalidArgument("indices must be a matrix, saw: ",
                                        indices_t.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(values_t.shape()),
                errors::InvalidArgument("values must be a vector, saw: ",
                                        values_t.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(default_value_t.shape()),
                errors::InvalidArgument("default_value must be a scalar, saw: ",
                                        default_value_t.shape().DebugString()));
    // TODO(ebrevdo): add shape checks between values, indices,
    // dense_shape.  Also add check that dense rank > 0.

    const T& default_value = default_value_t.scalar<T>()();
    const auto indices = indices_t.matrix<int64>();
    const auto values = values_t.vec<T>();
    const auto dense_shape = dense_shape_t.vec<int64>();

    const int64 N = indices_t.shape().dim_size(0);
    const int64 dense_rows = dense_shape(0);

    Tensor* empty_row_indicator_t;
    OP_REQUIRES_OK(context, context->allocate_output(kEmptyRowIndicatorOutput,
                                                     TensorShape({dense_rows}),
                                                     &empty_row_indicator_t));
    auto empty_row_indicator = empty_row_indicator_t->vec<bool>();
    Tensor* reverse_index_map_t;
    OP_REQUIRES_OK(context, context->allocate_output(kReverseIndexMapOutput,
                                                     TensorShape({N}),
                                                     &reverse_index_map_t));
    auto reverse_index_map = reverse_index_map_t->vec<int64>();

    int rank = indices_t.shape().dim_size(1);

    if (dense_rows == 0) {
      OP_REQUIRES(
          context, N == 0,
          errors::InvalidArgument("Received SparseTensor with dense_shape[0] = "
                                  "0 but indices.shape[0] = ",
                                  N));
      Tensor* output_indices_t;
      TensorShape output_indices_shape({0, rank});
      OP_REQUIRES_OK(context, context->allocate_output(kOutputIndicesOutput,
                                                       output_indices_shape,
                                                       &output_indices_t));
      Tensor* output_values_t;
      OP_REQUIRES_OK(context, context->allocate_output(kOutputValuesOutput,
                                                       TensorShape({0}),
                                                       &output_values_t));

      // Exit early, nothing more to do.
      return;
    }

    Tensor scratch_t;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({dense_rows}),
                                          &scratch_t));
    auto scratch = scratch_t.vec<int64>();
    scratch.device(d) = scratch.constant(0);
    for (int i = 0; i < N; ++i) {
      const int64 row = indices(i, 0);
      OP_REQUIRES(context, row >= 0 && row < dense_rows,
                  errors::InvalidArgument("indices(", i, ", 0) is invalid: ",
                                          row, " >= ", dense_rows));
      ++scratch(indices(i, 0));
    }
    for (int row = 0; row < dense_rows; ++row) {
      // Scratch here describes the number of elements in this dense row
      empty_row_indicator(row) = (scratch(row) == 0);
      // In filled version, each row has at least one element.
      scratch(row) = std::max(scratch(row), int64{1});
      // Update scratch to represent the number of elements up to and
      // including dense_row + 1:
      //  scratch(0) == #{elements of row 0}
      //  scratch(1) == #{elements of row 1} + #{elements of row 0}
      //  ..
      //  scratch(i) == starting index for elements in row i + 1.
      if (row > 0) {
        scratch(row) += scratch(row - 1);
      }
    }
    Tensor* output_indices_t;
    const int64 N_full = scratch(dense_rows - 1);
    TensorShape output_indices_shape({N_full, rank});
    OP_REQUIRES_OK(context, context->allocate_output(kOutputIndicesOutput,
                                                     output_indices_shape,
                                                     &output_indices_t));
    auto output_indices = output_indices_t->matrix<int64>();
    output_indices.device(d) = output_indices.constant(0);

    Tensor* output_values_t;
    OP_REQUIRES_OK(context, context->allocate_output(kOutputValuesOutput,
                                                     TensorShape({N_full}),
                                                     &output_values_t));
    auto output_values = output_values_t->vec<T>();
    output_values.device(d) = output_values.constant(default_value);

    Tensor filled_count_t;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({dense_rows}),
                                          &filled_count_t));
    auto filled_count = filled_count_t.vec<int64>();
    filled_count.device(d) = filled_count.constant(0);

    // Fill in values for rows that are not missing
    for (int64 i = 0; i < N; ++i) {
      const int64 row = indices(i, 0);
      int64& offset = filled_count(row);
      const int64 output_i = ((row == 0) ? 0 : scratch(row - 1)) + offset;
      offset++;  // Increment the filled count for this row.
      std::copy_n(&indices(i, 0), rank, &output_indices(output_i, 0));
      output_values(output_i) = values(i);
      // We'll need this reverse index map to backprop correctly.
      reverse_index_map(i) = output_i;
    }

    // Fill in values for rows that are missing
    for (int64 row = 0; row < dense_rows; ++row) {
      const int64 row_count = filled_count(row);
      if (row_count == 0) {  // We haven't filled this row
        const int64 starting_index = (row == 0) ? 0 : scratch(row - 1);
        // Remaining index values were set to zero already.
        // The value at this index was set to default_value already.
        // Just need to set the row index in the right location.
        output_indices(starting_index, 0) = row;
      }
    }
  }
};

#define REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("SparseFillEmptyRows")     \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<type>("T"), \
                          SparseFillEmptyRowsOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

template <typename T>
class SparseFillEmptyRowsGradOp : public OpKernel {
 public:
  explicit SparseFillEmptyRowsGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* reverse_index_map_t;
    const Tensor* grad_values_t;
    OP_REQUIRES_OK(context,
                   context->input("reverse_index_map", &reverse_index_map_t));
    OP_REQUIRES_OK(context, context->input("grad_values", &grad_values_t));

    const CPUDevice& d = context->eigen_device<CPUDevice>();

    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(reverse_index_map_t->shape()),
        errors::InvalidArgument("reverse_index_map must be a vector, saw: ",
                                reverse_index_map_t->shape().DebugString()));

    const auto reverse_index_map = reverse_index_map_t->vec<int64>();
    const auto grad_values = grad_values_t->vec<T>();

    const int64 N = reverse_index_map_t->shape().dim_size(0);
    const int64 N_full = grad_values_t->shape().dim_size(0);

    Tensor* d_values_t;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "d_values", TensorShape({N}), &d_values_t));
    auto d_values = d_values_t->vec<T>();
    Tensor* d_default_value_t;
    OP_REQUIRES_OK(context,
                   context->allocate_output("d_default_value", TensorShape({}),
                                            &d_default_value_t));
    T& d_default_value = d_default_value_t->scalar<T>()();
    d_default_value = T();

    Tensor visited_t;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DT_BOOL, TensorShape({N_full}), &visited_t));
    auto visited = visited_t.vec<bool>();
    visited.device(d) = visited.constant(false);

    for (int i = 0; i < N; ++i) {
      // Locate the index of the output of the forward prop associated
      // with this location in the input of the forward prop.  Copy
      // the gradient into it.  Mark it as visited.
      d_values(i) = grad_values(reverse_index_map(i));
      visited(reverse_index_map(i)) = true;
    }
    for (int j = 0; j < N_full; ++j) {
      // The default value gradient gets the accumulated remainder of
      // the backprop values (since the default value was used to fill
      // in these slots in the forward calculation).
      if (!visited(j)) {
        d_default_value += grad_values(j);
      }
    }
  }
};

#define REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("SparseFillEmptyRowsGrad") \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<type>("T"), \
                          SparseFillEmptyRowsGradOp<type>)

TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS
}  // namespace tensorflow
