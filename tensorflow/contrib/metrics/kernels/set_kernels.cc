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

// Ops for operating with sets. They are not checked in
// to TensorFlow because we would first like to demonstrate successful
// end-to-end use of these ops in eval and polush the api a bit like taking two
// SparseTensor rather than on edense and one sparse.

#define EIGEN_USE_THREADS

#include <algorithm>
// TODO(ptucker): Consider switching back to hash_set - I had trouble getting it
// to work with string values.
#include <set>
#include <string>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

// Validate rank >= 2.
void CheckRankAtLeast2(OpKernelContext* ctx, const TensorShape& shape) {
  const auto rank = shape.dims();
  OP_REQUIRES(ctx, rank >= 2,
              errors::InvalidArgument("Invalid rank ", rank, "."));
}

// Return group shape, which is the 1st n-1 dimensions of shape.
const TensorShape GroupShape(OpKernelContext* ctx,
                             const TensorShape& input_shape) {
  CheckRankAtLeast2(ctx, input_shape);
  TensorShape shape(input_shape);
  shape.RemoveDim(shape.dims() - 1);
  return TensorShape(shape);
}

// Validate sparse indices are valid. This is O(n), so use sparingly.
void CheckSparseTensorIndices(OpKernelContext* ctx,
                              const sparse::SparseTensor& st) {
  OP_REQUIRES_OK(ctx, st.IndicesValid());
}

// Build `SparseTensor` from indices, values, and shape in inputs
// [base_index, base_index + 3), and validate its rank and indices.
sparse::SparseTensor SparseTensorFromContext(OpKernelContext* ctx,
                                             const int32 base_index,
                                             bool validate_indices) {
  // Assume row-major order.
  const TensorShape shape =
      TensorShape(ctx->input(base_index + 2).vec<int64>());
  CheckRankAtLeast2(ctx, shape);
  std::vector<int64> order(shape.dims());
  std::iota(order.begin(), order.end(), 0);

  const sparse::SparseTensor st(ctx->input(base_index),
                                ctx->input(base_index + 1), shape, order);
  if (validate_indices) {
    CheckSparseTensorIndices(ctx, st);
  }
  return st;
}

// TODO(ptucker): CheckGroup is just a sanity check on the result of
// SparseTensor.group, consider removing.
// `sparse_tensor_shape` is the shape of the `SparseTensor` from which group
// was created, and is used to sanity check the indices in `group'.
template <typename T>
void CheckGroup(OpKernelContext* ctx, const sparse::Group& group,
                const TensorShape& sparse_tensor_shape) {
  const auto& indices = group.indices();
  const auto& values = group.values<T>();

  // Sanity check: group is non-empty, and indices and values are same size.
  const auto num_values = values.dimension(0);
  OP_REQUIRES(ctx, indices.size() > 0, errors::Internal("Empty group."));
  OP_REQUIRES(
      ctx, indices.dimension(0) == num_values,
      errors::Internal("shape[0] of group indices ", indices.dimension(0),
                       " != values ", num_values, "."));

  // Sanity check: valid indices.
  const auto group_rank = indices.dimension(1);
  const auto expected_rank = sparse_tensor_shape.dims();
  OP_REQUIRES(ctx, expected_rank == group_rank,
              errors::Internal("Rank expected ", expected_rank, ", got ",
                               group_rank, "."));
  for (int32 j = 0; j < expected_rank; ++j) {
    const auto dim_size = sparse_tensor_shape.dim_size(j);
    OP_REQUIRES(ctx, dim_size > 0, errors::Internal("Invalid dim_size[", j,
                                                    "] = ", dim_size, "."));
    for (int64 i = 0; i < num_values; ++i) {
      const auto index = indices(i, j);
      OP_REQUIRES(ctx, dim_size > index,
                  errors::Internal("indices[", i, ", ", j, "] expected < ",
                                   dim_size, ", got ", index, "."));
    }
  }
}

// This lets us calculate the row-major index into flattened output.
const gtl::InlinedVector<int64, 8> Strides(const TensorShape& shape) {
  gtl::InlinedVector<int64, 8> result(shape.dims());
  int64 product = 1;
  for (auto i = shape.dims() - 1; i >= 0; --i) {
    result[i] = product;
    product *= shape.dim_size(i);
  }
  return result;
}

// TODO(ptucker): If memory becomes an issue, consider a 2-pass approach to
// eliminate the intermediate `values` data structure - iterate once to
// determine `num_values`, allocate output tensors, then write results directly
// to output tensors.

// TODO(ptucker): Consider sharding work across multiple threads. See
// SparseCrossOp for an example.

// Output `SparseTensor` of shape `output_shape`. `sets` contains a map of
// group indices (i.e., values for all but the last dimension of `output_shape`)
// to set values, each of which will occupy the last dimension of
// `output_shape`.
template <typename T>
void OutputSparseTensor(OpKernelContext* ctx, const TensorShape& output_shape,
                        const int64 num_values,
                        const std::map<std::vector<int64>, std::set<T>>& sets) {
  // Allocate 3 output tensors for sparse data.
  Tensor *out_indices_t, *out_values_t, *out_shape_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(
                          0, TensorShape({num_values, output_shape.dims()}),
                          &out_indices_t));
  OP_REQUIRES_OK(
      ctx, ctx->allocate_output(1, TensorShape({num_values}), &out_values_t));
  OP_REQUIRES_OK(ctx, ctx->allocate_output(
                          2, TensorShape({output_shape.dims()}), &out_shape_t));
  auto out_indices_mat = out_indices_t->matrix<int64>();
  auto out_values_flat = out_values_t->vec<T>();

  // For each set, write its indices and values to output tensors.
  int64 value_index = 0;
  for (auto it = sets.begin(); it != sets.end(); ++it) {
    const auto& group_indices = it->first;
    OP_REQUIRES(
        ctx, group_indices.size() == output_shape.dims() - 1,
        errors::Internal("Invalid number of indices ", group_indices.size(),
                         ", expected ", output_shape.dims() - 1, "."));
    const auto& set = it->second;

    // For each set item, write its indices and value to output tensors.
    int64 group_value_index = 0;
    for (auto value = set.begin(); value != set.end();
         ++value, ++value_index, ++group_value_index) {
      // First n-1 dimensions are the group, last dimension is the position in
      // the set.
      for (int32 i = 0; i < group_indices.size(); ++i) {
        out_indices_mat(value_index, i) = group_indices[i];
      }
      out_indices_mat(value_index, group_indices.size()) = group_value_index;

      out_values_flat(value_index) = *value;
    }
  }

  // Write output shape.
  auto out_shape_flat = out_shape_t->vec<int64>();
  for (int32 i = 0; i < output_shape.dims(); ++i) {
    out_shape_flat(i) = output_shape.dim_size(i);
  }
}

bool ValidateIndicesFromContext(OpKernelConstruction* ctx) {
  bool result;
  if (ctx->GetAttr("validate_indices", &result).ok()) {
    return result;
  }
  return true;
}

// Populate `result` set from group in `tensor`. "Group" is defined by
// `group_indices`, which are values for the first n-1 dimensions of
// `input_tensor`. `input_strides` is provided to avoid recalculating it
// multiple times, and is used to calculate the flat index into `input_tensor`
// values.
template <typename T>
void PopulateFromDenseGroup(OpKernelContext* ctx, const Tensor& input_tensor,
                            const gtl::InlinedVector<int64, 8>& input_strides,
                            const std::vector<int64>& group_indices,
                            std::set<T>* result) {
  OP_REQUIRES(ctx, group_indices.size() == input_strides.size() - 1,
              errors::Internal("group_indices.size ", group_indices.size(),
                               ", !=  input_strides.size-1 ",
                               input_strides.size() - 1, "."));
  result->clear();
  auto input_flat = input_tensor.flat<T>();
  const auto start = std::inner_product(
      group_indices.begin(), group_indices.end(), input_strides.begin(), 0);
  const TensorShape& input_shape = input_tensor.shape();
  const auto end = start + input_shape.dim_size(input_shape.dims() - 1);
  for (int64 i = start; i < end; ++i) {
    result->insert(input_flat(i));
  }
}

// Populate `result` set from `group`. `sparse_tensor_shape` is the shape of the
// `SparseTensor` from which group was created, and is used to sanity check the
// indices in `group'.
template <typename T>
void PopulateFromSparseGroup(OpKernelContext* ctx, const sparse::Group& group,
                             const TensorShape& sparse_tensor_shape,
                             std::set<T>* result) {
  CheckGroup<T>(ctx, group, sparse_tensor_shape);
  result->clear();
  const auto& group_values = group.values<T>();
  for (int64 i = 0; i < group_values.size(); ++i) {
    result->insert(group_values(i));
  }
}

template <typename T>
class SetSizeOp : public OpKernel {
 public:
  explicit SetSizeOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), validate_indices_(ValidateIndicesFromContext(ctx)) {}

  void Compute(OpKernelContext* ctx) override;

 private:
  const bool validate_indices_;
};

template <typename T>
void SetSizeOp<T>::Compute(OpKernelContext* ctx) {
  const sparse::SparseTensor set_st =
      SparseTensorFromContext(ctx, 0, validate_indices_);

  // Output shape is same as input except for last dimension, which reduces to
  // the set size of values along that dimension.
  const TensorShape output_shape = GroupShape(ctx, set_st.shape());
  const auto output_strides = Strides(output_shape);

  Tensor* out_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out_t));
  auto out = out_t->flat<int32>();
  out.device(ctx->eigen_cpu_device()) = out.constant(static_cast<int32>(0.0));

  // Group by all but last dimension, create a set of group values, and add set
  // size to output.
  sparse::SparseTensor::VarDimArray group_ix(set_st.order(), 0,
                                             set_st.order().size() - 1);
  std::set<T> group_set;
  for (const auto& group : set_st.group(group_ix)) {
    PopulateFromSparseGroup<T>(ctx, group, set_st.shape(), &group_set);

    const auto group_key = group.group();
    const auto output_index = std::inner_product(
        group_key.begin(), group_key.end(), output_strides.begin(), 0);
    out(output_index) = group_set.size();
  }
}

#define _SET_SIZE_REGISTER_KERNEL_BUILDER(T)                     \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("SetSize").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SetSizeOp<T>);
_SET_SIZE_REGISTER_KERNEL_BUILDER(int8);
_SET_SIZE_REGISTER_KERNEL_BUILDER(int16);
_SET_SIZE_REGISTER_KERNEL_BUILDER(int32);
_SET_SIZE_REGISTER_KERNEL_BUILDER(int64);
_SET_SIZE_REGISTER_KERNEL_BUILDER(uint8);
_SET_SIZE_REGISTER_KERNEL_BUILDER(uint16);
_SET_SIZE_REGISTER_KERNEL_BUILDER(string);
#undef _SET_SIZE_REGISTER_KERNEL_BUILDER

enum InputTypes {
  DENSE_DENSE = 0,
  DENSE_SPARSE = 1,
  SPARSE_SPARSE = 2,
};

enum SetOperation { A_MINUS_B = 0, B_MINUS_A = 1, INTERSECTION = 2, UNION = 3 };

SetOperation SetOperationFromContext(OpKernelConstruction* ctx) {
  string set_operation_str;
  if (!ctx->GetAttr("set_operation", &set_operation_str).ok()) {
    ctx->CtxFailure(errors::InvalidArgument("Missing set_operation."));
  } else {
    std::transform(set_operation_str.begin(), set_operation_str.end(),
                   set_operation_str.begin(), ::tolower);
    if ("a-b" == set_operation_str) {
      return A_MINUS_B;
    }
    if ("b-a" == set_operation_str) {
      return B_MINUS_A;
    }
    if ("intersection" == set_operation_str) {
      return INTERSECTION;
    }
    if ("union" != set_operation_str) {
      ctx->CtxFailure(errors::InvalidArgument("Invalid set_operation ",
                                              set_operation_str, "."));
    }
  }
  // NOTE: This is not the default, this function fails if no 'set_operation'
  // attribute is provided.
  return UNION;
}

// Abstract base class for performing set operations across the last dimension
// of 2 input tensors.
template <typename T>
class SetOperationOp : public OpKernel {
 public:
  SetOperationOp(OpKernelConstruction* ctx, InputTypes input_types)
      : OpKernel(ctx),
        set_operation_(SetOperationFromContext(ctx)),
        validate_indices_(ValidateIndicesFromContext(ctx)),
        input_types_(input_types) {}

  void Compute(OpKernelContext* ctx) override;

 private:
  void ApplySetOperation(const std::set<T>& set1, const std::set<T>& set2,
                         std::set<T>* result) const;
  void ComputeDenseToDense(OpKernelContext* ctx) const;
  void ComputeDenseToSparse(OpKernelContext* ctx) const;
  void ComputeSparseToSparse(OpKernelContext* ctx) const;
  const SetOperation set_operation_;
  const bool validate_indices_;
  const InputTypes input_types_;
};

template <typename T>
void SetOperationOp<T>::ApplySetOperation(const std::set<T>& set1,
                                          const std::set<T>& set2,
                                          std::set<T>* result) const {
  switch (set_operation_) {
    case A_MINUS_B:
      std::set_difference(set1.begin(), set1.end(), set2.begin(), set2.end(),
                          std::inserter(*result, result->begin()));
      break;
    case B_MINUS_A:
      std::set_difference(set2.begin(), set2.end(), set1.begin(), set1.end(),
                          std::inserter(*result, result->begin()));
      break;
    case INTERSECTION:
      std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
                            std::inserter(*result, result->begin()));
      break;
    case UNION:
      std::set_union(set1.begin(), set1.end(), set2.begin(), set2.end(),
                     std::inserter(*result, result->begin()));
      break;
  }
}

// Validate shapes have the same dimensions.
void CheckShapesMatch(OpKernelContext* ctx, const TensorShape& shape1,
                      const TensorShape& shape2) {
  OP_REQUIRES(
      ctx, shape1 == shape2,
      errors::InvalidArgument("Mismatched shapes ", shape1.DebugString(),
                              " vs ", shape2.DebugString(), "."));
}

// Validate ranks are the same, and all but last dimension are the same.
// Return GroupShape.
const TensorShape GroupShapeFromInputs(OpKernelContext* ctx,
                                       const TensorShape& shape1,
                                       const TensorShape& shape2) {
  const TensorShape group_shape = GroupShape(ctx, shape1);
  CheckShapesMatch(ctx, group_shape, GroupShape(ctx, shape2));
  return group_shape;
}

// Split `flat_group_index` into separate dimensions based on `group_shape`.
void PopulateGroupIndices(const int64 flat_group_index,
                          const TensorShape& group_shape,
                          std::vector<int64>* group_indices) {
  group_indices->clear();
  int64 running_flat_group_index = flat_group_index;
  for (auto group_dim_index = group_shape.dims() - 1; group_dim_index >= 0;
       --group_dim_index) {
    const auto group_dim = group_shape.dim_size(group_dim_index);
    group_indices->insert(group_indices->begin(),
                          running_flat_group_index % group_dim);
    running_flat_group_index /= group_dim;
  }
}

// `ctx` contains set1 and set2 dense tensors.
// Iterate over groups in set1 and set2, applying `ApplySetOperation` to each,
// and outputing the result `SparseTensor`. A "group" is a collection of values
// with the same first n-1 dimensions in set1 and set2.
template <typename T>
void SetOperationOp<T>::ComputeDenseToDense(OpKernelContext* ctx) const {
  const Tensor& set1_t = ctx->input(0);
  const Tensor& set2_t = ctx->input(1);
  const TensorShape group_shape =
      GroupShapeFromInputs(ctx, set1_t.shape(), set2_t.shape());

  const auto set1_strides = Strides(set1_t.shape());
  const auto set2_strides = Strides(set2_t.shape());

  std::map<std::vector<int64>, std::set<T>> group_sets;
  int64 num_result_values = 0;
  int64 max_set_size = 0;

  std::set<T> set1_group_set;
  std::set<T> set2_group_set;
  std::vector<int64> group_indices;
  for (int64 flat_group_index = 0;
       flat_group_index < group_shape.num_elements(); ++flat_group_index) {
    PopulateGroupIndices(flat_group_index, group_shape, &group_indices);
    PopulateFromDenseGroup<T>(ctx, set1_t, set1_strides, group_indices,
                              &set1_group_set);
    PopulateFromDenseGroup<T>(ctx, set2_t, set2_strides, group_indices,
                              &set2_group_set);

    std::set<T> group_set;
    ApplySetOperation(set1_group_set, set2_group_set, &group_set);
    if (group_set.size() > 0) {
      group_sets[group_indices] = group_set;
      const auto set_size = group_set.size();
      if (set_size > max_set_size) {
        max_set_size = set_size;
      }
      num_result_values += set_size;
    }
  }

  TensorShape output_shape(group_shape);
  output_shape.AddDim(max_set_size);
  OutputSparseTensor<T>(ctx, output_shape, num_result_values, group_sets);
}

// `ctx` contains dense set1 and sparse set2 tensors.
// Iterate over groups in set1 and set2, applying `ApplySetOperation` to each,
// and outputing the result `SparseTensor`. A "group" is a collection of values
// with the same first n-1 dimensions in set1 and set2.
template <typename T>
void SetOperationOp<T>::ComputeDenseToSparse(OpKernelContext* ctx) const {
  const Tensor& set1_t = ctx->input(0);
  const sparse::SparseTensor set2_st =
      SparseTensorFromContext(ctx, 1, validate_indices_);
  const TensorShape group_shape =
      GroupShapeFromInputs(ctx, set1_t.shape(), set2_st.shape());

  const auto set1_strides = Strides(set1_t.shape());

  std::map<std::vector<int64>, std::set<T>> group_sets;
  int64 num_result_values = 0;
  int64 max_set_size = 0;

  std::set<T> set1_group_set;
  std::set<T> set2_group_set;
  auto set2_grouper = set2_st.group(sparse::SparseTensor::VarDimArray(
      set2_st.order(), 0, set2_st.order().size() - 1));
  auto set2_group_it = set2_grouper.begin();
  std::vector<int64> group_indices;
  for (int64 flat_group_index = 0;
       flat_group_index < group_shape.num_elements(); ++flat_group_index) {
    PopulateGroupIndices(flat_group_index, group_shape, &group_indices);

    // Get values from set1.
    PopulateFromDenseGroup<T>(ctx, set1_t, set1_strides, group_indices,
                              &set1_group_set);

    // Get values from set2, if applicable.
    set2_group_set.clear();
    if (set2_group_it != set2_grouper.end()) {
      const auto& group = *set2_group_it;
      const auto set2_group_indices = group.group();
      OP_REQUIRES(
          ctx, set2_group_indices.size() == group_indices.size(),
          errors::InvalidArgument("Invalid number of group indices ",
                                  set2_group_indices.size(), ", expected ",
                                  group_indices.size(), "."));
      bool group_match = true;
      for (int32 i = 0; group_match && (i < set2_group_indices.size()); ++i) {
        if (set2_group_indices[i] != group_indices[i]) {
          group_match = false;
        }
      }
      if (group_match) {
        PopulateFromSparseGroup<T>(ctx, group, set2_st.shape(),
                                   &set2_group_set);
        ++set2_group_it;
      }
    }

    std::set<T> group_set;
    ApplySetOperation(set1_group_set, set2_group_set, &group_set);
    if (group_set.size() > 0) {
      group_sets[group_indices] = group_set;
      const auto set_size = group_set.size();
      if (set_size > max_set_size) {
        max_set_size = set_size;
      }
      num_result_values += set_size;
    }
  }

  TensorShape output_shape(group_shape);
  output_shape.AddDim(max_set_size);
  OutputSparseTensor<T>(ctx, output_shape, num_result_values, group_sets);
}

// This is used to determine which group iterator is less than the other, based
// on row-major ordering of indices.
// An empty index list indicates end of iteration, which is interpreted as "max"
// for the purposes of comparison; i.e., non-empty < empty.
// Return 0 if both groups are empty, or both non-empty with the same values.
// Return <0 if set1 <= set2, or set2 is empty.
// Return >0 if set2 <= set1, or set1 is empty.
void CompareGroups(OpKernelContext* ctx,
                   const std::vector<int64>& set1_group_indices,
                   const std::vector<int64>& set2_group_indices,
                   int64* result) {
  if (set1_group_indices.empty()) {
    *result = set2_group_indices.empty() ? 0 : 1;
    return;
  }
  if (set2_group_indices.empty()) {
    *result = set1_group_indices.empty() ? 0 : -1;
    return;
  }
  OP_REQUIRES(ctx, set1_group_indices.size() == set2_group_indices.size(),
              errors::InvalidArgument("Mismatched group dims ",
                                      set1_group_indices.size(), " vs ",
                                      set2_group_indices.size(), "."));
  for (int32 i = 0; i < set1_group_indices.size(); ++i) {
    *result = set1_group_indices[i] - set2_group_indices[i];
    if (*result != 0) {
      return;
    }
  }
}

// Empty indices vector represents iteration end in `CompareGroups`.
const std::vector<int64> GROUP_ITER_END;

// `ctx` contains set1 and set2 sparse tensors.
// Iterate over groups in set1 and set2, applying `ApplySetOperation` to each,
// and outputing the result `SparseTensor`. A "group" is a collection of values
// with the same first n-1 dimensions in set1 and set2.
template <typename T>
void SetOperationOp<T>::ComputeSparseToSparse(OpKernelContext* ctx) const {
  const sparse::SparseTensor set1_st =
      SparseTensorFromContext(ctx, 0, validate_indices_);
  const sparse::SparseTensor set2_st =
      SparseTensorFromContext(ctx, 3, validate_indices_);
  const TensorShape group_shape =
      GroupShapeFromInputs(ctx, set1_st.shape(), set2_st.shape());

  const auto set1_strides = Strides(set1_st.shape());
  const auto set2_strides = Strides(set2_st.shape());

  std::map<std::vector<int64>, std::set<T>> group_sets;
  int64 num_result_values = 0;
  int64 max_set_size = 0;

  std::set<T> set1_group_set;
  std::set<T> set2_group_set;
  auto set1_grouper = set1_st.group(sparse::SparseTensor::VarDimArray(
      set1_st.order(), 0, set1_st.order().size() - 1));
  auto set1_group_it = set1_grouper.begin();
  auto set2_grouper = set2_st.group(sparse::SparseTensor::VarDimArray(
      set2_st.order(), 0, set2_st.order().size() - 1));
  auto set2_group_it = set2_grouper.begin();

  // Group by rows, and iterate over rows of both sets in parallel, creating a
  // set for each row.
  while ((set1_group_it != set1_grouper.end()) ||
         (set2_group_it != set2_grouper.end())) {
    const std::vector<int64>& set1_group_indices =
        (set1_group_it == set1_grouper.end()) ? GROUP_ITER_END
                                              : (*set1_group_it).group();
    const std::vector<int64>& set2_group_indices =
        (set2_group_it == set2_grouper.end()) ? GROUP_ITER_END
                                              : (*set2_group_it).group();

    int64 compare_groups;
    CompareGroups(ctx, set1_group_indices, set2_group_indices, &compare_groups);
    const std::vector<int64>* group_indices;

    // Get values from set1, if applicable.
    set1_group_set.clear();
    if (compare_groups <= 0) {
      PopulateFromSparseGroup<T>(ctx, *set1_group_it, set1_st.shape(),
                                 &set1_group_set);
      ++set1_group_it;
      group_indices = &set1_group_indices;
    }

    // Get values from set2, if applicable.
    set2_group_set.clear();
    if (compare_groups >= 0) {
      PopulateFromSparseGroup<T>(ctx, *set2_group_it, set2_st.shape(),
                                 &set2_group_set);
      ++set2_group_it;
      group_indices = &set2_group_indices;
    }

    std::set<T> group_set;
    ApplySetOperation(set1_group_set, set2_group_set, &group_set);
    if (group_set.size() > 0) {
      group_sets[*group_indices] = group_set;
      const auto set_size = group_set.size();
      if (set_size > max_set_size) {
        max_set_size = set_size;
      }
      num_result_values += set_size;
    }
  }

  TensorShape output_shape(group_shape);
  output_shape.AddDim(max_set_size);
  OutputSparseTensor<T>(ctx, output_shape, num_result_values, group_sets);
}

// Given set1 of shape [b, n1] and data_2 of shape [b, n2], populate result
// sparse tendor with [b, n3] values, where each row `i` contains the result of
// the set operation on elements from set1[i] and set2[i]. `n3` is the number
// of elements in that result row.
template <typename T>
void SetOperationOp<T>::Compute(OpKernelContext* ctx) {
  switch (input_types_) {
    case DENSE_DENSE:
      ComputeDenseToDense(ctx);
      break;
    case DENSE_SPARSE:
      ComputeDenseToSparse(ctx);
      break;
    case SPARSE_SPARSE:
      ComputeSparseToSparse(ctx);
      break;
  }
}

template <typename T>
class DenseToDenseSetOperationOp : public SetOperationOp<T> {
 public:
  explicit DenseToDenseSetOperationOp(OpKernelConstruction* ctx)
      : SetOperationOp<T>(ctx, DENSE_DENSE) {}
};

#define _DENSE_TO_DENSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(T) \
  REGISTER_KERNEL_BUILDER(Name("DenseToDenseSetOperation")       \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<T>("T"),           \
                          DenseToDenseSetOperationOp<T>);
_DENSE_TO_DENSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int8);
_DENSE_TO_DENSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int16);
_DENSE_TO_DENSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int32);
_DENSE_TO_DENSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int64);
_DENSE_TO_DENSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(uint8);
_DENSE_TO_DENSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(uint16);
_DENSE_TO_DENSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(string);
#undef _DENSE_TO_DENSE_SET_OPERATION_REGISTER_KERNEL_BUILDER

template <typename T>
class DenseToSparseSetOperationOp : public SetOperationOp<T> {
 public:
  explicit DenseToSparseSetOperationOp(OpKernelConstruction* ctx)
      : SetOperationOp<T>(ctx, DENSE_SPARSE) {}
};

#define _DENSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(T) \
  REGISTER_KERNEL_BUILDER(Name("DenseToSparseSetOperation")       \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<T>("T"),            \
                          DenseToSparseSetOperationOp<T>);
_DENSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int8);
_DENSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int16);
_DENSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int32);
_DENSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int64);
_DENSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(uint8);
_DENSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(uint16);
_DENSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(string);
#undef _DENSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER

template <typename T>
class SparseToSparseSetOperationOp : public SetOperationOp<T> {
 public:
  explicit SparseToSparseSetOperationOp(OpKernelConstruction* ctx)
      : SetOperationOp<T>(ctx, SPARSE_SPARSE) {}
};

#define _SPARSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(T) \
  REGISTER_KERNEL_BUILDER(Name("SparseToSparseSetOperation")       \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<T>("T"),             \
                          SparseToSparseSetOperationOp<T>);
_SPARSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int8);
_SPARSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int16);
_SPARSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int32);
_SPARSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(int64);
_SPARSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(uint8);
_SPARSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(uint16);
_SPARSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER(string);
#undef _SPARSE_TO_SPARSE_SET_OPERATION_REGISTER_KERNEL_BUILDER

}  // namespace tensorflow
