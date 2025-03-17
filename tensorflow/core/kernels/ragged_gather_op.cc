/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

namespace {

// For each slice in `(start, limit)` in `value_slices`, append
// `params_dense_values_in[start:limit] to `values_out`.  `value_size` indicates
// the number of scalars contained in each value params_dense_values_in[i].
template <typename VALUE_TYPE, typename SPLITS_TYPE>
void WriteValueSlices(
    const Tensor& params_dense_values_in,
    const std::vector<std::pair<SPLITS_TYPE, SPLITS_TYPE>>& value_slices,
    SPLITS_TYPE value_size, Tensor* values_out) {
  const auto& params_dense_values =
      params_dense_values_in.flat_outer_dims<VALUE_TYPE, 2>();
  auto values = values_out->flat_outer_dims<VALUE_TYPE, 2>();
  int out_pos = 0;
  for (const auto& slice : value_slices) {
    for (int i = slice.first; i < slice.second; ++i) {
      for (int j = 0; j < value_size; ++j) {
        values(out_pos, j) = params_dense_values(i, j);
      }
      ++out_pos;
    }
  }
}

}  // namespace

template <typename INDEX_TYPE, typename SPLITS_TYPE>
class RaggedGatherOpBase : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* context) override {
    // Get the input Tensors.

    OpInputList params_nested_splits_in;
    OP_REQUIRES_OK(context, context->input_list("params_nested_splits",
                                                &params_nested_splits_in));
    OP_REQUIRES(
        context, params_nested_splits_in.size() > 0,
        errors::InvalidArgument("params_nested_splits must be non empty"));

    const Tensor& params_dense_values_in =
        context->input(params_nested_splits_in.size());
    const Tensor& indices_in =
        context->input(params_nested_splits_in.size() + 1);

    OP_REQUIRES(context, params_nested_splits_in[0].dims() > 0,
                errors::InvalidArgument("Split tensors must not be scalars"));
    SPLITS_TYPE num_params = params_nested_splits_in[0].dim_size(0) - 1;
    OP_REQUIRES_OK(context, ValidateIndices(indices_in, num_params));

    OP_REQUIRES(context, params_dense_values_in.dims() > 0,
                errors::InvalidArgument("params.rank must be nonzero"));
    SPLITS_TYPE num_params_dense_values = params_dense_values_in.dim_size(0);

    // Calculate the `splits`, and store the value slices that we need to
    // copy in `value_slices`.
    std::vector<std::pair<SPLITS_TYPE, SPLITS_TYPE>> value_slices;
    SPLITS_TYPE num_values = 0;
    std::vector<std::vector<SPLITS_TYPE>> out_splits;
    OP_REQUIRES_OK(context, MakeSplits(indices_in, params_nested_splits_in,
                                       num_params_dense_values, &out_splits,
                                       &value_slices, &num_values));

    // Write the output tensors.
    OP_REQUIRES_OK(context, WriteSplits(out_splits, context));
    OP_REQUIRES_OK(context,
                   WriteValues(params_dense_values_in, value_slices,
                               out_splits.size(), num_values, context));
  }

 private:
  using ConstFlatType = typename TTypes<SPLITS_TYPE>::ConstFlat;

  // Check if any indices are out-of-bounds.
  absl::Status ValidateIndices(const Tensor& indices_in,
                               SPLITS_TYPE num_params) {
    const auto& indices = indices_in.flat<INDEX_TYPE>();
    for (SPLITS_TYPE i = 0; i < indices.size(); ++i) {
      SPLITS_TYPE index = indices(i);
      if (index < 0 || index >= num_params) {
        return errors::InvalidArgument(
            "indices", SliceDebugString(indices_in.shape(), i), " = ", index,
            " is not in [0, ", num_params, ")");
      }
    }
    return absl::OkStatus();
  }

  // Construct the `splits` output tensors, encoded using a nested vector.
  // Also find the slices of values that need to be copied, and store them
  // in `value_slices`.  The total number of values that will be copied (which
  // we need for allocating the output values tensor) is stored in `num_values`.
  absl::Status MakeSplits(
      const Tensor& indices_in, const OpInputList& params_nested_splits_in,
      SPLITS_TYPE num_params_dense_values,
      std::vector<std::vector<SPLITS_TYPE>>* out_splits,
      std::vector<std::pair<SPLITS_TYPE, SPLITS_TYPE>>* value_slices,
      SPLITS_TYPE* num_values) {
    *num_values = 0;
    value_slices->clear();

    int num_splits = indices_in.dims() - 1 + params_nested_splits_in.size();
    out_splits->assign(num_splits, {0});

    // Get Eigen tensors.
    const auto& indices = indices_in.flat<INDEX_TYPE>();
    std::vector<ConstFlatType> params_nested_splits;
    params_nested_splits.reserve(params_nested_splits_in.size());
    for (const auto& splits_in : params_nested_splits_in) {
      params_nested_splits.push_back(splits_in.flat<SPLITS_TYPE>());
    }

    TF_RETURN_IF_ERROR(
        ValidateSplits(params_nested_splits, num_params_dense_values));

    // Add `splits` that come from all but the last dimension of the dense
    // Tensor `indices`.  In particular, for each dimension D, we add a
    // splits tensor whose values are:
    //   range(reduce_prod(splits.shape[:D]) + 1) * splits.shape[D+1]
    // E.g., if indices.shape=[2, 3, 4] then we will add splits tensors:
    //   [0, 3, 6]                    # length=2+1, stride=3
    //   [0, 4, 8, 12, 16, 20, 24]    # length=2*3+1, stride=4
    int nrows = 1;
    for (int dim = 0; dim < indices_in.dims() - 1; ++dim) {
      nrows *= indices_in.dim_size(dim);
      int row_length = indices_in.dim_size(dim + 1);
      for (int i = 1; i < nrows + 1; ++i) {
        out_splits->at(dim).push_back(i * row_length);
      }
    }

    // Add `splits` that come from `params_nested_splits`.  Starting with the
    // outermost ragged dimension (i.e., the first `splits` tensor), we work
    // our way in, finding the range of values that should be copied.  As we
    // go, we update the output `splits` for each dimension with the appropriate
    // values.  In particular, the *lengths* of the slices from `param_splits`
    // should be copied to generate corresponding slice lengths in the output
    // splits.  E.g., if we are copying a ragged row with length 4, then we
    // should add a new split point to out_splits that is 4 greater than the
    // previous split point in out_splits.
    for (int i = 0; i < indices.size(); ++i) {
      int start = indices(i);
      int limit = indices(i) + 1;

      // Copy splits.
      for (int dim = 0; dim < params_nested_splits.size(); ++dim) {
        const auto& splits = params_nested_splits[dim];
        int out_dim = dim + indices_in.dims() - 1;
        if (out_dim >= 0) {
          SPLITS_TYPE delta = out_splits->at(out_dim).back() - splits(start);
          for (int j = start; j < limit; ++j) {
            out_splits->at(out_dim).push_back(splits(j + 1) + delta);
          }
        }
        start = splits(start);
        limit = splits(limit);
      }
      if (limit != start) {
        value_slices->emplace_back(start, limit);
        *num_values += limit - start;
      }
    }
    return absl::OkStatus();
  }

  absl::Status ValidateSplits(
      const std::vector<ConstFlatType>& params_nested_splits,
      SPLITS_TYPE num_params_dense_values) {
    // Validate
    for (int dim = 0; dim < params_nested_splits.size(); ++dim) {
      const auto& splits = params_nested_splits[dim];
      SPLITS_TYPE last_split = (dim == params_nested_splits.size() - 1)
                                   ? num_params_dense_values
                                   : params_nested_splits[dim + 1].size();
      if (splits.size() == 0) {
        return errors::InvalidArgument("Ragged splits may not be empty");
      }
      if (splits(0) < 0) {
        return errors::InvalidArgument("Ragged splits must be non-negative");
      }
      if (splits(splits.size() - 1) > last_split) {
        return errors::InvalidArgument(
            "Ragged splits must not point past values");
      }
      for (int i = 1; i < splits.size(); ++i) {
        if (splits(i - 1) > splits(i)) {
          return errors::InvalidArgument("Ragged splits must be sorted");
        }
      }
    }
    return absl::OkStatus();
  }

  absl::Status WriteSplits(
      const std::vector<std::vector<SPLITS_TYPE>>& out_splits,
      OpKernelContext* context) {
    OpOutputList splits_out;
    TF_RETURN_IF_ERROR(
        context->output_list("output_nested_splits", &splits_out));
    for (int i = 0; i < out_splits.size(); ++i) {
      Tensor* splits;
      SPLITS_TYPE num_splits = out_splits[i].size();
      TF_RETURN_IF_ERROR(
          splits_out.allocate(i, TensorShape({num_splits}), &splits));
      auto splits_flat = splits->flat<SPLITS_TYPE>();
      std::copy_n(out_splits[i].data(), out_splits[i].size(),
                  splits_flat.data());
    }
    return absl::OkStatus();
  }

  absl::Status WriteValues(
      const Tensor& params_dense_values_in,
      const std::vector<std::pair<SPLITS_TYPE, SPLITS_TYPE>>& value_slices,
      int values_index, SPLITS_TYPE num_values,
      OpKernelContext* context) const {
    Tensor* values_out = nullptr;
    TensorShape values_shape = params_dense_values_in.shape();
    values_shape.set_dim(0, num_values);
    TF_RETURN_IF_ERROR(
        context->allocate_output(values_index, values_shape, &values_out));
    const SPLITS_TYPE num_elements = params_dense_values_in.NumElements();
    const SPLITS_TYPE value_size =
        num_elements == 0 ? 0
                          : (num_elements / params_dense_values_in.dim_size(0));
    CallWriteValueSlices(params_dense_values_in, value_slices, value_size,
                         values_out);
    return absl::OkStatus();
  }

 protected:
  // Call WriteValueSlices() using the appropriate VALUE_TYPE template
  // parameter.  This pattern is used to reduce binary size.  In particular,
  // this allows us to have two instantiations of this class (one for each
  // index type), rather than 14 (one for each index type and value type),
  // which cuts the binary size of this op from ~300k to <90k.
  virtual void CallWriteValueSlices(
      const Tensor& params_dense_values_in,
      const std::vector<std::pair<SPLITS_TYPE, SPLITS_TYPE>>& value_slices,
      SPLITS_TYPE value_size, Tensor* values_out) const = 0;
};

template <typename INDEX_TYPE, typename VALUE_TYPE, typename SPLITS_TYPE>
class RaggedGatherOp : public RaggedGatherOpBase<INDEX_TYPE, SPLITS_TYPE> {
 public:
  using RaggedGatherOpBase<INDEX_TYPE, SPLITS_TYPE>::RaggedGatherOpBase;

 private:
  void CallWriteValueSlices(
      const Tensor& params_dense_values_in,
      const std::vector<std::pair<SPLITS_TYPE, SPLITS_TYPE>>& value_slices,
      SPLITS_TYPE value_size, Tensor* values_out) const override {
    WriteValueSlices<VALUE_TYPE>(params_dense_values_in, value_slices,
                                 value_size, values_out);
  }
};

#define REGISTER_CPU_KERNEL_WITH_INDEX_TYPE(index_type, value_type, \
                                            splits_type)            \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("RaggedGather")                                          \
          .Device(DEVICE_CPU)                                       \
          .TypeConstraint<index_type>("Tindices")                   \
          .TypeConstraint<value_type>("Tvalues")                    \
          .TypeConstraint<splits_type>("Tsplits"),                  \
      RaggedGatherOp<index_type, value_type, splits_type>);
#define REGISTER_CPU_KERNEL(value_type)                           \
  REGISTER_CPU_KERNEL_WITH_INDEX_TYPE(int32, value_type, int32)   \
  REGISTER_CPU_KERNEL_WITH_INDEX_TYPE(int64_t, value_type, int32) \
  REGISTER_CPU_KERNEL_WITH_INDEX_TYPE(int32, value_type, int64_t) \
  REGISTER_CPU_KERNEL_WITH_INDEX_TYPE(int64_t, value_type, int64_t)
TF_CALL_POD_TYPES(REGISTER_CPU_KERNEL);
TF_CALL_tstring(REGISTER_CPU_KERNEL);
TF_CALL_QUANTIZED_TYPES(REGISTER_CPU_KERNEL);
TF_CALL_quint16(REGISTER_CPU_KERNEL);
TF_CALL_qint16(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL
#undef REGISTER_CPU_KERNEL_WITH_INDEX_TYPE

}  // namespace tensorflow
