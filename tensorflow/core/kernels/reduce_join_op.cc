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

// See docs in ../ops/string_ops.cc.

#include <string>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

namespace {

const gtl::InlinedVector<int64, 8> GetStrides(const TensorShape& shape) {
  gtl::InlinedVector<int64, 8> result(shape.dims());
  int64 product = 1;
  for (int32 i = shape.dims() - 1; i >= 0; --i) {
    result[i] = product;
    product *= shape.dim_size(i);
  }
  return result;
}

// Given a linear index to a subset of dimensions, full shape,
// precomputed list of running products of the full shape, and list of
// dimensions in the subset, outputs the linear index to the full shape with
// nonspecified dimensions set to 0.  Dimensions must be ordered from outer-most
// to inner-most with respect to the subset linear index.
inline int64 LinearSubIndexToFullIndex(
    int64 output_index, const gtl::InlinedVector<int32, 8>& dim_list,
    const TensorShape& input_shape,
    const gtl::InlinedVector<int64, 8>& strides) {
  int64 result = 0;
  int64 quotient = output_index;
  for (int32 i = dim_list.size() - 1; i >= 0; --i) {
    int32 dim = dim_list[i];
    int64 dim_value = quotient % input_shape.dim_size(dim);
    quotient = quotient / input_shape.dim_size(dim);
    result += strides[dim] * dim_value;
  }
  return result;
}

// Computes the number of input elements reduced per output element.
int64 GetReductionIterSize(const gtl::InlinedVector<int32, 8>& reduced_indices,
                           const TensorShape& input_shape) {
  int64 result = 1;
  for (int32 reduce_dim : reduced_indices) {
    result *= input_shape.dim_size(reduce_dim);
  }
  return result;
}

// Computes a list of all true reduced indices, accounting for negative
// indices.
gtl::InlinedVector<int32, 8> GetReducedIndices(const Tensor& reduction_indices,
                                               int32 input_dims) {
  const auto reduction_indices_flat = reduction_indices.flat<int32>();
  const int32 reduction_dims = reduction_indices_flat.size();

  gtl::InlinedVector<int32, 8> reduced_indices(reduction_dims);
  for (int32 i = 0; i < reduction_dims; ++i) {
    reduced_indices[i] = reduction_indices_flat(reduction_dims - i - 1);
    reduced_indices[i] += reduced_indices[i] < 0 ? input_dims : 0;
  }

  return reduced_indices;
}

// Appends all unreduced dimensions to the given vector.
void MakeUnreducedIndices(gtl::InlinedVector<bool, 8> index_is_reduced,
                          int32 input_dims,
                          gtl::InlinedVector<int32, 8>* unreduced_indices) {
  for (int32 index = 0; index < input_dims; ++index) {
    if (!index_is_reduced[index]) unreduced_indices->push_back(index);
  }
}

TensorShape GetOutputShape(gtl::InlinedVector<bool, 8> index_is_reduced,
                           const TensorShape& input_shape, bool keep_dims) {
  TensorShape output_shape;
  for (size_t index = 0; index < index_is_reduced.size(); ++index) {
    if (index_is_reduced[index]) {
      if (keep_dims) output_shape.AddDim(1);
    } else {
      output_shape.AddDim(input_shape.dim_size(index));
    }
  }
  return output_shape;
}

}  // namespace

class ReduceJoinOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  explicit ReduceJoinOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("separator", &separator_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const auto input_flat = input.flat<tstring>();
    const TensorShape& input_shape = input.shape();
    const int32 input_dims = input_shape.dims();

    const Tensor& reduction_indices = context->input(1);
    const auto reduction_indices_flat = reduction_indices.flat<int32>();
    const int32 reduction_dims = reduction_indices_flat.size();

    gtl::InlinedVector<bool, 8> index_is_reduced(input_dims, false);
    for (int32 i = 0; i < reduction_dims; i++) {
      int32 reduce_index = reduction_indices_flat(i);
      const int32 true_reduce_index =
          reduce_index < 0 ? reduce_index + input_dims : reduce_index;
      OP_REQUIRES(
          context, reduce_index >= -input_dims && reduce_index < input_dims,
          errors::OutOfRange("Invalid reduction dimension ", reduce_index,
                             " for input with ", input_dims, " dimension(s)"));
      OP_REQUIRES(context, !index_is_reduced[true_reduce_index],
                  errors::InvalidArgument("Duplicate reduction dimension ",
                                          reduce_index));
      index_is_reduced[true_reduce_index] = true;
    }

    gtl::InlinedVector<int32, 8> reduced_indices =
        GetReducedIndices(reduction_indices, input_dims);
    gtl::InlinedVector<int32, 8> unreduced_indices;
    MakeUnreducedIndices(index_is_reduced, input_dims, &unreduced_indices);
    const auto strides = GetStrides(input_shape);

    Tensor* output_tensor = nullptr;
    TensorShape output_shape =
        GetOutputShape(index_is_reduced, input_shape, keep_dims_);
    OP_REQUIRES_OK(context, context->allocate_output("output", output_shape,
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int64 reduction_iter_size =
        GetReductionIterSize(reduced_indices, input_shape);
    gtl::InlinedVector<StringPiece, 8> curr_strings(reduction_iter_size);
    for (int64 output_index = 0; output_index < output_shape.num_elements();
         ++output_index) {
      int64 output_full_index = LinearSubIndexToFullIndex(
          output_index, unreduced_indices, input_shape, strides);
      for (int64 reduction_index = 0; reduction_index < reduction_iter_size;
           ++reduction_index) {
        int64 reduction_full_index = LinearSubIndexToFullIndex(
            reduction_index, reduced_indices, input_shape, strides);
        curr_strings[reduction_index] =
            input_flat(output_full_index + reduction_full_index);
      }
      output_flat(output_index) = absl::StrJoin(curr_strings, separator_);
    }
  }

 private:
  bool keep_dims_;
  string separator_;
};

REGISTER_KERNEL_BUILDER(Name("ReduceJoin").Device(DEVICE_CPU), ReduceJoinOp);

}  // namespace tensorflow
