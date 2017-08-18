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

#include <sstream>
#include "tensorflow/compiler/tf2xla/kernels/cwise_ops.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace {

class UnsortedSegmentSum : public XlaOpKernel {
 public:
  explicit UnsortedSegmentSum(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // output = unsorted_segment_sum(data, indices, num_segments)
    // Compute a tensor such that:
    //    output[i] = sum over {j where indices[j] == i} of data[j]
    //    output[i] == 0 if i does not appear in indices
    //
    // Contrast with segment_sum(), which assumes indices are sorted and that
    // max(indices)+1 is the desired size of the output.
    //
    // The returned output tensor has the same type as data, and the same shape
    // as data with the first indices.rank dimensions are replaced
    // by a single dimension with size num_segments.

    xla::ComputationBuilder* builder = ctx->builder();

    auto data = ctx->Input(0);
    auto data_shape = ctx->InputShape(0);

    auto indices = ctx->Input(1);
    auto indices_shape = ctx->InputShape(1);

    OP_REQUIRES(ctx, data_shape.dims() >= indices_shape.dims(),
                errors::InvalidArgument(
                    "UnsortedSegmentSum requires that indices' rank be"
                    " less than or equal to data's rank."));
    // Validate that indices.shape is a prefix of data.shape.
    for (int d = 0; d < indices_shape.dims(); ++d) {
      OP_REQUIRES(ctx, (data_shape.dim_size(d) == indices_shape.dim_size(d)),
                  errors::InvalidArgument(
                      "UnsortedSegmentSum requires indices shape to be prefix"
                      " of data_shape, but dimension ",
                      d, " differs ", data_shape.dim_size(d), " vs. ",
                      indices_shape.dim_size(d)));
    }

    int64 num_segments;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(2, &num_segments));

    // Flatten the indices into 1-D.
    auto indices_1d = builder->Reshape(indices, {indices_shape.num_elements()});

    // flatten data for dynamic indexing.
    int64 out_tensor_dims = data_shape.dims() - indices_shape.dims();
    std::vector<int64> flat_shape(1 + out_tensor_dims);
    flat_shape[0] = indices_shape.num_elements();
    for (int64 k = 0; k < out_tensor_dims; ++k) {
      flat_shape[1 + k] = data_shape.dim_size(indices_shape.dims() + k);
    }
    auto data_flat = builder->Reshape(data, flat_shape);

    // output shape; same as data_shape, but dimension 0 is num_segments.
    std::vector<int64> out_shape(flat_shape);
    out_shape[0] = num_segments;

    // Pad the output array dims to rank >= 3 to work around lowering issues.
    // TODO(b/37575001) This is awkward, and could be improved.
    int64 extra_dims = 0;
    if (out_shape.size() < 3) {
      extra_dims = 3u - out_shape.size();
    }
    std::vector<int64> rshape(extra_dims + out_shape.size(), 1);
    for (unsigned k = 0; k < out_shape.size(); ++k) {
      rshape[extra_dims + k] = out_shape[k];
    }
    auto output = builder->Broadcast(XlaHelpers::Zero(builder, dtype_), rshape);

    auto zero = builder->ConstantR1<int32>({0});

    for (int64 i = 0; i < indices_shape.num_elements(); ++i) {
      // output[indices[i]] += data[i]

      std::vector<int64> data_start_indices(flat_shape.size());
      data_start_indices[0] = i;
      for (unsigned d = 1; d < flat_shape.size(); ++d) {
        data_start_indices[d] = 0;
      }
      std::vector<int64> data_limit_indices(flat_shape);
      data_limit_indices[0] = i + 1;
      std::vector<int64> stride(flat_shape.size(), 1);

      auto data_slice = builder->Slice(data_flat, data_start_indices,
                                       data_limit_indices, stride);

      // Reshape the sliced data into the R3+ shape to match output array.
      std::vector<int64> rdata_shape(extra_dims + flat_shape.size());
      for (int64 k = 0; k <= extra_dims; ++k) {
        rdata_shape[k] = 1;
      }
      for (unsigned k = 1; k < data_limit_indices.size(); ++k) {
        rdata_shape[extra_dims + k] = data_limit_indices[k];
      }
      auto rdata_slice = builder->Reshape(data_slice, rdata_shape);

      auto index = builder->Slice(indices_1d, {i}, {i + 1}, {1});

      // Construct the index into the R3+ output array 0, ..., <index>, 0, ...
      std::vector<xla::ComputationDataHandle> out_start_index_parts(
          extra_dims + flat_shape.size(), zero);
      out_start_index_parts[extra_dims] = builder->Reshape(index, {1});
      auto out_start_indices = builder->ConcatInDim(out_start_index_parts, 0);

      std::vector<int64> slice_size(rshape);
      slice_size[extra_dims] = 1;

      auto out_slice =
          builder->DynamicSlice(output, out_start_indices, slice_size);
      auto sumval = builder->Add(out_slice, rdata_slice);
      output = builder->DynamicUpdateSlice(output, sumval, out_start_indices);
    }
    auto reshaped_output = builder->Reshape(output, out_shape);
    ctx->SetOutput(0, reshaped_output);
  }

 private:
  DataType dtype_;
};

REGISTER_XLA_OP(Name("UnsortedSegmentSum"), UnsortedSegmentSum);

}  // namespace
}  // namespace tensorflow
