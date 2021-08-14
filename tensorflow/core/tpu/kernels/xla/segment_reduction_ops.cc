/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/lib/scatter.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/tpu/tpu_defs.h"

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
    xla::XlaOp data = ctx->Input(0);
    TensorShape data_shape = ctx->InputShape(0);

    xla::XlaOp indices = ctx->Input(1);
    TensorShape indices_shape = ctx->InputShape(1);

    int64_t num_segments;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsIntScalar(
                       2, &num_segments, xla::ValueInferenceMode::kUpperBound));

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
    xla::XlaBuilder* builder = ctx->builder();
    // data shape = [indices_shape, segment_shape]
    // buffer shape = [num_segment, segment_shape]
    // We now create the buffer shape by reverse enginerring data shape into
    // indices shape and segment shape.
    TensorShape buffer_shape = data_shape;
    buffer_shape.RemoveDimRange(0, indices_shape.dims());
    buffer_shape.InsertDim(0, num_segments);

    auto buffer = xla::Broadcast(XlaHelpers::Zero(builder, dtype_),
                                 buffer_shape.dim_sizes());

    // Build dynamic dim sizes for buffer, as well as whether each dimension
    // size is dynamic or static. We build two parts: num_sgement part and
    // segment_shape part.
    std::vector<xla::XlaOp> buffer_dims;
    std::vector<bool> buffer_dims_are_dynamic;
    // Build the "num_segment" part.
    bool num_segments_is_dynamic;
    OP_REQUIRES_OK(
        ctx, ctx->ResolveInputDynamismIntoPred(2, &num_segments_is_dynamic));

    buffer_dims.insert(buffer_dims.begin(), ctx->Input(2));
    buffer_dims_are_dynamic.insert(buffer_dims_are_dynamic.begin(),
                                   num_segments_is_dynamic);
    // Build the segment shape part.
    for (int64_t i = indices_shape.dims(); i < data_shape.dims(); ++i) {
      buffer_dims.push_back(xla::GetDimensionSize(data, i));
      buffer_dims_are_dynamic.push_back(
          ctx->InputXlaShape(0)->is_dynamic_dimension(i));
    }

    for (int64_t i = 0; i < buffer_dims.size(); ++i) {
      if (buffer_dims_are_dynamic[i]) {
        // For each dynamic dimension, call set-dimension-size on it.
        buffer = xla::SetDimensionSize(buffer, buffer_dims[i], i);
      }
    }

    auto combiner = [](xla::XlaOp a, xla::XlaOp b, xla::XlaBuilder* builder) {
      return a + b;
    };

    auto result = XlaScatter(buffer, /*updates=*/data, indices,
                             /*indices_are_vectors=*/false, combiner, builder);
    OP_REQUIRES_OK(ctx, result.status());
    ctx->SetOutput(0, result.ValueOrDie());
  }

 private:
  DataType dtype_;
};

REGISTER_XLA_OP(Name("UnsortedSegmentSum")
                    .Device(DEVICE_TPU_XLA_JIT)
                    .CompileTimeConstantInput("num_segments"),
                UnsortedSegmentSum);

}  // namespace
}  // namespace tensorflow
