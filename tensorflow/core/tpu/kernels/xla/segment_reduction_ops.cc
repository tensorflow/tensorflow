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
// TODO(b/32945756): Add a scatter op in XLA and move this to a HLO optimization
// pass. Optimization for UnsortedSegmentSum on TPU: use k-hot matmul. This
// optimization requires:
//     1. data has dtype supported by TPU matmul and has rank of 1 or 2.
//     2. indices has rank of 1.
//     3. matmul op count is less than 800 billion.
//
// Example of calculating UnsortedSegmentSum by k-hot matmul:
//     data shape        [A, B]
//     indices shape     [A]
//     num_segment        N
//     output shape      [N, B]
//     matmul op count    N * A * B
// Step 1: create k-hot matrix
//     k-hot matrix has shape of [A, N], where row i is responsible for
//     collecting the sum of the i-th segment, concretely
//            k-hot[i][j] = 1 if indices[i] = j
// Step 2: perform matmul
//     the final result is obtained by multiplying k-hot matrix with data
//     matrix, namely
//             k-hot  *  data   => result
// shape:      [N, A] *  [A, B] => [N, B]
xla::XlaOp KHotMatmul(XlaOpKernelContext* ctx, xla::XlaBuilder* builder,
                      const xla::XlaOp data, const xla::XlaOp indices,
                      int64 num_segments) {
  DataType data_dtype = ctx->input_type(0);
  xla::PrimitiveType indices_type = ctx->input_xla_type(1);
  TensorShape data_shape = ctx->InputShape(0);
  TensorShape indices_shape = ctx->InputShape(1);
  xla::XlaOp linspace = xla::Iota(builder, indices_type, num_segments);
  xla::XlaOp linspace_col = xla::Reshape(linspace, {num_segments, 1});
  TensorShape indices_row_shape = indices_shape;
  indices_row_shape.InsertDim(0, 1);
  xla::XlaOp indices_row = xla::Reshape(indices, indices_row_shape.dim_sizes());
  xla::XlaOp k_hot = xla::Eq(indices_row, linspace_col);
  xla::XlaOp k_hot_with_data_dtype =
      XlaHelpers::ConvertElementType(k_hot, data_dtype);
  // F32 version of the KHotMatmul. It splits the F32 data into three
  // BF16 partial data and run KHotMatmul for each of them. The final result
  // is the summation of three BF16 results.
  // Note that this still doesn't fully retain f32 precision.
  // In particular, values smaller than 2^-111 may see loss of precision.
  xla::PrecisionConfig precision_config;
  if (data_dtype == DT_FLOAT) {
    precision_config.add_operand_precision(xla::PrecisionConfig::HIGHEST);
  } else {
    CHECK_EQ(data_dtype, DT_BFLOAT16);
    precision_config.add_operand_precision(xla::PrecisionConfig::DEFAULT);
  }
  precision_config.add_operand_precision(xla::PrecisionConfig::DEFAULT);
  return xla::Dot(k_hot_with_data_dtype, data, &precision_config);
}

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

    int64 num_segments;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(2, &num_segments));

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
    for (int64 i = indices_shape.dims(); i < data_shape.dims(); ++i) {
      buffer_dims.push_back(xla::GetDimensionSize(data, i));
      buffer_dims_are_dynamic.push_back(
          ctx->InputXlaShape(0)->is_dynamic_dimension(i));
    }

    for (int64 i = 0; i < buffer_dims.size(); ++i) {
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
