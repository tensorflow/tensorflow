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

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

namespace tensorflow {
namespace {

void SpaceToBatch(XlaOpKernelContext* ctx,
                  const xla::ComputationDataHandle& input, DataType input_dtype,
                  const TensorShape& input_tensor_shape,
                  gtl::ArraySlice<int64> block_shape,
                  const xla::Literal& paddings) {
  const int input_rank = input_tensor_shape.dims();
  const gtl::InlinedVector<int64, 4> input_shape =
      input_tensor_shape.dim_sizes();
  const int block_rank = block_shape.size();

  OP_REQUIRES(
      ctx, input_rank >= 1 + block_rank,
      errors::InvalidArgument("input rank should be >= ", 1 + block_rank,
                              " instead of ", input_rank));
  gtl::ArraySlice<int64> remainder_shape(input_shape);
  remainder_shape.remove_prefix(1 + block_rank);

  OP_REQUIRES(
      ctx,
      xla::ShapeUtil::Rank(paddings.shape()) == 2 &&
          block_rank == xla::ShapeUtil::GetDimension(paddings.shape(), 0) &&
          2 == xla::ShapeUtil::GetDimension(paddings.shape(), 1),
      errors::InvalidArgument("paddings should have shape [", block_rank,
                              ", 2] instead of ",
                              xla::ShapeUtil::HumanString(paddings.shape())));

  xla::ComputationBuilder* b = ctx->builder();

  // 1. Zero-pad the start and end of dimensions `[1, ..., M]` of the
  //  input according to `paddings` to produce `padded` of shape `padded_shape`.
  xla::PaddingConfig padding_config;
  std::vector<int64> padded_shape(input_shape.begin(), input_shape.end());
  int64 block_num_elems = 1LL;
  padding_config.add_dimensions();  // Don't pad the batch dimension.
  for (int i = 0; i < block_rank; ++i) {
    auto* dim = padding_config.add_dimensions();
    int64 pad_start = paddings.Get<int64>({i, 0});
    int64 pad_end = paddings.Get<int64>({i, 1});
    OP_REQUIRES(ctx, pad_start >= 0 && pad_end >= 0,
                errors::InvalidArgument("Paddings must be non-negative"));
    dim->set_edge_padding_low(pad_start);
    dim->set_edge_padding_high(pad_end);
    padded_shape[1 + i] += pad_start + pad_end;
    block_num_elems *= block_shape[i];
  }
  // Don't pad the remainder dimensions.
  for (int i = 0; i < remainder_shape.size(); ++i) {
    padding_config.add_dimensions();
  }
  OP_REQUIRES(ctx, block_num_elems > 0,
              errors::InvalidArgument(
                  "The product of the block dimensions must be positive"));

  xla::ComputationDataHandle padded =
      b->Pad(input, XlaHelpers::Zero(b, input_dtype), padding_config);

  // 2. Reshape `padded` to `reshaped_padded` of shape:
  //
  //      [batch] +
  //      [padded_shape[1] / block_shape[0],
  //        block_shape[0],
  //       ...,
  //       padded_shape[M] / block_shape[M-1],
  //       block_shape[M-1]] +
  //      remaining_shape
  const int64 batch_size = input_shape[0];
  std::vector<int64> reshaped_padded_shape(input_rank + block_rank);
  reshaped_padded_shape[0] = batch_size;
  for (int i = 0; i < block_rank; ++i) {
    OP_REQUIRES(ctx, padded_shape[1 + i] % block_shape[i] == 0,
                errors::InvalidArgument("padded_shape[", 1 + i,
                                        "]=", padded_shape[1 + i],
                                        " is not divisible by block_shape[", i,
                                        "]=", block_shape[i]));

    reshaped_padded_shape[1 + i * 2] = padded_shape[1 + i] / block_shape[i];
    reshaped_padded_shape[1 + i * 2 + 1] = block_shape[i];
  }
  std::copy(remainder_shape.begin(), remainder_shape.end(),
            reshaped_padded_shape.begin() + 1 + 2 * block_rank);

  xla::ComputationDataHandle reshaped_padded =
      b->Reshape(padded, reshaped_padded_shape);

  // 3. Permute dimensions of `reshaped_padded` to produce
  //    `permuted_reshaped_padded` of shape:
  //
  //      block_shape +
  //      [batch] +
  //      [padded_shape[1] / block_shape[0],
  //       ...,
  //       padded_shape[M] / block_shape[M-1]] +
  //      remaining_shape
  std::vector<int64> permutation(reshaped_padded_shape.size());
  for (int i = 0; i < block_rank; ++i) {
    permutation[i] = 1 + 2 * i + 1;
    permutation[block_rank + 1 + i] = 1 + 2 * i;
  }
  permutation[block_rank] = 0;
  std::iota(permutation.begin() + 1 + block_rank * 2, permutation.end(),
            1 + block_rank * 2);
  xla::ComputationDataHandle permuted_reshaped_padded =
      b->Transpose(reshaped_padded, permutation);

  // 4. Reshape `permuted_reshaped_padded` to flatten `block_shape` into the
  //    batch dimension, producing an output tensor of shape:
  //
  //      [batch * prod(block_shape)] +
  //      [padded_shape[1] / block_shape[0],
  //       ...,
  //       padded_shape[M] / block_shape[M-1]] +
  //      remaining_shape
  // Determine the length of the prefix of block dims that can be combined
  // into the batch dimension due to having no padding and block_shape=1.
  std::vector<int64> output_shape(input_rank);
  output_shape[0] = batch_size * block_num_elems;
  for (int i = 0; i < block_rank; ++i) {
    output_shape[1 + i] = padded_shape[1 + i] / block_shape[i];
  }
  std::copy(remainder_shape.begin(), remainder_shape.end(),
            output_shape.begin() + 1 + block_rank);

  xla::ComputationDataHandle output =
      b->Reshape(permuted_reshaped_padded, output_shape);
  ctx->SetOutput(0, output);
}

class SpaceToBatchNDOp : public XlaOpKernel {
 public:
  explicit SpaceToBatchNDOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<int64> block_shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(1, &block_shape));

    xla::Literal paddings;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsInt64Literal(2, &paddings));

    SpaceToBatch(ctx, ctx->Input(0), input_type(0), ctx->InputShape(0),
                 block_shape, paddings);
  }
};
REGISTER_XLA_OP(Name("SpaceToBatchND")
                    .CompileTimeConstInput("paddings")
                    .CompileTimeConstInput("block_shape"),
                SpaceToBatchNDOp);

class SpaceToBatchOp : public XlaOpKernel {
 public:
  explicit SpaceToBatchOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("block_size", &block_size_));
    OP_REQUIRES(
        ctx, block_size_ > 1,
        errors::InvalidArgument("Block size should be > 1: ", block_size_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::Literal paddings;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsInt64Literal(1, &paddings));

    SpaceToBatch(ctx, ctx->Input(0), input_type(0), ctx->InputShape(0),
                 {block_size_, block_size_}, paddings);
  }

 private:
  int block_size_;
};
REGISTER_XLA_OP(Name("SpaceToBatch").CompileTimeConstInput("paddings"),
                SpaceToBatchOp);

}  // namespace
}  // namespace tensorflow
