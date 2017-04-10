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

void BatchToSpace(XlaOpKernelContext* ctx,
                  const xla::ComputationDataHandle& input, DataType input_dtype,
                  const TensorShape& input_tensor_shape,
                  gtl::ArraySlice<int64> block_shape,
                  const xla::Literal& crops) {
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
      xla::ShapeUtil::Rank(crops.shape()) == 2 &&
          block_rank == xla::ShapeUtil::GetDimension(crops.shape(), 0) &&
          2 == xla::ShapeUtil::GetDimension(crops.shape(), 1),
      errors::InvalidArgument("crops should have shape [", block_rank,
                              ", 2] instead of ",
                              xla::ShapeUtil::HumanString(crops.shape())));

  xla::ComputationBuilder* b = ctx->builder();
  const int64 batch_size = input_shape[0];

  // Compute the product of the block_shape values.
  int64 block_num_elems = 1;
  for (int i = 0; i < block_rank; ++i) {
    block_num_elems *= block_shape[i];
  }
  OP_REQUIRES(ctx, block_num_elems > 0,
              errors::InvalidArgument(
                  "The product of the block dimensions must be positive"));

  // 1. Reshape `input` to `reshaped` of shape:
  //      [block_shape[0], ..., block_shape[M-1],
  //       batch / prod(block_shape),
  //       input_shape[1], ..., input_shape[N-1]]

  OP_REQUIRES(
      ctx, batch_size % block_num_elems == 0,
      errors::InvalidArgument("Input batch dimension (", batch_size,
                              ") is not divisible by product of block sizes (",
                              block_num_elems, ")"));
  std::vector<int64> reshaped_shape(input_rank + block_rank);
  std::copy(block_shape.begin(), block_shape.end(), reshaped_shape.begin());
  reshaped_shape[block_rank] = batch_size / block_num_elems;
  std::copy(input_shape.begin() + 1, input_shape.end(),
            reshaped_shape.begin() + block_rank + 1);
  xla::ComputationDataHandle reshaped = b->Reshape(input, reshaped_shape);

  // 2. Permute dimensions of `reshaped` to produce `permuted` of shape
  //      [batch / prod(block_shape),
  //
  //       input_shape[1], block_shape[0],
  //       ...,
  //       input_shape[M], block_shape[M-1],
  //
  //       input_shape[M+1], ..., input_shape[N-1]]
  std::vector<int64> permutation(reshaped_shape.size());
  permutation[0] = block_rank;
  for (int i = 0; i < block_rank; ++i) {
    permutation[1 + 2 * i] = block_rank + 1 + i;
    permutation[1 + 2 * i + 1] = i;
  }
  std::iota(permutation.begin() + 1 + block_rank * 2, permutation.end(),
            1 + block_rank * 2);
  xla::ComputationDataHandle permuted = b->Transpose(reshaped, permutation);

  // 3. Reshape `permuted` to produce `reshaped_permuted` of shape
  //      [batch / prod(block_shape),
  //
  //       input_shape[1] * block_shape[0],
  //       ...,
  //       input_shape[M] * block_shape[M-1],
  //
  //       input_shape[M+1],
  //       ...,
  //       input_shape[N-1]]
  std::vector<int64> reshaped_permuted_shape(input_rank);
  reshaped_permuted_shape[0] = batch_size / block_num_elems;
  for (int i = 0; i < block_rank; ++i) {
    reshaped_permuted_shape[1 + i] = block_shape[i] * input_shape[1 + i];
  }
  std::copy(remainder_shape.begin(), remainder_shape.end(),
            reshaped_permuted_shape.begin() + 1 + block_rank);

  xla::ComputationDataHandle reshaped_permuted =
      b->Reshape(permuted, reshaped_permuted_shape);

  // 4. Crop the start and end of dimensions `[1, ..., M]` of
  //    `reshaped_permuted` according to `crops` to produce the output of shape:
  //      [batch / prod(block_shape),
  //
  //       input_shape[1] * block_shape[0] - crops[0,0] - crops[0,1],
  //       ...,
  //       input_shape[M] * block_shape[M-1] - crops[M-1,0] - crops[M-1,1],
  //
  //       input_shape[M+1], ..., input_shape[N-1]]
  std::vector<int64> start_indices(input_rank, 0);
  std::vector<int64> end_indices = reshaped_permuted_shape;
  for (int i = 0; i < block_rank; ++i) {
    int64 crop_start = xla::LiteralUtil::Get<int64>(crops, {i, 0});
    int64 crop_end = xla::LiteralUtil::Get<int64>(crops, {i, 1});
    OP_REQUIRES(ctx, crop_start >= 0 && crop_end >= 0,
                errors::InvalidArgument("Crops must be non-negative"));
    start_indices[1 + i] = crop_start;
    end_indices[1 + i] -= crop_end;
    OP_REQUIRES(
        ctx, start_indices[1 + i] <= end_indices[1 + i],
        errors::InvalidArgument(
            "Cropped size must be non-negative: start: ", crop_start,
            " end: ", crop_end, " size ", reshaped_permuted_shape[1 + i]));
  }
  xla::ComputationDataHandle output =
      b->Slice(reshaped_permuted, start_indices, end_indices);
  ctx->SetOutput(0, output);
}

class BatchToSpaceNDOp : public XlaOpKernel {
 public:
  explicit BatchToSpaceNDOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<int64> block_shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(1, &block_shape));

    xla::Literal crops;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsInt64Literal(2, &crops));

    BatchToSpace(ctx, ctx->Input(0), input_type(0), ctx->InputShape(0),
                 block_shape, crops);
  }
};
REGISTER_XLA_OP(Name("BatchToSpaceND"), BatchToSpaceNDOp);

class BatchToSpaceOp : public XlaOpKernel {
 public:
  explicit BatchToSpaceOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("block_size", &block_size_));
    OP_REQUIRES(
        ctx, block_size_ > 1,
        errors::InvalidArgument("Block size should be > 1: ", block_size_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::Literal crops;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsInt64Literal(1, &crops));

    BatchToSpace(ctx, ctx->Input(0), input_type(0), ctx->InputShape(0),
                 {block_size_, block_size_}, crops);
  }

 private:
  int block_size_;
};
REGISTER_XLA_OP(Name("BatchToSpace"), BatchToSpaceOp);

}  // namespace
}  // namespace tensorflow
