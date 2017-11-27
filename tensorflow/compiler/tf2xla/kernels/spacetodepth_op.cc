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

class SpaceToDepthOp : public XlaOpKernel {
 public:
  explicit SpaceToDepthOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("block_size", &block_size_));
    OP_REQUIRES(
        ctx, block_size_ > 1,
        errors::InvalidArgument("Block size should be > 1: ", block_size_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_tensor_shape = ctx->InputShape(0);
    // The input is presumed to be [batch, height, width, depth]
    int input_rank = input_tensor_shape.dims();
    static const int kRequiredDims = 4;
    OP_REQUIRES(ctx, kRequiredDims == input_rank,
                errors::InvalidArgument("Input rank should be: ", kRequiredDims,
                                        " instead of: ", input_rank));
    const gtl::InlinedVector<int64, 4> input_shape =
        input_tensor_shape.dim_sizes();

    xla::ComputationBuilder* b = ctx->builder();
    xla::ComputationDataHandle input = ctx->Input(0);

    // 1. Reshape `input` to `reshaped` of shape:
    //
    //      [batch,
    //       input_shape[1] / block_size_, block_size_,
    //       input_shape[2] / block_size_, block_size_,
    //       depth]
    const int block_rank = 2;
    for (int i = 0; i < block_rank; ++i) {
      OP_REQUIRES(ctx, input_shape[1 + i] % block_size_ == 0,
                  errors::InvalidArgument(
                      "input shape[", 1 + i, "]=", input_shape[1 + i],
                      " is not divisible by block_size=", block_size_));
    }
    xla::ComputationDataHandle reshaped = b->Reshape(
        input, {input_shape[0], input_shape[1] / block_size_, block_size_,
                input_shape[2] / block_size_, block_size_, input_shape[3]});

    // 2. Permute dimensions of `reshaped` to produce
    //    `permuted_reshaped` of shape:
    //
    //      [batch,
    //       input_shape[1] / block_size_,
    //       input_shape[2] / block_size_,
    //       block_size_, block_size_,
    //       depth]
    xla::ComputationDataHandle permuted_reshaped =
        b->Transpose(reshaped, {0, 1, 3, 2, 4, 5});

    // 3. Reshape `permuted_reshaped` to flatten `block_shape` into the
    //    batch dimension, producing an output tensor of shape:
    //
    //      [batch,
    //       input_shape[1] / block_size_,
    //       input_shape[2] / block_size_,
    //       block_size_ * block_size_ * depth]
    //
    xla::ComputationDataHandle output = b->Reshape(
        permuted_reshaped, {input_shape[0], input_shape[1] / block_size_,
                            input_shape[2] / block_size_,
                            block_size_ * block_size_ * input_shape[3]});

    ctx->SetOutput(0, output);
  }

 private:
  int block_size_;
};
REGISTER_XLA_OP(Name("SpaceToDepth"), SpaceToDepthOp);

}  // namespace
}  // namespace tensorflow
