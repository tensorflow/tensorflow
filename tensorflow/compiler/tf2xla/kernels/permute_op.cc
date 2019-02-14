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

#include <string>
#include <vector>

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

class DataFormatVecPermuteOp : public XlaOpKernel {
 public:
  explicit DataFormatVecPermuteOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("src_format", &src_format_));
    OP_REQUIRES(
        ctx, src_format_.size() == 4,
        errors::InvalidArgument("Data format should have 4 characters"));
    TensorFormat data_format;
    OP_REQUIRES(ctx, FormatFromString(src_format_, &data_format),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dst_format", &dst_format_));
    OP_REQUIRES(
        ctx, dst_format_.size() == 4,
        errors::InvalidArgument("Data format should have 4 characters"));
    OP_REQUIRES(ctx, FormatFromString(dst_format_, &data_format),
                errors::InvalidArgument("Invalid data format"));
  }
  void Compile(XlaOpKernelContext* ctx) override {
    auto builder = ctx->builder();
    const TensorShape input_tensor_shape = ctx->InputShape(0);
    int input_rank = input_tensor_shape.dims();
    OP_REQUIRES(ctx, input_rank == 1 || input_rank == 2,
                errors::InvalidArgument(
                    "Input must be a vector or matrix, but got shape ",
                    input_tensor_shape.DebugString()));
    OP_REQUIRES(
        ctx, input_tensor_shape.dim_size(0) == 4,
        errors::InvalidArgument(
            "First dimension of input must be of size 4, but got shape ",
            input_tensor_shape.DebugString()));
    if (input_rank == 2) {
      OP_REQUIRES(
          ctx, input_tensor_shape.dim_size(1) == 2,
          errors::InvalidArgument(
              "Second dimension of 2D input must be of size 2, but got shape ",
              input_tensor_shape.DebugString()));
    }
    std::vector<int32> dst_indices(4, 0);
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        if (src_format_[i] == dst_format_[j]) {
          dst_indices[i] = j;
          break;
        }
      }
    }
    auto keys = xla::ConstantR1(builder, absl::Span<const int32>(dst_indices));
    if (input_rank == 2) {
      keys = xla::BroadcastInDim(keys, {4, 2}, {0});
    }
    auto sorted = xla::Sort(keys, {ctx->Input(0)}, 0);
    auto output = xla::GetTupleElement(sorted, 1);
    ctx->SetOutput(0, output);
  }

 private:
  string src_format_;
  string dst_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(DataFormatVecPermuteOp);
};

REGISTER_XLA_OP(
    Name("DataFormatVecPermute").TypeConstraint("T", {DT_INT32, DT_INT64}),
    DataFormatVecPermuteOp);

}  // namespace
}  // namespace tensorflow
