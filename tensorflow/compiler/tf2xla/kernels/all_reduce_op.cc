/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

class CollectiveReduceV2Op : public XlaOpKernel {
 public:
  explicit CollectiveReduceV2Op(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("merge_op", &merge_op_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("final_op", &final_op_name_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("communication_hint", &communication_hint_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    int64 group_key, group_size;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar("group_key", &group_key));
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsIntScalar("group_size", &group_size));
    OP_REQUIRES(ctx,
                communication_hint_ == "nccl" || communication_hint_ == "auto",
                errors::InvalidArgument(
                    "Only compiling NCCL/auto collective is supported, got: ",
                    communication_hint_));

    // Store all traversed collective configurations.
    OP_REQUIRES_OK(ctx, ctx->xla_context()->RecordCollectiveReduceV2OpInfo(
                            group_key, group_size));

    DataType dtype = XlaHelpers::SumAccumulationType(ctx->input_type(0));
    OP_REQUIRES(ctx, merge_op_name_ == "Add" || merge_op_name_ == "Mul",
                errors::InvalidArgument("Only Add and Mul reduction supported "
                                        "for tf2xla all-reduce lowering, got: ",
                                        merge_op_name_));
    const xla::XlaComputation* reducer = [&] {
      if (merge_op_name_ == "Add") {
        return ctx->GetOrCreateAdd(dtype);
      }
      CHECK_EQ(merge_op_name_, "Mul");
      return ctx->GetOrCreateMul(dtype);
    }();

    OP_REQUIRES(
        ctx, final_op_name_ == "Id",
        errors::InvalidArgument("Only 'Id' is supported as a final operation "
                                "for all-reduce tf2xla lowering"));
    xla::ChannelHandle channel_handle;
    channel_handle.set_type(xla::ChannelHandle::DEVICE_TO_DEVICE);
    channel_handle.set_handle(group_key);
    ctx->SetOutput(0,
                   xla::AllReduce(ctx->Input(0), *reducer, {}, channel_handle));
  }

 private:
  DataType dtype_ = DT_INVALID;
  string merge_op_name_;
  string final_op_name_;
  string communication_hint_;

  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveReduceV2Op);
};

REGISTER_XLA_OP(Name("CollectiveReduceV2")
                    .CompileTimeConstantInput("group_key")
                    .CompileTimeConstantInput("group_size"),
                CollectiveReduceV2Op);

}  // namespace tensorflow
