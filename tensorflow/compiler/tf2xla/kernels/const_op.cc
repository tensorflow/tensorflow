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

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/tensor.pb.h"

namespace tensorflow {
namespace {

class ConstOp : public XlaOpKernel {
 public:
  explicit ConstOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    const TensorProto* proto = nullptr;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("value", &proto));
    proto_ = *proto;
    OP_REQUIRES(
        ctx, ctx->output_type(0) == proto_.dtype(),
        errors::InvalidArgument("Type mismatch between value (",
                                DataTypeString(proto_.dtype()), ") and dtype (",
                                DataTypeString(ctx->output_type(0)), ")"));
    OP_REQUIRES_OK(ctx, TensorShape::IsValidShape(proto_.tensor_shape()));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape shape(proto_.tensor_shape());

    if (proto_.dtype() == DT_STRING) {
      LOG(WARNING) << "Not computing Const of type DT_STRING";
      ctx->SetInvalidOutput(0);
      return;
    }
    xla::ComputationBuilder* b = ctx->builder();

    // To avoid blowups for large constants filled with the same value,
    // recognize that case and emit a scalar broadcast instead.
    if (shape.num_elements() > 1) {
      switch (proto_.dtype()) {
        case DT_BOOL:
          if (proto_.bool_val_size() == 1) {
            ctx->SetOutput(0,
                           b->Broadcast(b->ConstantR0<bool>(proto_.bool_val(0)),
                                        shape.dim_sizes()));
            return;
          }
          break;
        case DT_FLOAT:
          if (proto_.float_val_size() == 1) {
            ctx->SetOutput(
                0, b->Broadcast(b->ConstantR0<float>(proto_.float_val(0)),
                                shape.dim_sizes()));
            return;
          }
          break;
        case DT_DOUBLE:
          if (proto_.double_val_size() == 1) {
            ctx->SetOutput(
                0, b->Broadcast(b->ConstantR0<double>(proto_.double_val(0)),
                                shape.dim_sizes()));
            return;
          }
          break;
        case DT_INT32:
          if (proto_.int_val_size() == 1) {
            ctx->SetOutput(0,
                           b->Broadcast(b->ConstantR0<int32>(proto_.int_val(0)),
                                        shape.dim_sizes()));
            return;
          }
          break;
        case DT_INT64:
          if (proto_.int64_val_size() == 1) {
            ctx->SetOutput(
                0, b->Broadcast(b->ConstantR0<int64>(proto_.int64_val(0)),
                                shape.dim_sizes()));
            return;
          }
          break;
        default:
          break;
      }
    }

    // General case
    Tensor tensor(proto_.dtype());
    OP_REQUIRES(ctx, tensor.FromProto(cpu_allocator(), proto_),
                errors::InvalidArgument("Cannot parse tensor from proto: ",
                                        proto_.DebugString()));
    ctx->SetConstantOutput(0, tensor);
  }

 private:
  TensorProto proto_;
  TF_DISALLOW_COPY_AND_ASSIGN(ConstOp);
};

// XLA_* devices also register a "real" Const operator so we suppress the
// dummy operator using CompilationOnly().
REGISTER_XLA_OP(Name("Const").CompilationOnly(), ConstOp);

}  // namespace
}  // namespace tensorflow
