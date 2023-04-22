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

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace {

class SendOp : public XlaOpKernel {
 public:
  explicit SendOp(OpKernelConstruction* ctx);
  void Compile(XlaOpKernelContext* ctx) override;

 private:
  string tensor_name_;

  TF_DISALLOW_COPY_AND_ASSIGN(SendOp);
};

SendOp::SendOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name_));
}

void SendOp::Compile(XlaOpKernelContext* ctx) {
  XlaCompiler* compiler = ctx->compiler();
  xla::ChannelHandle channel;
  OP_REQUIRES_OK(ctx, compiler->GetChannelHandle(tensor_name_, &channel));
  xla::Send(ctx->Input(0), channel);
}

REGISTER_XLA_OP(Name("XlaSend"), SendOp);

class RecvOp : public XlaOpKernel {
 public:
  explicit RecvOp(OpKernelConstruction* ctx);
  void Compile(XlaOpKernelContext* ctx) override;

 private:
  string tensor_name_;
  xla::Shape shape_;

  TF_DISALLOW_COPY_AND_ASSIGN(RecvOp);
};

RecvOp::RecvOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name_));

  TensorShape tensor_shape;
  DataType dtype;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &tensor_shape));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype));
  OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, tensor_shape, &shape_));
}

void RecvOp::Compile(XlaOpKernelContext* ctx) {
  XlaCompiler* compiler = ctx->compiler();
  xla::ChannelHandle channel;
  OP_REQUIRES_OK(ctx, compiler->GetChannelHandle(tensor_name_, &channel));
  ctx->SetOutput(0, xla::Recv(ctx->builder(), shape_, channel));
}

REGISTER_XLA_OP(Name("XlaRecv"), RecvOp);

}  // namespace
}  // namespace tensorflow
