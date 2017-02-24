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

#include "tensorflow/compiler/tf2xla/kernels/cwise_ops.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/no_op.h"

namespace tensorflow {
namespace {

class ReadVariableOp : public XlaOpKernel {
 public:
  explicit ReadVariableOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationDataHandle handle;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, &handle));
    ctx->SetOutput(0, handle);
  }
};
REGISTER_XLA_OP("ReadVariableOp", ReadVariableOp);

class AssignVariableOp : public XlaOpKernel {
 public:
  explicit AssignVariableOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx,
                   ctx->AssignVariable(0, ctx->input_type(1), ctx->Input(1)));
  }
};
REGISTER_XLA_OP("AssignVariableOp", AssignVariableOp);

class AssignAddVariableOp : public XlaOpKernel {
 public:
  explicit AssignAddVariableOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationDataHandle handle;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, &handle));
    handle = ctx->builder()->Add(handle, ctx->Input(1));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, ctx->input_type(1), handle));
  }
};
REGISTER_XLA_OP("AssignAddVariableOp", AssignAddVariableOp);

class AssignSubVariableOp : public XlaOpKernel {
 public:
  explicit AssignSubVariableOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationDataHandle handle;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, &handle));
    handle = ctx->builder()->Sub(handle, ctx->Input(1));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, ctx->input_type(1), handle));
  }
};
REGISTER_XLA_OP("AssignSubVariableOp", AssignSubVariableOp);

class ResourceApplyGradientDescent : public XlaOpKernel {
 public:
  explicit ResourceApplyGradientDescent(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationDataHandle handle;
    xla::ComputationBuilder* b = ctx->builder();
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, &handle));
    handle = b->Sub(handle, b->Mul(ctx->Input(1), ctx->Input(2)));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, ctx->input_type(1), handle));
  }
};
REGISTER_XLA_OP("ResourceApplyGradientDescent", ResourceApplyGradientDescent);

}  // namespace
}  // namespace tensorflow
