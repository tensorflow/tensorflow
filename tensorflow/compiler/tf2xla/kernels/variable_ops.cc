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

class VarIsInitializedOp : public XlaOpKernel {
 public:
  explicit VarIsInitializedOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationDataHandle handle;
    bool initialized = ctx->ReadVariableInput(0, &handle).ok();
    ctx->SetOutput(0, ctx->builder()->ConstantR0<bool>(initialized));
  }
};
REGISTER_XLA_OP("VarIsInitializedOp", VarIsInitializedOp);

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
REGISTER_XLA_OP("_UnsafeReadVariable", ReadVariableOp);

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

class ResourceApplyMomentum : public XlaOpKernel {
 public:
  explicit ResourceApplyMomentum(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationBuilder* b = ctx->builder();

    DataType type = ctx->input_type(2);

    DataType var_type, accum_type;
    TensorShape var_shape, accum_shape;
    OP_REQUIRES_OK(ctx, ctx->GetVariableTypeAndShape(0, &var_type, &var_shape));
    OP_REQUIRES_OK(ctx,
                   ctx->GetVariableTypeAndShape(1, &accum_type, &accum_shape));

    OP_REQUIRES(
        ctx, type == var_type && type == accum_type,
        errors::InvalidArgument(
            "Types of variable arguments to ResourceApplyMomentum must match: ",
            DataTypeString(type), " vs. ", DataTypeString(var_type), " and ",
            DataTypeString(accum_type)));

    OP_REQUIRES(ctx, var_shape.IsSameSize(accum_shape),
                errors::InvalidArgument(
                    "var and accum do not have the same shape",
                    var_shape.DebugString(), " ", accum_shape.DebugString()));

    TensorShape lr_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr_shape),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr_shape.DebugString()));

    TensorShape grad_shape = ctx->InputShape(3);
    OP_REQUIRES(ctx, var_shape.IsSameSize(grad_shape),
                errors::InvalidArgument(
                    "var and grad do not have the same shape",
                    var_shape.DebugString(), " ", grad_shape.DebugString()));

    TensorShape momentum_shape = ctx->InputShape(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum_shape),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum_shape.DebugString()));

    xla::ComputationDataHandle var, accum;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, &var));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(1, &accum));

    xla::ComputationDataHandle lr = ctx->Input(2);
    xla::ComputationDataHandle grad = ctx->Input(3);
    xla::ComputationDataHandle momentum = ctx->Input(4);

    accum = b->Add(b->Mul(accum, momentum), grad);
    if (use_nesterov_) {
      // See https://github.com/tensorflow/tensorflow/pull/2798 for an
      // explanation of the reparameterization used here.
      var = b->Sub(
          var, b->Add(b->Mul(grad, lr), b->Mul(b->Mul(accum, momentum), lr)));
    } else {
      var = b->Sub(var, b->Mul(accum, lr));
    }
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, type, var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, type, accum));
  }

 private:
  bool use_nesterov_;
};
REGISTER_XLA_OP("ResourceApplyMomentum", ResourceApplyMomentum);

}  // namespace
}  // namespace tensorflow
