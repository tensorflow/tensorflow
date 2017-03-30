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
REGISTER_XLA_OP(Name("ResourceApplyGradientDescent"),
                ResourceApplyGradientDescent);

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
REGISTER_XLA_OP(Name("ResourceApplyMomentum"), ResourceApplyMomentum);

class ResourceApplyAdagrad : public XlaOpKernel {
 public:
  explicit ResourceApplyAdagrad(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

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
            "Types of variable arguments to ResourceApplyAdagrad must match: ",
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

    xla::ComputationDataHandle var, accum;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, &var));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(1, &accum));
    xla::ComputationDataHandle lr = ctx->Input(2);
    xla::ComputationDataHandle grad = ctx->Input(3);

    accum = b->Add(accum, b->Pow(grad, XlaHelpers::FloatLiteral(b, type, 2.0)));
    var = b->Sub(
        var, b->Mul(b->Mul(grad, lr),
                    b->Pow(accum, XlaHelpers::FloatLiteral(b, type, -0.5))));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, type, var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, type, accum));
  }
};
REGISTER_XLA_OP(Name("ResourceApplyAdagrad"), ResourceApplyAdagrad);

class ResourceApplyRMSProp : public XlaOpKernel {
 public:
  explicit ResourceApplyRMSProp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationBuilder* b = ctx->builder();

    DataType type = ctx->input_type(3);

    DataType var_type, ms_type, mom_type;
    TensorShape var_shape, ms_shape, mom_shape;
    OP_REQUIRES_OK(ctx, ctx->GetVariableTypeAndShape(0, &var_type, &var_shape));
    OP_REQUIRES_OK(ctx, ctx->GetVariableTypeAndShape(1, &ms_type, &ms_shape));
    OP_REQUIRES_OK(ctx, ctx->GetVariableTypeAndShape(2, &mom_type, &mom_shape));

    OP_REQUIRES(
        ctx, type == var_type && type == ms_type && type == mom_type,
        errors::InvalidArgument(
            "Types of variable arguments to ResourceApplyRMSProp must match: ",
            DataTypeString(type), " vs. ", DataTypeString(var_type), " vs. ",
            DataTypeString(ms_type), " vs. ", DataTypeString(mom_type)));

    TensorShape lr_shape = ctx->InputShape(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr_shape),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr_shape.DebugString()));
    TensorShape rho_shape = ctx->InputShape(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho_shape),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho_shape.DebugString()));
    TensorShape momentum_shape = ctx->InputShape(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum_shape),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum_shape.DebugString()));
    TensorShape epsilon_shape = ctx->InputShape(6);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon_shape),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon_shape.DebugString()));
    TensorShape grad_shape = ctx->InputShape(7);

    // var should be the same shape as mom and ms.
    OP_REQUIRES(ctx, var_shape.IsSameSize(ms_shape),
                errors::InvalidArgument("var and ms do not have the same shape",
                                        var_shape.DebugString(), " ",
                                        ms_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(mom_shape),
                errors::InvalidArgument(
                    "var and mom do not have the same shape",
                    var_shape.DebugString(), " ", mom_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(grad_shape),
                errors::InvalidArgument(
                    "var and grad do not have the same shape",
                    var_shape.DebugString(), " ", grad_shape.DebugString()));

    xla::ComputationDataHandle var, ms, mom;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, &var));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(1, &ms));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(2, &mom));
    xla::ComputationDataHandle lr = ctx->Input(3);
    xla::ComputationDataHandle rho = ctx->Input(4);
    xla::ComputationDataHandle momentum = ctx->Input(5);
    xla::ComputationDataHandle epsilon = ctx->Input(6);
    xla::ComputationDataHandle grad = ctx->Input(7);

    // ms <- rho * ms_{t-1} + (1-rho) * grad * grad
    // mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
    // var <- var - mom
    //
    // We use an alternate formulation of the ms equation:
    //
    //    ms <- ms + (grad**2 - ms) * (1 - rho)
    //
    // Which expands to:
    //
    //    ms <- ms + grad**2 - rho * grad ** 2 - ms + ms * rho
    //
    // Which simplifies to:
    //
    //    ms <- grad**2 (1 - rho) + ms * rho
    //
    // Which is the equation listed above.
    xla::ComputationDataHandle new_ms = b->Add(
        ms,
        b->Mul(b->Sub(b->Pow(grad, XlaHelpers::FloatLiteral(b, type, 2.0)), ms),
               b->Sub(XlaHelpers::FloatLiteral(b, type, 1.0), rho)));
    xla::ComputationDataHandle new_mom =
        b->Add(b->Mul(mom, momentum),
               b->Div(b->Mul(grad, lr),
                      b->Pow(b->Add(new_ms, epsilon),
                             XlaHelpers::FloatLiteral(b, type, 0.5))));
    xla::ComputationDataHandle new_var = b->Sub(var, new_mom);

    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, type, new_var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, type, new_ms));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(2, type, new_mom));
  }
};
REGISTER_XLA_OP(Name("ResourceApplyRMSProp"), ResourceApplyRMSProp);

}  // namespace
}  // namespace tensorflow
