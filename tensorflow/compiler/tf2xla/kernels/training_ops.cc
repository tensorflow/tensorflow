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
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
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
    xla::XlaOp handle;
    DataType type = ctx->input_type(1);
    TensorShape var_shape;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, type, &var_shape, &handle));

    TensorShape alpha_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(alpha_shape),
                errors::InvalidArgument("alpha is not a scalar: ",
                                        alpha_shape.DebugString()));

    TensorShape delta_shape = ctx->InputShape(2);
    OP_REQUIRES(
        ctx, var_shape.IsSameSize(delta_shape),
        errors::InvalidArgument("var and delta do not have the same shape: ",
                                var_shape.DebugString(), " vs ",
                                delta_shape.DebugString()));

    handle = xla::Sub(handle, xla::Mul(ctx->Input(1), ctx->Input(2)));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, type, handle));
  }
};
REGISTER_XLA_OP(
    Name("ResourceApplyGradientDescent").TypeConstraint("T", kFloatTypes),
    ResourceApplyGradientDescent);

class ResourceApplyMomentum : public XlaOpKernel {
 public:
  explicit ResourceApplyMomentum(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    DataType type = ctx->input_type(2);

    TensorShape var_shape, accum_shape;
    xla::XlaOp var, accum;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, type, &var_shape, &var));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(1, type, &accum_shape, &accum));

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

    xla::XlaOp lr = ctx->Input(2);
    xla::XlaOp grad = ctx->Input(3);
    xla::XlaOp momentum = ctx->Input(4);

    accum = xla::Add(xla::Mul(accum, momentum), grad);
    if (use_nesterov_) {
      // See https://github.com/tensorflow/tensorflow/pull/2798 for an
      // explanation of the reparameterization used here.
      var = xla::Sub(var, xla::Add(xla::Mul(grad, lr),
                                   xla::Mul(xla::Mul(accum, momentum), lr)));
    } else {
      var = xla::Sub(var, xla::Mul(accum, lr));
    }
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, type, var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, type, accum));
  }

 private:
  bool use_nesterov_;
};
REGISTER_XLA_OP(Name("ResourceApplyMomentum").TypeConstraint("T", kFloatTypes),
                ResourceApplyMomentum);

class ResourceApplyAdagrad : public XlaOpKernel {
 public:
  explicit ResourceApplyAdagrad(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();

    DataType type = ctx->input_type(2);

    TensorShape var_shape, accum_shape;
    xla::XlaOp var, accum;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, type, &var_shape, &var));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(1, type, &accum_shape, &accum));

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

    xla::XlaOp lr = ctx->Input(2);
    xla::XlaOp grad = ctx->Input(3);

    accum =
        xla::Add(accum, xla::Pow(grad, XlaHelpers::FloatLiteral(b, type, 2.0)));
    var = xla::Sub(
        var,
        xla::Mul(xla::Mul(grad, lr),
                 xla::Pow(accum, XlaHelpers::FloatLiteral(b, type, -0.5))));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, type, var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, type, accum));
  }
};
REGISTER_XLA_OP(Name("ResourceApplyAdagrad").TypeConstraint("T", kFloatTypes),
                ResourceApplyAdagrad);

class ResourceApplyAdam : public XlaOpKernel {
 public:
  explicit ResourceApplyAdam(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape var_shape, m_shape, v_shape;
    xla::XlaOp var, m, v;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, dtype_, &var_shape, &var));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(1, dtype_, &m_shape, &m));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(2, dtype_, &v_shape, &v));

    TensorShape beta1_power_shape = ctx->InputShape(3);
    TensorShape beta2_power_shape = ctx->InputShape(4);
    TensorShape lr_shape = ctx->InputShape(5);
    TensorShape beta1_shape = ctx->InputShape(6);
    TensorShape beta2_shape = ctx->InputShape(7);
    TensorShape epsilon_shape = ctx->InputShape(8);
    TensorShape grad_shape = ctx->InputShape(9);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power_shape),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power_shape),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr_shape),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_shape),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_shape),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon_shape),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon_shape.DebugString()));

    OP_REQUIRES(ctx, var_shape.IsSameSize(m_shape),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var_shape.DebugString(), " ",
                                        m_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(v_shape),
                errors::InvalidArgument("var and v do not have the same shape",
                                        var_shape.DebugString(), " ",
                                        v_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(grad_shape),
                errors::InvalidArgument(
                    "var and grad do not have the same shape",
                    var_shape.DebugString(), " ", grad_shape.DebugString()));

    xla::XlaOp beta1_power = ctx->Input(3);
    xla::XlaOp beta2_power = ctx->Input(4);
    xla::XlaOp lr = ctx->Input(5);
    xla::XlaOp beta1 = ctx->Input(6);
    xla::XlaOp beta2 = ctx->Input(7);
    xla::XlaOp epsilon = ctx->Input(8);
    xla::XlaOp grad = ctx->Input(9);

    // alpha <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
    // m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
    // v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
    // variable <- variable - alpha * m_t / (sqrt(v_t) + epsilon)

    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp half = XlaHelpers::FloatLiteral(b, dtype_, 0.5);
    xla::XlaOp one = XlaHelpers::FloatLiteral(b, dtype_, 1.0);
    xla::XlaOp two = XlaHelpers::FloatLiteral(b, dtype_, 2.0);

    xla::XlaOp alpha =
        xla::Div(xla::Mul(lr, xla::Pow(xla::Sub(one, beta2_power), half)),
                 xla::Sub(one, beta1_power));
    m = xla::Add(m, xla::Mul(xla::Sub(grad, m), xla::Sub(one, beta1)));
    v = xla::Add(
        v, xla::Mul(xla::Sub(xla::Pow(grad, two), v), xla::Sub(one, beta2)));
    var = xla::Sub(var, xla::Div(xla::Mul(m, alpha),
                                 xla::Add(xla::Pow(v, half), epsilon)));

    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, dtype_, var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, dtype_, m));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(2, dtype_, v));
  }

 private:
  DataType dtype_;
};
REGISTER_XLA_OP(Name("ResourceApplyAdam").TypeConstraint("T", kFloatTypes),
                ResourceApplyAdam);

class ResourceApplyRMSProp : public XlaOpKernel {
 public:
  explicit ResourceApplyRMSProp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();

    DataType type = ctx->input_type(3);

    TensorShape var_shape, ms_shape, mom_shape;
    xla::XlaOp var, ms, mom;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, type, &var_shape, &var));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(1, type, &ms_shape, &ms));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(2, type, &mom_shape, &mom));

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

    xla::XlaOp lr = ctx->Input(3);
    xla::XlaOp rho = ctx->Input(4);
    xla::XlaOp momentum = ctx->Input(5);
    xla::XlaOp epsilon = ctx->Input(6);
    xla::XlaOp grad = ctx->Input(7);

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
    xla::XlaOp new_ms = xla::Add(
        ms, xla::Mul(
                xla::Sub(xla::Pow(grad, XlaHelpers::FloatLiteral(b, type, 2.0)),
                         ms),
                xla::Sub(XlaHelpers::FloatLiteral(b, type, 1.0), rho)));
    xla::XlaOp new_mom =
        xla::Add(xla::Mul(mom, momentum),
                 xla::Mul(xla::Mul(grad, lr),
                          xla::Pow(xla::Add(new_ms, epsilon),
                                   XlaHelpers::FloatLiteral(b, type, -0.5))));
    xla::XlaOp new_var = xla::Sub(var, new_mom);

    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, type, new_var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, type, new_ms));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(2, type, new_mom));
  }
};
REGISTER_XLA_OP(Name("ResourceApplyRMSProp").TypeConstraint("T", kFloatTypes),
                ResourceApplyRMSProp);

void CompileFtrl(XlaOpKernelContext* ctx, DataType dtype,
                 bool has_l2_shrinkage) {
  xla::XlaBuilder* b = ctx->builder();

  TensorShape var_shape, accum_shape, linear_shape;
  xla::XlaOp var, accum, linear;
  OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, dtype, &var_shape, &var));
  OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(1, dtype, &accum_shape, &accum));
  OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(2, dtype, &linear_shape, &linear));

  OP_REQUIRES(ctx, var_shape.IsSameSize(accum_shape),
              errors::InvalidArgument(
                  "var and accum do not have the same shape",
                  var_shape.DebugString(), " ", accum_shape.DebugString()));

  OP_REQUIRES(ctx, var_shape.IsSameSize(linear_shape),
              errors::InvalidArgument(
                  "var and linear do not have the same shape",
                  var_shape.DebugString(), " ", linear_shape.DebugString()));

  TensorShape grad_shape = ctx->InputShape(3);
  TensorShape lr_shape = ctx->InputShape(4);
  TensorShape l1_shape = ctx->InputShape(5);
  TensorShape l2_shape = ctx->InputShape(6);
  TensorShape l2_shrinkage_shape;
  TensorShape lr_power_shape;
  if (has_l2_shrinkage) {
    l2_shrinkage_shape = ctx->InputShape(7);
    lr_power_shape = ctx->InputShape(8);
  } else {
    lr_power_shape = ctx->InputShape(7);
  }

  OP_REQUIRES(ctx, var_shape.IsSameSize(grad_shape),
              errors::InvalidArgument("var and grad do not have the same shape",
                                      var_shape.DebugString(), " ",
                                      grad_shape.DebugString()));

  OP_REQUIRES(
      ctx, TensorShapeUtils::IsScalar(lr_shape),
      errors::InvalidArgument("lr is not a scalar: ", lr_shape.DebugString()));

  OP_REQUIRES(
      ctx, TensorShapeUtils::IsScalar(l1_shape),
      errors::InvalidArgument("l1 is not a scalar: ", l1_shape.DebugString()));

  OP_REQUIRES(
      ctx, TensorShapeUtils::IsScalar(l2_shape),
      errors::InvalidArgument("l2 is not a scalar: ", l2_shape.DebugString()));

  if (has_l2_shrinkage) {
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(l2_shrinkage_shape),
                errors::InvalidArgument("l2_shrinkage is not a scalar: ",
                                        l2_shrinkage_shape.DebugString()));
  }

  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr_power_shape),
              errors::InvalidArgument("lr_power is not a scalar: ",
                                      lr_power_shape.DebugString()));

  xla::XlaOp grad = ctx->Input(3);
  xla::XlaOp lr = ctx->Input(4);
  xla::XlaOp l1 = ctx->Input(5);
  xla::XlaOp l2 = ctx->Input(6);
  xla::XlaOp l2_shrinkage;
  xla::XlaOp lr_power;
  if (has_l2_shrinkage) {
    l2_shrinkage = ctx->Input(7);
    lr_power = ctx->Input(8);
  } else {
    lr_power = ctx->Input(7);
  }

  // grad_to_use = grad + 2 * l2_shrinkage * var
  // new_accum = accum + grad_to_use * grad_to_use
  // linear += grad_to_use -
  //     (new_accum^(-lr_power) - accum^(-lr_power)) / lr * var
  // quadratic = (new_accum^(-lr_power) / lr) + 2 * l2
  // linear_clipped = clamp linear in [-l1, l1]
  // var = (linear_clipped - linear) / quadratic
  // accum = new_accum

  xla::XlaOp two = XlaHelpers::FloatLiteral(b, dtype, 2.0);
  xla::XlaOp grad_to_use;
  if (has_l2_shrinkage) {
    grad_to_use = xla::Add(grad, xla::Mul(two, xla::Mul(l2_shrinkage, var)));
  } else {
    grad_to_use = grad;
  }

  xla::XlaOp new_accum = xla::Add(accum, xla::Pow(grad_to_use, two));
  xla::XlaOp new_accum_lr_pow = xla::Pow(new_accum, xla::Neg(lr_power));
  xla::XlaOp accum_lr_pow = xla::Pow(accum, xla::Neg(lr_power));
  linear = xla::Add(
      linear,
      xla::Sub(grad_to_use,
               xla::Mul(xla::Div(xla::Sub(new_accum_lr_pow, accum_lr_pow), lr),
                        var)));
  xla::XlaOp linear_clipped = xla::Clamp(xla::Neg(l1), linear, l1);
  xla::XlaOp quadratic =
      xla::Add(xla::Div(new_accum_lr_pow, lr), xla::Mul(two, l2));
  var = xla::Div(xla::Sub(linear_clipped, linear), quadratic);
  accum = new_accum;

  OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, dtype, var));
  OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, dtype, accum));
  OP_REQUIRES_OK(ctx, ctx->AssignVariable(2, dtype, linear));
}

class ResourceApplyFtrl : public XlaOpKernel {
 public:
  explicit ResourceApplyFtrl(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    CompileFtrl(ctx, dtype_, /*has_l2_shrinkage=*/false);
  }

 private:
  DataType dtype_;
};
REGISTER_XLA_OP(Name("ResourceApplyFtrl").TypeConstraint("T", kFloatTypes),
                ResourceApplyFtrl);

class ResourceApplyFtrlV2 : public XlaOpKernel {
 public:
  explicit ResourceApplyFtrlV2(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    CompileFtrl(ctx, dtype_, /*has_l2_shrinkage=*/true);
  }

 private:
  DataType dtype_;
};
REGISTER_XLA_OP(Name("ResourceApplyFtrlV2").TypeConstraint("T", kFloatTypes),
                ResourceApplyFtrlV2);

}  // namespace
}  // namespace tensorflow
