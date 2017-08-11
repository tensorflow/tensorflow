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

class ResourceApplyAdam : public XlaOpKernel {
 public:
  explicit ResourceApplyAdam(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    DataType var_type, m_type, v_type;
    TensorShape var_shape, m_shape, v_shape;
    OP_REQUIRES_OK(ctx, ctx->GetVariableTypeAndShape(0, &var_type, &var_shape));
    OP_REQUIRES_OK(ctx, ctx->GetVariableTypeAndShape(1, &m_type, &m_shape));
    OP_REQUIRES_OK(ctx, ctx->GetVariableTypeAndShape(2, &v_type, &v_shape));

    OP_REQUIRES(
        ctx, dtype_ == var_type && dtype_ == m_type && dtype_ == v_type,
        errors::InvalidArgument(
            "Types of variable arguments to ResourceApplyRMSProp must match: ",
            DataTypeString(dtype_), " vs. ", DataTypeString(var_type), " vs. ",
            DataTypeString(m_type), " vs. ", DataTypeString(v_type)));

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

    xla::ComputationDataHandle var, m, v;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, &var));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(1, &m));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(2, &v));
    xla::ComputationDataHandle beta1_power = ctx->Input(3);
    xla::ComputationDataHandle beta2_power = ctx->Input(4);
    xla::ComputationDataHandle lr = ctx->Input(5);
    xla::ComputationDataHandle beta1 = ctx->Input(6);
    xla::ComputationDataHandle beta2 = ctx->Input(7);
    xla::ComputationDataHandle epsilon = ctx->Input(8);
    xla::ComputationDataHandle grad = ctx->Input(9);

    // alpha <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
    // m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
    // v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
    // variable <- variable - alpha * m_t / (sqrt(v_t) + epsilon)

    xla::ComputationBuilder* b = ctx->builder();
    xla::ComputationDataHandle half = XlaHelpers::FloatLiteral(b, dtype_, 0.5);
    xla::ComputationDataHandle one = XlaHelpers::FloatLiteral(b, dtype_, 1.0);
    xla::ComputationDataHandle two = XlaHelpers::FloatLiteral(b, dtype_, 2.0);

    xla::ComputationDataHandle alpha =
        b->Div(b->Mul(lr, b->Pow(b->Sub(one, beta2_power), half)),
               b->Sub(one, beta1_power));
    m = b->Add(m, b->Mul(b->Sub(grad, m), b->Sub(one, beta1)));
    v = b->Add(v, b->Mul(b->Sub(b->Pow(grad, two), v), b->Sub(one, beta2)));
    var =
        b->Sub(var, b->Div(b->Mul(m, alpha), b->Add(b->Pow(v, half), epsilon)));

    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, dtype_, var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, dtype_, m));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(2, dtype_, v));
  }

 private:
  DataType dtype_;
};
REGISTER_XLA_OP(Name("ResourceApplyAdam"), ResourceApplyAdam);

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
               b->Mul(b->Mul(grad, lr),
                      b->Pow(b->Add(new_ms, epsilon),
                             XlaHelpers::FloatLiteral(b, type, -0.5))));
    xla::ComputationDataHandle new_var = b->Sub(var, new_mom);

    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, type, new_var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, type, new_ms));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(2, type, new_mom));
  }
};
REGISTER_XLA_OP(Name("ResourceApplyRMSProp"), ResourceApplyRMSProp);

void CompileFtrl(XlaOpKernelContext* ctx, DataType dtype,
                 bool has_l2_shrinkage) {
  xla::ComputationBuilder* b = ctx->builder();

  DataType var_type, accum_type, linear_type;
  TensorShape var_shape, accum_shape, linear_shape;
  OP_REQUIRES_OK(ctx, ctx->GetVariableTypeAndShape(0, &var_type, &var_shape));
  OP_REQUIRES_OK(ctx,
                 ctx->GetVariableTypeAndShape(1, &accum_type, &accum_shape));
  OP_REQUIRES_OK(ctx,
                 ctx->GetVariableTypeAndShape(2, &linear_type, &linear_shape));

  OP_REQUIRES(
      ctx, dtype == var_type && dtype == accum_type && dtype == linear_type,
      errors::InvalidArgument(
          "Types of variable arguments to ResourceApplyFtrlV2 must match: ",
          DataTypeString(dtype), " vs. ", DataTypeString(var_type), " and ",
          DataTypeString(accum_type), " and ", DataTypeString(linear_type)));

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

  xla::ComputationDataHandle var, accum, linear;
  OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, &var));
  OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(1, &accum));
  OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(2, &linear));
  xla::ComputationDataHandle grad = ctx->Input(3);
  xla::ComputationDataHandle lr = ctx->Input(4);
  xla::ComputationDataHandle l1 = ctx->Input(5);
  xla::ComputationDataHandle l2 = ctx->Input(6);
  xla::ComputationDataHandle l2_shrinkage;
  xla::ComputationDataHandle lr_power;
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

  xla::ComputationDataHandle two = XlaHelpers::FloatLiteral(b, dtype, 2.0);
  xla::ComputationDataHandle grad_to_use;
  if (has_l2_shrinkage) {
    grad_to_use = b->Add(grad, b->Mul(two, b->Mul(l2_shrinkage, var)));
  } else {
    grad_to_use = grad;
  }

  xla::ComputationDataHandle new_accum =
      b->Add(accum, b->Pow(grad_to_use, two));
  xla::ComputationDataHandle new_accum_lr_pow =
      b->Pow(new_accum, b->Neg(lr_power));
  xla::ComputationDataHandle accum_lr_pow = b->Pow(accum, b->Neg(lr_power));
  linear = b->Add(
      linear,
      b->Sub(grad_to_use,
             b->Mul(b->Div(b->Sub(new_accum_lr_pow, accum_lr_pow), lr), var)));
  xla::ComputationDataHandle linear_clipped = b->Clamp(b->Neg(l1), linear, l1);
  xla::ComputationDataHandle quadratic =
      b->Add(b->Div(new_accum_lr_pow, lr), b->Mul(two, l2));
  var = b->Div(b->Sub(linear_clipped, linear), quadratic);
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
REGISTER_XLA_OP(Name("ResourceApplyFtrl"), ResourceApplyFtrl);

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
REGISTER_XLA_OP(Name("ResourceApplyFtrlV2"), ResourceApplyFtrlV2);

}  // namespace
}  // namespace tensorflow
