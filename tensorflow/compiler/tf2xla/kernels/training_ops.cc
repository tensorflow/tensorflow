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
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/hlo/builder/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"

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

    handle = handle - ctx->Input(1) * ctx->Input(2);
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, type, handle));
  }
};
REGISTER_XLA_OP(Name("ResourceApplyGradientDescent")
                    .TypeConstraint("T", kFloatAndComplexTypes),
                ResourceApplyGradientDescent);

xla::XlaOp ProximalGradientDescentUpdate(xla::XlaOp var, xla::XlaOp lr,
                                         xla::XlaOp l1, xla::XlaOp l2,
                                         xla::XlaOp grad) {
  xla::XlaOp one = xla::ScalarLike(lr, 1.0);
  xla::XlaOp zero = xla::ScalarLike(lr, 0.0);
  xla::XlaOp prox_var = var - grad * lr;
  xla::XlaOp l1_gt_zero =
      xla::Sign(prox_var) * xla::Max(xla::Abs(prox_var) - lr * l1, zero);
  xla::XlaOp l1_le_zero = prox_var;
  return xla::Select(xla::Gt(l1, zero), l1_gt_zero, l1_le_zero) /
         (one + lr * l2);
}

class ResourceApplyProximalGradientDescent : public XlaOpKernel {
 public:
  explicit ResourceApplyProximalGradientDescent(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp var;
    TensorShape var_shape;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, dtype_, &var_shape, &var));

    TensorShape alpha_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(alpha_shape),
                errors::InvalidArgument("alpha is not a scalar: ",
                                        alpha_shape.DebugString()));
    TensorShape l1_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(alpha_shape),
                errors::InvalidArgument("l1 is not a scalar: ",
                                        l1_shape.DebugString()));
    TensorShape l2_shape = ctx->InputShape(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(alpha_shape),
                errors::InvalidArgument("l2 is not a scalar: ",
                                        l2_shape.DebugString()));
    TensorShape delta_shape = ctx->InputShape(4);
    OP_REQUIRES(
        ctx, var_shape.IsSameSize(delta_shape),
        errors::InvalidArgument("var and delta do not have the same shape: ",
                                var_shape.DebugString(), " vs ",
                                delta_shape.DebugString()));
    xla::XlaOp alpha = ctx->Input(1);
    xla::XlaOp l1 = ctx->Input(2);
    xla::XlaOp l2 = ctx->Input(3);
    xla::XlaOp delta = ctx->Input(4);
    var = ProximalGradientDescentUpdate(var, alpha, l1, l2, delta);
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, dtype_, var));
  }

 private:
  DataType dtype_;
};
REGISTER_XLA_OP(Name("ResourceApplyProximalGradientDescent")
                    .TypeConstraint("T", kFloatAndComplexTypes),
                ResourceApplyProximalGradientDescent);

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

    accum = accum * momentum + grad;
    if (use_nesterov_) {
      // See https://github.com/tensorflow/tensorflow/pull/2798 for an
      // explanation of the reparameterization used here.
      var = var - (grad * lr + accum * momentum * lr);
    } else {
      var = var - accum * lr;
    }
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, type, var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, type, accum));
  }

 private:
  bool use_nesterov_;
};
REGISTER_XLA_OP(Name("ResourceApplyMomentum").TypeConstraint("T", kFloatTypes),
                ResourceApplyMomentum);

class ResourceApplyKerasMomentum : public XlaOpKernel {
 public:
  explicit ResourceApplyKerasMomentum(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
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

    accum = accum * momentum - grad * lr;
    if (use_nesterov_) {
      // See https://github.com/tensorflow/tensorflow/pull/2798 for an
      // explanation of the reparameterization used here.
      var = var + accum * momentum - grad * lr;
    } else {
      var = var + accum;
    }
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, type, var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, type, accum));
  }

 private:
  bool use_nesterov_;
};
REGISTER_XLA_OP(Name("ResourceApplyKerasMomentum")
                    .TypeConstraint("T", kFloatAndComplexTypes),
                ResourceApplyKerasMomentum);

class ResourceApplyAdagrad : public XlaOpKernel {
 public:
  explicit ResourceApplyAdagrad(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots_));
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

    xla::XlaOp lr = ctx->Input(2);
    xla::XlaOp grad = ctx->Input(3);

    if (update_slots_) {
      accum = accum + xla::Square(grad);
    }
    var = var - grad * lr * xla::Rsqrt(accum);
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, type, var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, type, accum));
  }

 private:
  bool update_slots_;
};
REGISTER_XLA_OP(
    Name("ResourceApplyAdagrad").TypeConstraint("T", kFloatAndComplexTypes),
    ResourceApplyAdagrad);

class ResourceApplyAdagradV2 : public XlaOpKernel {
 public:
  explicit ResourceApplyAdagradV2(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots_));
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

    TensorShape epsilon_shape = ctx->InputShape(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon_shape),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon_shape.DebugString()));

    TensorShape grad_shape = ctx->InputShape(4);
    OP_REQUIRES(ctx, var_shape.IsSameSize(grad_shape),
                errors::InvalidArgument(
                    "var and grad do not have the same shape",
                    var_shape.DebugString(), " ", grad_shape.DebugString()));

    xla::XlaOp lr = ctx->Input(2);
    xla::XlaOp epsilon = ctx->Input(3);
    xla::XlaOp grad = ctx->Input(4);

    if (update_slots_) {
      accum = accum + xla::Square(grad);
    }
    var = var - grad * lr / (xla::Sqrt(accum) + epsilon);
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, type, var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, type, accum));
  }

 private:
  bool update_slots_;
};
REGISTER_XLA_OP(
    Name("ResourceApplyAdagradV2").TypeConstraint("T", kFloatAndComplexTypes),
    ResourceApplyAdagradV2);

class ResourceApplyProximalAdagrad : public XlaOpKernel {
 public:
  explicit ResourceApplyProximalAdagrad(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape var_shape, accum_shape;
    xla::XlaOp var, accum;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, dtype_, &var_shape, &var));
    OP_REQUIRES_OK(ctx,
                   ctx->ReadVariableInput(1, dtype_, &accum_shape, &accum));

    OP_REQUIRES(ctx, var_shape.IsSameSize(accum_shape),
                errors::InvalidArgument(
                    "var and accum do not have the same shape",
                    var_shape.DebugString(), " ", accum_shape.DebugString()));

    TensorShape lr_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr_shape),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr_shape.DebugString()));
    TensorShape l1_shape = ctx->InputShape(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(l1_shape),
                errors::InvalidArgument("l1 is not a scalar: ",
                                        l1_shape.DebugString()));
    TensorShape l2_shape = ctx->InputShape(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(l2_shape),
                errors::InvalidArgument("l2 is not a scalar: ",
                                        l2_shape.DebugString()));
    TensorShape grad_shape = ctx->InputShape(5);
    OP_REQUIRES(ctx, var_shape.IsSameSize(grad_shape),
                errors::InvalidArgument(
                    "var and grad do not have the same shape: ",
                    var_shape.DebugString(), " vs ", grad_shape.DebugString()));

    xla::XlaOp lr = ctx->Input(2);
    xla::XlaOp l1 = ctx->Input(3);
    xla::XlaOp l2 = ctx->Input(4);
    xla::XlaOp grad = ctx->Input(5);
    accum = accum + xla::Square(grad);
    // Adagrad learning rate.
    xla::XlaOp adagrad_lr = lr * xla::Rsqrt(accum);
    var = ProximalGradientDescentUpdate(var, adagrad_lr, l1, l2, grad);
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, dtype_, var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, dtype_, accum));
  }

 private:
  DataType dtype_;
};
REGISTER_XLA_OP(Name("ResourceApplyProximalAdagrad")
                    .TypeConstraint("T", kFloatAndComplexTypes),
                ResourceApplyProximalAdagrad);

class ResourceApplyAdagradDA : public XlaOpKernel {
 public:
  explicit ResourceApplyAdagradDA(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape var_shape, accum_shape, squared_accum_shape;
    xla::XlaOp var, accum, squared_accum;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, dtype_, &var_shape, &var));
    OP_REQUIRES_OK(ctx,
                   ctx->ReadVariableInput(1, dtype_, &accum_shape, &accum));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(2, dtype_, &squared_accum_shape,
                                               &squared_accum));
    OP_REQUIRES(ctx, var_shape.IsSameSize(accum_shape),
                errors::InvalidArgument(
                    "var and accum do not have the same shape",
                    var_shape.DebugString(), " ", accum_shape.DebugString()));
    OP_REQUIRES(
        ctx, var_shape.IsSameSize(squared_accum_shape),
        errors::InvalidArgument(
            "var and squared accum do not have the same shape",
            var_shape.DebugString(), " ", squared_accum_shape.DebugString()));

    TensorShape grad_shape = ctx->InputShape(3);
    TensorShape lr_shape = ctx->InputShape(4);
    TensorShape l1_shape = ctx->InputShape(5);
    TensorShape l2_shape = ctx->InputShape(6);
    TensorShape global_step_shape = ctx->InputShape(7);

    OP_REQUIRES(ctx, var_shape.IsSameSize(grad_shape),
                errors::InvalidArgument(
                    "var and grad do not have the same shape",
                    var_shape.DebugString(), " ", grad_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr_shape),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(l1_shape),
                errors::InvalidArgument("l1 is not a scalar: ",
                                        l1_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(l2_shape),
                errors::InvalidArgument("l2 is not a scalar: ",
                                        l2_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(global_step_shape),
                errors::InvalidArgument("global step is not a scalar: ",
                                        global_step_shape.DebugString()));

    xla::XlaOp grad = ctx->Input(3);
    xla::XlaOp lr = ctx->Input(4);
    xla::XlaOp l1 = ctx->Input(5);
    xla::XlaOp l2 = ctx->Input(6);
    xla::XlaOp global_step =
        XlaHelpers::ConvertElementType(ctx->Input(7), dtype_);

    accum = accum + grad;
    squared_accum = squared_accum + xla::Square(grad);
    xla::XlaOp zero = xla::ScalarLike(lr, 0.0);
    xla::XlaOp denominator = global_step * lr * l2 + xla::Sqrt(squared_accum);
    xla::XlaOp l1_le_zero = -lr * accum / denominator;
    xla::XlaOp l1_gt_zero = -lr * xla::Sign(accum) *
                            xla::Max(xla::Abs(accum) - global_step * l1, zero) /
                            denominator;

    var = xla::Select(xla::Gt(l1, zero), l1_gt_zero, l1_le_zero);
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, dtype_, var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, dtype_, accum));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(2, dtype_, squared_accum));
  }

 private:
  DataType dtype_;
};
REGISTER_XLA_OP(Name("ResourceApplyAdagradDA").TypeConstraint("T", kFloatTypes),
                ResourceApplyAdagradDA);

class ResourceApplyAdam : public XlaOpKernel {
 public:
  explicit ResourceApplyAdam(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
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
    // if not use_nesterov:
    //   variable <- variable - alpha * m_t / (sqrt(v_t) + epsilon)
    // if use_nesterov:
    //   variable <- variable - alpha * (m_t * beta1 + (1 - beta1) * g_t) /
    //   (sqrt(v_t) + epsilon)

    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp one = XlaHelpers::FloatLiteral(b, dtype_, 1.0);

    xla::XlaOp alpha = lr * xla::Sqrt(one - beta2_power) / (one - beta1_power);
    auto m_t = m + (grad - m) * (one - beta1);
    v = v + (xla::Square(grad) - v) * (one - beta2);
    if (use_nesterov_) {
      var = var - alpha * (m_t * beta1 + (one - beta1) * grad) /
                      (xla::Sqrt(v) + epsilon);
    } else {
      var = var - m_t * alpha / (xla::Sqrt(v) + epsilon);
    }

    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, dtype_, var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, dtype_, m_t));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(2, dtype_, v));
  }

 private:
  DataType dtype_;
  bool use_nesterov_;
};
REGISTER_XLA_OP(
    Name("ResourceApplyAdam").TypeConstraint("T", kFloatAndComplexTypes),
    ResourceApplyAdam);

class ResourceApplyAdaMax : public XlaOpKernel {
 public:
  explicit ResourceApplyAdaMax(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape var_shape, m_shape, v_shape;
    xla::XlaOp var, m, v;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, dtype_, &var_shape, &var));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(1, dtype_, &m_shape, &m));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(2, dtype_, &v_shape, &v));

    TensorShape beta1_power_shape = ctx->InputShape(3);
    TensorShape lr_shape = ctx->InputShape(4);
    TensorShape beta1_shape = ctx->InputShape(5);
    TensorShape beta2_shape = ctx->InputShape(6);
    TensorShape epsilon_shape = ctx->InputShape(7);
    TensorShape grad_shape = ctx->InputShape(8);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power_shape),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power_shape.DebugString()));
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
    xla::XlaOp lr = ctx->Input(4);
    xla::XlaOp beta1 = ctx->Input(5);
    xla::XlaOp beta2 = ctx->Input(6);
    xla::XlaOp epsilon = ctx->Input(7);
    xla::XlaOp grad = ctx->Input(8);

    xla::XlaOp one = xla::ScalarLike(lr, 1.0);
    m = beta1 * m + (one - beta1) * grad;
    v = xla::Max(beta2 * v, xla::Abs(grad));
    var = var - lr / (one - beta1_power) * (m / (v + epsilon));

    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, dtype_, var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, dtype_, m));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(2, dtype_, v));
  }

 private:
  DataType dtype_;
};
REGISTER_XLA_OP(Name("ResourceApplyAdaMax").TypeConstraint("T", kFloatTypes),
                ResourceApplyAdaMax);

class ResourceApplyRMSProp : public XlaOpKernel {
 public:
  explicit ResourceApplyRMSProp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape var_shape, ms_shape, mom_shape, mg_shape;
    xla::XlaOp var, ms, mom, mg;
    OP_REQUIRES_OK(ctx,
                   ctx->ReadVariableInput("var", dtype_, &var_shape, &var));
    if (centered_) {
      OP_REQUIRES_OK(ctx, ctx->ReadVariableInput("mg", dtype_, &mg_shape, &mg));
    }
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput("ms", dtype_, &ms_shape, &ms));
    OP_REQUIRES_OK(ctx,
                   ctx->ReadVariableInput("mom", dtype_, &mom_shape, &mom));

    TensorShape lr_shape = ctx->InputShape("lr");
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr_shape),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr_shape.DebugString()));
    TensorShape rho_shape = ctx->InputShape("rho");
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho_shape),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho_shape.DebugString()));
    TensorShape momentum_shape = ctx->InputShape("momentum");
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum_shape),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum_shape.DebugString()));
    TensorShape epsilon_shape = ctx->InputShape("epsilon");
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon_shape),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon_shape.DebugString()));
    TensorShape grad_shape = ctx->InputShape("grad");

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

    xla::XlaOp lr = ctx->Input("lr");
    xla::XlaOp rho = ctx->Input("rho");
    xla::XlaOp momentum = ctx->Input("momentum");
    xla::XlaOp epsilon = ctx->Input("epsilon");
    xla::XlaOp grad = ctx->Input("grad");

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
    xla::XlaOp one = xla::ScalarLike(ms, 1.0);
    xla::XlaOp new_ms = xla::Square(grad) * (one - rho) + ms * rho;
    xla::XlaOp denominator;
    if (centered_) {
      mg = grad * (one - rho) + mg * rho;
      denominator = new_ms - xla::Square(mg) + epsilon;
    } else {
      denominator = new_ms + epsilon;
    }
    xla::XlaOp new_mom = mom * momentum + grad * lr * xla::Rsqrt(denominator);
    xla::XlaOp new_var = var - new_mom;

    OP_REQUIRES_OK(ctx, ctx->AssignVariable("var", dtype_, new_var));
    if (centered_) {
      OP_REQUIRES_OK(ctx, ctx->AssignVariable("mg", dtype_, mg));
    }
    OP_REQUIRES_OK(ctx, ctx->AssignVariable("ms", dtype_, new_ms));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable("mom", dtype_, new_mom));
  }

 protected:
  bool centered_ = false;

 private:
  DataType dtype_;
};
REGISTER_XLA_OP(
    Name("ResourceApplyRMSProp").TypeConstraint("T", kFloatAndComplexTypes),
    ResourceApplyRMSProp);

class ResourceApplyCenteredRMSProp : public ResourceApplyRMSProp {
 public:
  explicit ResourceApplyCenteredRMSProp(OpKernelConstruction* ctx)
      : ResourceApplyRMSProp(ctx) {
    centered_ = true;
  }
};
REGISTER_XLA_OP(Name("ResourceApplyCenteredRMSProp")
                    .TypeConstraint("T", kFloatAndComplexTypes),
                ResourceApplyCenteredRMSProp);

void CompileFtrl(XlaOpKernelContext* ctx, DataType dtype, bool has_l2_shrinkage,
                 bool multiply_linear_by_lr) {
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
  // new_accum = accum + grad * grad
  // linear += grad_to_use -
  //     (new_accum^(-lr_power) - accum^(-lr_power)) / lr * var
  // quadratic = (new_accum^(-lr_power) / lr) + 2 * l2
  // linear_clipped = clamp linear in [-l1, l1]
  // var = (linear_clipped - linear) / quadratic
  // accum = new_accum

  xla::XlaOp two = XlaHelpers::FloatLiteral(b, dtype, 2.0);
  xla::XlaOp grad_to_use;
  if (has_l2_shrinkage) {
    grad_to_use = grad + two * l2_shrinkage * var;
  } else {
    grad_to_use = grad;
  }

  xla::XlaOp new_accum = accum + xla::Square(grad);
  xla::XlaOp new_accum_lr_pow = xla::Pow(new_accum, -lr_power);
  xla::XlaOp accum_lr_pow = xla::Pow(accum, -lr_power);
  if (multiply_linear_by_lr) {
    linear =
        linear + grad_to_use * lr - (new_accum_lr_pow - accum_lr_pow) * var;
  } else {
    linear =
        linear + grad_to_use - (new_accum_lr_pow - accum_lr_pow) / lr * var;
  }
  xla::XlaOp linear_clipped =
      (multiply_linear_by_lr ? xla::Clamp(-l1 * lr, linear, l1 * lr)
                             : xla::Clamp(-l1, linear, l1));
  xla::XlaOp quadratic =
      (multiply_linear_by_lr ? new_accum_lr_pow + two * l2 * lr
                             : new_accum_lr_pow / lr + two * l2);
  var = (linear_clipped - linear) / quadratic;
  accum = new_accum;

  OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, dtype, var));
  OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, dtype, accum));
  OP_REQUIRES_OK(ctx, ctx->AssignVariable(2, dtype, linear));
}

class ResourceApplyFtrl : public XlaOpKernel {
 public:
  explicit ResourceApplyFtrl(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("multiply_linear_by_lr", &multiply_linear_by_lr_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    CompileFtrl(ctx, dtype_, /*has_l2_shrinkage=*/false,
                /*multiply_linear_by_lr=*/multiply_linear_by_lr_);
  }

 private:
  DataType dtype_;

  // Whether to keep the "linear" slot variable multiplied by the learning rate.
  bool multiply_linear_by_lr_;
};
REGISTER_XLA_OP(Name("ResourceApplyFtrl").TypeConstraint("T", kFloatTypes),
                ResourceApplyFtrl);

class ResourceApplyFtrlV2 : public XlaOpKernel {
 public:
  explicit ResourceApplyFtrlV2(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("multiply_linear_by_lr", &multiply_linear_by_lr_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    CompileFtrl(ctx, dtype_, /*has_l2_shrinkage=*/true,
                /*multiply_linear_by_lr=*/multiply_linear_by_lr_);
  }

 private:
  DataType dtype_;

  // Whether to keep the "linear" slot variable multiplied by the learning rate.
  bool multiply_linear_by_lr_;
};
REGISTER_XLA_OP(Name("ResourceApplyFtrlV2").TypeConstraint("T", kFloatTypes),
                ResourceApplyFtrlV2);

class ResourceApplyAdadelta : public XlaOpKernel {
 public:
  explicit ResourceApplyAdadelta(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape var_shape, accum_shape, accum_update_shape;
    xla::XlaOp var, accum, accum_update;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, dtype_, &var_shape, &var));
    OP_REQUIRES_OK(ctx,
                   ctx->ReadVariableInput(1, dtype_, &accum_shape, &accum));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(2, dtype_, &accum_update_shape,
                                               &accum_update));

    TensorShape lr_shape = ctx->InputShape(3);
    TensorShape rho_shape = ctx->InputShape(4);
    TensorShape epsilon_shape = ctx->InputShape(5);
    TensorShape grad_shape = ctx->InputShape(6);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr_shape),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr_shape.DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho_shape),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho_shape.DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon_shape),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon_shape.DebugString()));

    OP_REQUIRES(ctx, var_shape.IsSameSize(accum_shape),
                errors::InvalidArgument(
                    "var and accum do not have the same shape",
                    var_shape.DebugString(), " ", accum_shape.DebugString()));

    OP_REQUIRES(ctx, var_shape.IsSameSize(grad_shape),
                errors::InvalidArgument(
                    "var and grad do not have the same shape",
                    var_shape.DebugString(), " ", grad_shape.DebugString()));

    xla::XlaOp lr = ctx->Input(3);
    xla::XlaOp rho = ctx->Input(4);
    xla::XlaOp epsilon = ctx->Input(5);
    xla::XlaOp grad = ctx->Input(6);

    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp one = XlaHelpers::FloatLiteral(b, dtype_, 1.0);

    accum = rho * accum + (one - rho) * xla::Square(grad);
    xla::XlaOp update =
        xla::Sqrt(accum_update + epsilon) * xla::Rsqrt(accum + epsilon) * grad;
    accum_update = rho * accum_update + (one - rho) * xla::Square(update);
    var = var - update * lr;
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, dtype_, var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, dtype_, accum));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(2, dtype_, accum_update));
  }

 private:
  DataType dtype_;
};
REGISTER_XLA_OP(
    Name("ResourceApplyAdadelta").TypeConstraint("T", kFloatAndComplexTypes),
    ResourceApplyAdadelta);

class ResourceApplySignBase : public XlaOpKernel {
 public:
  explicit ResourceApplySignBase(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape var_shape, m_shape;
    xla::XlaOp var, m;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, dtype_, &var_shape, &var));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(1, dtype_, &m_shape, &m));
    OP_REQUIRES(ctx, var_shape.IsSameSize(m_shape),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var_shape.DebugString(), " ",
                                        m_shape.DebugString()));
    TensorShape grad_shape = ctx->InputShape(6);
    OP_REQUIRES(ctx, var_shape.IsSameSize(grad_shape),
                errors::InvalidArgument(
                    "var and grad do not have the same shape",
                    var_shape.DebugString(), " ", grad_shape.DebugString()));
    CheckScalarParams(ctx);

    xla::XlaOp lr = ctx->Input(2);
    xla::XlaOp alpha = ctx->Input(3);
    xla::XlaOp sign_decay = ctx->Input(4);
    xla::XlaOp beta = ctx->Input(5);
    xla::XlaOp grad = ctx->Input(6);

    m = m * beta + grad * (xla::ScalarLike(beta, 1.0) - beta);
    xla::XlaOp decay = xla::Sign(grad) * xla::Sign(m) * sign_decay;

    xla::XlaOp grad_scale = ComputeGradientScale(alpha, decay);
    var = var - lr * grad_scale * grad;
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, dtype_, var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, dtype_, m));
  }

  virtual void CheckScalarParams(XlaOpKernelContext* ctx) {
    TensorShape lr_shape = ctx->InputShape(2);
    TensorShape sign_decay_shape = ctx->InputShape(4);
    TensorShape beta_shape = ctx->InputShape(5);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr_shape),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr_shape.DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(sign_decay_shape),
                errors::InvalidArgument("sign_decay is not a scalar: ",
                                        sign_decay_shape.DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta_shape),
                errors::InvalidArgument("beta is not a scalar: ",
                                        beta_shape.DebugString()));
  }

  virtual xla::XlaOp ComputeGradientScale(xla::XlaOp alpha,
                                          xla::XlaOp decay) = 0;

 private:
  DataType dtype_;
};

class ResourceApplyAddSign : public ResourceApplySignBase {
 public:
  explicit ResourceApplyAddSign(OpKernelConstruction* ctx)
      : ResourceApplySignBase(ctx) {}

  void CheckScalarParams(XlaOpKernelContext* ctx) override {
    ResourceApplySignBase::CheckScalarParams(ctx);
    TensorShape alpha_shape = ctx->InputShape(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(alpha_shape),
                errors::InvalidArgument("alpha is not a scalar: ",
                                        alpha_shape.DebugString()));
  }

  xla::XlaOp ComputeGradientScale(xla::XlaOp alpha, xla::XlaOp decay) override {
    return alpha + decay;
  }
};
REGISTER_XLA_OP(Name("ResourceApplyAddSign").TypeConstraint("T", kFloatTypes),
                ResourceApplyAddSign);

class ResourceApplyPowerSign : public ResourceApplySignBase {
 public:
  explicit ResourceApplyPowerSign(OpKernelConstruction* ctx)
      : ResourceApplySignBase(ctx) {}

  void CheckScalarParams(XlaOpKernelContext* ctx) override {
    ResourceApplySignBase::CheckScalarParams(ctx);
    TensorShape logbase_shape = ctx->InputShape(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(logbase_shape),
                errors::InvalidArgument("logbase is not a scalar: ",
                                        logbase_shape.DebugString()));
  }

  xla::XlaOp ComputeGradientScale(xla::XlaOp alpha, xla::XlaOp decay) override {
    return xla::Exp(alpha * decay);
  }
};
REGISTER_XLA_OP(Name("ResourceApplyPowerSign").TypeConstraint("T", kFloatTypes),
                ResourceApplyPowerSign);

}  // namespace
}  // namespace tensorflow
