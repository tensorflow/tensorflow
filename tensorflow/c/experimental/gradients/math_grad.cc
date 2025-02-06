/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/gradients/math_grad.h"

#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/experimental/ops/nn_ops.h"

using std::vector;
using tensorflow::ops::AddV2;
using tensorflow::ops::Div;
using tensorflow::ops::DivNoNan;
using tensorflow::ops::MatMul;
using tensorflow::ops::Mul;
using tensorflow::ops::Neg;
using tensorflow::ops::OnesLike;
using tensorflow::ops::SqrtGrad;

namespace tensorflow {
namespace gradients {
namespace {

static absl::Status SafeConj(AbstractContext* ctx,
                             AbstractTensorHandle* const input,
                             AbstractTensorHandle** output, const char* name) {
  auto dtype = input->DataType();
  if (DataTypeIsFloating(BaseType(dtype)) ||
      DataTypeIsInteger(BaseType(dtype))) {
    return tensorflow::ops::Identity(ctx, input, output, name);
  } else if (!DataTypeIsComplex(BaseType(dtype)) &&
             BaseType(dtype) != DT_VARIANT) {
    return errors::InvalidArgument(
        "Expected numeric or variant tensor, got dtype ", dtype);
  }
  return tensorflow::ops::Conj(ctx, input, output, name);
}

class AddGradientFunction : public GradientFunction {
 public:
  absl::Status Compute(AbstractContext* ctx,
                       absl::Span<AbstractTensorHandle* const> grad_outputs,
                       absl::Span<AbstractTensorHandle*> grad_inputs) override {
    // TODO(b/161805092): Support broadcasting.

    DCHECK(grad_outputs[0]);
    grad_inputs[0] = grad_outputs[0];
    grad_inputs[1] = grad_outputs[0];

    grad_inputs[0]->Ref();
    grad_inputs[1]->Ref();
    return absl::OkStatus();
  }
  ~AddGradientFunction() override {}
};

class ExpGradientFunction : public GradientFunction {
 public:
  explicit ExpGradientFunction(AbstractTensorHandle* exp) : exp_(exp) {
    exp->Ref();
  }
  absl::Status Compute(AbstractContext* ctx,
                       absl::Span<AbstractTensorHandle* const> grad_outputs,
                       absl::Span<AbstractTensorHandle*> grad_inputs) override {
    AbstractTensorHandle* conj_output;
    std::string name = "Conj_Exp_Grad";
    TF_RETURN_IF_ERROR(SafeConj(ctx, exp_.get(), &conj_output, name.c_str()));
    AbstractTensorHandlePtr conj_output_releaser(conj_output);

    name = "Mul_Exp_Grad";
    TF_RETURN_IF_ERROR(
        Mul(ctx, conj_output, grad_outputs[0], &grad_inputs[0], name.c_str()));
    return absl::OkStatus();
  }
  ~ExpGradientFunction() override {}

 private:
  AbstractTensorHandlePtr exp_;
};

class SqrtGradientFunction : public GradientFunction {
 public:
  explicit SqrtGradientFunction(AbstractTensorHandle* sqrt) : sqrt_(sqrt) {
    sqrt->Ref();
  }
  absl::Status Compute(AbstractContext* ctx,
                       absl::Span<AbstractTensorHandle* const> grad_outputs,
                       absl::Span<AbstractTensorHandle*> grad_inputs) override {
    std::string name = "Sqrt_Grad";
    TF_RETURN_IF_ERROR(SqrtGrad(ctx, sqrt_.get(), grad_outputs[0],
                                &grad_inputs[0], name.c_str()));
    return absl::OkStatus();
  }
  ~SqrtGradientFunction() override {}

 private:
  AbstractTensorHandlePtr sqrt_;
};

class MatMulGradientFunction : public GradientFunction {
 public:
  explicit MatMulGradientFunction(vector<AbstractTensorHandle*> f_inputs,
                                  AttrBuilder f_attrs)
      : forward_inputs_(f_inputs), forward_attrs_(f_attrs) {
    for (auto input : forward_inputs_) {
      if (input) {
        input->Ref();
      }
    }
  }

  absl::Status Compute(AbstractContext* ctx,
                       absl::Span<AbstractTensorHandle* const> grad_outputs,
                       absl::Span<AbstractTensorHandle*> grad_inputs) override {
    /* Given upstream grad U and a matmul op A*B, the gradients are:
     *
     *    dA = U * B.T
     *    dB = A.T * U
     *
     *    where A.T means `transpose(A)`
     */
    AbstractTensorHandle* upstream_grad = grad_outputs[0];

    // Get transpose attrs
    bool t_a;
    TF_RETURN_IF_ERROR(forward_attrs_.Get("transpose_a", &t_a));

    bool t_b;
    TF_RETURN_IF_ERROR(forward_attrs_.Get("transpose_b", &t_b));

    // Conj each input
    AbstractTensorHandle* conj_output;
    std::string name = "Conj_A_MatMul_Grad";
    TF_RETURN_IF_ERROR(
        SafeConj(ctx, forward_inputs_[0], &conj_output, name.c_str()));

    AbstractTensorHandlePtr A(conj_output);

    name = "Conj_B_MatMul_Grad";
    TF_RETURN_IF_ERROR(
        SafeConj(ctx, forward_inputs_[1], &conj_output, name.c_str()));

    AbstractTensorHandlePtr B(conj_output);

    // Calc Grad
    AbstractTensorHandle* matmul_A_output;
    AbstractTensorHandle* matmul_B_output;
    std::string name_grad_A = "MatMul_Grad_A";
    std::string name_grad_B = "MatMul_Grad_B";
    if (!t_a && !t_b) {
      TF_RETURN_IF_ERROR(MatMul(ctx, upstream_grad, B.get(), &matmul_A_output,
                                /*transpose_a = */ false,
                                /*transpose_b = */ true, name_grad_A.c_str()));

      TF_RETURN_IF_ERROR(MatMul(ctx, A.get(), upstream_grad, &matmul_B_output,
                                /*transpose_a = */ true,
                                /*transpose_b = */ false, name_grad_B.c_str()));
    } else if (!t_a && t_b) {
      TF_RETURN_IF_ERROR(MatMul(ctx, upstream_grad, B.get(), &matmul_A_output,
                                /*transpose_a = */ false,
                                /*transpose_b = */ false, name_grad_A.c_str()));

      TF_RETURN_IF_ERROR(MatMul(ctx, upstream_grad, A.get(), &matmul_B_output,
                                /*transpose_a = */ true,
                                /*transpose_b = */ false, name_grad_B.c_str()));

    } else if (t_a && !t_b) {
      TF_RETURN_IF_ERROR(MatMul(ctx, B.get(), upstream_grad, &matmul_A_output,
                                /*transpose_a = */ false,
                                /*transpose_b = */ true, name_grad_A.c_str()));

      TF_RETURN_IF_ERROR(MatMul(ctx, A.get(), upstream_grad, &matmul_B_output,
                                /*transpose_a = */ false,
                                /*transpose_b = */ false, name_grad_B.c_str()));
    } else {  // t_a && t_b
      TF_RETURN_IF_ERROR(MatMul(ctx, B.get(), upstream_grad, &matmul_A_output,
                                /*transpose_a = */ true,
                                /*transpose_b = */ true, name_grad_A.c_str()));

      TF_RETURN_IF_ERROR(MatMul(ctx, upstream_grad, A.get(), &matmul_B_output,
                                /*transpose_a = */ true,
                                /*transpose_b = */ true, name_grad_B.c_str()));
    }

    // Gradient for A
    grad_inputs[0] = matmul_A_output;

    // Gradient for B
    grad_inputs[1] = matmul_B_output;
    return absl::OkStatus();
  }
  ~MatMulGradientFunction() override {
    for (auto input : forward_inputs_) {
      if (input) {
        input->Unref();
      }
    }
  }

 private:
  // TODO(b/174778737): Only hold needed inputs.
  vector<AbstractTensorHandle*> forward_inputs_;
  AttrBuilder forward_attrs_;
};

class NegGradientFunction : public GradientFunction {
 public:
  absl::Status Compute(AbstractContext* ctx,
                       absl::Span<AbstractTensorHandle* const> grad_outputs,
                       absl::Span<AbstractTensorHandle*> grad_inputs) override {
    /* Given upstream grad U and a Neg op Y = -X, the gradients are:
     *
     *    dX =  -U
     *
     */

    std::string name = "Neg_Grad";
    TF_RETURN_IF_ERROR(
        ops::Neg(ctx, grad_outputs[0], &grad_inputs[0], name.c_str()));
    return absl::OkStatus();
  }
  ~NegGradientFunction() override {}
};

class SubGradientFunction : public GradientFunction {
 public:
  absl::Status Compute(AbstractContext* ctx,
                       absl::Span<AbstractTensorHandle* const> grad_outputs,
                       absl::Span<AbstractTensorHandle*> grad_inputs) override {
    /* Given upstream grad U and a Sub op A-B, the gradients are:
     *
     *    dA =  U
     *    dB = -U
     *
     */

    // Grad for A
    DCHECK(grad_outputs[0]);
    grad_inputs[0] = grad_outputs[0];
    grad_inputs[0]->Ref();

    // Grad for B
    // negate the upstream grad
    std::string name = "Neg_Sub_Grad_B";
    TF_RETURN_IF_ERROR(
        ops::Neg(ctx, grad_outputs[0], &grad_inputs[1], name.c_str()));

    return absl::OkStatus();
  }
  ~SubGradientFunction() override {}
};

class MulGradientFunction : public GradientFunction {
 public:
  explicit MulGradientFunction(vector<AbstractTensorHandle*> f_inputs)
      : forward_inputs_(f_inputs) {
    for (auto input : forward_inputs_) {
      if (input) {
        input->Ref();
      }
    }
  }

  absl::Status Compute(AbstractContext* ctx,
                       absl::Span<AbstractTensorHandle* const> grad_outputs,
                       absl::Span<AbstractTensorHandle*> grad_inputs) override {
    /* Given upstream grad U and a mul op A*B, the gradients are:
     *
     *    dA = U * B
     *    dB = A * U
     *
     */

    AbstractTensorHandle* upstream_grad = grad_outputs[0];

    // Gradient for A
    std::string name = "Mul_Grad_A";
    TF_RETURN_IF_ERROR(Mul(ctx, upstream_grad, forward_inputs_[1],
                           &grad_inputs[0], name.c_str()));

    // Gradient for B
    name = "Mul_Grad_B";
    TF_RETURN_IF_ERROR(Mul(ctx, forward_inputs_[0], upstream_grad,
                           &grad_inputs[1], name.c_str()));
    return absl::OkStatus();
  }
  ~MulGradientFunction() override {
    for (auto input : forward_inputs_) {
      if (input) {
        input->Unref();
      }
    }
  }

 private:
  // TODO(b/174778737): Only hold needed inputs.
  vector<AbstractTensorHandle*> forward_inputs_;
};

class Log1pGradientFunction : public GradientFunction {
 public:
  explicit Log1pGradientFunction(vector<AbstractTensorHandle*> f_inputs)
      : forward_inputs_(f_inputs) {
    for (auto input : forward_inputs_) {
      if (input) {
        input->Ref();
      }
    }
  }

  absl::Status Compute(AbstractContext* ctx,
                       absl::Span<AbstractTensorHandle* const> grad_outputs,
                       absl::Span<AbstractTensorHandle*> grad_inputs) override {
    // TODO(vnvo2409): Add control dependency
    /* Given upstream grad U and a Log1p op: Y = log(1 + X), the gradients are:
     *
     *    dX = U / (1 + X)
     *
     */

    AbstractTensorHandle* upstream_grad = grad_outputs[0];
    AbstractTensorHandle* X = forward_inputs_[0];

    AbstractTensorHandle* temp_output;

    // Calculate conjugate of X
    std::string name = "Conj_Log1p_Grad_X";
    TF_RETURN_IF_ERROR(SafeConj(ctx, X, &temp_output, name.c_str()));

    AbstractTensorHandlePtr Conj_X(temp_output);

    // Creates Ones
    name = "OnesLike_Log1p_Grad_X";
    TF_RETURN_IF_ERROR(OnesLike(ctx, Conj_X.get(), &temp_output, name.c_str()));

    AbstractTensorHandlePtr Ones_X(temp_output);

    name = "Add_Log1p_Grad_X";
    // Calculate 1 + Conj(X)
    TF_RETURN_IF_ERROR(
        AddV2(ctx, Ones_X.get(), Conj_X.get(), &temp_output, name.c_str()));

    AbstractTensorHandlePtr Conj_XP1(temp_output);

    name = "Div_Log1p_Grad_X";
    // Calculate U / (1 + Conj(X))
    TF_RETURN_IF_ERROR(
        Div(ctx, upstream_grad, Conj_XP1.get(), &grad_inputs[0], name.c_str()));

    return absl::OkStatus();
  }
  ~Log1pGradientFunction() override {
    for (auto input : forward_inputs_) {
      if (input) {
        input->Unref();
      }
    }
  }

 private:
  // TODO(b/174778737): Only hold needed inputs.
  vector<AbstractTensorHandle*> forward_inputs_;
};

class DivNoNanGradientFunction : public GradientFunction {
 public:
  explicit DivNoNanGradientFunction(vector<AbstractTensorHandle*> f_inputs,
                                    vector<AbstractTensorHandle*> f_outputs)
      : forward_inputs_(f_inputs), forward_outputs_(f_outputs) {
    for (auto input : forward_inputs_) {
      if (input) {
        input->Ref();
      }
    }
    for (auto output : forward_outputs_) {
      if (output) {
        output->Ref();
      }
    }
  }

  absl::Status Compute(AbstractContext* ctx,
                       absl::Span<AbstractTensorHandle* const> grad_outputs,
                       absl::Span<AbstractTensorHandle*> grad_inputs) override {
    // TODO(vnvo2409): Add shape broadcasting
    /* Given upstream grad U and a Div op: Z = X/Y, the gradients are:
     *
     *    dX = U / Y
     *    dY = -U*X / Y^2 = (X/Y) * -U / Y = -U*Z / Y
     *
     */

    AbstractTensorHandle* upstream_grad = grad_outputs[0];
    AbstractTensorHandle* Y = forward_inputs_[1];
    AbstractTensorHandle* Z = forward_outputs_[0];

    // Calculate dX =  U / Y
    std::string name = "Div_Grad_X";
    TF_RETURN_IF_ERROR(
        DivNoNan(ctx, upstream_grad, Y, &grad_inputs[0], name.c_str()));

    AbstractTensorHandle* temp_output;
    // Calculate dY = -U*Z / Y
    name = "Neg_Div_Grad_Y";
    TF_RETURN_IF_ERROR(Neg(ctx, upstream_grad, &temp_output,
                           name.c_str()));  // -U
    AbstractTensorHandlePtr MinusU(temp_output);

    name = "Mul_Div_Grad_Y";
    TF_RETURN_IF_ERROR(Mul(ctx, MinusU.get(), Z, &temp_output,
                           name.c_str()));  // -U*Z
    AbstractTensorHandlePtr UZ(temp_output);

    name = "Div_Grad_Y";
    TF_RETURN_IF_ERROR(DivNoNan(ctx, UZ.get(), Y, &grad_inputs[1],
                                name.c_str()));  // -U*Z / Y

    return absl::OkStatus();
  }
  ~DivNoNanGradientFunction() override {
    for (auto input : forward_inputs_) {
      if (input) {
        input->Unref();
      }
    }
    for (auto output : forward_outputs_) {
      if (output) {
        output->Unref();
      }
    }
  }

 private:
  // TODO(b/174778737): Only hold needed inputs and outputs.
  vector<AbstractTensorHandle*> forward_inputs_;
  vector<AbstractTensorHandle*> forward_outputs_;
};

}  // namespace

GradientFunction* AddRegisterer(const ForwardOperation& op) {
  return new AddGradientFunction;
}

GradientFunction* ExpRegisterer(const ForwardOperation& op) {
  return new ExpGradientFunction(op.outputs[0]);
}

GradientFunction* MatMulRegisterer(const ForwardOperation& op) {
  return new MatMulGradientFunction(op.inputs, op.attrs);
}

GradientFunction* SqrtRegisterer(const ForwardOperation& op) {
  return new SqrtGradientFunction(op.outputs[0]);
}

GradientFunction* NegRegisterer(const ForwardOperation& op) {
  return new NegGradientFunction;
}

GradientFunction* SubRegisterer(const ForwardOperation& op) {
  return new SubGradientFunction;
}

GradientFunction* MulRegisterer(const ForwardOperation& op) {
  return new MulGradientFunction(op.inputs);
}

GradientFunction* Log1pRegisterer(const ForwardOperation& op) {
  return new Log1pGradientFunction(op.inputs);
}

GradientFunction* DivNoNanRegisterer(const ForwardOperation& op) {
  return new DivNoNanGradientFunction(op.inputs, op.outputs);
}

}  // namespace gradients
}  // namespace tensorflow
