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
using tensorflow::ops::Add;
using tensorflow::ops::Conj;
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

class AddGradientFunction : public GradientFunction {
 public:
  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    grad_outputs->resize(2);
    // TODO(b/161805092): Support broadcasting.

    DCHECK(grad_inputs[0]);
    (*grad_outputs)[0] = grad_inputs[0];
    (*grad_outputs)[1] = grad_inputs[0];

    (*grad_outputs)[0]->Ref();
    (*grad_outputs)[1]->Ref();
    return Status::OK();
  }
  ~AddGradientFunction() override {}
};

class ExpGradientFunction : public GradientFunction {
 public:
  explicit ExpGradientFunction(AbstractTensorHandle* exp) : exp_(exp) {
    exp->Ref();
  }
  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    vector<AbstractTensorHandle*> conj_outputs(1);
    std::string name = "Conj_Exp_Grad";
    TF_RETURN_IF_ERROR(Conj(ctx->ctx, {exp_.get()},
                            absl::MakeSpan(conj_outputs), name.c_str()));
    AbstractTensorHandlePtr conj_output_releaser(conj_outputs[0]);
    grad_outputs->resize(1);

    name = "Mul_Exp_Grad";
    TF_RETURN_IF_ERROR(Mul(ctx->ctx, {conj_outputs[0], grad_inputs[0]},
                           absl::MakeSpan(*grad_outputs), name.c_str()));
    return Status::OK();
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
  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    std::string name = "Sqrt_Grad";
    grad_outputs->resize(1);
    TF_RETURN_IF_ERROR(SqrtGrad(ctx->ctx, {sqrt_.get(), grad_inputs[0]},
                                absl::MakeSpan(*grad_outputs), name.c_str()));
    return Status::OK();
  }
  ~SqrtGradientFunction() override {}

 private:
  AbstractTensorHandlePtr sqrt_;
};

class MatMulGradientFunction : public GradientFunction {
 public:
  explicit MatMulGradientFunction(vector<AbstractTensorHandle*> f_inputs,
                                  AttrBuilder f_attrs)
      : forward_inputs(f_inputs), forward_attrs(f_attrs) {}

  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    /* Given upstream grad U and a matmul op A*B, the gradients are:
     *
     *    dA = U * B.T
     *    dB = A.T * U
     *
     *    where A.T means `transpose(A)`
     */
    AbstractTensorHandle* upstream_grad = grad_inputs[0];
    grad_outputs->resize(2);

    // Get transpose attrs
    bool t_a;
    TF_RETURN_IF_ERROR(forward_attrs.Get("transpose_a", &t_a));

    bool t_b;
    TF_RETURN_IF_ERROR(forward_attrs.Get("transpose_b", &t_b));

    // Conj each input
    vector<AbstractTensorHandle*> conj_outputs(1);
    std::string name = "Conj_A_MatMul_Grad";
    TF_RETURN_IF_ERROR(Conj(ctx->ctx, {forward_inputs[0]},
                            absl::MakeSpan(conj_outputs), name.c_str()));

    AbstractTensorHandle* A = conj_outputs[0];

    name = "Conj_B_MatMul_Grad";
    TF_RETURN_IF_ERROR(Conj(ctx->ctx, {forward_inputs[1]},
                            absl::MakeSpan(conj_outputs), name.c_str()));

    AbstractTensorHandle* B = conj_outputs[0];

    // Calc Grad
    vector<AbstractTensorHandle*> matmul_A_outputs(1);
    vector<AbstractTensorHandle*> matmul_B_outputs(1);
    std::string name_grad_A = "MatMul_Grad_A";
    std::string name_grad_B = "MatMul_Grad_B";
    if (!t_a && !t_b) {
      TF_RETURN_IF_ERROR(MatMul(ctx->ctx, {upstream_grad, B},
                                absl::MakeSpan(matmul_A_outputs),
                                name_grad_A.c_str(),
                                /*transpose_a = */ false,
                                /*transpose_b = */ true));

      TF_RETURN_IF_ERROR(MatMul(ctx->ctx, {A, upstream_grad},
                                absl::MakeSpan(matmul_B_outputs),
                                name_grad_B.c_str(),
                                /*transpose_a = */ true,
                                /*transpose_b = */ false));
    } else if (!t_a && t_b) {
      TF_RETURN_IF_ERROR(MatMul(ctx->ctx, {upstream_grad, B},
                                absl::MakeSpan(matmul_A_outputs),
                                name_grad_A.c_str(),
                                /*transpose_a = */ false,
                                /*transpose_b = */ false));

      TF_RETURN_IF_ERROR(MatMul(ctx->ctx, {upstream_grad, A},
                                absl::MakeSpan(matmul_B_outputs),
                                name_grad_B.c_str(),
                                /*transpose_a = */ true,
                                /*transpose_b = */ false));

    } else if (t_a && !t_b) {
      TF_RETURN_IF_ERROR(MatMul(ctx->ctx, {B, upstream_grad},
                                absl::MakeSpan(matmul_A_outputs),
                                name_grad_A.c_str(),
                                /*transpose_a = */ false,
                                /*transpose_b = */ true));

      TF_RETURN_IF_ERROR(MatMul(ctx->ctx, {A, upstream_grad},
                                absl::MakeSpan(matmul_B_outputs),
                                name_grad_B.c_str(),
                                /*transpose_a = */ false,
                                /*transpose_b = */ false));
    } else {  // t_a && t_b
      TF_RETURN_IF_ERROR(MatMul(ctx->ctx, {B, upstream_grad},
                                absl::MakeSpan(matmul_A_outputs),
                                name_grad_A.c_str(),
                                /*transpose_a = */ true,
                                /*transpose_b = */ true));

      TF_RETURN_IF_ERROR(MatMul(ctx->ctx, {upstream_grad, A},
                                absl::MakeSpan(matmul_B_outputs),
                                name_grad_B.c_str(),
                                /*transpose_a = */ true,
                                /*transpose_b = */ true));
    }

    // Gradient for A
    (*grad_outputs)[0] = matmul_A_outputs[0];

    // Gradient for B
    (*grad_outputs)[1] = matmul_B_outputs[0];
    return Status::OK();
  }
  ~MatMulGradientFunction() override {}

 private:
  vector<AbstractTensorHandle*> forward_inputs;
  AttrBuilder forward_attrs;
};

class NegGradientFunction : public GradientFunction {
 public:
  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    /* Given upstream grad U and a Neg op Y = -X, the gradients are:
     *
     *    dX =  -U
     *
     */

    grad_outputs->resize(1);
    std::string name = "Neg_Grad";
    TF_RETURN_IF_ERROR(ops::Neg(ctx->ctx, {grad_inputs[0]},
                                absl::MakeSpan(*grad_outputs), name.c_str()));
    return Status::OK();
  }
  ~NegGradientFunction() override {}
};

class SubGradientFunction : public GradientFunction {
 public:
  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    /* Given upstream grad U and a Sub op A-B, the gradients are:
     *
     *    dA =  U
     *    dB = -U
     *
     */

    grad_outputs->resize(2);

    // Grad for A
    DCHECK(grad_inputs[0]);
    (*grad_outputs)[0] = grad_inputs[0];
    (*grad_outputs)[0]->Ref();

    // Grad for B
    // negate the upstream grad
    std::vector<AbstractTensorHandle*> neg_outputs(1);
    std::string name = "Neg_Sub_Grad_B";
    TF_RETURN_IF_ERROR(ops::Neg(ctx->ctx, {grad_inputs[0]},
                                absl::MakeSpan(neg_outputs), name.c_str()));
    (*grad_outputs)[1] = neg_outputs[0];

    return Status::OK();
  }
  ~SubGradientFunction() override {}
};

class MulGradientFunction : public GradientFunction {
 public:
  explicit MulGradientFunction(vector<AbstractTensorHandle*> f_inputs)
      : forward_inputs(f_inputs) {}

  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    /* Given upstream grad U and a mul op A*B, the gradients are:
     *
     *    dA = U * B
     *    dB = A * U
     *
     */

    AbstractTensorHandle* upstream_grad = grad_inputs[0];
    grad_outputs->resize(2);
    std::vector<AbstractTensorHandle*> mul_outputs(1);

    // Gradient for A
    std::string name = "Mul_Grad_A";
    TF_RETURN_IF_ERROR(Mul(ctx->ctx, {upstream_grad, forward_inputs[1]},
                           absl::MakeSpan(mul_outputs), name.c_str()));
    (*grad_outputs)[0] = mul_outputs[0];

    // Gradient for B
    name = "Mul_Grad_B";
    TF_RETURN_IF_ERROR(Mul(ctx->ctx, {forward_inputs[0], upstream_grad},
                           absl::MakeSpan(mul_outputs), name.c_str()));
    (*grad_outputs)[1] = mul_outputs[0];
    return Status::OK();
  }
  ~MulGradientFunction() override {}

 private:
  vector<AbstractTensorHandle*> forward_inputs;
};

class Log1pGradientFunction : public GradientFunction {
 public:
  explicit Log1pGradientFunction(vector<AbstractTensorHandle*> f_inputs)
      : forward_inputs(f_inputs) {}

  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    // TODO(vnvo2409): Add control dependency
    /* Given upstream grad U and a Log1p op: Y = log(1 + X), the gradients are:
     *
     *    dX = U / (1 + X)
     *
     */

    AbstractTensorHandle* upstream_grad = grad_inputs[0];
    AbstractTensorHandle* X = forward_inputs[0];

    grad_outputs->resize(1);
    vector<AbstractTensorHandle*> temp_outputs(1);

    // Calculate conjugate of X
    std::string name = "Conj_Log1p_Grad_X";
    TF_RETURN_IF_ERROR(
        Conj(ctx->ctx, {X}, absl::MakeSpan(temp_outputs), name.c_str()));

    AbstractTensorHandle* Conj_X = temp_outputs[0];

    // Creates Ones
    name = "OnesLike_Log1p_Grad_X";
    TF_RETURN_IF_ERROR(OnesLike(ctx->ctx, {Conj_X},
                                absl::MakeSpan(temp_outputs), name.c_str()));

    AbstractTensorHandle* Ones_X = temp_outputs[0];

    name = "Add_Log1p_Grad_X";
    // Calculate 1 + Conj(X)
    TF_RETURN_IF_ERROR(Add(ctx->ctx, {Ones_X, Conj_X},
                           absl::MakeSpan(temp_outputs), name.c_str()));

    AbstractTensorHandle* Conj_XP1 = temp_outputs[0];

    name = "Div_Log1p_Grad_X";
    // Calculate U / (1 + Conj(X))
    TF_RETURN_IF_ERROR(Div(ctx->ctx, {upstream_grad, Conj_XP1},
                           absl::MakeSpan(temp_outputs), name.c_str()));

    (*grad_outputs)[0] = temp_outputs[0];

    return Status::OK();
  }
  ~Log1pGradientFunction() override {}

 private:
  vector<AbstractTensorHandle*> forward_inputs;
};

class DivNoNanGradientFunction : public GradientFunction {
 public:
  explicit DivNoNanGradientFunction(vector<AbstractTensorHandle*> f_inputs,
                                    vector<AbstractTensorHandle*> f_outputs)
      : forward_inputs(f_inputs), forward_outputs(f_outputs) {}

  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    // TODO(vnvo2409): Add shape broadcasting
    /* Given upstream grad U and a Div op: Z = X/Y, the gradients are:
     *
     *    dX = U / Y
     *    dY = -U*X / Y^2 = (X/Y) * -U / Y = -U*Z / Y
     *
     */

    AbstractTensorHandle* upstream_grad = grad_inputs[0];
    AbstractTensorHandle* Y = forward_inputs[1];
    AbstractTensorHandle* Z = forward_outputs[0];

    grad_outputs->resize(2);
    vector<AbstractTensorHandle*> temp_outputs(1);

    // Calculate dX =  U / Y
    std::string name = "Div_Grad_X";
    TF_RETURN_IF_ERROR(DivNoNan(ctx->ctx, {upstream_grad, Y},
                                absl::MakeSpan(temp_outputs), name.c_str()));

    (*grad_outputs)[0] = temp_outputs[0];

    // Calculate dY = -U*Z / Y
    name = "Neg_Div_Grad_Y";
    TF_RETURN_IF_ERROR(Neg(ctx->ctx, {upstream_grad},
                           absl::MakeSpan(temp_outputs), name.c_str()));  // -U
    AbstractTensorHandle* MinusU = temp_outputs[0];

    name = "Mul_Div_Grad_Y";
    TF_RETURN_IF_ERROR(Mul(ctx->ctx, {MinusU, Z}, absl::MakeSpan(temp_outputs),
                           name.c_str()));  // -U*Z
    AbstractTensorHandle* UZ = temp_outputs[0];

    name = "Div_Grad_Y";
    TF_RETURN_IF_ERROR(DivNoNan(ctx->ctx, {UZ, Y}, absl::MakeSpan(temp_outputs),
                                name.c_str()));  // -U*Z / Y

    (*grad_outputs)[1] = temp_outputs[0];
    return Status::OK();
  }
  ~DivNoNanGradientFunction() override {}

 private:
  vector<AbstractTensorHandle*> forward_inputs;
  vector<AbstractTensorHandle*> forward_outputs;
};

}  // namespace

BackwardFunction* AddRegisterer(const ForwardOperation& op) {
  auto gradient_function = new AddGradientFunction;
  // For ops with a single output, the gradient function is not called if there
  // is no incoming gradient. So we do not need to worry about creating zeros
  // grads in this case.
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

BackwardFunction* ExpRegisterer(const ForwardOperation& op) {
  auto gradient_function = new ExpGradientFunction(op.outputs[0]);
  // For ops with a single output, the gradient function is not called if there
  // is no incoming gradient. So we do not need to worry about creating zeros
  // grads in this case.
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

BackwardFunction* MatMulRegisterer(const ForwardOperation& op) {
  auto gradient_function = new MatMulGradientFunction(op.inputs, op.attrs);
  // For ops with a single output, the gradient function is not called if there
  // is no incoming gradient. So we do not need to worry about creating zeros
  // grads in this case.
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

BackwardFunction* SqrtRegisterer(const ForwardOperation& op) {
  auto gradient_function = new SqrtGradientFunction(op.outputs[0]);
  // For ops with a single output, the gradient function is not called if there
  // is no incoming gradient. So we do not need to worry about creating zeros
  // grads in this case.
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

BackwardFunction* NegRegisterer(const ForwardOperation& op) {
  auto gradient_function = new NegGradientFunction;
  // For ops with a single output, the gradient function is not called if there
  // is no incoming gradient. So we do not need to worry about creating zeros
  // grads in this case.
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

BackwardFunction* SubRegisterer(const ForwardOperation& op) {
  // For ops with a single output, the gradient function is not called if there
  // is no incoming gradient. So we do not need to worry about creating zeros
  // grads in this case.
  auto gradient_function = new SubGradientFunction;
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

BackwardFunction* MulRegisterer(const ForwardOperation& op) {
  // For ops with a single output, the gradient function is not called if there
  // is no incoming gradient. So we do not need to worry about creating zeros
  // grads in this case.
  auto gradient_function = new MulGradientFunction(op.inputs);
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

BackwardFunction* Log1pRegisterer(const ForwardOperation& op) {
  // For ops with a single output, the gradient function is not called if there
  // is no incoming gradient. So we do not need to worry about creating zeros
  // grads in this case.
  auto gradient_function = new Log1pGradientFunction(op.inputs);
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

BackwardFunction* DivNoNanRegisterer(const ForwardOperation& op) {
  // For ops with a single output, the gradient function is not called if there
  // is no incoming gradient. So we do not need to worry about creating zeros
  // grads in this case.
  auto gradient_function = new DivNoNanGradientFunction(op.inputs, op.outputs);
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

}  // namespace gradients
}  // namespace tensorflow
