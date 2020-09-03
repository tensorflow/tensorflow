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
#include "tensorflow/c/experimental/gradients/array_grad.h"
#include "tensorflow/c/eager/gradients_util.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/experimental/ops/nn_ops.h"

using tensorflow::ops::Slice;
using tensorflow::ops::Rank;
using tensorflow::ops::ConcatV2;
using tensorflow::ops::Shape;
using tensorflow::ops::Reshape;

namespace tensorflow {
namespace gradients {
namespace {
using std::vector;
class IdentityNGradientFunction : public GradientFunction {
 public:
  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    grad_outputs->resize(grad_inputs.size(), nullptr);
    for (int i = 0; i < grad_inputs.size(); i++) {
      auto grad_input = grad_inputs[i];
      // TODO(srbs): Should we add a copy contructor to AbstractTensorHandle
      // that takes care of this similar to `Tensor`?
      if (grad_input) {
        grad_input->Ref();
      }
      (*grad_outputs)[i] = grad_input;
    }
    return Status::OK();
  }
  ~IdentityNGradientFunction() override {}
};

class PadGradientFunction : public GradientFunction {
 public:
  explicit PadGradientFunction(vector<AbstractTensorHandle*> f_inputs)
       : forward_inputs(f_inputs) {}
  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    
    std::cout << "inside grad" << std::endl;
    grad_outputs->resize(2);

    AbstractTensorHandle* upstream_grad = grad_inputs[0];
    AbstractTensorHandle* x = forward_inputs[0];

    // Paddings will be shape [Rank(x), 2]
    AbstractTensorHandle* paddings = forward_inputs[1];

    // Get Rank(x)
    std::string name = "Rank_pad";
    std::vector<AbstractTensorHandle*> temp_outputs(1);
    TF_RETURN_IF_ERROR(Rank(ctx->ctx, {x},
                       absl::MakeSpan(temp_outputs), name.c_str()));
    AbstractTensorHandle* Rank_x = temp_outputs[0];
    std::cout << "got the rank" << std::endl;

    AbstractTensorHandlePtr zero = GetScalarTensorHandleUtilInt(ctx->ctx, 0);
    AbstractTensorHandlePtr one = GetScalarTensorHandleUtilInt(ctx->ctx, 1);
    
    std::cout << "made some tensors" << std::endl;
    // Concatenate Rank and 1 to strip first column from paddings.
    name = "Concat_Rank_&_1";
    TF_RETURN_IF_ERROR(ConcatV2(ctx->ctx, {Rank_x, one.get()}, {zero.get()},
                       absl::MakeSpan(temp_outputs), name.c_str()));
    AbstractTensorHandle* size = temp_outputs[0];

  
    std::cout << "concatenated rank & 1" << std::endl;
    // Make a Tensor with begin values [0, 0] to strip from paddings.
    int begin_vals[] = {0, 0};
    int64_t begin_dims[] = {2};
    AbstractTensorHandlePtr begin = GetTensorHandleUtilInt(ctx->ctx, begin_vals, begin_dims, 1);

    // Get first column from paddings.
    name = "Slice_from_paddings";
    TF_RETURN_IF_ERROR(Slice(ctx->ctx, {paddings, begin.get(), size},
            absl::MakeSpan(temp_outputs), name.c_str()));
    AbstractTensorHandle* first_col_2d = temp_outputs[0];  
    //AbstractTensorHandle* pad_left = temp_outputs[0];
    
    std::cout << "got the first col" << std::endl;

    // Reshape the first column to be a 1-D Tensor
    name = "Reshape_fc";
    AbstractTensorHandlePtr minus_one = GetScalarTensorHandleUtilInt(ctx->ctx, -1);
    TF_RETURN_IF_ERROR(Reshape(ctx->ctx, {first_col_2d, minus_one.get()},
            absl::MakeSpan(temp_outputs), name.c_str()));
    AbstractTensorHandle* pad_left = temp_outputs[0];
    
    // Get the original shape of the input.
    name = "Shape_x";
    TF_RETURN_IF_ERROR(Shape(ctx->ctx, {x},
                       absl::MakeSpan(temp_outputs), name.c_str()));
    AbstractTensorHandle* x_shape = temp_outputs[0];
    
    // Slice out the original shape from the upstream gradient
    name = "Slice_grad";
    TF_RETURN_IF_ERROR(Slice(ctx->ctx, {upstream_grad, pad_left, x_shape},
            absl::MakeSpan(temp_outputs), name.c_str()));

    (*grad_outputs)[0] = temp_outputs[0]; // dx
    (*grad_outputs)[1] = nullptr; // No grad for paddings
    return Status::OK();
  }
  ~PadGradientFunction() override {}
 private:
  vector<AbstractTensorHandle*> forward_inputs; 
};
}  // namespace

BackwardFunction* IdentityNRegisterer(const ForwardOperation& op) {
  auto gradient_function = new IdentityNGradientFunction;
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

BackwardFunction* PadRegisterer(const ForwardOperation& op) {
  auto gradient_function = new PadGradientFunction(op.inputs);
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

}  // namespace gradients
}  // namespace tensorflow
