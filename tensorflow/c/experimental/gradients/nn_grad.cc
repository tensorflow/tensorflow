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
#include "tensorflow/c/experimental/gradients/nn_grad.h"

#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/experimental/ops/nn_ops.h"

using std::vector;
using tensorflow::ops::BiasAddGrad;
using tensorflow::ops::Conj;
using tensorflow::ops::Conv2DBackpropFilter;
using tensorflow::ops::Conv2DBackpropInput;
using tensorflow::ops::DivNoNan;
using tensorflow::ops::FusedBatchNormGrad;
using tensorflow::ops::FusedBatchNormGradV3;
using tensorflow::ops::Identity;
using tensorflow::ops::MaxPoolGrad;
using tensorflow::ops::Mul;
using tensorflow::ops::OnesLike;
using tensorflow::ops::ReluGrad;
using tensorflow::ops::Shape;
using tensorflow::ops::SparseSoftmaxCrossEntropyLoss;
using tensorflow::ops::Sqrt;
using tensorflow::ops::ZerosLike;

namespace tensorflow {
namespace gradients {
namespace {

class ReluGradientFunction : public GradientFunction {
 public:
  explicit ReluGradientFunction(vector<AbstractTensorHandle*> f_outputs)
      : forward_outputs(f_outputs) {}

  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    AbstractTensorHandle* upstream_grad = grad_inputs[0];
    AbstractTensorHandle* activations = forward_outputs[0];
    grad_outputs->resize(1);
    vector<AbstractTensorHandle*> relugrad_outputs(1);

    // Calculate Grad
    std::string name = "relu_grad";
    TF_RETURN_IF_ERROR(ReluGrad(ctx->ctx, {upstream_grad, activations},
                                absl::MakeSpan(relugrad_outputs),
                                name.c_str()));
    (*grad_outputs)[0] = relugrad_outputs[0];

    return Status::OK();
  }
  ~ReluGradientFunction() override {}

 private:
  vector<AbstractTensorHandle*> forward_outputs;
};

class SparseSoftmaxCrossEntropyLossGradientFunction : public GradientFunction {
 public:
  explicit SparseSoftmaxCrossEntropyLossGradientFunction(
      vector<AbstractTensorHandle*> f_outputs)
      : forward_outputs(f_outputs) {}

  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    grad_outputs->resize(2);

    // Grad for Softmax Input
    std::string name = "Mul_Softmax_Grad";
    vector<AbstractTensorHandle*> mul_outputs(1);
    TF_RETURN_IF_ERROR(
        ops::Mul(ctx->ctx, {grad_inputs[0], forward_outputs[1]},
                 absl::MakeSpan(mul_outputs),
                 name.c_str()));  // upstream_grad * local softmax grad
    (*grad_outputs)[0] = mul_outputs[0];

    // Grad for labels is null
    (*grad_outputs)[1] = nullptr;

    return Status::OK();
  }
  ~SparseSoftmaxCrossEntropyLossGradientFunction() override {}

 private:
  vector<AbstractTensorHandle*> forward_outputs;
};

class BiasAddGradientFunction : public GradientFunction {
 public:
  explicit BiasAddGradientFunction(AttrBuilder f_attrs)
      : forward_attrs(f_attrs) {}

  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    /* Given upstream grad U and a BiasAdd: A + bias, the gradients are:
     *
     *    dA = U
     *    dbias = reduceSum(U, dims = channel_dim)
     */

    AbstractTensorHandle* upstream_grad = grad_inputs[0];
    grad_outputs->resize(2);

    // Recover data format from forward pass for gradient.
    string data_format;
    forward_attrs.Get("data_format", &data_format);

    // Calculate Grads
    // Grad for A
    vector<AbstractTensorHandle*> identity_outputs(1);
    std::string name = "Identity_bias_add";
    TF_RETURN_IF_ERROR(ops::Identity(ctx->ctx, {upstream_grad},
                                     absl::MakeSpan(identity_outputs),
                                     name.c_str()));

    (*grad_outputs)[0] = identity_outputs[0];

    // Grad for bias
    vector<AbstractTensorHandle*> bias_add_grad_outputs(1);
    name = "bias_add_grad";
    TF_RETURN_IF_ERROR(BiasAddGrad(ctx->ctx, {upstream_grad},
                                   absl::MakeSpan(bias_add_grad_outputs),
                                   data_format.c_str(), name.c_str()));

    (*grad_outputs)[1] = bias_add_grad_outputs[0];
    return Status::OK();
  }
  ~BiasAddGradientFunction() override {}

 private:
  AttrBuilder forward_attrs;
};

class Conv2DGradientFunction : public GradientFunction {
 public:
  explicit Conv2DGradientFunction(vector<AbstractTensorHandle*> f_inputs,
                                  AttrBuilder f_attrs)
      : forward_inputs(f_inputs), forward_attrs(f_attrs) {}

  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    grad_outputs->resize(2);
    AbstractTensorHandle* upstream_grad = grad_inputs[0];
    AbstractTensorHandle* input = forward_inputs[0];
    AbstractTensorHandle* filter = forward_inputs[1];

    // Recover forward attributes.
    string pad;
    forward_attrs.Get("padding", &pad);

    vector<int64> strides_vec;
    forward_attrs.Get("strides", &strides_vec);
    int num_dims = strides_vec.size();
    int64_t* strides = (int64_t*)strides_vec.data();

    // Get shapes for input and filter.
    vector<AbstractTensorHandle*> shape_outputs(1);
    std::string name = "conv_shape_input";
    TF_RETURN_IF_ERROR(
        Shape(ctx->ctx, {input}, absl::MakeSpan(shape_outputs), name.c_str()));
    AbstractTensorHandle* input_dims = shape_outputs[0];

    name = "conv_shape_filter";
    TF_RETURN_IF_ERROR(
        Shape(ctx->ctx, {filter}, absl::MakeSpan(shape_outputs), name.c_str()));
    AbstractTensorHandle* filter_dims = shape_outputs[0];

    // Gradient for input.
    vector<AbstractTensorHandle*> conv_backprop_input_outputs(1);
    name = "conv_bp_input";
    TF_RETURN_IF_ERROR(
        Conv2DBackpropInput(ctx->ctx, {input_dims, filter, upstream_grad},
                            absl::MakeSpan(conv_backprop_input_outputs),
                            strides, num_dims, pad.c_str(), name.c_str()));

    (*grad_outputs)[0] = conv_backprop_input_outputs[0];

    // Gradient for filter.
    vector<AbstractTensorHandle*> conv_backprop_filter_outputs(1);
    name = "conv_bp_filter";
    TF_RETURN_IF_ERROR(
        Conv2DBackpropFilter(ctx->ctx, {input, filter_dims, upstream_grad},
                             absl::MakeSpan(conv_backprop_filter_outputs),
                             strides, num_dims, pad.c_str(), name.c_str()));
    (*grad_outputs)[1] = conv_backprop_filter_outputs[0];

    return Status::OK();
  }
  ~Conv2DGradientFunction() override {}

 private:
  vector<AbstractTensorHandle*> forward_inputs;
  AttrBuilder forward_attrs;
};

class FusedBatchNormV3GradientFunction : public GradientFunction {
 public:
  explicit FusedBatchNormV3GradientFunction(
      vector<AbstractTensorHandle*> f_inputs,
      vector<AbstractTensorHandle*> f_outputs)
      : forward_inputs(f_inputs), forward_outputs(f_outputs) {}

  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    grad_outputs->resize(5);

    AbstractTensorHandle* upstream_grad = grad_inputs[0];
    AbstractTensorHandle* x_input = forward_inputs[0];
    AbstractTensorHandle* scale = forward_inputs[1];

    // Cached values from forward pass
    AbstractTensorHandle* batch_mean = forward_outputs[1];
    AbstractTensorHandle* batch_var = forward_outputs[2];
    AbstractTensorHandle* rs_1 = forward_outputs[3];
    AbstractTensorHandle* rs_2 = forward_outputs[4];
    AbstractTensorHandle* rs_3 = forward_outputs[5];

    // Calculate Grad
    std::string name = "FBN_V3_grad";
    vector<AbstractTensorHandle*> fbn_grad_outputs(5);

    // Returns[dX, dscale, doffset, rs_4, rs_5]
    TF_RETURN_IF_ERROR(FusedBatchNormGradV3(
        ctx->ctx, {upstream_grad, x_input, scale, rs_1, rs_2, rs_3},
        absl::MakeSpan(fbn_grad_outputs), name.c_str()));

    (*grad_outputs)[0] = fbn_grad_outputs[0];  // dX
    (*grad_outputs)[1] = fbn_grad_outputs[1];  // dscale
    (*grad_outputs)[2] = fbn_grad_outputs[2];  // doffset
    (*grad_outputs)[3] = nullptr;  // Don't pass grads for reserve_spaces
    (*grad_outputs)[4] = nullptr;  // Don't pass grads for reseve_spaces

    return Status::OK();
  }
  ~FusedBatchNormV3GradientFunction() override {}

 private:
  vector<AbstractTensorHandle*> forward_inputs;
  vector<AbstractTensorHandle*> forward_outputs;
};

class MaxPoolGradientFunction : public GradientFunction {
 public:
  explicit MaxPoolGradientFunction(vector<AbstractTensorHandle*> f_inputs,
                                   vector<AbstractTensorHandle*> f_outputs,
                                   AttrBuilder f_attrs)
      : forward_inputs(f_inputs),
        forward_outputs(f_outputs),
        forward_attrs(f_attrs) {}

  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    grad_outputs->resize(1);
    AbstractTensorHandle* upstream_grad = grad_inputs[0];
    AbstractTensorHandle* input = forward_inputs[0];
    AbstractTensorHandle* maxpool_out = forward_outputs[0];

    // Recover forward attributes.
    string pad;
    forward_attrs.Get("padding", &pad);

    vector<int64> strides_vec;
    forward_attrs.Get("strides", &strides_vec);
    int num_dims = strides_vec.size();
    int64_t* strides = (int64_t*)strides_vec.data();

    vector<int64> ksize_vec;
    forward_attrs.Get("ksize", &ksize_vec);
    int64_t* ksize = (int64_t*)ksize_vec.data();

    string data_format;
    forward_attrs.Get("data_format", &data_format);

    // Gradient for input.
    vector<AbstractTensorHandle*> maxpool_grad_outputs(1);
    std::string name = "mp_grad";
    TF_RETURN_IF_ERROR(
        MaxPoolGrad(ctx->ctx, {input, maxpool_out, upstream_grad},
                    absl::MakeSpan(maxpool_grad_outputs), num_dims, ksize,
                    strides, pad.c_str(), data_format.c_str(), name.c_str()));

    (*grad_outputs)[0] = maxpool_grad_outputs[0];

    return Status::OK();
  }
  ~MaxPoolGradientFunction() override {}

 private:
  vector<AbstractTensorHandle*> forward_inputs;
  vector<AbstractTensorHandle*> forward_outputs;
  AttrBuilder forward_attrs;
};

}  // namespace

BackwardFunction* ReluRegisterer(const ForwardOperation& op) {
  auto gradient_function = new ReluGradientFunction(op.outputs);
  // For ops with a single output, the gradient function is not called if there
  // is no incoming gradient. So we do not need to worry about creating zeros
  // grads in this case.
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

BackwardFunction* SparseSoftmaxCrossEntropyLossRegisterer(
    const ForwardOperation& op) {
  auto gradient_function =
      new SparseSoftmaxCrossEntropyLossGradientFunction(op.outputs);
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

BackwardFunction* BiasAddRegisterer(const ForwardOperation& op) {
  auto gradient_function = new BiasAddGradientFunction(op.attrs);
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

BackwardFunction* Conv2DRegisterer(const ForwardOperation& op) {
  auto gradient_function = new Conv2DGradientFunction(op.inputs, op.attrs);
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

BackwardFunction* FusedBatchNormV3Registerer(const ForwardOperation& op) {
  auto gradient_function =
      new FusedBatchNormV3GradientFunction(op.inputs, op.outputs);
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

BackwardFunction* MaxPoolRegisterer(const ForwardOperation& op) {
  auto gradient_function =
      new MaxPoolGradientFunction(op.inputs, op.outputs, op.attrs);
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

}  // namespace gradients
}  // namespace tensorflow
