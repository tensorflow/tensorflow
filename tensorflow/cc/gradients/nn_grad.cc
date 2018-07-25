/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"

namespace tensorflow {
namespace ops {
namespace {

Status SoftmaxGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
  // Softmax gradient function.
  // p = softmax(x) maps from [batch, n] to [batch, m]
  // dp/dx = [dp0/dx0   ... dp0/dxn-1  ]
  //         [  ...           ...      ]
  //         [dpm-1/dx0 ... dpm-1/dxn-1]
  // dL/dx = dp/dx * dL/dy
  //
  // Using alternative formula:
  // dL/dx = dL/dy * y - sum(dL/dy * y) * y
  //    = (dL/dy - sum(dL/dy * y)) * y
  auto y = op.output(0);
  auto dyy = Mul(scope, grad_inputs[0], y);
  auto sum = Reshape(scope, Sum(scope, dyy, {1}), {-1, 1});
  auto sub = Sub(scope, grad_inputs[0], sum);
  auto dx = Mul(scope, sub, y);
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Softmax", SoftmaxGrad);

bool IsZero(const Scope& scope, const Output& grad) {
  string op_type_name = grad.op().node()->type_string();
  if (op_type_name == "ZerosLike" || op_type_name == "Zeros") {
    return true;
  }
  // The Operation we were provided is not named something obvious so
  // we need to actually look at its contents.
  // The original python code did this by calling a utility function called
  // tensor_util.constant_value.
  // There is no C++ equivalent to tensor_util.constant_value so we do nothing
  // for the moment.
  return false;
}

// Multiply after broadcasting vec to match dimensions of mat.
//   Args:
//     vec: A 1-D tensor of dimension [D0]
//     mat: A 2-D tensor of dimesnion [D0, D1]
//
//   Returns:
//     A tensor of dimension [D0, D1], the result fo vec * mat.
Output BroadcastMul(const Scope& scope, const Output& vec, const Output& mat) {
  auto reshaped = ExpandDims(scope, vec, -1);
  return Multiply(scope, reshaped, mat);
}

Status SoftmaxCrossEntropyWithLogitsGrad(const Scope& scope,
                                         const Operation& op,
                                         const std::vector<Output>& grad_inputs,
                                         std::vector<Output>* grad_outputs) {
  // Softmax gradient with cross entropy logits function.
  // We multiply the backprop for cost with the gradients - op.output[1].
  // There is no gradient for labels.

  // The outputs of the network are at input index 0.
  auto logits = op.input(0);
  // The "truth" labels are at index 1.
  auto softmax_grad = op.output(1);

  // The loss is the output at index 0, and backprop is the output at index 1.
  auto grad_loss = grad_inputs[0];
  auto grad_grad = grad_inputs[1];

  auto grad = BroadcastMul(scope, grad_loss, softmax_grad);
  if (!IsZero(scope, grad_grad)) {
    std::vector<int> axis;
    auto logits_softmax = Softmax(scope, logits);

    auto grad_grad_expand = ExpandDims(scope, grad_grad, 1);
    auto logits_softmax_expand = ExpandDims(scope, logits_softmax, 2);
    auto matmul_result =
        BatchMatMul(scope, grad_grad_expand, logits_softmax_expand);
    axis.push_back(1);
    auto squeeze_result = Squeeze(scope, matmul_result, Squeeze::Axis(axis));
    auto subtraction_result = Subtract(scope, grad_grad, squeeze_result);
    auto multiply_result = Multiply(scope, subtraction_result, logits_softmax);
    grad = Add(scope, grad, multiply_result);
  }
  auto minus_log_softmax = Multiply(scope, LogSoftmax(scope, logits), -1.0f);
  grad_outputs->push_back(grad);
  grad_outputs->push_back(BroadcastMul(scope, grad_loss, minus_log_softmax));
  return scope.status();
}
REGISTER_GRADIENT_OP("SoftmaxCrossEntropyWithLogits",
                     SoftmaxCrossEntropyWithLogitsGrad);

Status LogSoftmaxGrad(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
  auto softmax = Exp(scope, op.output(0));
  auto sum = Sum(scope, grad_inputs[0], {1}, Sum::KeepDims(true));
  auto mul = Mul(scope, sum, softmax);
  auto dx = Sub(scope, grad_inputs[0], mul);
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("LogSoftmax", LogSoftmaxGrad);

Status ReluGradHelper(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
  auto dx = internal::ReluGrad(scope, grad_inputs[0], op.input(0));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Relu", ReluGradHelper);

Status Relu6GradHelper(const Scope& scope, const Operation& op,
                       const std::vector<Output>& grad_inputs,
                       std::vector<Output>* grad_outputs) {
  auto dx = internal::Relu6Grad(scope, grad_inputs[0], op.input(0));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Relu6", Relu6GradHelper);

Status EluGradHelper(const Scope& scope, const Operation& op,
                     const std::vector<Output>& grad_inputs,
                     std::vector<Output>* grad_outputs) {
  auto dx = internal::EluGrad(scope, grad_inputs[0], op.output(0));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Elu", EluGradHelper);

Status SeluGradHelper(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
  auto dx = internal::SeluGrad(scope, grad_inputs[0], op.output(0));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Selu", SeluGradHelper);

Status L2LossGrad(const Scope& scope, const Operation& op,
                  const std::vector<Output>& grad_inputs,
                  std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(Mul(scope, op.input(0), grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("L2Loss", L2LossGrad);

Status BiasAddGradHelper(const Scope& scope, const Operation& op,
                         const std::vector<Output>& grad_inputs,
                         std::vector<Output>* grad_outputs) {
  string data_format;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.output(0).node()->attrs(), "data_format", &data_format));
  auto dx_1 =
      BiasAddGrad(scope, grad_inputs[0], BiasAddGrad::DataFormat(data_format));
  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  grad_outputs->push_back(dx_1);
  return scope.status();
}
REGISTER_GRADIENT_OP("BiasAdd", BiasAddGradHelper);

Status Conv2DGrad(const Scope& scope, const Operation& op,
                  const std::vector<Output>& grad_inputs,
                  std::vector<Output>* grad_outputs) {
  string data_format;
  string padding;
  std::vector<int32> strides;
  bool use_cudnn_on_gpu;
  auto attrs = op.output(0).node()->attrs();
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "padding", &padding));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "strides", &strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "use_cudnn_on_gpu", &use_cudnn_on_gpu));
  auto dx_1 = Conv2DBackpropInput(scope, Shape(scope, op.input(0)), op.input(1),
                                  grad_inputs[0], strides, padding,
                                  Conv2DBackpropInput::DataFormat(data_format)
                                      .UseCudnnOnGpu(use_cudnn_on_gpu));
  grad_outputs->push_back(dx_1);
  auto dx_2 =
      Conv2DBackpropFilter(scope, op.input(0), Shape(scope, op.input(1)),
                           grad_inputs[0], strides, padding,
                           Conv2DBackpropFilter::DataFormat(data_format)
                               .UseCudnnOnGpu(use_cudnn_on_gpu));
  grad_outputs->push_back(dx_2);
  return scope.status();
}
REGISTER_GRADIENT_OP("Conv2D", Conv2DGrad);

Status MaxPoolGradHelper(const Scope& scope, const Operation& op,
                         const std::vector<Output>& grad_inputs,
                         std::vector<Output>* grad_outputs) {
  string data_format;
  string padding;
  std::vector<int32> strides;
  std::vector<int32> ksize;
  auto attrs = op.output(0).node()->attrs();
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "ksize", &ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "padding", &padding));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "strides", &strides));
  auto dx = internal::MaxPoolGrad(
      scope, op.input(0), op.output(0), grad_inputs[0], ksize, strides, padding,
      internal::MaxPoolGrad::DataFormat(data_format));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("MaxPool", MaxPoolGradHelper);

Status MaxPoolGradV2Helper(const Scope& scope, const Operation& op,
                           const std::vector<Output>& grad_inputs,
                           std::vector<Output>* grad_outputs) {
  string data_format;
  string padding;
  auto attrs = op.output(0).node()->attrs();
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "padding", &padding));
  auto dx = MaxPoolGradV2(scope, op.input(0), op.output(0), grad_inputs[0],
                          op.input(1), op.input(2), padding,
                          MaxPoolGradV2::DataFormat(data_format));
  grad_outputs->push_back(dx);
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("MaxPoolV2", MaxPoolGradV2Helper);

Status MaxPool3DGradHelper(const Scope& scope, const Operation& op,
                           const std::vector<Output>& grad_inputs,
                           std::vector<Output>* grad_outputs) {
  std::vector<int32> ksize;
  std::vector<int32> strides;
  string padding;
  string data_format;
  auto attrs = op.output(0).node()->attrs();
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "ksize", &ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "strides", &strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "padding", &padding));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  MaxPool3DGrad::Attrs grad_attrs;
  auto dx =
      MaxPool3DGrad(scope, op.input(0), op.output(0), grad_inputs[0], ksize,
                    strides, padding, grad_attrs.DataFormat(data_format));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("MaxPool3D", MaxPool3DGradHelper);

Status AvgPoolGradHelper(const Scope& scope, const Operation& op,
                         const std::vector<Output>& grad_inputs,
                         std::vector<Output>* grad_outputs) {
  std::vector<int32> ksize;
  std::vector<int32> strides;
  string padding;
  string data_format;
  auto attrs = op.output(0).node()->attrs();
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "ksize", &ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "strides", &strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "padding", &padding));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  internal::AvgPoolGrad::Attrs grad_attrs;
  auto dx = internal::AvgPoolGrad(scope, Shape(scope, op.input(0)),
                                  grad_inputs[0], ksize, strides, padding,
                                  grad_attrs.DataFormat(data_format));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("AvgPool", AvgPoolGradHelper);

Status AvgPool3DGradHelper(const Scope& scope, const Operation& op,
                           const std::vector<Output>& grad_inputs,
                           std::vector<Output>* grad_outputs) {
  std::vector<int32> ksize;
  std::vector<int32> strides;
  string padding;
  string data_format;
  auto attrs = op.output(0).node()->attrs();
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "ksize", &ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "strides", &strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "padding", &padding));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  AvgPool3DGrad::Attrs grad_attrs;
  auto dx =
      AvgPool3DGrad(scope, Shape(scope, op.input(0)), grad_inputs[0], ksize,
                    strides, padding, grad_attrs.DataFormat(data_format));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("AvgPool3D", AvgPool3DGradHelper);

Status LRNGradHelper(const Scope& scope, const Operation& op,
                     const std::vector<Output>& grad_inputs,
                     std::vector<Output>* grad_outputs) {
  auto dx = internal::LRNGrad(scope, grad_inputs[0], op.input(0), op.output(0));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("LRN", LRNGradHelper);

Status SoftplusGradHelper(const Scope& scope, const Operation& op,
                          const std::vector<Output>& grad_inputs,
                          std::vector<Output>* grad_outputs) {
  auto dx = internal::SoftplusGrad(scope, grad_inputs[0], op.input(0));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Softplus", SoftplusGradHelper);

Status SoftsignGradHelper(const Scope& scope, const Operation& op,
                          const std::vector<Output>& grad_inputs,
                          std::vector<Output>* grad_outputs) {
  auto dx = internal::SoftsignGrad(scope, grad_inputs[0], op.input(0));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Softsign", SoftsignGradHelper);

Status FractionalAvgPoolGradHelper(const Scope& scope, const Operation& op,
                                   const std::vector<Output>& grad_inputs,
                                   std::vector<Output>* grad_outputs) {
  bool overlapping;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.output(0).node()->attrs(), "overlapping", &overlapping));
  auto dx = internal::FractionalAvgPoolGrad(
      scope, Shape(scope, op.input(0), Shape::OutType(DT_INT64)),
      grad_inputs[0], op.output(1), op.output(2),
      internal::FractionalAvgPoolGrad::Overlapping(overlapping));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("FractionalAvgPool", FractionalAvgPoolGradHelper);

Status FractionalMaxPoolGradHelper(const Scope& scope, const Operation& op,
                                   const std::vector<Output>& grad_inputs,
                                   std::vector<Output>* grad_outputs) {
  bool overlapping;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.output(0).node()->attrs(), "overlapping", &overlapping));
  auto dx = internal::FractionalMaxPoolGrad(
      scope, op.input(0), op.output(0), grad_inputs[0], op.output(1),
      op.output(2), internal::FractionalMaxPoolGrad::Overlapping(overlapping));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("FractionalMaxPool", FractionalMaxPoolGradHelper);

}  // anonymous namespace
}  // namespace ops
}  // namespace tensorflow
