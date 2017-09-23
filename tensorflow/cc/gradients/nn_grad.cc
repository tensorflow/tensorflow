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
  BiasAddGrad::Attrs input_attrs;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.output(0).node()->attrs(), "data_format", &data_format));
  input_attrs.DataFormat(data_format);
  auto dx_1 = BiasAddGrad(scope, grad_inputs[0], input_attrs);
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
  Conv2DBackpropInput::Attrs input_attrs;
  input_attrs.DataFormat(data_format);
  input_attrs.UseCudnnOnGpu(use_cudnn_on_gpu);
  auto dx_1 = Conv2DBackpropInput(scope, Shape(scope, op.input(0)),
                                  op.input(1), grad_inputs[0],
                                  strides, padding, input_attrs);
  grad_outputs->push_back(dx_1);
  Conv2DBackpropFilter::Attrs filter_attrs;
  filter_attrs.DataFormat(data_format);
  filter_attrs.UseCudnnOnGpu(use_cudnn_on_gpu);
  auto dx_2 = Conv2DBackpropFilter(scope, op.input(0),
                                   Shape(scope, op.input(1)), grad_inputs[0],
                                   strides, padding, filter_attrs);
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
  internal::MaxPoolGrad::Attrs grad_attrs;
  grad_attrs.DataFormat(data_format);
  auto dx = internal::MaxPoolGrad(scope, op.input(0),
                                  op.output(0),
                                  grad_inputs[0],
                                  ksize, strides,
                                  padding, grad_attrs);
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
  MaxPoolGradV2::Attrs grad_attrs;
  grad_attrs.DataFormat(data_format);
  auto dx = MaxPoolGradV2(scope, op.input(0),
                          op.output(0),
                          grad_inputs[0],
                          op.input(1),
                          op.input(2),
                          padding,
                          grad_attrs);
  grad_outputs->push_back(dx);
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("MaxPoolV2", MaxPoolGradV2Helper);

}  // anonymous namespace
}  // namespace ops
}  // namespace tensorflow
