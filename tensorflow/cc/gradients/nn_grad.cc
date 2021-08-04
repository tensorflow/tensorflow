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
//     mat: A 2-D tensor of dimension [D0, D1]
//
//   Returns:
//     A tensor of dimension [D0, D1], the result for vec * mat.
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

Status LeakyReluGradHelper(const Scope& scope, const Operation& op,
                           const std::vector<Output>& grad_inputs,
                           std::vector<Output>* grad_outputs) {
  float alpha;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "alpha", &alpha));
  internal::LeakyReluGrad::Attrs attrs;
  auto dx = internal::LeakyReluGrad(scope, grad_inputs[0], op.input(0),
                                    attrs.Alpha(alpha));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("LeakyRelu", LeakyReluGradHelper);

Status LeakyReluGradGradHelper(const Scope& scope, const Operation& op,
                               const std::vector<Output>& grad_inputs,
                               std::vector<Output>* grad_outputs) {
  float alpha;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "alpha", &alpha));
  internal::LeakyReluGrad::Attrs attrs;
  auto dx = internal::LeakyReluGrad(scope, grad_inputs[0], op.input(1),
                                    attrs.Alpha(alpha));
  grad_outputs->push_back(dx);
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("LeakyReluGrad", LeakyReluGradGradHelper);

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

// Templated constructor for FusedBatchNormGrad[..]::Attrs.
template <typename T>
T FusedBatchNormGradAttrs(float epsilon, std::string data_format,
                          bool is_training) {
  T result;
  result.epsilon_ = epsilon;
  result.data_format_ = data_format;
  result.is_training_ = is_training;
  return result;
}

using BatchNormGradFn =
    std::function<Status(const Scope&, Output x, Output grad_y, Output scale,
                         const std::vector<Output>& reserve_spaces,
                         float epsilon, std::string data_format,
                         bool is_training, std::vector<Output>* grad_outputs)>;

Status BaseFusedBatchNormGrad(const Scope& scope, const Operation& op,
                              const std::vector<Output>& grad_inputs,
                              BatchNormGradFn grad_fn,
                              std::vector<Output>* grad_outputs) {
  if (op.num_outputs() < 5) {
    return errors::InvalidArgument(
        "FusedBatchNorm requires at least 5 outputs");
  }
  if (grad_inputs.empty()) {
    return errors::InvalidArgument("FusedBatchNorm grad requires 1 grad input");
  }
  if (op.num_inputs() < 3) {
    return errors::InvalidArgument("FusedBatchNorm has too few inputs");
  }

  Output x = op.input(0);
  Output grad_y = grad_inputs[0];
  Output scale = op.input(1);
  float epsilon;
  std::string data_format;
  bool is_training;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "epsilon", &epsilon));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "data_format", &data_format));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "is_training", &is_training));

  std::vector<Output> reserve_spaces;
  reserve_spaces.push_back(op.output(3));
  reserve_spaces.push_back(op.output(4));
  if (op.num_outputs() > 5) {
    reserve_spaces.push_back(op.output(5));
  }

  if (is_training) {
    return grad_fn(scope, x, grad_y, scale, reserve_spaces, epsilon,
                   data_format, is_training, grad_outputs);
  } else {
    if (op.num_inputs() < 5) {
      return errors::InvalidArgument(
          "FusedBatchNorm requires 5 inputs in eval mode");
    }

    reserve_spaces[0] = op.input(3);  // pop_mean
    reserve_spaces[1] = op.input(4);  // pop_var
    if (data_format == "NCHW") {
      x = Transpose(scope, x, {0, 2, 3, 1});
      grad_y = Transpose(scope, grad_y, {0, 2, 3, 1});
    } else if (data_format == "NCDHW") {
      x = Transpose(scope, x, {0, 2, 3, 4, 1});
      grad_y = Transpose(scope, grad_y, {0, 2, 3, 4, 1});
    }

    std::string target_data_format;
    if (data_format == "NCHW" || data_format == "NHWC") {
      target_data_format = "NHWC";
    } else {
      target_data_format = "NDHWC";
    }

    TF_RETURN_IF_ERROR(grad_fn(scope, x, grad_y, scale, reserve_spaces, epsilon,
                               target_data_format, is_training, grad_outputs));
    if (data_format == "NCHW") {
      (*grad_outputs)[0] = Transpose(scope, (*grad_outputs)[0], {0, 3, 1, 2});
    } else if (data_format == "NCDHW") {
      (*grad_outputs)[0] =
          Transpose(scope, (*grad_outputs)[0], {0, 4, 1, 2, 3});
    }
    return scope.status();
  }
}

Status FusedBatchNormV3Grad(const Scope& scope, const Operation& op,
                            const std::vector<Output>& grad_inputs,
                            std::vector<Output>* grad_outputs) {
  return BaseFusedBatchNormGrad(
      scope, op, grad_inputs,
      [](const Scope& scope, Output x, Output grad_y, Output scale,
         const std::vector<Output>& reserve_spaces, float epsilon,
         std::string data_format, bool is_training,
         std::vector<Output>* grad_outputs) {
        FusedBatchNormGradV3 grad(
            scope, grad_y, x, scale, reserve_spaces[0], reserve_spaces[1],
            reserve_spaces[2],
            FusedBatchNormGradAttrs<FusedBatchNormGradV3::Attrs>(
                epsilon, data_format, is_training));
        grad_outputs->push_back(grad.x_backprop);
        grad_outputs->push_back(grad.scale_backprop);
        grad_outputs->push_back(grad.offset_backprop);
        grad_outputs->push_back(NoGradient());
        grad_outputs->push_back(NoGradient());
        return scope.status();
      },
      grad_outputs);
}

REGISTER_GRADIENT_OP("FusedBatchNormV3", FusedBatchNormV3Grad);

Status Conv2DBackpropInputGrad(const Scope& scope, const Operation& op,
                               const std::vector<Output>& grad_inputs,
                               std::vector<Output>* grad_outputs) {
  if (op.num_inputs() != 3) {
    return errors::InvalidArgument("Conv2DBackpropInput requires 3 inputs.");
  }
  if (grad_inputs.empty()) {
    return errors::InvalidArgument(
        "Conv2DBackpropInput grad requires 1 grad input");
  }

  std::vector<int> dilations, strides, explicit_paddings;
  bool use_cudnn_on_gpu;
  std::string data_format, padding;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "dilations", &dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "strides", &strides));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "explicit_paddings", &explicit_paddings));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "use_cudnn_on_gpu", &use_cudnn_on_gpu));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "padding", &padding));

  grad_outputs->push_back(NoGradient());

  Conv2DBackpropFilter::Attrs filter_attrs;
  filter_attrs.use_cudnn_on_gpu_ = use_cudnn_on_gpu;
  filter_attrs.explicit_paddings_ = explicit_paddings;
  filter_attrs.data_format_ = data_format;
  filter_attrs.dilations_ = dilations;
  grad_outputs->push_back(
      Conv2DBackpropFilter(scope, grad_inputs[0], Shape(scope, op.input(1)),
                           op.input(2), strides, padding, filter_attrs));

  Conv2D::Attrs conv_attrs;
  conv_attrs.use_cudnn_on_gpu_ = use_cudnn_on_gpu;
  conv_attrs.explicit_paddings_ = explicit_paddings;
  conv_attrs.data_format_ = data_format;
  conv_attrs.dilations_ = dilations;
  grad_outputs->push_back(
      Conv2D(scope, grad_inputs[0], op.input(1), strides, padding, conv_attrs));
  return scope.status();
}
REGISTER_GRADIENT_OP("Conv2DBackpropInput", Conv2DBackpropInputGrad);

Status DepthwiseConv2dNativeGrad(const Scope& scope, const Operation& op,
                                 const std::vector<Output>& grad_inputs,
                                 std::vector<Output>* grad_outputs) {
  if (op.num_inputs() != 2) {
    return errors::InvalidArgument("DepthwiseConv2dNative requires 2 inputs.");
  }
  if (grad_inputs.empty()) {
    return errors::InvalidArgument(
        "DepthwiseConv2dNative grad requires 1 grad input");
  }

  std::vector<int> dilations, strides, explicit_paddings;
  std::string data_format, padding;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "dilations", &dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "strides", &strides));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "explicit_paddings", &explicit_paddings));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "padding", &padding));

  DepthwiseConv2dNativeBackpropInput::Attrs input_attrs;
  input_attrs.explicit_paddings_ = explicit_paddings;
  input_attrs.data_format_ = data_format;
  input_attrs.dilations_ = dilations;
  grad_outputs->push_back(DepthwiseConv2dNativeBackpropInput(
      scope, Shape(scope, op.input(0)), op.input(1), grad_inputs[0], strides,
      padding, input_attrs));

  DepthwiseConv2dNativeBackpropFilter::Attrs filter_attrs;
  filter_attrs.explicit_paddings_ = explicit_paddings;
  filter_attrs.data_format_ = data_format;
  filter_attrs.dilations_ = dilations;
  grad_outputs->push_back(DepthwiseConv2dNativeBackpropFilter(
      scope, op.input(0), Shape(scope, op.input(1)), grad_inputs[0], strides,
      padding, filter_attrs));
  return scope.status();
}
REGISTER_GRADIENT_OP("DepthwiseConv2dNative", DepthwiseConv2dNativeGrad);

}  // anonymous namespace
}  // namespace ops
}  // namespace tensorflow
