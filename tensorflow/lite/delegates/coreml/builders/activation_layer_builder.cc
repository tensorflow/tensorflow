/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/coreml/builders/activation_layer_builder.h"

#include <string>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/coreml/builders/op_factory.h"
#include "tensorflow/lite/delegates/coreml/builders/threshold_layer_builder.h"

namespace tflite {
namespace delegates {
namespace coreml {

const std::string& ActivationLayerBuilder::DebugName() {
  if (debug_name_.empty()) SetDebugName("ActivationLayerBuilder", node_id_);
  return debug_name_;
}

CoreML::Specification::NeuralNetworkLayer* ActivationLayerBuilder::Build() {
  layer_->set_name(DebugName());
  switch (activation_) {
    // ActNone is used for sclalar multiplication (linear activation)
    case kTfLiteActNone:
      layer_->mutable_activation()->mutable_linear()->set_alpha(alpha_);
      break;
    case kTfLiteActRelu:
      layer_->mutable_activation()->mutable_relu();
      break;
    // Relu1 and Relu6 layers are fully composed in PopulateSubgraph().
    case kTfLiteActReluN1To1:  // clip(-1, 1)
      layer_->mutable_unary()->set_alpha(-1);
      layer_->mutable_unary()->set_type(
          CoreML::Specification::UnaryFunctionLayerParams::THRESHOLD);
      break;
    case kTfLiteActRelu6:  // clip(0, 6)
      layer_->mutable_activation()->mutable_relu();
      break;
    case kTfLiteActTanh:
      layer_->mutable_activation()->mutable_tanh();
      break;
    case kTfLiteActSigmoid:
      layer_->mutable_activation()->mutable_sigmoid();
      break;
    // TODO(taeheej): signbit is not implemented.
    default:
      fprintf(stderr, "Activation %d is not supported.\n", activation_);
      break;
  }
  return layer_.release();
}

TfLiteStatus ActivationLayerBuilder::PopulateSubgraph(TfLiteContext* context) {
  if (!(activation_ == kTfLiteActRelu6 || activation_ == kTfLiteActReluN1To1)) {
    builder_output_ = AddOutput();
    return kTfLiteOk;
  }

  // Relu1: Threshold(-1) -> Threshold(-1) with scale: -1 -> Negation
  // Relu6: ReLU -> Threshold(-6) with scale: -1 -> Negation
  const int relu_threshold = activation_ == kTfLiteActRelu6 ? 6 : 1;
  ThresholdLayerBuilder* threshold_builder =
      reinterpret_cast<ThresholdLayerBuilder*>(
          graph_builder_->AddBuilder(CreateThresholdLayerBuilder, nullptr));

  threshold_builder->SetAlpha(-relu_threshold);
  threshold_builder->SetScale(-1);

  threshold_builder->AddInput(AddOutput());

  ActivationLayerBuilder* negation_builder =
      reinterpret_cast<ActivationLayerBuilder*>(
          graph_builder_->AddBuilder(CreateActivationLayerBuilder, nullptr));
  negation_builder->SetActivation(kTfLiteActNone);
  negation_builder->SetAlpha(-1);

  negation_builder->AddInput(threshold_builder->AddOutput());
  builder_output_ = negation_builder->AddOutput();
  return kTfLiteOk;
}

TfLiteStatus ActivationLayerBuilder::RegisterInputs(
    const TfLiteIntArray* inputs, TfLiteContext* context) {
  if (inputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Activation: Wrong # of inputs!.");
    return kTfLiteError;
  }
  AddInput(inputs->data[0]);
  return kTfLiteOk;
}

TfLiteStatus ActivationLayerBuilder::RegisterOutputs(
    const TfLiteIntArray* outputs, TfLiteContext* context) {
  if (outputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Activation: Wrong # of outputs!.");
    return kTfLiteError;
  }
  graph_builder_->AddTensorWithID(outputs->data[0], GetOutput(context));
  return kTfLiteOk;
}

OpBuilder* CreateActivationLayerBuilder(GraphBuilder* graph_builder) {
  return new ActivationLayerBuilder(graph_builder);
}

OpBuilder* CreateLogisticOpBuilder(GraphBuilder* graph_builder) {
  return new ActivationLayerBuilder(graph_builder, kTfLiteActSigmoid);
}

OpBuilder* CreateReluOpBuilder(GraphBuilder* graph_builder) {
  return new ActivationLayerBuilder(graph_builder, kTfLiteActRelu);
}

OpBuilder* CreateReluN1To1OpBuilder(GraphBuilder* graph_builder) {
  return new ActivationLayerBuilder(graph_builder, kTfLiteActReluN1To1);
}

OpBuilder* CreateRelu6OpBuilder(GraphBuilder* graph_builder) {
  return new ActivationLayerBuilder(graph_builder, kTfLiteActRelu6);
}

OpBuilder* CreateTanhOpBuilder(GraphBuilder* graph_builder) {
  return new ActivationLayerBuilder(graph_builder, kTfLiteActTanh);
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
