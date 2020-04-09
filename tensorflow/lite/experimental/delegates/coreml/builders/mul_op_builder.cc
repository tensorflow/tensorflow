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
#include "tensorflow/lite/experimental/delegates/coreml/builders/mul_op_builder.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/delegates/coreml/builders/activation_layer_builder.h"
#include "tensorflow/lite/experimental/delegates/coreml/builders/op_factory.h"
#include "tensorflow/lite/experimental/delegates/coreml/builders/op_validator.h"
#include "tensorflow/lite/experimental/delegates/coreml/builders/util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace coreml {
const char* MulOpBuilder::DebugName() {
  if (!str_debug_name_[0])
    GetDebugName("MulOpBuilder", node_id_, str_debug_name_);
  return str_debug_name_;
}

CoreML::Specification::NeuralNetworkLayer* MulOpBuilder::Build() {
  if (layer_ == nullptr) {
    layer_.reset(new CoreML::Specification::NeuralNetworkLayer);
  }
  // MultiplyLayerParams only has limited broadcasting support. For example:
  // [B, 1, 1, 1], [B, C, 1, 1], [B, 1, H, W], [B, C, H, W]. other shapes
  // will make broadcasting fail.
  layer_->set_name(DebugName());
  layer_->mutable_multiply();
  if (alpha_ != 1.0f) {
    layer_->mutable_multiply()->set_alpha(alpha_);
  }

  return layer_.release();
}

TfLiteStatus MulOpBuilder::PopulateSubgraph(TfLiteContext* context) {
  TfLiteMulParams* params = reinterpret_cast<TfLiteMulParams*>(builtin_data_);

  TfLiteFusedActivation activation = params->activation;
  if (activation == kTfLiteActNone) {
    builder_output_ = AddOutput();
  } else {
    ActivationLayerBuilder* activation_builder =
        reinterpret_cast<ActivationLayerBuilder*>(
            graph_builder_->AddBuilder(CreateActivationLayerBuilder, nullptr));
    activation_builder->SetActivation(activation);
    activation_builder->AddInput(AddOutput());
    activation_builder->PopulateSubgraph(context);
    builder_output_ = activation_builder->GetOutput(context);
  }
  return kTfLiteOk;
}

TfLiteStatus MulOpBuilder::RegisterInputs(const TfLiteIntArray* inputs,
                                          TfLiteContext* context) {
  // TFL MUL op always has 2 inputs.
  if (inputs->size != 2) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of inputs to mul!.");
    return kTfLiteError;
  }
  const auto* input_0 = &context->tensors[inputs->data[0]];
  const auto* input_1 = &context->tensors[inputs->data[1]];
  // store constant, scalar value into MultiplyLayerParams directly.
  if (IsConstantTensor(input_0) && NumElements(input_0) == 1) {
    AddInput(inputs->data[1]);
    SetAlpha(GetTensorData<float>(input_0)[0]);
  } else if (IsConstantTensor(input_1) && NumElements(input_1) == 1) {
    AddInput(inputs->data[0]);
    SetAlpha(GetTensorData<float>(input_1)[0]);
  } else {
    AddInput(inputs->data[0]);
    AddInput(inputs->data[1]);
  }
  return kTfLiteOk;
}

TfLiteStatus MulOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                           TfLiteContext* context) {
  if (outputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of outputs to mul!.");
    return kTfLiteError;
  }
  TensorID output_tensor = GetOutput(context);
  if (output_tensor.NodeID() == -1) {
    TF_LITE_KERNEL_LOG(context, "Failed to build output tensor.");
    return kTfLiteError;
  }
  graph_builder_->AddTensorWithID(outputs->data[0], output_tensor);
  return kTfLiteOk;
}

void MulOpBuilder::SetAlpha(float alpha) { alpha_ = alpha; }

OpBuilder* CreateMulOpBuilder(GraphBuilder* graph_builder) {
  return new MulOpBuilder(graph_builder);
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
