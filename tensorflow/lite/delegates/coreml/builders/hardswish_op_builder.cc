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
#include "tensorflow/lite/delegates/coreml/builders/hardswish_op_builder.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/coreml/builders/add_op_builder.h"
#include "tensorflow/lite/delegates/coreml/builders/mul_op_builder.h"
#include "tensorflow/lite/delegates/coreml/builders/op_factory.h"

namespace tflite {
namespace delegates {
namespace coreml {
const std::string& HardSwishOpBuilder::DebugName() {
  if (debug_name_.empty()) SetDebugName("HardSwishOpBuilder", node_id_);
  return debug_name_;
}

CoreML::Specification::NeuralNetworkLayer* HardSwishOpBuilder::Build() {
  layer_->set_name(DebugName());
  layer_->mutable_multiply()->set_alpha(1.0f / 6.0f);

  return layer_.release();
}

TfLiteStatus HardSwishOpBuilder::PopulateSubgraph(TfLiteContext* context) {
  // hswish(x) = (x/6) * ReLU6(x+3). main layer_ contains the first part, x/6.
  // ReLU6(x +3) constructed as add op with fused ReLU6 activation.
  AddOpBuilder* add_builder = reinterpret_cast<AddOpBuilder*>(
      graph_builder_->AddBuilder(CreateAddOpBuilder, nullptr));
  TfLiteAddParams add_param{kTfLiteActRelu6};
  add_builder->SetBuiltinData(&add_param);
  add_builder->SetAlpha(3.0f);
  add_builder->AddInput(layer_->input(0));
  add_builder->PopulateSubgraph(context);

  // multiplies (x/6) from main layer_ and ReLU6(x+3) from the above code.
  MulOpBuilder* mul_builder = reinterpret_cast<MulOpBuilder*>(
      graph_builder_->AddBuilder(CreateMulOpBuilder, nullptr));
  mul_builder->AddInput(AddOutput());
  mul_builder->AddInput(add_builder->GetOutput(context));
  builder_output_ = mul_builder->AddOutput();
  return kTfLiteOk;
}

TfLiteStatus HardSwishOpBuilder::RegisterInputs(const TfLiteIntArray* inputs,
                                                TfLiteContext* context) {
  if (inputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of inputs to hardswish!.");
    return kTfLiteError;
  }
  AddInput(inputs->data[0]);
  return kTfLiteOk;
}

TfLiteStatus HardSwishOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                                 TfLiteContext* context) {
  if (outputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of outputs to hardswish!.");
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

OpBuilder* CreateHardSwishOpBuilder(GraphBuilder* graph_builder) {
  return new HardSwishOpBuilder(graph_builder);
}
}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
