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
#include "tensorflow/lite/delegates/coreml/builders/softmax_op_builder.h"

#include <memory>
#include <string>

#include "mlmodel/format/NeuralNetwork.pb.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/coreml/builders/op_builder.h"

namespace tflite {
namespace delegates {
namespace coreml {
const std::string& SoftmaxOpBuilder::DebugName() {
  if (debug_name_.empty()) SetDebugName("SoftmaxOpBuilder", node_id_);
  return debug_name_;
}

CoreML::Specification::NeuralNetworkLayer* SoftmaxOpBuilder::Build() {
  if (layer_ == nullptr) {
    layer_ = std::make_unique<CoreML::Specification::NeuralNetworkLayer>();
  }
  layer_->set_name(DebugName());
  layer_->mutable_softmax();

  return layer_.release();
}

TfLiteStatus SoftmaxOpBuilder::RegisterInputs(const TfLiteIntArray* inputs,
                                              TfLiteContext* context) {
  if (inputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of inputs to softmax!.");
    return kTfLiteError;
  }
  AddInput(inputs->data[0]);
  return kTfLiteOk;
}

TfLiteStatus SoftmaxOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                               TfLiteContext* context) {
  if (outputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of outputs to softmax!.");
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

OpBuilder* CreateSoftmaxOpBuilder(GraphBuilder* graph_builder) {
  return new SoftmaxOpBuilder(graph_builder);
}
}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
