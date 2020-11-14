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
#include "tensorflow/lite/experimental/delegates/coreml/builders/threshold_layer_builder.h"

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace delegates {
namespace coreml {

const char* ThresholdLayerBuilder::DebugName() {
  if (!str_debug_name_[0])
    GetDebugName("ThresholdLayerBuilder", node_id_, str_debug_name_);
  return str_debug_name_;
}

CoreML::Specification::NeuralNetworkLayer* ThresholdLayerBuilder::Build() {
  if (layer_ == nullptr) {
    layer_.reset(new CoreML::Specification::NeuralNetworkLayer);
  }
  layer_->set_name(DebugName());
  layer_->mutable_unary()->set_alpha(alpha_);
  layer_->mutable_unary()->set_scale(scale_);
  layer_->mutable_unary()->set_type(
      CoreML::Specification::UnaryFunctionLayerParams::THRESHOLD);
  return layer_.release();
}

TfLiteStatus ThresholdLayerBuilder::RegisterInputs(const TfLiteIntArray* inputs,
                                                   TfLiteContext* context) {
  if (inputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Threshold: Wrong # of inputs!.");
    return kTfLiteError;
  }
  AddInput(inputs->data[0]);
  return kTfLiteOk;
}

TfLiteStatus ThresholdLayerBuilder::RegisterOutputs(
    const TfLiteIntArray* outputs, TfLiteContext* context) {
  if (outputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Threshold: Wrong # of outputs!.");
    return kTfLiteError;
  }
  graph_builder_->AddTensorWithID(outputs->data[0], GetOutput(context));
  return kTfLiteOk;
}

OpBuilder* CreateThresholdLayerBuilder(GraphBuilder* graph_builder) {
  return new ThresholdLayerBuilder(graph_builder);
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
