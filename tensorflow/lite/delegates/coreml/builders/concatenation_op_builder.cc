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
#include "tensorflow/lite/delegates/coreml/builders/concatenation_op_builder.h"

#include <memory>

#include "mlmodel/format/NeuralNetwork.pb.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/coreml/builders/op_builder.h"
#include "tensorflow/lite/delegates/coreml/builders/op_validator.h"

namespace tflite {
namespace delegates {
namespace coreml {

CoreML::Specification::NeuralNetworkLayer* ConcatenationOpBuilder::Build() {
  if (layer_ == nullptr) {
    layer_ = std::make_unique<CoreML::Specification::NeuralNetworkLayer>();
  }
  layer_->set_name(DebugName());
  layer_->mutable_concat()->set_sequenceconcat(false);
  return layer_.release();
}

TfLiteStatus ConcatenationOpBuilder::RegisterInputs(
    const TfLiteIntArray* inputs, TfLiteContext* context) {
  if (inputs->size < 2) {
    TF_LITE_KERNEL_LOG(
        context, "ConcatenationOpBuilder: at least 2 inputs are required.");
    return kTfLiteError;
  }
  for (int i = 0; i < inputs->size; ++i) {
    AddInput(inputs->data[i]);
  }
  return kTfLiteOk;
}

TfLiteStatus ConcatenationOpBuilder::RegisterOutputs(
    const TfLiteIntArray* outputs, TfLiteContext* context) {
  if (outputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of outputs to Concat!.");
    return kTfLiteError;
  }
  graph_builder_->AddTensorWithID(outputs->data[0], GetOutput(context));
  return kTfLiteOk;
}

OpBuilder* CreateConcatenationOpBuilder(GraphBuilder* graph_builder) {
  return new ConcatenationOpBuilder(graph_builder);
}

bool IsConcatenationOpSupported(const TfLiteRegistration* registration,
                                const TfLiteNode* node,
                                TfLiteContext* context) {
  if (node->builtin_data == nullptr) return false;
  auto params =
      reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data);
  int input_dims = context->tensors[node->inputs->data[0]].dims->size;

  // Not supported in TfLite kernel.
  if (params->activation != kTfLiteActNone) return false;
  if (node->inputs->size < 2) return false;

  // Only supports concatenation by channel. Core ML concatenation supports
  // concatenation by channel and by sequence (axis -5) only.
  // TODO(b/145642128): support stack layer here with Core ML 3 support.
  if (input_dims == 3) return params->axis == 2 || params->axis == -1;
  if (input_dims == 4) return params->axis == 3 || params->axis == -1;
  return false;
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
