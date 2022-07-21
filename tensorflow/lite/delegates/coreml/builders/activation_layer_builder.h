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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_COREML_BUILDERS_ACTIVATION_LAYER_BUILDER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_COREML_BUILDERS_ACTIVATION_LAYER_BUILDER_H_

#include <string>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/coreml/builders/op_builder.h"

namespace tflite {
namespace delegates {
namespace coreml {

class ActivationLayerBuilder : public OpBuilder {
 public:
  explicit ActivationLayerBuilder(GraphBuilder* graph_builder)
      : OpBuilder(graph_builder) {}

  explicit ActivationLayerBuilder(GraphBuilder* graph_builder,
                                  TfLiteFusedActivation activation)
      : OpBuilder(graph_builder), activation_(activation) {}

  const std::string& DebugName() override;

  CoreML::Specification::NeuralNetworkLayer* Build() override;

  void SetActivation(TfLiteFusedActivation activation) {
    activation_ = activation;
  }

  void SetAlpha(float alpha) { alpha_ = alpha; }

  TfLiteStatus PopulateSubgraph(TfLiteContext* context) override;

  TfLiteStatus RegisterInputs(const TfLiteIntArray* inputs,
                              TfLiteContext* context) override;

  TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                               TfLiteContext* context) override;

 private:
  TfLiteFusedActivation activation_;
  float alpha_ = 1.0f;
};

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_COREML_BUILDERS_ACTIVATION_LAYER_BUILDER_H_
