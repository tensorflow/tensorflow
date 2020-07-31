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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_COREML_BUILDERS_CONCATENATION_OP_BUILDER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_COREML_BUILDERS_CONCATENATION_OP_BUILDER_H_

#include "tensorflow/lite/experimental/delegates/coreml/builders/op_builder.h"

namespace tflite {
namespace delegates {
namespace coreml {

class ConcatenationOpBuilder : public OpBuilder {
 public:
  explicit ConcatenationOpBuilder(GraphBuilder* graph_builder)
      : OpBuilder(graph_builder) {}

  const char* DebugName() override {
    if (!str_debug_name_[0])
      GetDebugName("ConcatOpBuilder", node_id_, str_debug_name_);
    return str_debug_name_;
  }

  CoreML::Specification::NeuralNetworkLayer* Build() override;

  TfLiteStatus RegisterInputs(const TfLiteIntArray* inputs,
                              TfLiteContext* context) override;

  TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                               TfLiteContext* context) override;
};

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_COREML_BUILDERS_CONCATENATION_OP_BUILDER_H_
