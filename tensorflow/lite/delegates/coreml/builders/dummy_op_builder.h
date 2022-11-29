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
#ifndef TENSORFLOW_LITE_DELEGATES_COREML_BUILDERS_DUMMY_OP_BUILDER_H_
#define TENSORFLOW_LITE_DELEGATES_COREML_BUILDERS_DUMMY_OP_BUILDER_H_

#include <string>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/coreml/builders/op_builder.h"

namespace tflite {
namespace delegates {
namespace coreml {

// Dummy Opbuilder for nodes that are claimed but not used. ex) FP16 dequantize
class DummyOpBuilder : public OpBuilder {
 public:
  explicit DummyOpBuilder(GraphBuilder* graph_builder)
      : OpBuilder(graph_builder) {}
  CoreML::Specification::NeuralNetworkLayer* Build() override;
  TfLiteStatus PopulateSubgraph(TfLiteContext* context) override;
  const std::string& DebugName() override;

  TfLiteStatus RegisterInputs(const TfLiteIntArray* inputs,
                              TfLiteContext* context) override;

  TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                               TfLiteContext* context) override;
};

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_COREML_BUILDERS_DUMMY_OP_BUILDER_H_
