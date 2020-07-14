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
#include "tensorflow/lite/experimental/delegates/coreml/builders/dummy_op_builder.h"

#include "tensorflow/lite/experimental/delegates/coreml/builders/op_factory.h"

namespace tflite {
namespace delegates {
namespace coreml {

CoreML::Specification::NeuralNetworkLayer* DummyOpBuilder::Build() {
  return nullptr;
}

const char* DummyOpBuilder::DebugName() { return "Dummy OpBuilder"; }

TfLiteStatus DummyOpBuilder::PopulateSubgraph(TfLiteContext* context) {
  return kTfLiteOk;
}

OpBuilder* CreateDummyOpBuilder(GraphBuilder* graph_builder) {
  return new DummyOpBuilder(graph_builder);
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
