/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"

namespace tensorflow::fuzzing {
namespace {

void FuzzImportGraphDef(const GraphDef& graph_def) {
  ImportGraphDefOptions options;
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  ImportGraphDef(options, graph_def, graph.get(), nullptr, nullptr)
      .IgnoreError();
}
FUZZ_TEST(GraphDefFuzz, FuzzImportGraphDef);

}  // namespace
}  // namespace tensorflow::fuzzing
