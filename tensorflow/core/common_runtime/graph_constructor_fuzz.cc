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
#include "tensorflow/core/public/session.h"

namespace tensorflow::fuzzing {
namespace {

void FuzzImportGraphDef(const GraphDef& graph_def) {
  ImportGraphDefOptions options;
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  ImportGraphDef(options, graph_def, graph.get(), nullptr, nullptr)
      .IgnoreError();
}
FUZZ_TEST(GraphDefFuzz, FuzzImportGraphDef);

void FuzzGraphEndToEndSimpleFixedInput(const GraphDef& graph_def) {
  // Load an arbitrary graph and run a session on it using simple input.
  ImportGraphDefOptions options;
  std::unique_ptr<Graph> graph = std::make_unique<Graph>(OpRegistry::Global());
  Status status =
      ImportGraphDef(options, graph_def, graph.get(), nullptr, nullptr);
  if (!status.ok()) {
    return;
  }
  GraphDef gdef;
  graph->ToGraphDef(&gdef);
  SessionOptions sess_options;
  std::unique_ptr<Session> sess =
      std::unique_ptr<Session>(NewSession(sess_options));
  status = sess.get()->Create(gdef);
  if (!status.ok()) {
    return;
  }

  // Use the same input for each fuzz iteration. The benefit of this is the
  // fuzzer will focus on exploring graphs that match this input, which
  // gives it a more narrow space to search, as opposed to any given input.
  Tensor p1(DT_FLOAT, TensorShape({1}));
  p1.scalar<float>()() = 1.0;
  Tensor p2(DT_FLOAT, TensorShape({1}));
  p2.scalar<float>()() = 2.0;
  std::vector<std::pair<string, Tensor>> inputs = {{"Placeholder", p1},
                                                   {"Placeholder_1", p2}};
  std::vector<string> output_names = {"O_FUZZ"};
  std::vector<string> target_names;
  std::vector<Tensor> outputs;
  status = sess->Run(inputs, output_names, target_names, &outputs);
}
FUZZ_TEST(GraphDefFuzz, FuzzGraphEndToEndSimpleFixedInput);

}  // namespace
}  // namespace tensorflow::fuzzing
