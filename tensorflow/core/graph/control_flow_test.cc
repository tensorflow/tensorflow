/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/control_flow.h"

#include <string>
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/while_loop.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {
Status LessThanTenCond(const Scope& scope, const std::vector<Output>& inputs,
                       Output* output) {
  *output = ops::Less(scope, inputs[0], 10);
  return scope.status();
}

Status AddOneBody(const Scope& scope, const std::vector<Output>& inputs,
                  std::vector<Output>* outputs) {
  outputs->push_back(ops::AddN(scope, {inputs[0], 1}));
  return scope.status();
}

Status NestedLoopBody(const Scope& scope, const std::vector<Output>& inputs,
                      std::vector<Output>* outputs) {
  return ops::BuildWhileLoop(scope.NewSubScope("inner"), inputs,
                             LessThanTenCond, AddOneBody, "inner_loop",
                             outputs);
}

TEST(ValidateControlFlowTest, InputsFromDifferentFrames) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  std::vector<Output> inputs;
  inputs.push_back(ops::Placeholder(scope, DT_INT32));
  std::vector<Output> outputs;
  TF_ASSERT_OK(ops::BuildWhileLoop(scope.NewSubScope("outer"), inputs,
                                   LessThanTenCond, NestedLoopBody,
                                   "outer_loop", &outputs));
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));
  // {inner/Enter', 'outer/Switch'} --> 'inner/Merge'. 'inner/Enter' is in frame
  // 'inner_loop'. 'outer/Switch' is in frame 'outer_loop'.
  std::vector<ControlFlowInfo> info;
  Status status = BuildControlFlowInfo(graph.get(), &info);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.error_message(),
                                "has inputs from different frames"))
      << status.error_message();
  EXPECT_TRUE(absl::StrContains(status.error_message(),
                                "{{node outer/body/inner/Merge}}"))
      << status.error_message();
  EXPECT_TRUE(absl::StrContains(status.error_message(),
                                "{{node outer/body/inner/Enter}}"))
      << status.error_message();
  EXPECT_TRUE(
      absl::StrContains(status.error_message(), "{{node outer/Switch}}"))
      << status.error_message();
}

TEST(ValidateControlFlowTest, MismatchedParentFrames) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  std::vector<Output> inputs;
  inputs.push_back(ops::Placeholder(scope, DT_INT32));
  std::vector<Output> outputs;
  TF_ASSERT_OK(ops::BuildWhileLoop(scope, inputs, LessThanTenCond, AddOneBody,
                                   "test_loop", &outputs));
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));
  Node* enter_1 = nullptr;
  for (Node* node : graph->op_nodes()) {
    if (IsEnter(node)) {
      enter_1 = node;
    }
  }
  ASSERT_TRUE(enter_1 != nullptr);

  NodeDef enter;
  enter.set_name("Enter2");
  enter.set_op("Enter");
  (*enter.mutable_attr())["T"].set_type(DT_INT32);
  (*enter.mutable_attr())["frame_name"].set_s("test_loop");
  *enter.add_input() = "Enter";
  Status status;
  Node* enter_2 = graph->AddNode(enter, &status);
  TF_ASSERT_OK(status);
  graph->AddControlEdge(enter_1, enter_2);

  // SOURCE("") --> Enter("test_loop") --> Enter2("test_loop")
  // For node 'Enter', the parent frame of "test_loop" is empty.
  // For node 'Enter2', the parent frame of "test_loop" is "test_loop".
  std::vector<ControlFlowInfo> info;
  status = BuildControlFlowInfo(graph.get(), &info);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(
      absl::StrContains(status.error_message(), "Mismatched parent frames"))
      << status.error_message();
  EXPECT_TRUE(absl::StrContains(status.error_message(), "{{node Enter2}}"))
      << status.error_message();
}

TEST(ValidateControlFlowTest, TwoLoopCond) {
  // Test that one frame has at most one LoopCond node. This is necessary for
  // functionalize control flow.
  Scope scope = Scope::NewRootScope().ExitOnError();
  std::vector<Output> inputs;
  inputs.push_back(ops::Placeholder(scope, DT_INT32));
  std::vector<Output> outputs;
  TF_ASSERT_OK(ops::BuildWhileLoop(scope, inputs, LessThanTenCond, AddOneBody,
                                   "test_loop", &outputs));
  outputs.clear();
  TF_ASSERT_OK(ops::BuildWhileLoop(scope.NewSubScope("sub"), inputs,
                                   LessThanTenCond, AddOneBody, "test_loop",
                                   &outputs, false));
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));
  std::vector<ControlFlowInfo> info;
  Status status = BuildControlFlowInfo(graph.get(), &info);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(
      absl::StrContains(status.error_message(), "more than one LoopCond node"))
      << status.error_message();
  EXPECT_TRUE(
      absl::StrContains(status.error_message(), "{{node sub/LoopCond}}"))
      << status.error_message();
  EXPECT_TRUE(absl::StrContains(status.error_message(), "{{node LoopCond}}"))
      << status.error_message();
}

}  // namespace
}  // namespace tensorflow
