/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/placer_device_propagation.h"

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status_test_util.h"

using ::testing::UnorderedElementsAreArray;

namespace tensorflow {

namespace {

const char Tpu0[] = "/job:localhost/replica:0/task:0/device:TPU:0";
const char Tpu1[] = "/job:localhost/replica:0/task:0/device:TPU:1";
const char Tpu2[] = "/job:localhost/replica:0/task:0/device:TPU:2";
const char Gpu0[] = "/job:localhost/replica:0/task:0/device:GPU:0";

bool IsTPUDevice(StringPiece device_name) {
  return absl::StrContains(device_name, "device:TPU:");
}

absl::flat_hash_map<std::string, std::string> GetNodeNameDevices(
    const Graph& graph) {
  absl::flat_hash_map<std::string, std::string> node_name_devices;
  for (const Node* node : graph.nodes()) {
    if (node->IsSource() || node->IsSink()) {
      continue;
    }
    const string& device = node->assigned_device_name().empty()
                               ? node->requested_device()
                               : node->assigned_device_name();
    node_name_devices[node->name()] = device;
  }
  return node_name_devices;
}

TEST(DevicePropagationTest, PropagateTPUDevices) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::Placeholder(scope.WithOpName("A"), DT_FLOAT);
  a.node()->set_assigned_device_name(Tpu0);
  auto b = ops::Placeholder(scope.WithOpName("B"), DT_FLOAT);
  b.node()->set_assigned_device_name(Tpu1);
  auto c = ops::Identity(scope.WithOpName("C"), a);
  auto d =
      ops::Merge(scope.WithOpName("D"), std::initializer_list<Input>{a, c});
  auto e =
      ops::Merge(scope.WithOpName("E"), std::initializer_list<Input>{b, c});
  auto f = ops::Identity(scope.WithOpName("F"), a);
  f.node()->set_assigned_device_name(Tpu2);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  PropagateDevices({"Identity", "Merge"}, IsTPUDevice, &graph);

  EXPECT_THAT(
      GetNodeNameDevices(graph),
      UnorderedElementsAreArray(
          std::vector<std::pair<std::string, std::string>>{
              {"A", Tpu0},  // A is originally placed on Tpu0.
              {"B", Tpu1},  // B is originally placed on Tpu1.
              // C is automatically placed on Tpu0 because A is on Tpu0.
              {"C", Tpu0},
              // D is automatically placed on Tpu0 because both inputs A and C
              // are on Tpu0.
              {"D", Tpu0},
              // E has no assigned device because inputs B and C are on
              // different devices.
              {"E", ""},
              // F device doesn't change because it is originally on Tpu2.
              {"F", Tpu2},
          }));
}

TEST(DevicePropagationTest, DoNotPropagateToUnsupportedOps) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::Placeholder(scope.WithOpName("A"), DT_FLOAT);
  a.node()->set_assigned_device_name(Tpu0);
  auto b = ops::Identity(scope.WithOpName("B"), a);
  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  PropagateDevices({"Merge"}, IsTPUDevice, &graph);

  EXPECT_THAT(GetNodeNameDevices(graph),
              UnorderedElementsAreArray(
                  std::vector<std::pair<std::string, std::string>>{
                      {"A", Tpu0},  // A is originally placed on Tpu0.
                      {"B", ""},    // Tpu0 is not propagated to "B" because
                                    // "Identity" is not in the target op list.
                  }));
}

TEST(DevicePropagationTest, DoNotPropagateUnmatchedDevices) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::Placeholder(scope.WithOpName("A"), DT_FLOAT);
  a.node()->set_assigned_device_name(Gpu0);
  auto b = ops::Identity(scope.WithOpName("B"), a);
  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  PropagateDevices({"Identity"}, IsTPUDevice, &graph);

  EXPECT_THAT(GetNodeNameDevices(graph),
              UnorderedElementsAreArray(
                  std::vector<std::pair<std::string, std::string>>{
                      {"A", Gpu0},  // A is originally placed on Tpu0.
                      {"B", ""},    // Gpu0 is not propagated to "B" because it
                                    // does not match `IsTPUDevice`.
                  }));
}

TEST(DevicePropagationTest, SwitchOpShouldIgnoreLoopCondOp) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::Placeholder(scope.WithOpName("A"), DT_BOOL);
  auto b = ops::LoopCond(scope.WithOpName("B"), a);
  auto c = ops::Placeholder(scope.WithOpName("C"), DT_FLOAT);
  c.node()->set_assigned_device_name(Tpu2);
  auto d = ops::Switch(scope.WithOpName("D"), c, b);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  PropagateDevices({"Switch", "LoopCond"}, IsTPUDevice, &graph);

  EXPECT_THAT(
      GetNodeNameDevices(graph),
      UnorderedElementsAreArray(std::vector<
                                std::pair<std::string, std::string>>{
          // A device remains empty because it cannot be derived from inputs.
          {"A", ""},
          // B device remains empty because it cannot be derived from inputs.
          {"B", ""},
          // C device remains empty because it cannot be derived from inputs.
          {"C", Tpu2},  // C is originally placed on Tpu2.
          // D device is derived from its input C. Input B is ignored because it
          // is a LoopCond op and D is a Switch op.
          {"D", Tpu2},
      }));
}

TEST(DevicePropagationTest, MergeOpShouldIgnoreEnterOp) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::Placeholder(scope.WithOpName("A"), DT_FLOAT);
  auto b = ops::Placeholder(scope.WithOpName("B"), DT_FLOAT);
  b.node()->set_assigned_device_name(Tpu2);
  auto c = ops::internal::Enter(scope.WithOpName("C"), a, "Enter");
  auto d = ops::NextIteration(scope.WithOpName("D"), b);
  auto e =
      ops::Merge(scope.WithOpName("E"), std::initializer_list<Input>{c, d});

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  PropagateDevices({"Enter", "Merge", "NextIteration"}, IsTPUDevice, &graph);

  EXPECT_THAT(
      GetNodeNameDevices(graph),
      UnorderedElementsAreArray(std::vector<
                                std::pair<std::string, std::string>>{
          // A device remains empty because it cannot be derived from inputs.
          {"A", ""},
          {"B", Tpu2},  // B is originally placed on Tpu2.
          // C device remains empty because it cannot be derived from inputs.
          {"C", ""},
          // D device is derived from its input B.
          {"D", Tpu2},
          // E device is derived from its input D. Input C is ignored because it
          // is an Enter op and E is a MERGE op.
          {"E", Tpu2},
      }));
}

}  // namespace
}  // namespace tensorflow
