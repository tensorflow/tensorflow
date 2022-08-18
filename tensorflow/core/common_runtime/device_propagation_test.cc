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

#include "tensorflow/core/common_runtime/device_propagation.h"

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
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

const char kTpu0[] = "/job:localhost/replica:0/task:0/device:TPU:0";
const char kTpu1[] = "/job:localhost/replica:0/task:0/device:TPU:1";
const char kTpu2[] = "/job:localhost/replica:0/task:0/device:TPU:2";
const char kGpu0[] = "/job:localhost/replica:0/task:0/device:GPU:0";

bool IsTPUDevice(StringPiece device_name) {
  return absl::StrContains(device_name, "device:TPU:");
}

device_propagation::NodeFilter TargetOps(
    const absl::flat_hash_set<std::string>& ops) {
  return [&ops](const Node& n) { return ops.contains(n.type_string()); };
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
  a.node()->set_assigned_device_name(kTpu0);
  auto b = ops::Placeholder(scope.WithOpName("B"), DT_FLOAT);
  b.node()->set_assigned_device_name(kTpu1);
  auto c = ops::Identity(scope.WithOpName("C"), a);
  auto d =
      ops::Merge(scope.WithOpName("D"), std::initializer_list<Input>{a, c});
  auto e =
      ops::Merge(scope.WithOpName("E"), std::initializer_list<Input>{b, c});
  auto f = ops::Identity(scope.WithOpName("F"), a);
  f.node()->set_assigned_device_name(kTpu2);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  PropagateDevices(TargetOps({"Identity", "Merge"}), IsTPUDevice, &graph);

  EXPECT_THAT(
      GetNodeNameDevices(graph),
      UnorderedElementsAreArray(
          std::vector<std::pair<std::string, std::string>>{
              {"A", kTpu0},  // A is originally placed on kTpu0.
              {"B", kTpu1},  // B is originally placed on kTpu1.
              // C is automatically placed on kTpu0 because A is on kTpu0.
              {"C", kTpu0},
              // D is automatically placed on kTpu0 because both inputs A and C
              // are on kTpu0.
              {"D", kTpu0},
              // E has no assigned device because inputs B and C are on
              // different devices.
              {"E", ""},
              // F device doesn't change because it is originally on kTpu2.
              {"F", kTpu2},
          }));
}

TEST(DevicePropagationTest, DoNotPropagateToUnsupportedOps) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::Placeholder(scope.WithOpName("A"), DT_FLOAT);
  a.node()->set_assigned_device_name(kTpu0);
  auto b = ops::Identity(scope.WithOpName("B"), a);
  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  PropagateDevices(TargetOps({"Merge"}), IsTPUDevice, &graph);

  EXPECT_THAT(GetNodeNameDevices(graph),
              UnorderedElementsAreArray(
                  std::vector<std::pair<std::string, std::string>>{
                      {"A", kTpu0},  // A is originally placed on kTpu0.
                      {"B", ""},     // kTpu0 is not propagated to "B" because
                                     // "Identity" is not in the target op list.
                  }));
}

TEST(DevicePropagationTest, DoNotPropagateUnmatchedDevices) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::Placeholder(scope.WithOpName("A"), DT_FLOAT);
  a.node()->set_assigned_device_name(kGpu0);
  auto b = ops::Identity(scope.WithOpName("B"), a);
  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  PropagateDevices(TargetOps({"Identity"}), IsTPUDevice, &graph);

  EXPECT_THAT(GetNodeNameDevices(graph),
              UnorderedElementsAreArray(
                  std::vector<std::pair<std::string, std::string>>{
                      {"A", kGpu0},  // A is originally placed on kTpu0.
                      {"B", ""},  // kGpu0 is not propagated to "B" because it
                                  // does not match `IsTPUDevice`.
                  }));
}

TEST(DevicePropagationTest, SwitchOpShouldIgnoreLoopCondOp) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::Placeholder(scope.WithOpName("A"), DT_BOOL);
  auto b = ops::LoopCond(scope.WithOpName("B"), a);
  auto c = ops::Placeholder(scope.WithOpName("C"), DT_FLOAT);
  c.node()->set_assigned_device_name(kTpu2);
  auto d = ops::Switch(scope.WithOpName("D"), c, b);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  PropagateDevices(TargetOps({"Switch", "LoopCond"}), IsTPUDevice, &graph);

  EXPECT_THAT(
      GetNodeNameDevices(graph),
      UnorderedElementsAreArray(std::vector<
                                std::pair<std::string, std::string>>{
          // A device remains empty because it cannot be derived from inputs.
          {"A", ""},
          // B device remains empty because it cannot be derived from inputs.
          {"B", ""},
          // C device remains empty because it cannot be derived from inputs.
          {"C", kTpu2},  // C is originally placed on kTpu2.
          // D device is derived from its input C. Input B is ignored because it
          // is a LoopCond op and D is a Switch op.
          {"D", kTpu2},
      }));
}

TEST(DevicePropagationTest, MergeOpShouldIgnoreEnterOp) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::Placeholder(scope.WithOpName("A"), DT_FLOAT);
  auto b = ops::Placeholder(scope.WithOpName("B"), DT_FLOAT);
  b.node()->set_assigned_device_name(kTpu2);
  auto c = ops::internal::Enter(scope.WithOpName("C"), a, "Enter");
  auto d = ops::NextIteration(scope.WithOpName("D"), b);
  auto e =
      ops::Merge(scope.WithOpName("E"), std::initializer_list<Input>{c, d});

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  PropagateDevices(TargetOps({"Enter", "Merge", "NextIteration"}), IsTPUDevice,
                   &graph);

  EXPECT_THAT(
      GetNodeNameDevices(graph),
      UnorderedElementsAreArray(std::vector<
                                std::pair<std::string, std::string>>{
          // A device remains empty because it cannot be derived from inputs.
          {"A", ""},
          {"B", kTpu2},  // B is originally placed on kTpu2.
          // C device remains empty because it cannot be derived from inputs.
          {"C", ""},
          // D device is derived from its input B.
          {"D", kTpu2},
          // E device is derived from its input D. Input C is ignored because it
          // is an Enter op and E is a MERGE op.
          {"E", kTpu2},
      }));
}

}  // namespace
}  // namespace tensorflow
