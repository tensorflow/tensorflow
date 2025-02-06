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

#include "tensorflow/compiler/jit/clone_constants_for_better_clustering.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/compiler/jit/node_matchers.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {
using ::tensorflow::testing::FindNodeByName;

absl::Status CloneConstantsForBetterClustering(const Scope& s,
                                               std::unique_ptr<Graph>* result) {
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(OptimizerOptions::ON_2);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  options.session_options = &session_options;

  // Scope::ToGraph seems to drop assigned devices, probably because it goes
  // through a GraphDef.  So explicitly maintain the device assignment.
  // std::unordered_map<string, string> assigned_device_names;
  // for (Node* n : s.graph()->nodes()) {
  //   assigned_device_names[n->name()] = n->assigned_device_name();
  // }
  GraphConstructorOptions opts;
  opts.expect_device_spec = true;
  TF_RETURN_IF_ERROR(s.ToGraph(graph.get(), opts));
  // for (Node* n : graph->nodes()) {
  //   n->set_assigned_device_name(assigned_device_names[n->name()]);
  // }

  CloneConstantsForBetterClusteringPass rewriter;
  TF_RETURN_IF_ERROR(rewriter.Run(options));
  *result = std::move(graph);
  return absl::OkStatus();
}

const char* kCPU = "/job:localhost/replica:0/task:0/device:CPU:0";
const char* kGPU = "/job:localhost/replica:0/task:0/device:GPU:0";

TEST(CloneConstantsForBetterClusteringTest, ScalarConstantPlacedOnGpu) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Scope on_gpu = root.WithAssignedDevice(kGPU).WithDevice(kGPU);

  Output in = ops::Placeholder(on_gpu.WithOpName("in"), DT_FLOAT);
  Output c = ops::Const(on_gpu.WithOpName("const"), 1.0f, {});
  Output add1 = ops::AddV2(on_gpu.WithOpName("add1"), in, c);
  Output add2 = ops::AddV2(on_gpu.WithOpName("add2"), add1, c);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(CloneConstantsForBetterClustering(root, &result));

  OutputTensor add1_operand;
  TF_ASSERT_OK(
      FindNodeByName(result.get(), "add1")->input_tensor(1, &add1_operand));

  OutputTensor add2_operand;
  TF_ASSERT_OK(
      FindNodeByName(result.get(), "add2")->input_tensor(1, &add2_operand));

  EXPECT_NE(add1_operand.node, add2_operand.node);
}

TEST(CloneConstantsForBetterClusteringTest, HostConstantPlacedOnCpu) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Scope on_gpu = root.WithAssignedDevice(kGPU).WithDevice(kGPU);
  Scope on_cpu = root.WithAssignedDevice(kCPU).WithDevice(kCPU);

  Output in0 = ops::Placeholder(on_gpu.WithOpName("in0"), DT_FLOAT);
  Output in1 = ops::Placeholder(on_gpu.WithOpName("in1"), DT_FLOAT);

  Output perm = ops::Const(on_cpu.WithOpName("perm"), {3, 1, 2, 0});

  {
    Output tr0 = ops::Transpose(on_gpu.WithOpName("tr0"), in0, perm);
    Output tr1 = ops::Transpose(on_gpu.WithOpName("tr1"), in1, perm);
  }

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(CloneConstantsForBetterClustering(root, &result));

  OutputTensor tr0_perm;
  TF_ASSERT_OK(FindNodeByName(result.get(), "tr0")->input_tensor(1, &tr0_perm));

  OutputTensor tr1_perm;
  TF_ASSERT_OK(FindNodeByName(result.get(), "tr1")->input_tensor(1, &tr1_perm));

  EXPECT_NE(tr0_perm.node, tr1_perm.node);
}

TEST(CloneConstantsForBetterClusteringTest, HostConstantPlacedOnGpu) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Scope on_gpu = root.WithAssignedDevice(kGPU).WithDevice(kGPU);

  Output in0 = ops::Placeholder(on_gpu.WithOpName("in0"), DT_FLOAT);
  Output in1 = ops::Placeholder(on_gpu.WithOpName("in1"), DT_FLOAT);

  Output perm = ops::Const(on_gpu.WithOpName("perm"), {3, 1, 2, 0});

  {
    Output tr0 = ops::Transpose(on_gpu.WithOpName("tr0"), in0, perm);
    Output tr1 = ops::Transpose(on_gpu.WithOpName("tr1"), in1, perm);
  }

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(CloneConstantsForBetterClustering(root, &result));

  OutputTensor tr0_perm;
  TF_ASSERT_OK(FindNodeByName(result.get(), "tr0")->input_tensor(1, &tr0_perm));

  OutputTensor tr1_perm;
  TF_ASSERT_OK(FindNodeByName(result.get(), "tr1")->input_tensor(1, &tr1_perm));

  EXPECT_NE(tr0_perm.node, tr1_perm.node);
}

TEST(CloneConstantsForBetterClusteringTest, CloneSmallDeviceConstants) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Scope on_gpu = root.WithAssignedDevice(kGPU).WithDevice(kGPU);

  Output in0 = ops::Placeholder(on_gpu.WithOpName("in0"), DT_FLOAT);
  Output in1 = ops::Placeholder(on_gpu.WithOpName("in1"), DT_FLOAT);

  Output perm_f32 = ops::Const(on_gpu.WithOpName("perm"), {3.0, 1.0, 2.0, 0.0});
  Output perm_int0 =
      ops::Cast(on_gpu.WithOpName("perm_cast_0"), perm_f32, DT_INT32);
  Output perm_int1 =
      ops::Cast(on_gpu.WithOpName("perm_cast_1"), perm_f32, DT_INT32);

  {
    Output tr0 = ops::Transpose(on_gpu.WithOpName("tr0"), in0, perm_int0);
    Output tr1 = ops::Transpose(on_gpu.WithOpName("tr1"), in1, perm_int1);
  }

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(CloneConstantsForBetterClustering(root, &result));

  OutputTensor tr0_perm;
  TF_ASSERT_OK(
      FindNodeByName(result.get(), "perm_cast_0")->input_tensor(0, &tr0_perm));

  OutputTensor tr1_perm;
  TF_ASSERT_OK(
      FindNodeByName(result.get(), "perm_cast_1")->input_tensor(0, &tr1_perm));

  EXPECT_NE(tr0_perm.node, tr1_perm.node);
}

TEST(CloneConstantsForBetterClusteringTest, DontCloneLargeConstants) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Scope on_gpu = root.WithAssignedDevice(kGPU).WithDevice(kGPU);
  Scope on_cpu = root.WithAssignedDevice(kCPU).WithDevice(kCPU);

  Output in0 = ops::Placeholder(on_gpu.WithOpName("in0"), DT_FLOAT);
  Output in1 = ops::Placeholder(on_gpu.WithOpName("in1"), DT_FLOAT);

  Output perm = ops::Const(
      on_cpu.WithOpName("perm"),
      {17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0});

  {
    Output tr0 = ops::Transpose(on_gpu.WithOpName("tr0"), in0, perm);
    Output tr1 = ops::Transpose(on_gpu.WithOpName("tr1"), in1, perm);
  }

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(CloneConstantsForBetterClustering(root, &result));

  OutputTensor tr0_perm;
  TF_ASSERT_OK(FindNodeByName(result.get(), "tr0")->input_tensor(1, &tr0_perm));

  OutputTensor tr1_perm;
  TF_ASSERT_OK(FindNodeByName(result.get(), "tr1")->input_tensor(1, &tr1_perm));

  EXPECT_EQ(tr0_perm.node, tr1_perm.node);
}

TEST(CloneConstantsForBetterClusteringTest, InplaceOps) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Scope on_gpu = root.WithAssignedDevice(kGPU).WithDevice(kGPU);
  Scope on_cpu = root.WithAssignedDevice(kCPU).WithDevice(kCPU);

  Output in0 = ops::Placeholder(on_gpu.WithOpName("in0"), DT_FLOAT);
  Output in1 = ops::Placeholder(on_gpu.WithOpName("in1"), DT_FLOAT);

  Output perm = ops::Const(on_cpu.WithOpName("perm"), {3, 1, 2, 0});

  {
    Output tr0 = ops::Transpose(on_gpu.WithOpName("tr0"), in0, perm);
    Output tr1 = ops::Transpose(on_gpu.WithOpName("tr1"), in1, perm);
  }

  Output in_place_add =
      ops::InplaceAdd(on_cpu.WithOpName("tr0"), perm,
                      ops::Placeholder(on_cpu.WithOpName("i"), DT_INT32), perm);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(CloneConstantsForBetterClustering(root, &result));

  OutputTensor tr0_perm;
  TF_ASSERT_OK(FindNodeByName(result.get(), "tr0")->input_tensor(1, &tr0_perm));

  OutputTensor tr1_perm;
  TF_ASSERT_OK(FindNodeByName(result.get(), "tr1")->input_tensor(1, &tr1_perm));

  EXPECT_EQ(tr0_perm.node, tr1_perm.node);
}
}  // namespace
}  // namespace tensorflow
