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
#include "tensorflow/compiler/jit/node_matchers.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {
using ::tensorflow::testing::FindNodeByName;

Status CloneConstantsForBetterClustering(const Scope& s,
                                         std::unique_ptr<Graph>* result) {
  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
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
  return Status::OK();
}

const char* kCPU = "/job:localhost/replica:0/task:0/device:CPU:0";
const char* kGPU = "/job:localhost/replica:0/task:0/device:GPU:0";

TEST(CloneConstantsForBetterClusteringTest, Basic) {
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

TEST(CloneConstantsForBetterClusteringTest, DontCloneNonHostConstants) {
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

  EXPECT_EQ(tr0_perm.node, tr1_perm.node);
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
