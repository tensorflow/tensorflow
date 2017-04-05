/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace grappler {
namespace {

class ConstantFoldingTest : public ::testing::Test {
 protected:
  std::vector<Tensor> EvaluateNodes(const GraphDef& graph,
                                    const std::vector<string>& fetch) {
    SessionOptions options;
    std::unique_ptr<tensorflow::Session> session(NewSession(options));
    TF_CHECK_OK(session->Create(graph));
    RunOptions run_options;
    std::vector<Tensor> output_tensors;
    TF_CHECK_OK(
        session->Run(run_options, {}, fetch, fetch, &output_tensors, nullptr));
    TF_CHECK_OK(session->Close());
    return output_tensors;
  }
};

TEST_F(ConstantFoldingTest, SimpleFolding) {
  // Build a simple graph with a few trivially prunable ops.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Const(s.WithOpName("a"), 1.0f, {1});
  Output b = ops::Const(s.WithOpName("b"), 2.0f, {1});
  Output c = ops::AddN(s.WithOpName("c"), {a, b});
  Output d = ops::AddN(s.WithOpName("d"), {b, c});

  GrapplerItem item;
  item.fetch.push_back("d");
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding fold;
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(5, output.node_size());

  const NodeDef& new_c = output.node(0);
  EXPECT_EQ("ConstantFolding-c", new_c.name());
  EXPECT_EQ("Const", new_c.op());

  const NodeDef& new_a = output.node(1);
  EXPECT_EQ("a", new_a.name());

  const NodeDef& new_b = output.node(2);
  EXPECT_EQ("b", new_b.name());

  const NodeDef& old_c = output.node(3);
  EXPECT_EQ("c", old_c.name());

  const NodeDef& new_d = output.node(4);
  EXPECT_EQ("d", new_d.name());
  EXPECT_EQ("ConstantFolding-c", new_d.input(1));

  std::vector<string> fetch = {"a", "b", "c", "d"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(4, tensors_expected.size());
  EXPECT_EQ(4, tensors.size());
  for (int i = 0; i < 4; i++) {
    test::ExpectTensorEqual<float>(tensors_expected[i], tensors[i]);
  }
}

TEST_F(ConstantFoldingTest, FoldingNodeWithTwoOutputs) {
  // Build a simple graph with a few trivially prunable ops.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Const(s.WithOpName("a"), 10, {3});
  auto b = ops::Unique(s.WithOpName("b"), {a});
  Output c = ops::Identity(s.WithOpName("c"), {b.y});
  Output d = ops::Identity(s.WithOpName("d"), {b.idx});

  GrapplerItem item;
  item.fetch.push_back("c");
  item.fetch.push_back("d");
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding fold;
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(6, output.node_size());

  const NodeDef& new_b_0 = output.node(0);
  EXPECT_EQ("ConstantFolding-b-0", new_b_0.name());
  EXPECT_EQ("Const", new_b_0.op());

  const NodeDef& new_b_1 = output.node(1);
  EXPECT_EQ("ConstantFolding-b-1", new_b_1.name());
  EXPECT_EQ("Const", new_b_1.op());

  const NodeDef& new_a = output.node(2);
  EXPECT_EQ("a", new_a.name());

  const NodeDef& new_b = output.node(3);
  EXPECT_EQ("b", new_b.name());

  const NodeDef& new_c = output.node(4);
  EXPECT_EQ("c", new_c.name());
  EXPECT_EQ("ConstantFolding-b-0", new_c.input(0));

  const NodeDef& new_d = output.node(5);
  EXPECT_EQ("d", new_d.name());
  EXPECT_EQ("ConstantFolding-b-1", new_d.input(0));

  std::vector<string> fetch = {"a", "b", "c", "d"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(4, tensors_expected.size());
  EXPECT_EQ(4, tensors.size());
  for (int i = 0; i < 4; i++) {
    test::ExpectTensorEqual<int>(tensors_expected[i], tensors[i]);
  }
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
