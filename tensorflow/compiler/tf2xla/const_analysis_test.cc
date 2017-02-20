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

// Tests for the backward const analysis.

#include "tensorflow/compiler/tf2xla/const_analysis.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(ConstAnalysisTest, Basics) {
  Scope root = Scope::NewRootScope();

  auto arg0 = ops::_Arg(root.WithOpName("Arg0"), DT_INT32, 0);
  auto arg1 = ops::_Arg(root.WithOpName("Arg1"), DT_INT32, 1);
  auto arg2 = ops::_Arg(root.WithOpName("Arg2"), DT_INT32, 2);
  auto arg3 = ops::_Arg(root.WithOpName("Arg3"), DT_INT32, 3);
  auto a = ops::Shape(root, arg0);
  auto b = ops::Add(root, a, arg1);
  auto c = ops::Reshape(root, arg2, b);
  auto d = ops::Mul(root, c, ops::Sum(root, arg3, arg3));

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph));

  std::vector<bool> const_args(4, false);
  TF_ASSERT_OK(BackwardsConstAnalysis(graph, &const_args));

  // Arg 0 doesn't need to be constant since the graph only uses its shape.
  // Arg 1 must be constant because it flows to the shape argument of a Reshape.
  // Arg 2 is used only as the value input to a Reshape and need not be const.
  // Arg 3 is used as the reduction-indices argument to Sum and must be const.
  EXPECT_EQ(const_args, std::vector<bool>({false, true, false, true}));
}

// Regression test for a case where the backward const analysis did
// not visit nodes in topological order.
TEST(ConstAnalysisTest, TopologicalOrder) {
  for (bool order : {false, true}) {
    Scope root = Scope::NewRootScope();

    auto arg0 = ops::_Arg(root.WithOpName("Arg0"), DT_INT32, 0);
    auto arg1 = ops::_Arg(root.WithOpName("Arg1"), DT_INT32, 1);
    auto arg2 = ops::_Arg(root.WithOpName("Arg2"), DT_INT32, 2);
    auto a = ops::Reshape(root, arg0, arg1);
    auto b = ops::Reshape(root, arg2, a);
    if (order) {
      // Consider both orders for arguments to the Sum so we aren't sensitive
      // to the DFS traversal order.
      std::swap(a, b);
    }
    auto c = ops::Add(root, a, b);

    Graph graph(OpRegistry::Global());
    TF_ASSERT_OK(root.ToGraph(&graph));

    std::vector<bool> const_args(3, false);
    TF_ASSERT_OK(BackwardsConstAnalysis(graph, &const_args));

    EXPECT_EQ(const_args, std::vector<bool>({true, true, false}));
  }
}

}  // namespace
}  // namespace tensorflow
