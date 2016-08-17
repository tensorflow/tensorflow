/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/equal_graph_def.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
using namespace ops;  // NOLINT(build/namespaces)

namespace {

// TODO(andydavis) Add more unit tests once more gradient functions are ported.
// TODO(andydavis) Add unit test that adds gradients to compute two Outputs,
// where the gradient w.r.t. one Output depends on the other.
class GradientsTest : public ::testing::Test {
 protected:
  GradientsTest()
      : scope_expected_(Scope::NewRootScope()),
        scope_test_(Scope::NewRootScope()) {}

  void CompareTestAndExpectedGraphs() {
    GraphDef gdef_test;
    TF_EXPECT_OK(scope_test_.ToGraphDef(&gdef_test));
    GraphDef gdef_exp;
    TF_EXPECT_OK(scope_expected_.ToGraphDef(&gdef_exp));
    TF_EXPECT_GRAPH_EQ(gdef_test, gdef_exp);
  }

  Scope scope_expected_;
  Scope scope_test_;
};

// EX.
//      ^             ^
//    dy|           dx|        // MatMul Gradient Graph
//      |             |
//   MatMul_1      MatMul_2
//   ^   ^          ^    ^
//   |   |----------|    |
//   |        ^          |
//   |      dz|          |
//   |        |          |
//   |     Const_3       |
//   |                   |
//   |        ^          |
//   |       z|          |     // MatMul Forward Graph
//   |        |          |
//   |      MatMul_0     |
//   |     /        \    |
//   |    ^          ^   |
//   |    |          |   |
//   |---x|         y|---|
//        |          |
//        |          |
//      Const_0   Const_1
//

TEST_F(GradientsTest, OneMatMul) {
  bool expected = false;
  for (Scope scope : {scope_test_, scope_expected_}) {
    // Construct forward graph.
    auto x = Const(scope, {{1.0, 2.0}, {3.0, 4.0}});
    auto y = Const(scope, {{1.0, 0.0}, {0.0, 1.0}});
    auto z = MatMul(scope, x, y);
    TF_EXPECT_OK(scope.status());
    CHECK_NOTNULL(z.node());

    if (expected) {
      // Construct backward graph.
      auto dz = Const(scope, {{1.0, 1.0}, {1.0, 1.0}});
      auto dx = MatMul(scope, dz, y, MatMul::TransposeB(true));
      auto dy = MatMul(scope, x, dz, MatMul::TransposeA(true));
    } else {
      // Call AddSymbolicGradients.
      auto dz = Const(scope, {{1.0, 1.0}, {1.0, 1.0}});
      std::vector<ops::Output> grad_outputs;
      TF_EXPECT_OK(
          AddSymbolicGradients(scope, {z}, {x, y}, {dz}, &grad_outputs));
    }
    expected = true;
  }
  CompareTestAndExpectedGraphs();
}

TEST_F(GradientsTest, TwoMatMuls_Chained) {
  bool expected = false;
  for (Scope scope : {scope_test_, scope_expected_}) {
    // Construct forward graph.
    auto u = Const(scope, {{1.0, 2.0}, {3.0, 4.0}});
    auto v = Const(scope, {{1.0, 0.0}, {0.0, 1.0}});
    auto x = MatMul(scope, u, v);

    auto y = Const(scope, {{1.0, 0.0}, {0.0, 1.0}});
    auto z = MatMul(scope, x, y);

    TF_EXPECT_OK(scope.status());
    CHECK_NOTNULL(z.node());

    if (expected) {
      // Construct backward graph.
      auto dz = Const(scope, {{1.0, 1.0}, {1.0, 1.0}});
      auto dx = MatMul(scope, dz, y, MatMul::TransposeB(true));
      auto dy = MatMul(scope, x, dz, MatMul::TransposeA(true));

      auto du = MatMul(scope, dx, v, MatMul::TransposeB(true));
      auto dv = MatMul(scope, u, dx, MatMul::TransposeA(true));
    } else {
      // Call AddSymbolicGradients.
      auto dz = Const(scope, {{1.0, 1.0}, {1.0, 1.0}});
      std::vector<ops::Output> grad_outputs;
      TF_EXPECT_OK(
          AddSymbolicGradients(scope, {z}, {u, v}, {dz}, &grad_outputs));
    }
    expected = true;
  }
  CompareTestAndExpectedGraphs();
}

TEST_F(GradientsTest, TwoMatMuls_Independent) {
  bool expected = false;
  for (Scope scope : {scope_test_, scope_expected_}) {
    // Construct forward graph.
    auto t = Const(scope, {{1.0, 2.0}, {3.0, 4.0}});
    auto u = Const(scope, {{1.0, 0.0}, {0.0, 1.0}});
    auto v = MatMul(scope, t, u);
    TF_EXPECT_OK(scope.status());
    CHECK_NOTNULL(v.node());

    auto x = Const(scope, {{5.0, 6.0}, {7.0, 8.0}});
    auto y = Const(scope, {{1.0, 0.0}, {0.0, 1.0}});
    auto z = MatMul(scope, x, y);
    TF_EXPECT_OK(scope.status());
    CHECK_NOTNULL(z.node());

    if (expected) {
      // Construct backward graph.
      auto dv = Const(scope, {{1.0, 1.0}, {1.0, 1.0}});
      auto dt = MatMul(scope, dv, u, MatMul::TransposeB(true));
      auto du = MatMul(scope, t, dv, MatMul::TransposeA(true));

      auto dz = Const(scope, {{1.0, 1.0}, {1.0, 1.0}});
      auto dx = MatMul(scope, dz, y, MatMul::TransposeB(true));
      auto dy = MatMul(scope, x, dz, MatMul::TransposeA(true));
    } else {
      // Call AddSymbolicGradients.
      auto dv = Const(scope_test_, {{1.0, 1.0}, {1.0, 1.0}});
      auto dz = Const(scope_test_, {{1.0, 1.0}, {1.0, 1.0}});
      std::vector<ops::Output> grad_outputs;
      TF_EXPECT_OK(AddSymbolicGradients(scope, {v, z}, {t, u, x, y}, {dv, dz},
                                        &grad_outputs));
    }
    expected = true;
  }
  CompareTestAndExpectedGraphs();
}

}  // namespace
}  // namespace tensorflow
