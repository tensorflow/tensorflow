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

#include "tensorflow/cc/training/optimizer.h"
#include "tensorflow/cc/training/gradient_descent_optimizer.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/util/equal_graph_def.h"
#include "tensorflow/cc/client/client_session.h"

namespace tensorflow {
using namespace ops;  // NOLINT(build/namespaces)

namespace {

class OptimizerTest : public ::testing::Test {
 protected:
  OptimizerTest()
      : scope_expected_(Scope::NewRootScope()),
        scope_test_(Scope::NewRootScope()) {}

  void CompareTestAndExpectedGraphs() {
    GraphDef gdef_test;
    TF_ASSERT_OK(scope_test_.ToGraphDef(&gdef_test));
    GraphDef gdef_exp;
    TF_ASSERT_OK(scope_expected_.ToGraphDef(&gdef_exp));
    TF_EXPECT_GRAPH_EQ(gdef_exp, gdef_test);
  }

  Scope scope_expected_;
  Scope scope_test_;
};

class OptimizerValueTest : public ::testing::Test {
 protected:
  OptimizerValueTest()
      : scope_(Scope::NewRootScope()) {}

  Scope scope_;
};

// test that the graph is correctly built
TEST_F(OptimizerTest, OneMatMul) {

  for (const bool expected : {false, true}) {

    const Scope& scope = expected ? scope_expected_ : scope_test_;

    // the forward node should be the same for the test and expected scope
    // TODO(theflofly): merge Const and Assign using one constructor as in python
    auto x = Variable(scope.WithOpName("x"), {2, 2}, DT_FLOAT);
    auto assign_x = Assign(scope.WithOpName("Assign_x"), x, Const(scope, {{1.0f, 2.0f}, {3.0f, 4.0f}}));

    auto y = Variable(scope.WithOpName("y"), {2, 2}, DT_FLOAT);
    auto assign_y = Assign(scope.WithOpName("Assign_y"), y, Const(scope, {{1.0f, 0.0f}, {0.0f, 1.0f}}));

    // the assign node is only used once, it should not be used in the graph
    auto z = MatMul(scope.WithOpName("z"), x,  y);

    TF_ASSERT_OK(scope.status());
    CHECK_NOTNULL(z.node());

    if (expected) {

      // we manually add the gradient node to the expected scope
      Scope scope_gradient = scope.NewSubScope("Gradients");
      Scope scope_optimizer = scope.NewSubScope("GradientDescent");

      // gradients
      auto dz = ops::OnesLike(scope_gradient, z);
      auto dx = MatMul(scope_gradient, dz, y, MatMul::TransposeB(true));
      auto dy = MatMul(scope_gradient, x, dz, MatMul::TransposeA(true));

      // update
      ApplyGradientDescent(scope_optimizer.NewSubScope("update"), 
                           {x}, 
                           Cast(scope_optimizer.NewSubScope("learning_rate"), 0.01f, static_cast<DataType>(Output{x}.type() - 100)), 
                           {dx});

      ApplyGradientDescent(scope_optimizer.NewSubScope("update"), 
                           {y}, 
                           Cast(scope_optimizer.NewSubScope("learning_rate"), 0.01f, static_cast<DataType>(Output{y}.type() - 100)), 
                           {dy});

    } else {
      
      // the gradient nodes and update nodes are added to the graph
      auto train = GradientDescentOptimizer(0.01).Minimize(scope, {z}, {x, y});

      TF_ASSERT_OK(scope.status());

      ClientSession session(scope);

      // TODO(theflofly): a global initializer would be nice
      TF_CHECK_OK(session.Run({assign_x, assign_y}, nullptr));

    }
  }

  CompareTestAndExpectedGraphs();

}

// test that the value produced by the optimizer are correct
TEST_F(OptimizerValueTest, NeuralNetworkValues) {
  
  auto x = Const(scope_, {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
  auto y = Const(scope_, {{1.0f}, {2.0f}, {3.0f}});

  auto w1 = Variable(scope_, {2, 1}, DT_FLOAT);
  auto assign_w1 = Assign(scope_, w1, Const(scope_, {{0.1f}, {0.2f}}));

  auto layer_1 = Tanh(scope_, MatMul(scope_, x, w1));

  //TODO(theflofly): add the gradients and do
  //auto b1 = Variable(scope_, {1, 1}, DT_FLOAT);
  //auto assign_b1 = Assign(scope_, b1, Const(scope_, {{0.45f}}));
  //auto layer_1 = Tanh(scope_, Add(scope_, MatMul(scope_, x, w1), b1));
  //auto loss = Mean(scope_, Square(scope_, Subtract(scope_, layer_1, y)), 1);
  //finally compare the decreasing loss

  auto train = GradientDescentOptimizer(0.01).Minimize(scope_, {layer_1}, {w1});

  TF_ASSERT_OK(scope_.status());

  ClientSession session(scope_);
  
  // TODO(theflofly): a global initializer would be nice
  TF_CHECK_OK(session.Run({assign_w1}, nullptr));

  for (int i = 0; i < 10; i++) {
    TF_CHECK_OK(session.Run({train}, nullptr));
  }
  std::vector<Tensor> outputs;
  TF_CHECK_OK(session.Run({layer_1}, &outputs));

  test::ExpectTensorEqual<float>(outputs[0], test::AsTensor<float>({-0.66430414, -0.95039594, -0.99360687}, {3, 1}));

}

}  // namespace
}  // namespace tensorflow