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

#include "tensorflow/cc/framework/gradient_checker.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/testutil.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/equal_graph_def.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
using namespace ops;  // NOLINT(build/namespaces)

namespace {

TEST(GradientCheckerTest, BasicFloat) {
  Scope scope = Scope::NewRootScope();
  TensorShape shape({2, 4, 3});
  auto x = Placeholder(scope, DT_FLOAT, Placeholder::Shape(shape));
  auto y = Square(scope, x);
  float max_error;
  TF_ASSERT_OK(
      ComputeGradientError<float>(scope, x, shape, y, shape, &max_error));
  EXPECT_LT(max_error, 1e-4);
}

TEST(GradientCheckerTest, BasicDouble) {
  Scope scope = Scope::NewRootScope();
  TensorShape shape({2, 4, 3});
  auto x = Placeholder(scope, DT_DOUBLE, Placeholder::Shape(shape));
  auto y = Square(scope, x);
  double max_error;
  TF_ASSERT_OK(
      ComputeGradientError<double>(scope, x, shape, y, shape, &max_error));
  EXPECT_LT(max_error, 1e-10);
}

TEST(GradientCheckerTest, MatMulGrad) {
  Scope scope = Scope::NewRootScope();

  TensorShape x_shape({4, 3});
  TensorShape y_shape({3, 2});
  TensorShape z_shape({4, 2});

  auto x = Placeholder(scope, DT_DOUBLE, Placeholder::Shape(x_shape));
  auto y = Const(scope, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, y_shape);
  auto z = MatMul(scope, x, y);
  double max_error;
  TF_ASSERT_OK(
      ComputeGradientError<double>(scope, x, x_shape, z, z_shape, &max_error));
  EXPECT_LT(max_error, 1e-10);
}

}  // namespace
}  // namespace tensorflow
