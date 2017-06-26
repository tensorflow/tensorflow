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

#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradient_checker.h"
#include "tensorflow/cc/framework/testutil.h"
#include "tensorflow/cc/gradients/grad_testutil.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
using namespace ops;  // NOLINT(build/namespaces)

namespace {

class NNGradTest : public ::testing::Test {
 protected:
  NNGradTest() : scope_(Scope::NewRootScope()) {}

  void RunTest(const Output& x, const TensorShape& x_shape, const Output& y,
               const TensorShape& y_shape) {
    float max_error;
    TF_ASSERT_OK(ComputeGradientError(scope_, {x}, {x_shape}, {y}, {y_shape},
                                      &max_error));
    EXPECT_LT(max_error, 1e-4);
  }

  void RunTest(const Output& x, const Tensor& x_init_value, const Output& y,
               const TensorShape& y_shape) {
    float max_error;
    TF_ASSERT_OK(
        ComputeGradientError(scope_, x, x_init_value, y, y_shape, &max_error));
    EXPECT_LT(max_error, 1e-4);
  }

  Scope scope_;
};

TEST_F(NNGradTest, SoftmaxGrad) {
  TensorShape shape({32, 10});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = Softmax(scope_, x);
  RunTest(x, shape, y, shape);
}

TEST_F(NNGradTest, ReluGrad) {
  TensorShape shape({5, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = Relu(scope_, x);
  // Avoid input values where ReLU gradient is not well defined (around zero).
  Tensor x_init_value = test::AsTensor<float>(
      {-0.9f, -0.7f, -0.5f, -0.3f, -0.1f, 0.1f, 0.3f, 0.5f, 0.7f, 0.9f},
      {5, 2});
  RunTest(x, x_init_value, y, shape);
}

TEST_F(NNGradTest, Relu6Grad) {
  TensorShape shape({5, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = Relu6(scope_, x);
  // Avoid input values where ReLU gradient is not well defined (around zero
  // and six).
  Tensor x_init_value = test::AsTensor<float>(
      {-0.9f, -0.7f, -0.5f, -0.3f, -0.1f, 6.1f, 6.3f, 6.5f, 6.7f, 6.9f},
      {5, 2});
  RunTest(x, x_init_value, y, shape);
}

TEST_F(NNGradTest, EluGrad) {
  TensorShape shape({5, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = Elu(scope_, x);
  Tensor x_init_value = test::AsTensor<float>(
      {-0.9f, -0.7f, -0.5f, -0.3f, -0.1f, 0.1f, 0.3f, 0.5f, 0.7f, 0.9f},
      {5, 2});
  RunTest(x, x_init_value, y, shape);
}

}  // namespace
}  // namespace tensorflow
