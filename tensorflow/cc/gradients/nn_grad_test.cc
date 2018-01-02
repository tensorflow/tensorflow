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
    TF_ASSERT_OK((ComputeGradientError<float, float, float>(
        scope_, {x}, {x_shape}, {y}, {y_shape}, &max_error)));
    EXPECT_LT(max_error, 1e-3);
  }

  void RunTest(const Output& x, const Tensor& x_init_value, const Output& y,
               const TensorShape& y_shape) {
    float max_error;
    TF_ASSERT_OK((ComputeGradientError<float, float, float>(
        scope_, x, x_init_value, y, y_shape, &max_error)));
    EXPECT_LT(max_error, 1e-3);
  }

  void RunTest(const OutputList& xs, const std::vector<TensorShape>& x_shapes,
               const OutputList& ys, const std::vector<TensorShape>& y_shapes) {
    TF_ASSERT_OK(scope_.status());
    float max_error;
    TF_ASSERT_OK((ComputeGradientError<float, float, float>(
        scope_, xs, x_shapes, ys, y_shapes, &max_error)));
    EXPECT_LT(max_error, 1e-3);
  }

  // Sets tensor with random values, ensuring that the max value is largest by
  // a reasonable amount.
  // This is an issue for MaxPool and MaxPoolV2, in which perturbations by the
  // numeric gradient computation in the gradient checker can change the max
  // value if values are too close together.
  template <typename T>
  void SetRandomValuesWithBumpedMax(Tensor* tensor) {
    auto tensor_flat = tensor->flat<T>();
    tensor_flat.setRandom();
    int32 max_index = 0;
    for (size_t i = 1; i < tensor->NumElements(); i++) {
      if (tensor_flat(i) > tensor_flat(max_index)) {
        max_index = i;
      }
    }
    tensor_flat(max_index) += 1e-2;
  }

  Scope scope_;
};

TEST_F(NNGradTest, SoftmaxGrad) {
  TensorShape shape({32, 10});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = Softmax(scope_, x);
  RunTest(x, shape, y, shape);
}

TEST_F(NNGradTest, LogSoftmaxGrad) {
  TensorShape shape({5, 3});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = LogSoftmax(scope_, x);
  // Avoid numerical instability when computing finite differences.
  Tensor x_init_value =
      test::AsTensor<float>({-0.9f, -0.7f, -0.5f, -0.3f, -0.1f, 0.1f, 0.3f,
                             0.5f, 0.7f, 0.8f, -0.1f, 0.1f, 0.1f, 0.1f, 1.2f},
                            {5, 3});
  RunTest(x, x_init_value, y, shape);
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

TEST_F(NNGradTest, SeluGrad) {
  TensorShape shape({5, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = Selu(scope_, x);
  Tensor x_init_value = test::AsTensor<float>(
      {-0.9f, -0.7f, -0.5f, -0.3f, -0.1f, 0.1f, 0.3f, 0.5f, 0.7f, 0.9f},
      {5, 2});
  RunTest(x, x_init_value, y, shape);
}

TEST_F(NNGradTest, L2LossGrad) {
  TensorShape x_shape({5, 2});
  TensorShape y_shape({1});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = L2Loss(scope_, x);
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(NNGradTest, BiasAddGradHelper) {
  TensorShape shape({4, 5});
  TensorShape bias_shape({5});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto bias = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(bias_shape));
  auto y = BiasAdd(scope_, x, bias);
  RunTest({x, bias}, {shape, bias_shape}, {y}, {shape});
}

TEST_F(NNGradTest, Conv2DGrad) {
  TensorShape shape({1, 2, 2, 1});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  Tensor filter = test::AsTensor<float>({0.5f}, {1, 1, 1, 1});
  const std::vector<int> strides{1, 1, 1, 1};
  auto y = Conv2D(scope_, x, filter, strides, "SAME");
  RunTest(x, shape, y, shape);
}

TEST_F(NNGradTest, MaxPoolGradHelper) {
  TensorShape x_shape({1, 2, 2, 1});
  TensorShape y_shape({1, 1, 1, 1});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  // Setup window and strides so that we only do one MaxPool.
  const std::vector<int> ksize{1, 2, 2, 1};
  const std::vector<int> strides{1, 2, 2, 1};
  auto y = MaxPool(scope_, x, ksize, strides, "VALID");
  Tensor x_init_value = Tensor(DT_FLOAT, x_shape);
  SetRandomValuesWithBumpedMax<float>(&x_init_value);
  RunTest(x, x_init_value, y, y_shape);
}

TEST_F(NNGradTest, MaxPoolGradV2Helper) {
  TensorShape x_shape({1, 2, 2, 1});
  TensorShape y_shape({1, 1, 1, 1});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  // Setup window and strides so that we only do one MaxPool.
  Tensor ksize = test::AsTensor<int>({1, 2, 2, 1}, {4});
  Tensor strides = test::AsTensor<int>({1, 2, 2, 1}, {4});
  auto y = MaxPoolV2(scope_, x, ksize, strides, "VALID");
  Tensor x_init_value = Tensor(DT_FLOAT, x_shape);
  SetRandomValuesWithBumpedMax<float>(&x_init_value);
  RunTest(x, x_init_value, y, y_shape);
}

TEST_F(NNGradTest, LRN){
  TensorShape x_shape({1, 1, 2, 1});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = LRN(scope_, x);
  RunTest(x, x_shape, y, x_shape);
}

}  // namespace
}  // namespace tensorflow
