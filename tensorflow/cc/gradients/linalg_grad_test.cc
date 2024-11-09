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

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradient_checker.h"
#include "tensorflow/cc/framework/testutil.h"
#include "tensorflow/cc/gradients/grad_testutil.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

using tensorflow::ops::Einsum;
using tensorflow::ops::Placeholder;

class LinalgGradTest : public ::testing::Test {
 protected:
  LinalgGradTest() : scope_(Scope::NewRootScope()) {}

  void RunTest(const Output& x, const TensorShape& x_shape, const Output& y,
               const TensorShape& y_shape) {
    TF_ASSERT_OK(scope_.status());
    float max_error;
    TF_ASSERT_OK((ComputeGradientError<float, float, float>(
        scope_, {x}, {x_shape}, {y}, {y_shape}, &max_error)));
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

  Scope scope_;
};

TEST_F(LinalgGradTest, Einsum_Transpose) {
  TensorShape x_shape({2, 3});
  Output x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = Einsum(scope_, {x}, "ij->ji");
  TensorShape y_shape({3, 2});
  RunTest({x}, {x_shape}, {y}, {y_shape});
}

TEST_F(LinalgGradTest, Einsum_TransposeBroadcast) {
  TensorShape x_shape({3, 2, 3});
  Output x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = Einsum(scope_, {x}, "...ij->...ji");
  TensorShape y_shape({3, 3, 2});
  RunTest({x}, {x_shape}, {y}, {y_shape});
}

TEST_F(LinalgGradTest, Einsum_MatMul) {
  TensorShape x_shape({2, 3});
  TensorShape y_shape({3, 3});
  Output x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  Output y = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(y_shape));
  auto z = Einsum(scope_, {x, y}, "ij,jk->ik");
  TensorShape z_shape({2, 3});
  RunTest({x, y}, {x_shape, y_shape}, {z}, {z_shape});
}

TEST_F(LinalgGradTest, Einsum_MatMulComplex) {
  TensorShape x_shape({2, 3});
  TensorShape y_shape({3, 3});
  Output x = Placeholder(scope_, DT_COMPLEX64, Placeholder::Shape(x_shape));
  Output y = Placeholder(scope_, DT_COMPLEX64, Placeholder::Shape(y_shape));
  auto z = Einsum(scope_, {x, y}, "ij,jk->ik");
  TensorShape z_shape({2, 3});
  TF_ASSERT_OK(scope_.status());
  float max_error;
  TF_ASSERT_OK((ComputeGradientError<complex64, complex64, float>(
      scope_, {x, y}, {x_shape, y_shape}, {z}, {z_shape}, &max_error)));
  EXPECT_LT(max_error, 1e-3);
}

TEST_F(LinalgGradTest, Einsum_MatMulBroadcast) {
  TensorShape x_shape({3, 2, 3});
  TensorShape y_shape({3, 3});
  Output x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  Output y = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(y_shape));
  auto z = Einsum(scope_, {x, y}, "...ij,...jk->...ik");
  TensorShape z_shape({3, 2, 3});
  RunTest({x, y}, {x_shape, y_shape}, {z}, {z_shape});
}

TEST_F(LinalgGradTest, Einsum_Trace) {
  TensorShape x_shape({3, 3});
  Output x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  // Note: In Python this could just be "ii" becuase tf.einsum normalizes the
  // equation, but c++ doesn't do that.
  auto z = Einsum(scope_, {x}, "ii->");
  TensorShape z_shape({});
  RunTest({x}, {x_shape}, {z}, {z_shape});
}

TEST_F(LinalgGradTest, Einsum_TraceBroadcast) {
  TensorShape x_shape({4, 3, 3});
  Output x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  // Note: In Python this could just be "ii" becuase tf.einsum normalizes the
  // equation, but c++ doesn't do that.
  auto z = Einsum(scope_, {x}, "...ii->...");
  TensorShape z_shape({4});
  RunTest({x}, {x_shape}, {z}, {z_shape});
}

TEST_F(LinalgGradTest, Einsum_DotProduct) {
  TensorShape x_shape({3});
  TensorShape y_shape({3});
  Output x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  Output y = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(y_shape));
  auto z = Einsum(scope_, {x, y}, "i,i->");
  TensorShape z_shape({});
  RunTest({x, y}, {x_shape, y_shape}, {z}, {z_shape});
}

TEST_F(LinalgGradTest, Einsum_OuterProduct) {
  TensorShape x_shape({3});
  TensorShape y_shape({5});
  Output x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  Output y = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(y_shape));
  auto z = Einsum(scope_, {x, y}, "i,j->ij");
  TensorShape z_shape({3, 5});
  RunTest({x, y}, {x_shape, y_shape}, {z}, {z_shape});
}

TEST_F(LinalgGradTest, Einsum_TwoInputReduction) {
  TensorShape x_shape({3, 2, 4});
  TensorShape y_shape({4, 5});
  Output x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  Output y = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(y_shape));
  auto z = Einsum(scope_, {x, y}, "abc,cd->ad");
  TensorShape z_shape({3, 5});
  RunTest({x, y}, {x_shape, y_shape}, {z}, {z_shape});
}

}  // namespace
}  // namespace tensorflow
