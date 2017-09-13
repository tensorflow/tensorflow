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

class DataFlowGradTest : public ::testing::Test {
 protected:
  DataFlowGradTest() : scope_(Scope::NewRootScope()) {}

  void RunTest(const OutputList& xs, const std::vector<TensorShape>& x_shapes,
               const OutputList& ys, const std::vector<TensorShape>& y_shapes) {
    TF_ASSERT_OK(scope_.status());
    float max_error;
    TF_ASSERT_OK((ComputeGradientError<float, float, float>(
        scope_, xs, x_shapes, ys, y_shapes, &max_error)));
    EXPECT_LT(max_error, 1e-4);
  }

  Scope scope_;
};

TEST_F(DataFlowGradTest, DynamicPartitionGrad) {
  TensorShape data_shape({2, 3, 2});
  auto data = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(data_shape));
  auto partitions = Const(scope_, {{2, 1, 0}, {1, 2, 0}});
  auto y = DynamicPartition(scope_, data, partitions, 3);
  TensorShape partition_shape({2, 2});
  RunTest({data}, {data_shape}, y.outputs,
          {partition_shape, partition_shape, partition_shape});
}

TEST_F(DataFlowGradTest, DynamicStitchGrad) {
  TensorShape d1_shape({2});
  TensorShape d2_shape({2, 2});
  std::vector<Output> indices = {Const(scope_, 2), Const(scope_, {1, 0})};
  std::vector<Output> data = {
      Placeholder(scope_, DT_FLOAT, Placeholder::Shape(d1_shape)),
      Placeholder(scope_, DT_FLOAT, Placeholder::Shape(d2_shape))};
  auto y = DynamicStitch(scope_, indices, data);
  TensorShape y_shape({3, 2});
  RunTest(data, {d1_shape, d2_shape}, {y}, {y_shape});
}

}  // namespace
}  // namespace tensorflow
