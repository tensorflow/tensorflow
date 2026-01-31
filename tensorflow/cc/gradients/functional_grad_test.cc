/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradient_checker.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/testutil.h"
#include "tensorflow/cc/gradients/grad_testutil.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace ops {
namespace {

class FunctionGradTest : public ::testing::Test {
 protected:
  FunctionGradTest() : scope_(Scope::NewRootScope()) {}

  void RunTest(const Output& x, const TensorShape& x_shape, const Output& y,
               const TensorShape& y_shape) {
    TF_ASSERT_OK(scope_.status());
    float max_error;
    auto result = (ComputeGradientError<float, float, float>(
        scope_, {x}, {x_shape}, {y}, {y_shape}, &max_error));
    TF_CHECK_OK(result);
    TF_ASSERT_OK(result);
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

TEST_F(FunctionGradTest, PartitionedCallGrad) {
  FunctionDefLibrary f_lib_proto;
  *(f_lib_proto.add_function()) = test::function::XTimesTwo();

  // Construct a graph:
  //   A = Placeholder[dtype=int32]
  //   B = XTimesTwo[_tpu_replicate="cluster"](A)
  //   C = XTimesTwo[_xla_compile_id="cluster"](A)
  TF_ASSERT_OK(scope_.graph()->AddFunctionLibrary(f_lib_proto));

  Output x = Placeholder(scope_, DT_FLOAT);
  NameAttrList f;
  f.set_name("XTimesTwo");
  (*f.mutable_attr())["T"].set_type(DT_FLOAT);
  auto results =
      PartitionedCall(scope_, std::initializer_list<Input>{x}, {DT_FLOAT}, f);
  RunTest(x, {}, results[0], {});

  auto stateful_results = StatefulPartitionedCall(
      scope_, std::initializer_list<Input>{x}, {DT_FLOAT}, f);
  RunTest(x, {}, stateful_results[0], {});
}

}  // namespace
}  // namespace ops
}  // namespace tensorflow
