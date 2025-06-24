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

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradient_checker.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/testutil.h"
#include "tensorflow/cc/gradients/grad_testutil.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace ops {
namespace {

TEST(ResourceVariableGradTest, ReadVariableOpGrad) {
  TensorShape shape({});
  auto scope = Scope::NewRootScope();
  auto x = Placeholder(scope, DT_FLOAT, Placeholder::Shape(shape));

  auto var = VarHandleOp(scope, DT_FLOAT, shape);
  auto init = AssignVariableOp(scope, var, Const(scope, 2.0f, shape));

  auto temp = ReadVariableOp(scope, var, DT_FLOAT);

  auto y = Mul(scope, temp, x);

  auto dy = Placeholder(scope, DT_FLOAT, Placeholder::Shape(shape));

  OutputList dxs;
  TF_ASSERT_OK(AddSymbolicGradients(scope, {y}, {var}, {dy}, &dxs));

  ClientSession::FeedType feed_list;
  feed_list.insert({x, 5.0f});
  feed_list.insert({dy, 1.0f});

  std::vector<Tensor> dxout;
  ClientSession session(scope);
  TF_ASSERT_OK(session.Run(feed_list, dxs, &dxout));

  auto grad = dxout[0].scalar<float>()();
  EXPECT_EQ(grad, 5.0f);
}

}  // namespace
}  // namespace ops
}  // namespace tensorflow
