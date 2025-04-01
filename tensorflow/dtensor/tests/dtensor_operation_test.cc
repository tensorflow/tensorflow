/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/cc/dtensor_operation.h"

#include <gtest/gtest.h>
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {
namespace dtensor {
namespace {

// Register a few dummy ops with resource and stateful traits.

REGISTER_OP("OutputResource").Output("resource: resource");

REGISTER_OP("InputResource").Input("resource: resource");

REGISTER_OP("Stateful").SetIsStateful();

REGISTER_OP("Pure");

TEST(DTensorOperationTest, TestEagerIsNotPure) {
  DTensorOperation output{"OutputResource", nullptr, {}, {}};
  DTensorOperation input{"InputResource", nullptr, {}, {}};
  DTensorOperation stateful{"Stateful", nullptr, {}, {}};
  DTensorOperation pure{"Pure", nullptr, {}, {}};

  EXPECT_FALSE(output.is_pure());
  EXPECT_FALSE(input.is_pure());
  EXPECT_FALSE(stateful.is_pure());
  EXPECT_TRUE(pure.is_pure());
}

TEST(DTensorOperationTest, TestFunctionIsNotPure) {
  FunctionDef fdef;
  DTensorOperation op{"func", &fdef, {}, {}};
  EXPECT_FALSE(op.is_pure());
}

TEST(DTensorOperationTest, TestIsFunc) {
  FunctionDef fdef;
  DTensorOperation func_op{"func", &fdef, {}, {}};
  DTensorOperation eager_op{"Pure", nullptr, {}, {}};
  EXPECT_TRUE(func_op.is_func());
  EXPECT_FALSE(eager_op.is_func());
}
}  // namespace
}  // namespace dtensor
}  // namespace tensorflow
