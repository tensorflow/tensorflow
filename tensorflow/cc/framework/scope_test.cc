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

#include "tensorflow/cc/framework/scope.h"

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(ScopeTest, BasicNames) {
  Scope root = Scope::NewRootScope();
  EXPECT_EQ(root.GetUniqueNameForOp("add"), "add");
  EXPECT_EQ(root.GetUniqueNameForOp("add"), "add_1");
  EXPECT_EQ(root.GetUniqueNameForOp("add"), "add_2");
  EXPECT_EQ(root.GetUniqueNameForOp("mul"), "mul");
}

TEST(ScopeTest, OpAndScopeNameCollision) {
  Scope root = Scope::NewRootScope();
  EXPECT_EQ(root.GetUniqueNameForOp("foo"), "foo");
  EXPECT_EQ(root.GetUniqueNameForOp("foo"), "foo_1");
  EXPECT_EQ(root.GetUniqueNameForOp("foo_1"), "foo_1_1");
  EXPECT_EQ(root.GetUniqueNameForOp("foo_2"), "foo_2");
  EXPECT_EQ(root.GetUniqueNameForOp("foo"), "foo_3");
  EXPECT_EQ(root.GetUniqueNameForOp("foo_2"), "foo_2_1");
}

TEST(ScopeTest, HierarchicalNames) {
  Scope root = Scope::NewRootScope();
  Scope child = root.NewSubScope("child");
  EXPECT_EQ(child.GetUniqueNameForOp("add"), "child/add");
  EXPECT_EQ(child.GetUniqueNameForOp("add"), "child/add_1");
  EXPECT_EQ(child.GetUniqueNameForOp("mul"), "child/mul");

  Scope child_1 = root.NewSubScope("child");
  EXPECT_EQ(child_1.GetUniqueNameForOp("add"), "child_1/add");
  EXPECT_EQ(child_1.GetUniqueNameForOp("add"), "child_1/add_1");
  EXPECT_EQ(child_1.GetUniqueNameForOp("mul"), "child_1/mul");

  Scope c_c = root.NewSubScope("c").NewSubScope("c");
  EXPECT_EQ(c_c.GetUniqueNameForOp("add"), "c/c/add");

  Scope c_1 = root.NewSubScope("c");
  Scope c_1_c = c_1.NewSubScope("c");
  EXPECT_EQ(c_1_c.GetUniqueNameForOp("add"), "c_1/c/add");

  Scope c_1_c_1 = c_1.NewSubScope("c");
  EXPECT_EQ(c_1_c_1.GetUniqueNameForOp("add"), "c_1/c_1/add");

  EXPECT_EQ(root.NewSubScope("").NewSubScope("").GetUniqueNameForOp("d"), "d");
  EXPECT_EQ(root.NewSubScope("").GetUniqueNameForOp("d"), "d_1");
  EXPECT_EQ(root.GetUniqueNameForOp("d"), "d_2");
}

TEST(ScopeTest, ScopeAndOpNames) {
  Scope root = Scope::NewRootScope();
  Scope child = root.NewSubScope("child");

  EXPECT_EQ(child.GetUniqueNameForOp("add"), "child/add");
  EXPECT_EQ(root.GetUniqueNameForOp("child"), "child_1");

  EXPECT_EQ(root.NewSubScope("child").GetUniqueNameForOp("p"), "child_2/p");
}

namespace {

string LastOp(const Scope& scope) { return scope.GetUniqueNameForOp("Last"); }

std::vector<string> AnotherCompositeOp(const Scope& scope) {
  auto cop_scopes = scope.GetCompositeOpScopes("another_cop");
  const string c1 = cop_scopes.child.GetUniqueNameForOp("c1");
  const string c2 = cop_scopes.child.GetUniqueNameForOp("mul");
  return {c1, c2, LastOp(cop_scopes.last)};
}

std::vector<string> LinearOp(const Scope& scope) {
  auto cop_scopes = scope.GetCompositeOpScopes("linear");
  Scope linear = cop_scopes.child;
  const string mul_op_name = linear.GetUniqueNameForOp("mul");
  const string bias_add_op_name = linear.GetUniqueNameForOp("bias_add");
  auto cop_names = AnotherCompositeOp(cop_scopes.last);
  return {mul_op_name, bias_add_op_name, cop_names[0], cop_names[1],
          cop_names[2]};
}

}  // namespace

TEST(ScopeTest, CompositeOp) {
  Scope root = Scope::NewRootScope();
  const auto names1 = LinearOp(root);

  EXPECT_EQ(names1[0], "linear/mul");
  EXPECT_EQ(names1[1], "linear/bias_add");
  EXPECT_EQ(names1[2], "linear/c1");
  EXPECT_EQ(names1[3], "linear/mul_1");
  EXPECT_EQ(names1[4], "linear");

  EXPECT_EQ(root.GetUniqueNameForOp("linear"), "linear_1");

  const auto names2 = LinearOp(root);

  EXPECT_EQ(names2[0], "linear_2/mul");
  EXPECT_EQ(names2[1], "linear_2/bias_add");
  EXPECT_EQ(names2[2], "linear_2/c1");
  EXPECT_EQ(names2[3], "linear_2/mul_1");
  EXPECT_EQ(names2[4], "linear_2");

  const auto names3 = LinearOp(root.WithOpName("c"));

  EXPECT_EQ(names3[0], "c/mul");
  EXPECT_EQ(names3[1], "c/bias_add");
  EXPECT_EQ(names3[2], "c/c1");
  EXPECT_EQ(names3[3], "c/mul_1");
  EXPECT_EQ(names3[4], "c");
}

TEST(ScopeTest, SingleUseScope) {
  Scope root = Scope::NewRootScope();
  auto cop_scopes = root.GetCompositeOpScopes("cop");
  // cop_scopes.last is a single use scope
  EXPECT_EQ(cop_scopes.last.GetUniqueNameForOp("foo"), "cop");
  cop_scopes.last.GetUniqueNameForOp("foo");
  // Error status should be set on cop_scopes.last
  EXPECT_FALSE(cop_scopes.last.ok());
}

TEST(ScopeTest, ControlDeps) {
  Scope root = Scope::NewRootScope();
  auto c1 = Operation();
  auto c2 = Operation();
  Scope c = root.WithControlDependencies({c1, c2});
  EXPECT_EQ(c.control_deps().size(), 2);
  Scope c_c = c.WithControlDependencies({Operation()});
  EXPECT_EQ(c_c.control_deps().size(), 3);
}

TEST(ScopeTest, CreateOutput) {
  Scope root = Scope::NewRootScope();
  Output a = ops::Placeholder(root.WithOpName("a"), DT_FLOAT);
  Output add;
  ASSERT_TRUE(
      CreateOutputWithScope("Add", {a, a}, root.WithOpName("add"), &add).ok());
  EXPECT_EQ(add.node()->name(), "add");
  EXPECT_EQ(add.node()->type_string(), "Add");
}

}  // namespace tensorflow
