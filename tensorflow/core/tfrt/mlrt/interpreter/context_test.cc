/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>

namespace mlrt {
namespace {

struct A : KernelFrame {
  static constexpr char kName[] = "A";
  using KernelFrame::KernelFrame;
  void Invoke() {}
};

struct B : KernelFrame {
  static constexpr char kName[] = "B";
  using KernelFrame::KernelFrame;
  void Invoke() {}
};

struct C : KernelFrame {
  static constexpr char kName[] = "C";
  using KernelFrame::KernelFrame;
  void Invoke() {}
};

TEST(ContextTest, MergeKernelRegistry) {
  KernelRegistry reg_a;
  reg_a.Register<A>();
  reg_a.Register<B>();

  KernelRegistry reg_b;
  reg_b.Register<B>();
  reg_b.Register<C>();

  EXPECT_TRUE(reg_a.Get(A::kName));
  EXPECT_TRUE(reg_a.Get(B::kName));

  reg_a.Merge(reg_b);

  EXPECT_TRUE(reg_a.Get(A::kName));
  EXPECT_TRUE(reg_a.Get(B::kName));
  EXPECT_TRUE(reg_a.Get(C::kName));
}

struct TestContext0 : UserContext<TestContext0> {
  int v = 0;
};
struct TestContext1 : UserContext<TestContext1> {
  int v = 1;
};

TEST(ContextTest, UserContext) {
  EXPECT_EQ(TestContext0::id(), 0);
  EXPECT_EQ(TestContext1::id(), 1);

  ExecutionContext execution_context(/*loaded_executable=*/nullptr);

  auto test_1 = std::make_unique<TestContext1>();
  auto* test_1_ptr = test_1.get();
  execution_context.AddUserContext(std::move(test_1));

  auto test_0 = std::make_unique<TestContext0>();
  auto* test_0_ptr = test_0.get();
  execution_context.AddUserContext(std::move(test_0));

  EXPECT_EQ(&execution_context.GetUserContext<TestContext0>(), test_0_ptr);
  EXPECT_EQ(&execution_context.GetUserContext<TestContext1>(), test_1_ptr);
  EXPECT_EQ(execution_context.GetUserContext<TestContext0>().v, 0);
  EXPECT_EQ(execution_context.GetUserContext<TestContext1>().v, 1);

  ExecutionContext execution_context_copy(/*loaded_executable=*/nullptr,
                                          execution_context.CopyUserContexts());
  EXPECT_NE(&execution_context_copy.GetUserContext<TestContext0>(), test_0_ptr);
  EXPECT_NE(&execution_context_copy.GetUserContext<TestContext1>(), test_1_ptr);

  EXPECT_EQ(execution_context_copy.GetUserContext<TestContext0>().v, 0);
  EXPECT_EQ(execution_context_copy.GetUserContext<TestContext1>().v, 1);
}

TEST(ContextTest, PartialUserContext) {
  EXPECT_EQ(TestContext0::id(), 0);
  EXPECT_EQ(TestContext1::id(), 1);

  ExecutionContext execution_context(/*loaded_executable=*/nullptr);

  auto test_1 = std::make_unique<TestContext1>();
  auto* test_1_ptr = test_1.get();
  execution_context.AddUserContext(std::move(test_1));

  EXPECT_EQ(&execution_context.GetUserContext<TestContext1>(), test_1_ptr);
  EXPECT_EQ(execution_context.GetUserContext<TestContext1>().v, 1);

  ExecutionContext execution_context_copy(/*loaded_executable=*/nullptr,
                                          execution_context.CopyUserContexts());
  EXPECT_NE(&execution_context_copy.GetUserContext<TestContext1>(), test_1_ptr);

  EXPECT_EQ(execution_context_copy.GetUserContext<TestContext1>().v, 1);
}

}  // namespace
}  // namespace mlrt
