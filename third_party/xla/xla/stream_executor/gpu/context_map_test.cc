/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/gpu/context_map.h"

#include "tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {

// Test context.
class TestContext {
 public:
  TestContext(void *context, int device_ordinal)
      : context_(context), device_ordinal_(device_ordinal) {}

  void *context() const { return context_; }
  int device_ordinal() const { return device_ordinal_; }

 private:
  void *context_;
  int device_ordinal_;
};

TEST(ContextMapTest, AddRemoveAndHasWorks) {
  int device_ordinal = 1;
  void *context = &device_ordinal;
  auto ordinal_finder = [device_ordinal](void *ptr) { return device_ordinal; };
  ContextMap<void *, TestContext> map(ordinal_finder);
  auto *test_context = map.Add(context, device_ordinal);
  EXPECT_EQ(test_context->context(), context);
  EXPECT_EQ(test_context->device_ordinal(), device_ordinal);
  EXPECT_TRUE(map.Has(context));
  map.Remove(context);
  EXPECT_FALSE(map.Has(context));
}

TEST(ContextMapTest, AddTwiceReturnsSameContext) {
  void *context = reinterpret_cast<void *>(2);
  constexpr int device_ordinal = 1;
  auto ordinal_finder = [](void *ptr) { return device_ordinal; };
  ContextMap<void *, TestContext> map(ordinal_finder);
  auto *test_context1 = map.Add(context, device_ordinal);
  auto *test_context2 = map.Add(context, device_ordinal);
  EXPECT_EQ(test_context1, test_context2);
}

TEST(ContextMapTest, GetAnyContextReturnsCorrectContext) {
  // Add two contexts.
  void *context1 = reinterpret_cast<void *>(2);
  void *context2 = reinterpret_cast<void *>(3);
  constexpr int device_ordinal1 = 1;
  constexpr int device_ordinal2 = 2;

  // Make the first call to GetAnyContext return device_ordinal1, everything
  // after device_ordinal2.
  auto ordinal_finder = [](void *ptr) {
    static int calls = 0;
    ++calls;
    if (calls <= 1) {
      return device_ordinal1;
    } else {
      return device_ordinal2;
    }
  };
  ContextMap<void *, TestContext> map(ordinal_finder);
  auto *test_context1 = map.Add(context1, device_ordinal1);
  auto *test_context2 = map.Add(context2, device_ordinal2);
  EXPECT_NE(test_context1, test_context2);
  auto first_context = map.GetAnyContext(context1);
  EXPECT_EQ(first_context, context1);
  auto second_context = map.GetAnyContext(context2);
  EXPECT_EQ(second_context, context2);
}

TEST(ContextMapTest, GetAnyContextShouldDieWithBadInput) {
  // Add two contexts.
  void *context1 = reinterpret_cast<void *>(2);
  void *context2 = reinterpret_cast<void *>(3);
  constexpr int device_ordinal1 = 1;
  constexpr int device_ordinal2 = 2;

  // Make the first call to GetAnyContext return device_ordinal1, everything
  // after device_ordinal2.
  auto ordinal_finder = [](void *ptr) {
    static int calls = 0;
    ++calls;
    if (calls <= 1) {
      return device_ordinal1;
    } else {
      return device_ordinal2;
    }
  };
  ContextMap<void *, TestContext> map(ordinal_finder);
  auto *test_context1 = map.Add(context1, device_ordinal1);
  auto *test_context2 = map.Add(context2, device_ordinal2);
  EXPECT_NE(test_context1, test_context2);
  auto first_context = map.GetAnyContext(context1);
  EXPECT_EQ(first_context, context1);
  auto second_context = map.GetAnyContext(context2);
  EXPECT_EQ(second_context, context2);

  // Remove second context, and GetAnyContext should fail.
  map.Remove(context2);
  EXPECT_DEATH(map.GetAnyContext(context2), "Check failed");
}

}  // namespace
}  // namespace stream_executor::gpu
