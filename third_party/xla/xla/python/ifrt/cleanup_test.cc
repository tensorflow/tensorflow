/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/python/ifrt/cleanup.h"

#include <functional>
#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"

namespace xla {
namespace ifrt {
namespace {

int function_pointer_calls = 0;
void IncrementGlobalCounter() { ++function_pointer_calls; }

TEST(CleanupTest, NullptrReturnsNull) {
  std::shared_ptr<const void> hold = MakeSharedCleanup(nullptr);
  EXPECT_EQ(hold, nullptr);
}

TEST(CleanupTest, EmptyStdFunctionReturnsNull) {
  std::function<void()> empty_fn;
  std::shared_ptr<const void> hold = MakeSharedCleanup(std::move(empty_fn));
  EXPECT_EQ(hold, nullptr);
}

TEST(CleanupTest, EmptyAnyInvocableReturnsNull) {
  absl::AnyInvocable<void() &&> empty_fn;
  std::shared_ptr<const void> hold = MakeSharedCleanup(std::move(empty_fn));
  EXPECT_EQ(hold, nullptr);
}

TEST(CleanupTest, NullFunctionPointerReturnsNull) {
  void (*null_fn)() = nullptr;
  std::shared_ptr<const void> hold = MakeSharedCleanup(null_fn);
  EXPECT_EQ(hold, nullptr);
}

TEST(CleanupTest, InvokesCallableAndReleasesCapturesOnReset) {
  std::shared_ptr<int> resource = std::make_shared<int>(42);
  int call_count = 0;

  std::shared_ptr<const void> hold =
      MakeSharedCleanup([resource, &call_count]() { ++call_count; });
  EXPECT_NE(hold, nullptr);
  EXPECT_TRUE(static_cast<bool>(hold));
  EXPECT_EQ(call_count, 0);
  EXPECT_EQ(resource.use_count(), 2);

  hold.reset();
  EXPECT_EQ(call_count, 1);
  EXPECT_EQ(resource.use_count(), 1);
}

TEST(CleanupTest, SharedCopiesInvokeOnceWhenLastCopyResets) {
  int call_count = 0;
  std::shared_ptr<const void> hold1 =
      MakeSharedCleanup([&call_count]() { ++call_count; });
  std::shared_ptr<const void> hold2 = hold1;

  hold1.reset();
  EXPECT_EQ(call_count, 0);

  hold2.reset();
  EXPECT_EQ(call_count, 1);
}

TEST(CleanupTest, SupportsMoveOnlyMutableLambda) {
  int destroyed_value = 0;
  std::unique_ptr<int> owned = std::make_unique<int>(7);

  std::shared_ptr<const void> hold =
      MakeSharedCleanup([owned = std::move(owned), &destroyed_value]() mutable {
        destroyed_value = *owned;
        owned.reset();
      });
  hold.reset();

  EXPECT_EQ(destroyed_value, 7);
}

TEST(CleanupTest, SupportsRvalueQualifiedAnyInvocable) {
  int call_count = 0;
  absl::AnyInvocable<void() &&> fn = [&call_count]() { ++call_count; };

  std::shared_ptr<const void> hold = MakeSharedCleanup(std::move(fn));
  hold.reset();

  EXPECT_EQ(call_count, 1);
}

TEST(CleanupTest, SupportsFunctionPointer) {
  function_pointer_calls = 0;
  {
    std::shared_ptr<const void> hold =
        MakeSharedCleanup(&IncrementGlobalCounter);
    EXPECT_NE(hold, nullptr);
    EXPECT_EQ(function_pointer_calls, 0);
  }
  EXPECT_EQ(function_pointer_calls, 1);
}

TEST(CleanupTest, SupportsLvalueCallable) {
  int call_count = 0;
  std::function<void()> fn = [&call_count]() { ++call_count; };
  {
    std::shared_ptr<const void> hold = MakeSharedCleanup(fn);
    EXPECT_EQ(call_count, 0);
  }
  EXPECT_EQ(call_count, 1);
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
