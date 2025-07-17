/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/python/ifrt/user_context.h"

#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace ifrt {

namespace {

class TestUserContext : public llvm::RTTIExtends<TestUserContext, UserContext> {
 public:
  static UserContextRef Create() { return tsl::MakeRef<TestUserContext>(); }

  uint64_t Fingerprint() const override { return 1; }

  std::string DebugString() const override { return ""; }

  // No new `ID` is not defined because tests below do not exercise RTTI.
};

TEST(UserContextScopeTest, NullContext) {
  EXPECT_EQ(UserContextScope::current(), nullptr);
}

TEST(UserContextScopeTest, SingleScope) {
  UserContextRef context = TestUserContext::Create();
  UserContextScope scope(context);
  EXPECT_EQ(UserContextScope::current(), context);
}

TEST(UserContextScopeTest, SingleScopeWithInlineContextCreation) {
  UserContextScope scope(TestUserContext::Create());
  EXPECT_EQ(UserContextScope::current()->Fingerprint(), 1);
}

TEST(UserContextScopeTest, NestedScopes) {
  UserContextRef context1 = TestUserContext::Create();
  UserContextRef context2 = TestUserContext::Create();
  UserContextScope scope1(context1);
  EXPECT_EQ(UserContextScope::current(), context1);
  {
    UserContextScope scope2(context2);
    EXPECT_EQ(UserContextScope::current(), context2);
  }
  EXPECT_EQ(UserContextScope::current(), context1);
}

TEST(UserContextScopeTest, ThreadLocalScopes) {
  UserContextRef context = TestUserContext::Create();
  UserContextScope scope(context);
  EXPECT_EQ(UserContextScope::current(), context);

  tsl::thread::ThreadPool thread_pool1(tsl::Env::Default(),
                                       tsl::ThreadOptions(), "pool1", 10);
  tsl::thread::ThreadPool thread_pool2(tsl::Env::Default(),
                                       tsl::ThreadOptions(), "pool2", 10);

  // The effect of UserContextScope set is limited to the current thread.
  for (int i = 0; i < 100; ++i) {
    thread_pool1.Schedule([&]() {
      UserContextRef context1 = TestUserContext::Create();
      UserContextScope scope1(context1);
      EXPECT_EQ(UserContextScope::current(), context1);
      absl::SleepFor(absl::Microseconds(10));
    });
  }
  for (int i = 0; i < 100; ++i) {
    thread_pool2.Schedule([&]() {
      UserContextRef context2 = TestUserContext::Create();
      UserContextScope scope1(context2);
      EXPECT_EQ(UserContextScope::current(), context2);
      absl::SleepFor(absl::Microseconds(10));
    });
  }
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(UserContextScope::current(), context);
    absl::SleepFor(absl::Microseconds(10));
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
