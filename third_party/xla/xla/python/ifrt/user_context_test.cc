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

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
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
  static UserContextRef Create(UserContextId id) {
    return tsl::TakeRef<TestUserContext>(new TestUserContext(id));
  }

  UserContextId Id() const override { return id_; }

  std::string DebugString() const override {
    return absl::StrCat("user context ", id_.value());
  }

  // No new `ID` is not defined because tests below do not exercise RTTI.

 private:
  explicit TestUserContext(UserContextId id) : id_(id) {}

  UserContextId id_;
};

TEST(AnnotatedUserContextTest, Id) {
  const UserContextId kUserContextId(100);
  UserContextRef context = TestUserContext::Create(kUserContextId);

  UserContextRef annotated_context1 =
      AnnotatedUserContext::Create(context, "test annotation");
  EXPECT_NE(annotated_context1->Id(), context->Id());

  UserContextRef annotated_context2 =
      AnnotatedUserContext::Create(context, "test annotation 2");
  EXPECT_NE(annotated_context2->Id(), annotated_context1->Id());

  UserContextRef annotated_context3 =
      AnnotatedUserContext::Create(UserContextRef(), "test annotation");
  EXPECT_NE(annotated_context3->Id(), annotated_context1->Id());
}

TEST(AnnotatedUserContextTest, DebugString) {
  {
    const UserContextId kUserContextId(100);
    UserContextRef context = TestUserContext::Create(kUserContextId);
    UserContextRef annotated_context =
        AnnotatedUserContext::Create(context, "test annotation");
    EXPECT_EQ(annotated_context->DebugString(),
              "user context 100; test annotation");
  }
  {
    UserContextRef annotated_context =
        AnnotatedUserContext::Create(UserContextRef(), "test annotation");
    EXPECT_EQ(annotated_context->DebugString(),
              "(nullptr user context); test annotation");
  }
}

TEST(ChainedUserContextTest, Id) {
  const UserContextId kUserContextId1(100);
  const UserContextId kUserContextId2(200);
  UserContextRef context1 = TestUserContext::Create(kUserContextId1);
  UserContextRef context2 = TestUserContext::Create(kUserContextId2);
  UserContextRef chained_context =
      ChainedUserContext::Create({context1, UserContextRef(), context2});
  EXPECT_NE(chained_context->Id(), context1->Id());
  EXPECT_NE(chained_context->Id(), context2->Id());
}

TEST(ChainedUserContextTest, DebugString) {
  const UserContextId kUserContextId1(100);
  const UserContextId kUserContextId2(200);
  UserContextRef context1 = TestUserContext::Create(kUserContextId1);
  UserContextRef context2 = TestUserContext::Create(kUserContextId2);
  UserContextRef chained_context =
      ChainedUserContext::Create({context1, UserContextRef(), context2});
  EXPECT_EQ(chained_context->DebugString(),
            "user context 100\n\n ->\n\n(nullptr user context)\n\n ->\n\nuser "
            "context 200");
}

TEST(FusedUserContextTest, Id) {
  const UserContextId kUserContextId1(100);
  const UserContextId kUserContextId2(200);
  UserContextRef context1 = TestUserContext::Create(kUserContextId1);
  UserContextRef context2 = TestUserContext::Create(kUserContextId2);
  UserContextRef fused_context =
      FusedUserContext::Create({context1, UserContextRef(), context2});
  EXPECT_NE(fused_context->Id(), context1->Id());
  EXPECT_NE(fused_context->Id(), context2->Id());
}

TEST(FusedUserContextTest, DebugString) {
  const UserContextId kUserContextId1(100);
  const UserContextId kUserContextId2(200);
  UserContextRef context1 = TestUserContext::Create(kUserContextId1);
  UserContextRef context2 = TestUserContext::Create(kUserContextId2);
  UserContextRef fused_context =
      FusedUserContext::Create({context1, UserContextRef(), context2});
  EXPECT_EQ(fused_context->DebugString(),
            "Fused user context: {\n\nuser context 100\n\n(nullptr user "
            "context)\n\nuser context 200\n\n}");
}

TEST(UserContextScopeTest, NullContext) {
  EXPECT_EQ(UserContextScope::current(), nullptr);
}

TEST(UserContextScopeTest, SingleScope) {
  const UserContextId kUserContextId(100);
  UserContextRef context = TestUserContext::Create(kUserContextId);
  UserContextScope scope(context);
  EXPECT_EQ(UserContextScope::current(), context);
}

TEST(UserContextScopeTest, SingleScopeWithInlineContextCreation) {
  const UserContextId kUserContextId(100);
  UserContextScope scope(TestUserContext::Create(kUserContextId));
  EXPECT_EQ(UserContextScope::current()->Id(), kUserContextId);
}

TEST(UserContextScopeTest, NestedScopes) {
  const UserContextId kUserContextId1(100);
  const UserContextId kUserContextId2(200);
  UserContextRef context1 = TestUserContext::Create(kUserContextId1);
  UserContextRef context2 = TestUserContext::Create(kUserContextId2);
  UserContextScope scope1(context1);
  EXPECT_EQ(UserContextScope::current(), context1);
  {
    UserContextScope scope2(context2);
    EXPECT_EQ(UserContextScope::current(), context2);
  }
  EXPECT_EQ(UserContextScope::current(), context1);
}

TEST(UserContextScopeTest, ThreadLocalScopes) {
  const UserContextId kUserContextId(100);
  UserContextRef context = TestUserContext::Create(kUserContextId);
  UserContextScope scope(context);
  EXPECT_EQ(UserContextScope::current(), context);

  tsl::thread::ThreadPool thread_pool1(tsl::Env::Default(),
                                       tsl::ThreadOptions(), "pool1", 10);
  tsl::thread::ThreadPool thread_pool2(tsl::Env::Default(),
                                       tsl::ThreadOptions(), "pool2", 10);

  // The effect of UserContextScope set is limited to the current thread.
  for (int i = 0; i < 100; ++i) {
    thread_pool1.Schedule([&]() {
      UserContextRef context1 = TestUserContext::Create(kUserContextId);
      UserContextScope scope1(context1);
      EXPECT_EQ(UserContextScope::current(), context1);
      absl::SleepFor(absl::Microseconds(10));
    });
  }
  for (int i = 0; i < 100; ++i) {
    thread_pool2.Schedule([&]() {
      UserContextRef context2 = TestUserContext::Create(kUserContextId);
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
