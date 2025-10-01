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

#include "xla/python/ifrt/user_context_registry.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/barrier.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/env.h"

namespace xla {
namespace ifrt {
namespace {

class TestUserContext : public llvm::RTTIExtends<TestUserContext, UserContext> {
 public:
  static UserContextRef Create(UserContextId id) {
    return tsl::TakeRef<TestUserContext>(new TestUserContext(id));
  }

  UserContextId Id() const override { return UserContextId(id_); }

  std::string DebugString() const override {
    return absl::StrCat("user context ", id_.value());
  }

  // No new `ID` is not defined because tests below do not exercise RTTI.

 private:
  explicit TestUserContext(UserContextId id) : id_(id) {}

  const UserContextId id_;
};

TEST(UserContextRegistryTest, NullptrUserContext) {
  TrackedUserContextRef tracked_user_context =
      UserContextRegistry::Get().Register(UserContextRef());
  EXPECT_EQ(tracked_user_context, nullptr);
}

TEST(UserContextRegistryTest, RegisterAndLookup) {
  const UserContextId kUserContextId1(100);
  UserContextRef user_context1 = TestUserContext::Create(kUserContextId1);
  TrackedUserContextRef tracked_user_context1 =
      UserContextRegistry::Get().Register(user_context1);
  ASSERT_EQ(UserContextRegistry::Get().Lookup(kUserContextId1),
            tracked_user_context1);
  EXPECT_EQ(UserContextRegistry::Get().Lookup(kUserContextId1)->user_context(),
            user_context1);

  const UserContextId kUserContextId2(200);
  UserContextRef user_context2 = TestUserContext::Create(kUserContextId2);
  TrackedUserContextRef tracked_user_context2 =
      UserContextRegistry::Get().Register(user_context2);
  ASSERT_EQ(UserContextRegistry::Get().Lookup(kUserContextId2),
            tracked_user_context2);
  EXPECT_EQ(UserContextRegistry::Get().Lookup(kUserContextId2)->user_context(),
            user_context2);
}

TEST(UserContextRegistryTest, LookupAll) {
  const UserContextId kUserContextId1(100);
  UserContextRef user_context1 = TestUserContext::Create(kUserContextId1);
  TrackedUserContextRef tracked_user_context1 =
      UserContextRegistry::Get().Register(user_context1);
  const UserContextId kUserContextId2(200);
  UserContextRef user_context2 = TestUserContext::Create(kUserContextId2);
  TrackedUserContextRef tracked_user_context2 =
      UserContextRegistry::Get().Register(user_context2);

  EXPECT_THAT(UserContextRegistry::Get().LookupAll(),
              testing::UnorderedElementsAre(tracked_user_context1,
                                            tracked_user_context2));
}

TEST(UserContextRegistryTest, Unregister) {
  const UserContextId kUserContextId(100);
  EXPECT_EQ(UserContextRegistry::Get().Lookup(kUserContextId), nullptr);
  {
    TrackedUserContextRef tracked_user_context =
        UserContextRegistry::Get().Register(
            TestUserContext::Create(kUserContextId));
    EXPECT_NE(UserContextRegistry::Get().Lookup(kUserContextId), nullptr);
  }
  EXPECT_EQ(UserContextRegistry::Get().Lookup(kUserContextId), nullptr);
}

TEST(UserContextRegistryTest, ConcurrentAccess) {
  const int kNumThreads = 4;
  const int kRepeats = 100000;

  absl::Barrier barrier(kNumThreads);
  std::vector<std::unique_ptr<tsl::Thread>> threads;

  const UserContextId kUserContextId(100);
  for (int i = 0; i < 2; ++i) {
    threads.push_back(absl::WrapUnique(
        tsl::Env::Default()->StartThread(tsl::ThreadOptions(), "test", [&]() {
          barrier.Block();
          for (int i = 0; i < kRepeats; ++i) {
            UserContextRef user_context =
                TestUserContext::Create(kUserContextId);
            TrackedUserContextRef tracked_user_context =
                UserContextRegistry::Get().Register(user_context);
            CHECK_NE(tracked_user_context, nullptr);
            CHECK_EQ(tracked_user_context->user_context()->Id(),
                     kUserContextId);
          }
        })));
  }
  threads.push_back(absl::WrapUnique(
      tsl::Env::Default()->StartThread(tsl::ThreadOptions(), "test", [&]() {
        barrier.Block();
        for (int i = 0; i < kRepeats; ++i) {
          TrackedUserContextRef tracked_user_context =
              UserContextRegistry::Get().Lookup(kUserContextId);
          if (tracked_user_context != nullptr) {
            CHECK_EQ(tracked_user_context->user_context()->Id(),
                     kUserContextId);
          }
        }
      })));
  threads.push_back(absl::WrapUnique(
      tsl::Env::Default()->StartThread(tsl::ThreadOptions(), "test", [&]() {
        barrier.Block();
        for (int i = 0; i < kRepeats; ++i) {
          std::vector<TrackedUserContextRef> tracked_user_contexts =
              UserContextRegistry::Get().LookupAll();
          if (!tracked_user_contexts.empty()) {
            CHECK_EQ(tracked_user_contexts.size(), 1);
            CHECK_EQ(tracked_user_contexts.front()->user_context()->Id(),
                     kUserContextId);
          }
        }
      })));
  CHECK_EQ(threads.size(), kNumThreads);
  threads.clear();  // Join all threads.
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
