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

#include "xla/python/ifrt_proxy/server/ifrt_backend_user_context.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/python/ifrt/user_context.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(IfrtBackendDestroyedUserContextIdsTest, AddAndConsume) {
  auto ids = std::make_shared<IfrtBackendDestroyedUserContextIds>();
  ids->Add(UserContextId(1));
  ids->Add(UserContextId(2));
  ids->Add(UserContextId(3));
  EXPECT_THAT(ids->Consume(), ElementsAre(1, 2, 3));
  EXPECT_THAT(ids->Consume(), IsEmpty());
}

TEST(IfrtBackendUserContextTest, CreateWithZeroId) {
  UserContextRef user_context = IfrtBackendUserContext::Create(
      std::make_shared<IfrtBackendDestroyedUserContextIds>(), UserContextId(0));
  EXPECT_EQ(user_context, UserContextRef());
}

TEST(IfrtBackendUserContextTest, CreateWithNonZeroId) {
  auto ids = std::make_shared<IfrtBackendDestroyedUserContextIds>();
  UserContextRef user_context =
      IfrtBackendUserContext::Create(ids, UserContextId(100));
  EXPECT_EQ(user_context->Id(), UserContextId(100));
  EXPECT_EQ(user_context->DebugString(), "IfrtBackendUserContext(100)");
}

TEST(IfrtBackendUserContextTest, DestructionAddsIdToDestroyedIds) {
  auto ids = std::make_shared<IfrtBackendDestroyedUserContextIds>();
  EXPECT_THAT(ids->Consume(), IsEmpty());
  {
    UserContextRef user_context =
        IfrtBackendUserContext::Create(ids, UserContextId(100));
    EXPECT_THAT(ids->Consume(), IsEmpty());
  }
  EXPECT_THAT(ids->Consume(), ElementsAre(100));
}

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
