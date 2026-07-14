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

#include <utility>

#include <gtest/gtest.h>
#include "xla/python/ifrt/user_context.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

TEST(IfrtBackendUserContextTest, CreateWithZeroId) {
  UserContextRef user_context =
      IfrtBackendUserContext::Create(UserContextId(0), [](UserContextId) {});
  EXPECT_EQ(user_context, UserContextRef());
}

TEST(IfrtBackendUserContextTest, CreateWithNonZeroId) {
  UserContextRef user_context =
      IfrtBackendUserContext::Create(UserContextId(100), [](UserContextId) {});
  EXPECT_EQ(user_context->Id(), UserContextId(100));
  EXPECT_EQ(user_context->DebugString(), "IfrtProxyServerUserContext(100)");
}

TEST(IfrtBackendUserContextTest, DestroyUserContext) {
  auto [promise, future] = tsl::MakePromise<UserContextId>();
  {
    UserContextRef user_context = IfrtBackendUserContext::Create(
        UserContextId(100), [promise = std::move(promise)](
                                UserContextId id) mutable { promise.Set(id); });
  }
  TF_ASSERT_OK_AND_ASSIGN(UserContextId id, future.Await());
  EXPECT_EQ(id, UserContextId(100));
}

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
