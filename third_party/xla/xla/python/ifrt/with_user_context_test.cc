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

#include "xla/python/ifrt/with_user_context.h"

#include <functional>
#include <optional>
#include <utility>

#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/user_context_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla {
namespace ifrt {
namespace {

absl::Status CheckUserContext() {
  if (UserContextScope::current() == nullptr) {
    return absl::InternalError("no user context");
  }
  if (UserContextScope::current()->Id() != UserContextId(100)) {
    return absl::InternalError(absl::StrCat(
        "wrong user context:", UserContextScope::current()->Id().value()));
  }
  return absl::OkStatus();
}

TEST(WithUserContextTest, Function) {
  {
    std::function<absl::Status()> f =
        WithUserContext(std::function<absl::Status()>(CheckUserContext),
                        TestUserContext::Create(UserContextId(100)));
    TF_EXPECT_OK(f());
    TF_EXPECT_OK(std::move(f)());
  }
  {
    const std::function<absl::Status()> f =
        WithUserContext(std::function<absl::Status()>(CheckUserContext),
                        TestUserContext::Create(UserContextId(100)));
    TF_EXPECT_OK(f());
    TF_EXPECT_OK(std::move(f)());
  }
}

TEST(WithUserContextTest, AnyInvocable) {
  {
    absl::AnyInvocable<absl::Status()> f =
        WithUserContext(absl::AnyInvocable<absl::Status()>(CheckUserContext),
                        TestUserContext::Create(UserContextId(100)));
    TF_EXPECT_OK(f());
    TF_EXPECT_OK(std::move(f)());
  }
  {
    const absl::AnyInvocable<absl::Status() const> f = WithUserContext(
        absl::AnyInvocable<absl::Status() const>(CheckUserContext),
        TestUserContext::Create(UserContextId(100)));
    TF_EXPECT_OK(f());
    TF_EXPECT_OK(std::move(f)());
  }
}

TEST(WithCurrentUserContextTest, Function) {
  std::function<absl::Status()> f;
  {
    UserContextScope user_context_scope(
        TestUserContext::Create(UserContextId(100)));
    f = WithCurrentUserContext(std::function<absl::Status()>(CheckUserContext));
    TF_EXPECT_OK(f());
    TF_EXPECT_OK(std::move(f)());
  }
  {
    std::optional<const std::function<absl::Status()>> f;
    {
      UserContextScope user_context_scope(
          TestUserContext::Create(UserContextId(100)));
      f.emplace(WithCurrentUserContext(
          std::function<absl::Status()>(CheckUserContext)));
      TF_EXPECT_OK((*f)());
      TF_EXPECT_OK((*std::move(f))());
    }
  }
}

TEST(WithCurrentUserContextTest, AnyInvocable) {
  {
    absl::AnyInvocable<absl::Status()> f;
    {
      UserContextScope user_context_scope(
          TestUserContext::Create(UserContextId(100)));
      f = WithCurrentUserContext(
          absl::AnyInvocable<absl::Status()>(CheckUserContext));
    }
    TF_EXPECT_OK(f());
    TF_EXPECT_OK(std::move(f)());
  }
  {
    std::optional<const absl::AnyInvocable<absl::Status() const>> f;
    {
      UserContextScope user_context_scope(
          TestUserContext::Create(UserContextId(100)));
      f.emplace(WithCurrentUserContext(
          absl::AnyInvocable<absl::Status() const>(CheckUserContext)));
    }
    TF_EXPECT_OK((*f)());
    TF_EXPECT_OK((*std::move(f))());
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
