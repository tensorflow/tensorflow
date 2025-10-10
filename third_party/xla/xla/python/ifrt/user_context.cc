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
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/random.h"

namespace xla {
namespace ifrt {

// For llvm::RTTI
[[maybe_unused]] char UserContext::ID = 0;
[[maybe_unused]] char AnnotatedUserContext::ID = 0;
[[maybe_unused]] char ChainedUserContext::ID = 0;
[[maybe_unused]] char FusedUserContext::ID = 0;

namespace {

const auto kNullContext = absl::NoDestructor<UserContextRef>(UserContextRef());

ABSL_CONST_INIT thread_local
    absl_nullable const UserContextRef* current_context = nullptr;

}  // namespace

absl_nonnull UserContextRef
AnnotatedUserContext::Create(UserContextRef user_context, std::string msg) {
  return tsl::MakeRef<AnnotatedUserContext>(std::move(user_context),
                                            std::move(msg));
}

AnnotatedUserContext::AnnotatedUserContext(UserContextRef user_context,
                                           std::string msg)
    : id_(tsl::random::ThreadLocalNew64()),
      user_context_(std::move(user_context)),
      msg_(std::move(msg)) {}

UserContextId AnnotatedUserContext::Id() const { return id_; }

std::string AnnotatedUserContext::DebugString() const {
  return absl::StrCat(
      (user_context_ ? user_context_->DebugString() : "(nullptr user context)"),
      "; ", msg_);
}

absl_nonnull UserContextRef
ChainedUserContext::Create(absl::Span<const UserContextRef> user_contexts) {
  return tsl::MakeRef<ChainedUserContext>(user_contexts);
}

ChainedUserContext::ChainedUserContext(
    absl::Span<const UserContextRef> user_contexts)
    : id_(tsl::random::ThreadLocalNew64()),
      user_contexts_(user_contexts.begin(), user_contexts.end()) {}

UserContextId ChainedUserContext::Id() const { return id_; }

std::string ChainedUserContext::DebugString() const {
  return absl::StrJoin(
      user_contexts_, "\n\n ->\n\n",
      [](std::string* out, const UserContextRef& user_context) {
        absl::StrAppend(out, (user_context ? user_context->DebugString()
                                           : "(nullptr user context)"));
      });
}

absl_nonnull UserContextRef
FusedUserContext::Create(absl::Span<const UserContextRef> user_contexts) {
  return tsl::MakeRef<FusedUserContext>(user_contexts);
}

FusedUserContext::FusedUserContext(
    absl::Span<const UserContextRef> user_contexts)
    : id_(tsl::random::ThreadLocalNew64()),
      user_contexts_(user_contexts.begin(), user_contexts.end()) {}

UserContextId FusedUserContext::Id() const { return id_; }

std::string FusedUserContext::DebugString() const {
  return absl::StrCat(
      "Fused user context: {\n\n",
      absl::StrJoin(user_contexts_, "\n\n",
                    [](std::string* out, const UserContextRef& user_context) {
                      absl::StrAppend(
                          out, (user_context ? user_context->DebugString()
                                             : "(nullptr user context)"));
                    }),
      "\n\n}");
}

UserContextScope::UserContextScope(absl_nullable UserContextRef context)
    : outer_context_(current_context), context_(std::move(context)) {
  current_context = &context_;
}

UserContextScope::~UserContextScope() {
  CHECK(current_context == &context_);
  current_context = outer_context_;
}

absl_nullable const UserContextRef& UserContextScope::current() {
  if (current_context == nullptr) {
    return *kNullContext;
  }
  return *current_context;
}

}  // namespace ifrt
}  // namespace xla
