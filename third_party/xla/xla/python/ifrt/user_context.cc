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
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/random.h"

namespace xla {
namespace ifrt {

char UserContext::ID = 0;  // For llvm::RTTI

namespace {

const auto kNullContext = absl::NoDestructor<UserContextRef>(UserContextRef());

ABSL_CONST_INIT thread_local
    absl_nullable const UserContextRef* current_context = nullptr;

}  // namespace

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

namespace internal {

class UserContextWithCustomFormatter
    : public llvm::RTTIExtends<UserContextWithCustomFormatter, UserContext> {
 public:
  explicit UserContextWithCustomFormatter(
      absl::AnyInvocable<std::string(absl_nullable const UserContextRef&) const>
          formatter,
      absl_nullable UserContextRef user_context)
      : id_(tsl::random::ThreadLocalNew64()),
        formatter_(std::move(formatter)),
        user_context_(std::move(user_context)) {}

  // Not copyable or movable.
  UserContextWithCustomFormatter(const UserContextWithCustomFormatter&) =
      delete;
  UserContextWithCustomFormatter& operator=(
      const UserContextWithCustomFormatter&) = delete;

  // `UserContext` implementation.

  uint64_t Fingerprint() const override {
    // This is not a correct fingerprint because
    // `UserContextWithCustomFormatter` with different formatters but the same
    // user context would give the same fingerprint. We accept this limitation
    // for now because will remove `Fingerprint()` and only use `Id()` instead.
    if (user_context_) {
      return user_context_->Fingerprint();
    }
    return 0;
  }

  UserContextId Id() const override { return id_; }

  std::string DebugString() const override { return formatter_(user_context_); }

  // No new `ID` is not defined because this class is not exposed for RTTI.

 private:
  UserContextId id_;
  absl::AnyInvocable<std::string(absl_nullable const UserContextRef&) const>
      formatter_;
  absl_nullable UserContextRef user_context_;
};

}  // namespace internal

UserContextRef UserContextFormat(
    absl::AnyInvocable<std::string(absl_nullable const UserContextRef&) const>
        formatter,
    absl_nullable UserContextRef user_context) {
  return tsl::MakeRef<internal::UserContextWithCustomFormatter>(
      std::move(formatter), std::move(user_context));
}

}  // namespace ifrt
}  // namespace xla
