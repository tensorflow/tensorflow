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

#ifndef XLA_PYTHON_IFRT_USER_CONTEXT_TEST_UTIL_H_
#define XLA_PYTHON_IFRT_USER_CONTEXT_TEST_UTIL_H_

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

class TestUserContext : public llvm::RTTIExtends<TestUserContext, UserContext> {
 public:
  static UserContextRef Create(UserContextId id) {
    return Create(id, absl::StrCat("TestUserContext(", id.value(), ")"));
  }

  static UserContextRef Create(UserContextId id, std::string debug_string) {
    return tsl::TakeRef<TestUserContext>(
        new TestUserContext(id, std::move(debug_string)));
  }

  UserContextId Id() const override { return id_; }

  std::string DebugString() const override { return debug_string_; }

  // No new `ID` is not defined because tests below do not exercise RTTI.

 private:
  explicit TestUserContext(UserContextId id, std::string debug_string)
      : id_(id), debug_string_(std::move(debug_string)) {}

  UserContextId id_;
  std::string debug_string_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_USER_CONTEXT_TEST_UTIL_H_
