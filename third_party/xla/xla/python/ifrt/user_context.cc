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

#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/no_destructor.h"
#include "absl/log/check.h"

namespace xla {
namespace ifrt {

char UserContext::ID = 0;  // For llvm::RTTI

namespace {

const auto kNullContext = absl::NoDestructor<UserContextRef>(UserContextRef());

ABSL_CONST_INIT thread_local const UserContextRef* current_context = nullptr;

}  // namespace

UserContextScope::UserContextScope(UserContextRef context)
    : outer_context_(current_context), context_(std::move(context)) {
  current_context = &context_;
}

UserContextScope::~UserContextScope() {
  CHECK(current_context == &context_);
  current_context = outer_context_;
}

const UserContextRef& UserContextScope::current() {
  if (current_context == nullptr) {
    return *kNullContext;
  }
  return *current_context;
}

}  // namespace ifrt
}  // namespace xla
