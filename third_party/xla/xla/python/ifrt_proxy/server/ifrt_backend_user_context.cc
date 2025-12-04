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

#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/strings/str_cat.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {
namespace proxy {

[[maybe_unused]] char IfrtBackendUserContext::ID = 0;

UserContextRef IfrtBackendUserContext::Create(
    UserContextId original_id,
    absl::AnyInvocable<void(UserContextId) &&> on_destroyed) {
  if (original_id.value() == 0) {
    return UserContextRef();
  }
  return UserContextRef(tsl::MakeRef<IfrtBackendUserContext>(
      original_id, std::move(on_destroyed)));
}

IfrtBackendUserContext::IfrtBackendUserContext(
    UserContextId original_id,
    absl::AnyInvocable<void(UserContextId) &&> on_destroyed)
    : original_id_(original_id), on_destroyed_(std::move(on_destroyed)) {}

IfrtBackendUserContext::~IfrtBackendUserContext() {
  std::move(on_destroyed_)(original_id_);
}

std::string IfrtBackendUserContext::DebugString() const {
  return absl::StrCat("IfrtProxyServerUserContext(", original_id_.value(), ")");
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
