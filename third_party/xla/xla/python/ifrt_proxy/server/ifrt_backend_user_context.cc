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
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {
namespace proxy {

[[maybe_unused]] char IfrtBackendUserContext::ID = 0;

void IfrtBackendDestroyedUserContextIds::Add(UserContextId id) {
  absl::MutexLock l(mu_);
  ids_.push_back(id);
}

std::vector<UserContextId> IfrtBackendDestroyedUserContextIds::Consume() {
  absl::MutexLock l(mu_);
  std::vector<UserContextId> destroyed_user_contexts;
  destroyed_user_contexts.swap(ids_);
  return destroyed_user_contexts;
}

UserContextRef IfrtBackendUserContext::Create(
    std::shared_ptr<IfrtBackendDestroyedUserContextIds>
        destroyed_user_context_ids,
    UserContextId id) {
  if (id.value() == 0) {
    return UserContextRef();
  }
  return UserContextRef(tsl::MakeRef<IfrtBackendUserContext>(
      std::move(destroyed_user_context_ids), id));
}

IfrtBackendUserContext::IfrtBackendUserContext(
    std::shared_ptr<IfrtBackendDestroyedUserContextIds>
        destroyed_user_context_ids,
    UserContextId id)
    : destroyed_user_context_ids_(destroyed_user_context_ids), id_(id) {}

IfrtBackendUserContext::~IfrtBackendUserContext() {
  destroyed_user_context_ids_->Add(id_);
}

std::string IfrtBackendUserContext::DebugString() const {
  return absl::StrCat("IfrtBackendUserContext(", id_.value(), ")");
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
