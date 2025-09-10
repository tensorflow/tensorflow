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
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "xla/python/ifrt/user_context.h"

namespace xla {
namespace ifrt {

UserContextRegistry& UserContextRegistry::Get() {
  static absl::NoDestructor<UserContextRegistry> registry;
  return *registry;
}

TrackedUserContextRef UserContextRegistry::Register(
    UserContextRef user_context) {
  const UserContextId id = user_context->Id();
  absl::MutexLock lock(&mu_);
  auto it = registry_.find(id);
  if (it != registry_.end()) {
    // If the user context is already registered, return the existing
    // `TrackedUserContextRef`. This will avoid duplicate `Unregister()` calls
    // for the same user context ID.
    TrackedUserContextRef tracked_user_context = it->second.lock();
    CHECK(tracked_user_context != nullptr)
        << "Unexpected dangling reference for user context ID: " << id;
    return tracked_user_context;
  }
  auto tracked_user_context = std::shared_ptr<TrackedUserContext>(
      new TrackedUserContext(id, std::move(user_context)));
  registry_.insert(
      {id, std::weak_ptr<TrackedUserContext>(tracked_user_context)});
  return tracked_user_context;
}

TrackedUserContextRef UserContextRegistry::Lookup(UserContextId id) const {
  absl::MutexLock lock(&mu_);
  auto it = registry_.find(id);
  if (it != registry_.end()) {
    TrackedUserContextRef tracked_user_context = it->second.lock();
    CHECK(tracked_user_context != nullptr)
        << "Unexpected dangling reference for user context ID: " << id;
    return tracked_user_context;
  }
  return nullptr;
}

std::vector<TrackedUserContextRef> UserContextRegistry::LookupAll() const {
  absl::MutexLock lock(&mu_);
  std::vector<TrackedUserContextRef> tracked_user_contexts;
  tracked_user_contexts.reserve(registry_.size());
  for (auto it = registry_.begin(); it != registry_.end(); ++it) {
    TrackedUserContextRef tracked_user_context = it->second.lock();
    CHECK(tracked_user_context != nullptr)
        << "Unexpected dangling reference for user context ID: " << it->first;
    tracked_user_contexts.push_back(std::move(tracked_user_context));
  }
  return tracked_user_contexts;
}

void UserContextRegistry::Unregister(UserContextId id) {
  absl::MutexLock lock(&mu_);
  CHECK_EQ(registry_.erase(id), 1);
}

}  // namespace ifrt
}  // namespace xla
