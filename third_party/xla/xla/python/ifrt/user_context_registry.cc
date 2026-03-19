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

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/python/ifrt/user_context.h"

namespace xla {
namespace ifrt {

UserContextRegistry& UserContextRegistry::Get() {
  static absl::NoDestructor<UserContextRegistry> registry;
  return *registry;
}

absl_nullable TrackedUserContextRef
UserContextRegistry::Register(absl_nullable UserContextRef user_context) {
  if (user_context == nullptr) {
    return nullptr;
  }
  const UserContextId id = user_context->Id();
  absl::MutexLock lock(mu_);
  auto it = registry_.find(id);
  if (it != registry_.end()) {
    // If the user context is already registered, return the existing
    // `TrackedUserContextRef`. This will avoid duplicate `Unregister()` calls
    // for the same user context ID.
    TrackedUserContextRef tracked_user_context = it->second.first.lock();
    if (tracked_user_context != nullptr) {
      return tracked_user_context;
    }
    // We can fail to obtain a shared pointer when `TrackedUserContext` just has
    // lost the last reference, but it has not called `Unregister()` yet. We
    // proactively unregister the stale entry and proceed to register a new one.
    registry_.erase(it);
  }
  auto tracked_user_context = std::shared_ptr<TrackedUserContext>(
      new TrackedUserContext(id, std::move(user_context)));
  registry_.insert({id,
                    {std::weak_ptr<TrackedUserContext>(tracked_user_context),
                     tracked_user_context.get()}});
  return tracked_user_context;
}

absl_nullable TrackedUserContextRef
UserContextRegistry::Lookup(UserContextId id) const {
  absl::MutexLock lock(mu_);
  auto it = registry_.find(id);
  if (it != registry_.end()) {
    // This may return `nullptr` if the `TrackedUserContext` has been destroyed
    // but `Unregister()` has not been called yet. This is acceptable behavior
    // because `nullptr` means that no matching `UserContext` is found.
    return it->second.first.lock();
  }
  return nullptr;
}

std::vector<absl_nonnull TrackedUserContextRef> UserContextRegistry::LookupAll()
    const {
  absl::MutexLock lock(mu_);
  std::vector<absl_nonnull TrackedUserContextRef> tracked_user_contexts;
  tracked_user_contexts.reserve(registry_.size());
  for (auto it = registry_.begin(); it != registry_.end(); ++it) {
    TrackedUserContextRef tracked_user_context = it->second.first.lock();
    if (tracked_user_context != nullptr) {
      tracked_user_contexts.push_back(std::move(tracked_user_context));
    }
  }
  return tracked_user_contexts;
}

void UserContextRegistry::Unregister(
    UserContextId id, const TrackedUserContext* tracked_user_context) {
  absl::MutexLock lock(mu_);
  auto it = registry_.find(id);
  if (it != registry_.end() && it->second.second == tracked_user_context) {
    registry_.erase(it);
  }
}

CustomStatusExpanderRegistry& CustomStatusExpanderRegistry::Get() {
  static absl::NoDestructor<CustomStatusExpanderRegistry> registry;
  return *registry;
}

void CustomStatusExpanderRegistry::Register(absl::string_view payload_name,
                                            PayloadExpanderFn expander,
                                            std::optional<int> process_order) {
  absl::WriterMutexLock lock(mu_);
  std::pair<int, std::string> key = {
      process_order.value_or(std::numeric_limits<int>::max()),
      std::string(payload_name)};
  CHECK(registry_.insert({std::move(key), std::move(expander)}).second);
}

void CustomStatusExpanderRegistry::Process(absl::Status& status) {
  absl::ReaderMutexLock lock(mu_);
  for (const auto& [_, expander] : registry_) {
    expander(status);
  }
}

}  // namespace ifrt
}  // namespace xla
