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

#ifndef XLA_PYTHON_IFRT_USER_CONTEXT_REGISTRY_H_
#define XLA_PYTHON_IFRT_USER_CONTEXT_REGISTRY_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/python/ifrt/user_context.h"

namespace xla {
namespace ifrt {

class TrackedUserContext;
using TrackedUserContextRef = std::shared_ptr<const TrackedUserContext>;

// Global registry for tracking live `UserContextRef`.
//
// `UserContextRef` can be registered in the process-wide registry. As long as
// the returned `TrackedUserContextRef` is alive, the corresponding
// `UserContextRef` will be kept alive and can be looked up via the registry by
// using its ID. Once all `TrackedUserContextRef`s are destroyed, the
// corresponding `UserContextRef` will be removed from the registry (and
// destroyed if there is no other copy of `UserContextRef`).
//
// This registry serves two primary purposes:
//
// * Allow IFRT implementations to attach `UserContextRef` to an error status
// (e.g., `absl::Status`) and propagate it across the IFRT API boundary while
// keeping it alive until the IFRT user expands the error status with user
// context IDs. The reason why a registry is required even when we can keep
// `UserContextRef` alive with a status payload is that the IFRT user still
// needs a way to retrieve the original `UserContextRef`, and they will only
// have the ID to do so and cannot access `UserContextRef` embedded in the
// status payload directly.
//
// * Provide a way for IFRT implementations to enumerate all live
// `UserContextRef`s that may be referenced in an error status. To use this
// feature, IFRT implementations are expected to call `Register()` in the
// beginning of any operation that performs async dispatch, and hold the
// returned `TrackedUserContextRef` until the operation is complete.
//
// All methods are thread-safe.
class UserContextRegistry {
 public:
  // Returns the singleton registry.
  static UserContextRegistry& Get();

  // Ensures that `user_context` is registered in the registry (if not) and
  // returns `TrackedUserContextRef` for `user_context`.
  absl_nullable TrackedUserContextRef
  Register(absl_nullable UserContextRef user_context);

  // Returns `TrackedUserContextRef` for `id`.
  // If no such `id` is found, returns `nullptr`.
  absl_nullable TrackedUserContextRef Lookup(UserContextId id) const;

  // Returns all `TrackedUserContextRef`s in the registry. Note that since the
  // registry is process-wide, the result will contain `TrackedUserContextRef`s
  // seen from all local IFRT client instances.
  std::vector<absl_nonnull TrackedUserContextRef> LookupAll() const;

 private:
  friend TrackedUserContext;

  // Removes a `TrackedUserContext` entry identified by `id` from the registry.
  // If the existing entry does not point to `tracked_user_context`, this is a
  // no-op.
  void Unregister(UserContextId id,
                  const TrackedUserContext* tracked_user_context);

  mutable absl::Mutex mu_;
  // A map from `UserContext::Id()` to a weak reference of `TrackedUserContext`.
  // The raw pointer is used for handling a race condition between `Register()`
  // and destruction of `TrackedUserContext` for the same ID.
  absl::flat_hash_map<
      UserContextId,
      std::pair<std::weak_ptr<TrackedUserContext>, const TrackedUserContext*>>
      registry_ ABSL_GUARDED_BY(mu_);
};

// RAII wrapper around `UserContextRef` to allow querying them via the registry
// while the RAII wrapper is alive.
class TrackedUserContext {
 public:
  // Not copyable or movable. Use `TrackedUserContextRef`.
  TrackedUserContext(const TrackedUserContext&) = delete;
  TrackedUserContext(TrackedUserContext&&) = delete;

  ~TrackedUserContext() { UserContextRegistry::Get().Unregister(id_, this); }

  absl_nonnull const UserContextRef& user_context() const {
    return user_context_;
  }

 private:
  friend UserContextRegistry;

  explicit TrackedUserContext(UserContextId id,
                              absl_nonnull UserContextRef user_context)
      : id_(id), user_context_(std::move(user_context)) {}

  const UserContextId id_;
  absl_nonnull const UserContextRef user_context_;
};

// CustomStatusExpanderRegistry allows registering 'payload expanders' that
// errors returned by the IFRT backend are processed through before the error
// message is returned to IFRT users.
class CustomStatusExpanderRegistry {
 public:
  static CustomStatusExpanderRegistry& Get();

  using PayloadExpanderFn = std::function<void(absl::Status&)>;
  // Registers a payload expander. `expander` is expected to take the entire
  // `absl::Status` object, remove the payload from the object, and modify the
  // contents of the `absl::Status` accordingly.
  //
  // The optional `process_order`, if supplied, determines the order in which
  // the expander is processed in relation to other expanders. Expanders with
  // lower process orders are processed first; please use a positive value
  // unless you have discussed with IFRT maintainers about writing a
  // a critical expander function that needs to be processed earlier. Order
  // among expanders of the same `process_order` is unspecified.
  void Register(absl::string_view payload_name, PayloadExpanderFn expander,
                std::optional<int> process_order = std::nullopt);

  // Invokes all registered expanders on the given status.
  void Process(absl::Status& status);

 private:
  mutable absl::Mutex mu_;
  absl::btree_map<std::pair<int, std::string>, PayloadExpanderFn> registry_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_USER_CONTEXT_REGISTRY_H_
