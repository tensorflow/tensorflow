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

#ifndef XLA_PYTHON_IFRT_USER_CONTEXT_H_
#define XLA_PYTHON_IFRT_USER_CONTEXT_H_

#include <cstdint>
#include <string>

#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

// UserContext is an interface that must be implemented by any object that the
// user would like to be associated with the runtime operations triggered by an
// IFRT call. For example, a UserContext can be based on a stack trace for
// Python frameworks (e.g.: JAX), or on a "request_id" in case of request
// serving applications.
class UserContext : public tsl::ReferenceCounted<UserContext>,
                    public llvm::RTTIExtends<UserContext, llvm::RTTIRoot> {
 public:
  ~UserContext() override = default;

  // Returns a fingerprint of the UserContext. The returned fingerprint must
  // be non-zero, as the special value of zero is reserved for the IFRT
  // implementations for their internal default UserContext.  IFRT
  // implementations may use internally. IFRT implementations
  // may also use this as a key for holding the UserContexts in a container, and
  // so this should be efficient enough to called multiple times.
  //
  // TODO(hyeontaek): Remove this method once we migrate the UserContext to
  // a unique id scheme, where the id can be chosen without fingerprinting the
  // content of the UserContext.
  virtual uint64_t Fingerprint() const = 0;

  // Returns a human readable string. Meant for debugging, logging, and for
  // putting together statusz-like pages.
  //
  // Caution: A call to this method is often expensive, and may accompany a
  // certain precondition. For instance, this method might internally acquire a
  // mutex lock that is visible to the external world (e.g., Python GIL), and
  // the caller must ensure that doing so would not cause a deadlock (e.g., no
  // one who is holding such a mutex lock never blocks on a call to this
  // method).
  virtual std::string DebugString() const = 0;

  // For llvm::RTTI
  static char ID;  // NOLINT
};

using UserContextRef = tsl::RCReference<UserContext>;

// Tracks the active `UserContext` within the scope. It holds a pointer to the
// `UserContext` instance and uses a thread-local variable to make it
// discoverable through a static method.
class UserContextScope {
 public:
  // Sets up the current thread's `UserContextRef` to the given `context`.
  // `context` must be valid throughout the lifetime of the scope.
  explicit UserContextScope(UserContextRef context);

  // Restores the current thread's `UserContextRef` to the state before this
  // scope was created.
  ~UserContextScope();

  // Not copyable or moveable. The current scope's UserContextScope will be
  // referenced by a thread-local raw pointer.
  UserContextScope(const UserContextScope&) = delete;
  UserContextScope(UserContextScope&&) = delete;
  UserContextScope& operator=(const UserContextScope&) = delete;
  UserContextScope& operator=(UserContextScope&&) = delete;

  // Returns the active `UserContextRef`. The returned reference is stable only
  // during the lifetime of the current scope. If the `UserContextRef` should be
  // used outside the current scope, it must be copied as a new
  // `UserContextRef`.
  //
  // Returns `nullptr` if there is no `UserContextRef` in the scope.
  static const UserContextRef& current();

 private:
  // The outer scope's `UserContext`. When this scope is destroyed, the current
  // scope's `UserContext` will be restored to it.
  const UserContextRef* const outer_context_;  // Not owned.
  // The current scope's `UserContext`.
  const UserContextRef context_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_USER_CONTEXT_H_
