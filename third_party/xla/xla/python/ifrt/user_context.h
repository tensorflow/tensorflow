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
#include <vector>

#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla {
namespace ifrt {

// Globally unique ID for a `UserContext`.
TSL_LIB_GTL_DEFINE_INT_TYPE(UserContextId, uint64_t);

// `UserContext` is an interface that must be implemented by any object that the
// user would like to be associated with the runtime operations triggered by an
// IFRT call. For example, a `UserContext` can be based on a stack trace for
// Python frameworks (e.g.: JAX), or on a "request_id" in case of request
// serving applications.
class UserContext : public tsl::ReferenceCounted<UserContext>,
                    public llvm::RTTIExtends<UserContext, llvm::RTTIRoot> {
 public:
  ~UserContext() override = default;

  // Returns the unique ID of the `UserContext`. This ID is expected to be
  // globally unique for a certain context. For instance, both a global random
  // ID and the fingerprint of the `UserContext` content may be used as the ID.
  virtual UserContextId Id() const = 0;

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

// 'AnnotatedUserContext` represents a `UserContext` with a human-readable short
// message. The annotation adds extra contextual information that is known after
// creation time of the original `UserContext`, but before actually observing
// any error. For example, if the dispatch logic of a runtime takes a certain
// internal implementation path, the runtime can mention this choice in the
// annotation. If an error ends up happening later, this annotation will provide
// an extra context to the user.
class AnnotatedUserContext
    : public llvm::RTTIExtends<AnnotatedUserContext, UserContext> {
 public:
  static absl_nonnull UserContextRef Create(UserContextRef user_context,
                                            std::string msg);

  const UserContextRef& user_context() const { return user_context_; }
  absl::string_view msg() const { return msg_; }

  // `UserContext` implementation.

  UserContextId Id() const override;
  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  template <typename T, typename... Args>
  friend tsl::RCReference<T> tsl::MakeRef(Args&&... args);

  explicit AnnotatedUserContext(UserContextRef user_context, std::string msg);

  UserContextId id_;
  UserContextRef user_context_;
  std::string msg_;
};

// `ChainedUserContext` represents a chain of `UserContext`s of the operations,
// each of which is dependent on its preceding operation. This expresses how an
// error is propagated from one operation to another operation that uses the
// result of the former operation. `user_contexts` is order-dependent: for
// example, `user_contexts.front()` indicates the context of the earliest
// operation that is the original source of the error, and
// `user_contexts.back()` indicates the context of the latest operation that is
// finally surfacing the error to the user.
class ChainedUserContext
    : public llvm::RTTIExtends<ChainedUserContext, UserContext> {
 public:
  static absl_nonnull UserContextRef
  Create(absl::Span<const UserContextRef> user_contexts);

  // Not copyable or movable.
  ChainedUserContext(const ChainedUserContext&) = delete;
  ChainedUserContext& operator=(const ChainedUserContext&) = delete;

  absl::Span<const UserContextRef> user_contexts() const {
    return user_contexts_;
  }

  // `UserContext` implementation.

  UserContextId Id() const override;
  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  template <typename T, typename... Args>
  friend tsl::RCReference<T> tsl::MakeRef(Args&&... args);

  explicit ChainedUserContext(absl::Span<const UserContextRef> user_contexts);

  UserContextId id_;
  std::vector<UserContextRef> user_contexts_;
};

// `FusedUserContext` represents a set of `UserContext`s whose operations may
// contribute to an error, but are indistinguishable from each other from a
// runtime's perspective. For instance, the runtime can batch multiple API calls
// together while getting a single aggregate error status for the whole batch.
// Then, the runtime can use this user context to indicate that the error can
// come from any of the batched operations. The ordering of user contexts in a
// fused user context is insignificant.
class FusedUserContext
    : public llvm::RTTIExtends<FusedUserContext, UserContext> {
 public:
  static absl_nonnull UserContextRef
  Create(absl::Span<const UserContextRef> user_contexts);

  // Not copyable or movable.
  FusedUserContext(const FusedUserContext&) = delete;
  FusedUserContext& operator=(const FusedUserContext&) = delete;

  absl::Span<const UserContextRef> user_contexts() const {
    return user_contexts_;
  }

  // `UserContext` implementation.

  UserContextId Id() const override;
  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  template <typename T, typename... Args>
  friend tsl::RCReference<T> tsl::MakeRef(Args&&... args);

  explicit FusedUserContext(absl::Span<const UserContextRef> user_contexts);

  UserContextId id_;
  std::vector<UserContextRef> user_contexts_;
};

// Tracks the active `UserContext` within the scope. It holds a pointer to the
// `UserContext` instance and uses a thread-local variable to make it
// discoverable through a static method.
class UserContextScope {
 public:
  // Sets up the current thread's `UserContextRef` to the given `context`.
  // `context` must be valid throughout the lifetime of the scope.
  explicit UserContextScope(absl_nullable UserContextRef context);

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
  static absl_nullable const UserContextRef& current();

 private:
  // The outer scope's `UserContext`. When this scope is destroyed, the current
  // scope's `UserContext` will be restored to it.
  absl_nullable const UserContextRef* const outer_context_;  // Not owned.
  // The current scope's `UserContext`.
  absl_nullable const UserContextRef context_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_USER_CONTEXT_H_
