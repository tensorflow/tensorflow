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

#ifndef XLA_PYTHON_IFRT_PROXY_SERVER_IFRT_BACKEND_USER_CONTEXT_H_
#define XLA_PYTHON_IFRT_PROXY_SERVER_IFRT_BACKEND_USER_CONTEXT_H_

#include <string>

#include "absl/functional/any_invocable.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {
namespace proxy {

class IfrtBackendUserContext
    : public llvm::RTTIExtends<IfrtBackendUserContext, UserContext> {
 public:
  // Creates a UserContextRef with the given `original_id` and `on_destroyed`
  // callback called at its destruction. If `original_id` is 0, returns a
  // nullptr; `on_destroyed` will not be called this case.
  static UserContextRef Create(
      UserContextId original_id,
      absl::AnyInvocable<void(UserContextId) &&> on_destroyed);

  // UserContext implementation.

  ~IfrtBackendUserContext() override;

  UserContextId Id() const override { return original_id_; }

  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  template <typename T, typename... Args>
  friend tsl::RCReference<T> tsl::MakeRef(Args&&... args);

  IfrtBackendUserContext(
      UserContextId original_id,
      absl::AnyInvocable<void(UserContextId) &&> on_destroyed);

  UserContextId original_id_;
  absl::AnyInvocable<void(UserContextId) &&> on_destroyed_;
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_SERVER_IFRT_BACKEND_USER_CONTEXT_H_
