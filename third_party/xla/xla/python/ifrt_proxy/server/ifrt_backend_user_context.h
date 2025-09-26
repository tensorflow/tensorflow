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

#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/user_context.h"

namespace xla {
namespace ifrt {
namespace proxy {

class IfrtBackendDestroyedUserContextIds {
 public:
  // Adds a destroyed user context ID.
  void Add(UserContextId id);

  // Returns all destroyed user context IDs since the last call.
  std::vector<UserContextId> Consume();

 private:
  absl::Mutex mu_;
  std::vector<UserContextId> ids_ ABSL_GUARDED_BY(mu_);
};

class IfrtBackendUserContext
    : public llvm::RTTIExtends<IfrtBackendUserContext, UserContext> {
 public:
  static UserContextRef Create(
      std::shared_ptr<IfrtBackendDestroyedUserContextIds>
          destroyed_user_context_ids,
      UserContextId id);

  explicit IfrtBackendUserContext(
      std::shared_ptr<IfrtBackendDestroyedUserContextIds>
          destroyed_user_context_ids,
      UserContextId id);

  // UserContext implementation.

  ~IfrtBackendUserContext() override;

  UserContextId Id() const override { return id_; }

  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  std::shared_ptr<IfrtBackendDestroyedUserContextIds>
      destroyed_user_context_ids_;
  UserContextId id_;
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_SERVER_IFRT_BACKEND_USER_CONTEXT_H_
