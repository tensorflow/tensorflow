/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_FFI_EXECUTION_CONTEXT_H_
#define XLA_FFI_EXECUTION_CONTEXT_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/ffi/type_id_registry.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::ffi {

// Execution context is a container for forwarding arbitrary user data to FFI
// handlers in the scope of a single XLA execution. Execution context allows to
// pass arbitrary user data to FFI handlers via the side channel that does not
// require modifying HLO modules.
//
// From XLA FFI perspective user data is an opaque pointer that can be forwarded
// to the FFI handler. We rely on type id to guarantee that we forward user data
// of correct type.
//
// Examples: FFI handler can register a per-execution cache in the execution
// context and get access to it in the FFI handler, with a guarantee that it is
// unique between separate calls to XLA execute.
class ExecutionContext {
 public:
  using TypeId = TypeIdRegistry::TypeId;

  template <typename T>
  using Deleter = std::function<void(T*)>;

  // Inserts opaque user data with a given type id and optional deleter.
  absl::Status Insert(TypeId type_id, void* data,
                      Deleter<void> deleter = nullptr);

  // Inserts typed user data of type `T` and optional deleter.
  template <typename T>
  absl::Status Insert(T* data, Deleter<T> deleter = nullptr);

  // Emplaces typed user data constructed from `args`. Execution context
  // becomes the owner of the constructed object.
  template <typename T, typename... Args>
  absl::Status Emplace(Args&&... args);

  // Looks up typed execution context data of type `T`.
  template <typename T>
  absl::StatusOr<T*> Lookup() const {
    TF_ASSIGN_OR_RETURN(auto user_data,
                        LookupUserData(TypeIdRegistry::GetTypeId<T>()));
    return static_cast<T*>(user_data->data());
  }

  // Looks up opaque execution context data with given `type_id`.
  absl::StatusOr<void*> Lookup(TypeId type_id) const {
    TF_ASSIGN_OR_RETURN(auto user_data, LookupUserData(type_id));
    return user_data->data();
  }

 private:
  // An RAII wrapper for opaque user data. Optional deleter will be called when
  // UserData is destroyed together with the execution context. If deleter is
  // nullptr then the caller is responsible for making sure that the pointer
  // stays valid during the XLA execution and correctly destroyed afterwards.
  class UserData {
   public:
    UserData(void* data, Deleter<void> deleter);
    ~UserData();

    UserData(UserData&) = delete;
    UserData& operator=(const UserData&) = delete;

    void* data() const { return data_; }

   private:
    void* data_;
    Deleter<void> deleter_;
  };

  absl::Status InsertUserData(TypeId type_id, std::unique_ptr<UserData> data);
  absl::StatusOr<UserData*> LookupUserData(TypeId type_id) const;

  absl::flat_hash_map<TypeId, std::unique_ptr<UserData>> user_data_;
};

template <typename T>
absl::Status ExecutionContext::Insert(T* data, Deleter<T> deleter) {
  return InsertUserData(TypeIdRegistry::GetTypeId<T>(),
                        std::make_unique<UserData>(
                            data, [deleter = std::move(deleter)](void* data) {
                              if (deleter) deleter(static_cast<T*>(data));
                            }));
}

template <typename T, typename... Args>
absl::Status ExecutionContext::Emplace(Args&&... args) {
  return InsertUserData(TypeIdRegistry::GetTypeId<T>(),
                        std::make_unique<UserData>(
                            new T(std::forward<Args>(args)...),
                            [](void* data) { delete static_cast<T*>(data); }));
}

}  // namespace xla::ffi

#endif  // XLA_FFI_EXECUTION_CONTEXT_H_
