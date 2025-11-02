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

#include <functional>
#include <memory>
#include <utility>

#include "absl/container/node_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/ffi/type_registry.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"

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
// We have two kinds of TypeIds:
//
// 1. Internal type id. When FFI handler defined in the same binary we rely
//    on a global static registry to automatically assign type ids.
//
// 2. External type id. When FFI handlers defined in a dynamically loaded
//    library, they must register types used in the execution context ahead
//    of time and explicitly get a unique type id for them.
//
// See `TypeRegistry` documentation for more details about different type ids.
//
// Examples: FFI handler can register a per-execution cache in the execution
// context and get access to it in the FFI handler, with a guarantee that it is
// unique between separate calls to XLA execute.
class ExecutionContext {
 public:
  using TypeId = TypeRegistry::TypeId;

  // Inserts user data with a given type id. Caller is responsible for making
  // sure that the pointer stays valid during the XLA execution and correctly
  // destroyed afterwards.
  absl::Status Insert(TypeId type_id, void* data);

  // Looks up opaque execution context data with given `type_id`.
  absl::StatusOr<void*> Lookup(TypeId type_id) const {
    TF_ASSIGN_OR_RETURN(auto user_data, LookupUserData(type_id));
    return user_data->get();
  }

  // Inserts typed user data of type `T`. Caller is responsible for making sure
  // that the pointer stays valid during the XLA execution and correctly
  // destroyed afterwards.
  template <typename T>
  absl::Status Insert(T* data);

  // Emplaces typed user data constructed from `args`. Execution context
  // becomes the owner of the constructed object.
  template <typename T, typename... Args>
  absl::Status Emplace(Args&&... args);

  // Looks up typed execution context data of type `T`.
  template <typename T>
  absl::StatusOr<T*> Lookup() const;

  // Visit all user data in the execution context.
  void ForEach(absl::FunctionRef<void(TypeId type_id, void* data)> fn) const;
  absl::Status ForEachWithStatus(
      absl::FunctionRef<absl::Status(TypeId type_id, void* data)> fn) const;

 private:
  // An RAII wrapper for opaque user data. If deleter is no-op then the caller
  // is responsible for making sure that the pointer stays valid during the XLA
  // execution and correctly destroyed afterwards
  using UserData = std::unique_ptr<void, std::function<void(void*)>>;

  absl::Status InsertUserData(TypeId type_id, UserData data);
  absl::StatusOr<const UserData*> LookupUserData(TypeId type_id) const;

  absl::node_hash_map<TypeId, UserData> user_data_;
};

template <typename T>
absl::StatusOr<T*> ExecutionContext::Lookup() const {
  TF_ASSIGN_OR_RETURN(auto user_data,
                      LookupUserData(TypeRegistry::GetTypeId<T>()));
  return static_cast<T*>(user_data->get());
}

template <typename T>
absl::Status ExecutionContext::Insert(T* data) {
  return InsertUserData(TypeRegistry::GetTypeId<T>(),
                        UserData(data, /*deleter=*/[](void*) {}));
}

template <typename T, typename... Args>
absl::Status ExecutionContext::Emplace(Args&&... args) {
  auto type_info = TypeRegistry::GetTypeInfo<T>();
  return InsertUserData(
      TypeRegistry::GetTypeId<T>(),
      UserData(new T(std::forward<Args>(args)...), type_info.deleter));
}

}  // namespace xla::ffi

#endif  // XLA_FFI_EXECUTION_CONTEXT_H_
