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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xla::ffi {

// Execution context is a container for forwarding arbitrary user data to FFI
// handlers in the scope of a single execution. Execution context allows to pass
// arbitrary user data to FFI handlers via the side channel that does not
// require modifying HLO modules. There are two kinds of user data that can be
// passed to FFI handlers:
//
// 1. Opaque data. This is a wrapper for an opaque user data pointer that is
//    useful when FFI handler is registered in the dynamically loaded library
//    and we do not know the type of the data and can only work with the opaque
//    pointer.
//
// 2. Typed data. This is useful when the FFI handler is registered in the same
//    process and we can rely on global static variable to assign ids to types
//    and we don't need to worry about breaking C++ ABI.
//
// For internal FFI handlers we always use typed data, and use opaque data only
// if FFI handler has to be defined in a separate dynamically loaded library.
//
// Examples: FFI handler can register a per-execution cache in the execution
// context and get access to it in the FFI handler, with a guarantee that it is
// unique between separate calls to XLA execute.
class ExecutionContext {
 public:
  // A base class for typed user data used for FFI handlers registered in the
  // same process where we can safely pass around C++ objects.
  class UserData {
   public:
    virtual ~UserData() = default;
  };

  template <typename T>
  using IsUserData = std::enable_if_t<std::is_base_of_v<UserData, T>>;

  // An RAII wrapper for opaque user data that is useful when FFI handler is
  // registered in the dynamically loaded library and we do not know the type of
  // the data and can only work with the opaque pointer.
  class OpaqueUserData {
   public:
    using Deleter = std::function<void(void*)>;

    OpaqueUserData(void* data, Deleter deleter);
    ~OpaqueUserData();

    OpaqueUserData(OpaqueUserData&) = delete;
    OpaqueUserData& operator=(const OpaqueUserData&) = delete;

    void* data() const { return data_; }

   private:
    void* data_;
    Deleter deleter_;
  };

  // Emplaces opaque user data keyed by `id`.
  absl::Status Emplace(std::string id, void* data,
                       OpaqueUserData::Deleter deleter);

  // Looks up opaque user data keyed by `id`.
  absl::StatusOr<std::shared_ptr<OpaqueUserData>> Lookup(
      std::string_view id) const;

  // Emplaces typed user data constructed from `args`.
  template <typename T, typename... Args, IsUserData<T>* = nullptr>
  absl::Status Emplace(Args&&... args) {
    return Insert(GetTypeId<T>(),
                  std::make_shared<T>(std::forward<Args>(args)...));
  }

  // Looks up typed execution context data of type `T`.
  template <typename T, IsUserData<T>* = nullptr>
  absl::StatusOr<std::shared_ptr<T>> Lookup() const {
    auto user_data = Lookup(GetTypeId<T>());
    if (!user_data.ok()) return user_data.status();
    return std::static_pointer_cast<T>(*std::move(user_data));
  }

 private:
  template <typename T, IsUserData<T>* = nullptr>
  static int64_t GetTypeId() {
    static const char id = 0;
    return reinterpret_cast<int64_t>(&id);
  }

  absl::Status Insert(int64_t type_id, std::shared_ptr<UserData> data);
  absl::StatusOr<std::shared_ptr<UserData>> Lookup(int64_t type_id) const;

  absl::flat_hash_map<int64_t, std::shared_ptr<UserData>> typed_;
  absl::flat_hash_map<std::string, std::shared_ptr<OpaqueUserData>> opaque_;
};

}  // namespace xla::ffi

#endif  // XLA_FFI_EXECUTION_CONTEXT_H_
