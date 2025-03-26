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

#ifndef XLA_FFI_EXECUTION_STATE_H_
#define XLA_FFI_EXECUTION_STATE_H_

#include <functional>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/ffi/type_id_registry.h"
#include "tsl/platform/statusor.h"

namespace xla::ffi {

// ExecutionState is an RAII wrapper for an opaque state that can be attached
// to the instance of the XLA FFI handler, and allows to implement "stateful"
// custom calls:
//
//   (1) At instantiation stage XLA FFI handler creates a state object and
//       passes ownership to the XLA runtime.
//
//   (2) At prepare/initialize/execute stages XLA runtime passes state back to
//       the FFI handler via the execution context.
//
//   (3) XLA runtime automatically destroys the state object when parent XLA
//       executable is destroyed.
//
class ExecutionState {
 public:
  using TypeId = TypeIdRegistry::TypeId;

  template <typename T>
  using Deleter = std::function<void(T*)>;

  ExecutionState();
  ~ExecutionState();

  ExecutionState(const ExecutionState&) = delete;
  ExecutionState& operator=(const ExecutionState&) = delete;

  // Sets opaque state with a given type id and deleter. Returns an error if
  // state is already set.
  absl::Status Set(TypeId type_id, void* state, Deleter<void> deleter);

  // Returns opaque state of the given type id. If set state type id does not
  // match the requested one, returns an error.
  absl::StatusOr<void*> Get(TypeId type_id) const;

  // Sets typed state of type `T` and optional deleter. Returns an
  // error if state is already set.
  template <typename T>
  absl::Status Set(std::unique_ptr<T> state);

  // Gets typed state of type `T`. If set state type id does not match the
  // requested one, returns an error.
  template <typename T>
  absl::StatusOr<T*> Get() const;

  bool IsSet() const;

 private:
  TypeId type_id_;
  void* state_;
  Deleter<void> deleter_;
};

template <typename T>
absl::Status ExecutionState::Set(std::unique_ptr<T> state) {
  return Set(TypeIdRegistry::GetTypeId<T>(), state.release(),
             [](void* state) { delete reinterpret_cast<T*>(state); });
}

template <typename T>
absl::StatusOr<T*> ExecutionState::Get() const {
  TF_ASSIGN_OR_RETURN(void* state, Get(TypeIdRegistry::GetTypeId<T>()));
  return reinterpret_cast<T*>(state);
}

}  // namespace xla::ffi

#endif  // XLA_FFI_EXECUTION_STATE_H_
