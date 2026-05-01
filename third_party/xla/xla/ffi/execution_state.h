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

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/ffi/execution_state.pb.h"
#include "xla/ffi/type_registry.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"

namespace xla::ffi {

// ExecutionState is an RAII wrapper for an opaque state object that can be
// created by the FFI handler. It allows creating "stateful" custom calls
// and pass arbitrary user types between execution stages.
//
// There are two types of execution state supported by FFI handlers:
//
// (1) Per-instance state: at instantiation stage XLA FFI handler creates a
//     state object and passes ownership to XLA. XLA attaches this state to
//     an instance of the FFI handler in the XLA program (custom call thunk
//     corresponding to custom call HLO) and then passes it back to prepare,
//     initialize and execute stages of FFI handler. This state destroyed by
//     the XLA runtime together with the executable.
//
// (2) Per-execution state: this is a transient state that can be created in
//     prepare or initialize state and it is accessible in the execute stage.
//     This state destroyed by the XLA runtime when XLA program finishes
//     execution.
//
// IMPORTANT: Note that single XLA program can be executed concurrently, and
// each individual execution will share access to per-instance state (it must
// be thread safe), but will get a unique per-execution state (no need to worry
// about data races).
class ExecutionState {
 public:
  using TypeId = TypeRegistry::TypeId;
  using TypeInfo = TypeRegistry::TypeInfo;

  // Sets opaque state with a given type id. Returns an error if state is
  // already set, or if type id is not supported as a state.
  absl::Status Set(TypeId type_id, void* state);

  // Returns opaque state of the given type id. If set state type id does not
  // match the requested one, returns an error.
  absl::StatusOr<void*> Get(TypeId type_id) const;

  absl::StatusOr<ExecutionStateProto> ToProto() const;
  static absl::StatusOr<ExecutionState> FromProto(
      const ExecutionStateProto& proto);

  // Sets typed state of type `T` and optional deleter. Returns an
  // error if state is already set.
  template <typename T>
  absl::Status Set(std::unique_ptr<T> state);

  // Gets typed state of type `T`. If set state type id does not match the
  // requested one, returns an error.
  template <typename T>
  absl::StatusOr<T*> Get() const;

  bool IsSet() const;
  bool IsSerializable() const;

 private:
  struct Deleter {
    void operator()(void* state);
    TypeId type_id;
    TypeInfo type_info;
  };

  absl::Status Set(TypeId type_id, TypeInfo type_info, void* state);

  std::unique_ptr<void, Deleter> state_;
};

inline void ExecutionState::Deleter::operator()(void* state) {
  if (type_info.deleter) {
    type_info.deleter(state);
  }
}

template <typename T>
absl::Status ExecutionState::Set(std::unique_ptr<T> state) {
  return Set(TypeRegistry::GetTypeId<T>(), TypeRegistry::GetTypeInfo<T>(),
             state.release());
}

template <typename T>
absl::StatusOr<T*> ExecutionState::Get() const {
  TF_ASSIGN_OR_RETURN(void* state, Get(TypeRegistry::GetTypeId<T>()));
  return tsl::safe_reinterpret_cast<T*>(state);
}

}  // namespace xla::ffi

#endif  // XLA_FFI_EXECUTION_STATE_H_
