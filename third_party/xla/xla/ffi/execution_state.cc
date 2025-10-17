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

#include "xla/ffi/execution_state.h"

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/ffi/type_registry.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::ffi {

ExecutionState::ExecutionState()
    : type_id_(TypeRegistry::kUnknownTypeId), state_(nullptr) {}

ExecutionState::~ExecutionState() {
  if (type_info_.deleter) {
    type_info_.deleter(state_);
  }
}

absl::Status ExecutionState::Set(TypeId type_id, void* state) {
  TF_ASSIGN_OR_RETURN(auto type_info,
                      TypeRegistry::GetExternalTypeInfo(type_id));
  if (type_info.deleter == nullptr) {
    return InvalidArgument(
        "Type id %d does not have a registered type info with a deleter",
        type_id.value());
  }
  return Set(type_id, type_info, state);
}

absl::Status ExecutionState::Set(TypeId type_id, TypeInfo type_info,
                                 void* state) {
  DCHECK(state && type_info.deleter) << "State and deleter must not be null";

  if (type_id_ != TypeRegistry::kUnknownTypeId) {
    return FailedPrecondition("State is already set with a type id %d",
                              type_id_.value());
  }

  type_id_ = type_id;
  type_info_ = type_info;
  state_ = state;

  return absl::OkStatus();
}

// Returns opaque state of the given type id. If set state type id does not
// match the requested one, returns an error.
absl::StatusOr<void*> ExecutionState::Get(TypeId type_id) const {
  if (type_id_ == TypeRegistry::kUnknownTypeId) {
    return NotFound("State is not set");
  }

  if (type_id_ != type_id) {
    return InvalidArgument(
        "Set state type id %d does not match the requested one %d",
        type_id_.value(), type_id.value());
  }

  return state_;
}

bool ExecutionState::IsSet() const {
  return type_id_ != TypeRegistry::kUnknownTypeId;
}

}  // namespace xla::ffi
