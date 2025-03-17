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

#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/ffi/type_id_registry.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"

namespace xla::ffi {

ExecutionState::ExecutionState()
    : type_id_(TypeIdRegistry::kUnknownTypeId),
      state_(nullptr),
      deleter_(nullptr) {}

ExecutionState::~ExecutionState() {
  if (deleter_) deleter_(state_);
}

absl::Status ExecutionState::Set(TypeId type_id, void* state,
                                 Deleter<void> deleter) {
  DCHECK(state && deleter) << "State and deleter must not be null";

  if (type_id_ != TypeIdRegistry::kUnknownTypeId) {
    return FailedPrecondition("State is already set with a type id %d",
                              type_id_.value());
  }

  type_id_ = type_id;
  state_ = state;
  deleter_ = std::move(deleter);

  return absl::OkStatus();
}

// Returns opaque state of the given type id. If set state type id does not
// match the requested one, returns an error.
absl::StatusOr<void*> ExecutionState::Get(TypeId type_id) const {
  if (type_id_ == TypeIdRegistry::kUnknownTypeId) {
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
  return type_id_ != TypeIdRegistry::kUnknownTypeId;
}

}  // namespace xla::ffi
