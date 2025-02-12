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

#include "xla/ffi/execution_context.h"

#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"

namespace xla::ffi {

ExecutionContext::UserData::UserData(void* data, Deleter<void> deleter)
    : data_(data), deleter_(std::move(deleter)) {}

ExecutionContext::UserData::~UserData() {
  if (deleter_) deleter_(data_);
}

absl::Status ExecutionContext::Insert(TypeId type_id, void* data,
                                      Deleter<void> deleter) {
  return InsertUserData(type_id,
                        std::make_unique<UserData>(data, std::move(deleter)));
}

absl::Status ExecutionContext::InsertUserData(TypeId type_id,
                                              std::unique_ptr<UserData> data) {
  if (!data) return absl::InvalidArgumentError("User data must be not null");

  auto emplaced = user_data_.emplace(type_id, std::move(data));
  if (!emplaced.second) {
    return Internal(
        "User data with type id %d already exists in execution context",
        type_id.value());
  }
  return absl::OkStatus();
}

absl::StatusOr<ExecutionContext::UserData*> ExecutionContext::LookupUserData(
    TypeId type_id) const {
  auto it = user_data_.find(type_id);
  if (it == user_data_.end()) {
    return NotFound("User data with type id %d not found in execution context",
                    type_id.value());
  }
  return it->second.get();
}

void ExecutionContext::ForEach(
    absl::FunctionRef<void(TypeId type_id, void* data)> fn) const {
  for (auto& [type_id, user_data] : user_data_) {
    fn(type_id, user_data->data());
  }
}

absl::Status ExecutionContext::ForEachWithStatus(
    absl::FunctionRef<absl::Status(TypeId type_id, void* data)> fn) const {
  for (auto& [type_id, user_data] : user_data_) {
    TF_RETURN_IF_ERROR(fn(type_id, user_data->data()));
  }
  return absl::OkStatus();
}

}  // namespace xla::ffi
