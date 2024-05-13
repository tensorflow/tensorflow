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

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace xla::ffi {

ExecutionContext::OpaqueUserData::OpaqueUserData(
    void* data, OpaqueUserData::Deleter deleter)
    : data_(data), deleter_(std::move(deleter)) {}

ExecutionContext::OpaqueUserData::~OpaqueUserData() {
  if (deleter_) deleter_(data_);
}

absl::Status ExecutionContext::Emplace(std::string id, void* data,
                                       OpaqueUserData::Deleter deleter) {
  if (!data) return absl::InvalidArgumentError("User data must be not null");

  auto emplaced = opaque_.emplace(
      id, std::make_shared<OpaqueUserData>(data, std::move(deleter)));
  if (!emplaced.second) {
    return absl::AlreadyExistsError(
        absl::StrCat("Opaque user data with id ", id,
                     " already exists in execution context"));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<ExecutionContext::OpaqueUserData>>
ExecutionContext::Lookup(std::string_view id) const {
  auto it = opaque_.find(id);
  if (it == opaque_.end()) {
    return absl::NotFoundError(absl::StrCat("Opaque user data with id ", id,
                                            " not found in execution context"));
  }
  return it->second;
}

absl::Status ExecutionContext::Insert(int64_t type_id,
                                      std::shared_ptr<UserData> data) {
  if (!data) return absl::InvalidArgumentError("User data must be not null");

  auto emplaced = typed_.emplace(type_id, std::move(data));
  if (!emplaced.second) {
    return absl::AlreadyExistsError(
        absl::StrCat("User data with type id ", type_id,
                     " already exists in execution context"));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<ExecutionContext::UserData>>
ExecutionContext::Lookup(int64_t type_id) const {
  auto it = typed_.find(type_id);
  if (it == typed_.end()) {
    return absl::NotFoundError(absl::StrCat("User data with type id ", type_id,
                                            " not found in execution context"));
  }
  return it->second;
}

}  // namespace xla::ffi
