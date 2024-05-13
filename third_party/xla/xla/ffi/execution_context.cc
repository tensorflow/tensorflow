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

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"

namespace xla::ffi {

ABSL_CONST_INIT absl::Mutex type_registry_mutex(absl::kConstInit);

using TypeRegistry = absl::flat_hash_map<std::string, ExecutionContext::TypeId>;
static TypeRegistry& StaticTypeRegistry() {
  static auto* registry = new TypeRegistry();
  return *registry;
}

ExecutionContext::TypeId ExecutionContext::GetNextTypeId() {
  static auto* counter = new std::atomic<int64_t>(1);
  return TypeId(counter->fetch_add(1));
}

ExecutionContext::UserData::UserData(void* data, Deleter<void> deleter)
    : data_(data), deleter_(std::move(deleter)) {}

ExecutionContext::UserData::~UserData() {
  if (deleter_) deleter_(data_);
}

absl::StatusOr<ExecutionContext::TypeId>
ExecutionContext::RegisterExternalTypeId(std::string_view name) {
  absl::MutexLock lock(&type_registry_mutex);
  auto& registry = StaticTypeRegistry();

  // Try to emplace with type id zero and fill it with real type id only if we
  // successfully acquired an entry for a given name.
  auto emplaced = registry.emplace(name, TypeId(0));
  if (!emplaced.second) {
    return absl::AlreadyExistsError(
        absl::StrCat("Type id ", emplaced.first->second.value(),
                     " already registered for type name ", name));
  }
  return emplaced.first->second = GetNextTypeId();
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
    return absl::AlreadyExistsError(
        absl::StrCat("User data with type id ", type_id.value(),
                     " already exists in execution context"));
  }
  return absl::OkStatus();
}

absl::StatusOr<ExecutionContext::UserData*> ExecutionContext::LookupUserData(
    TypeId type_id) const {
  auto it = user_data_.find(type_id);
  if (it == user_data_.end()) {
    return absl::NotFoundError(absl::StrCat("User data with type id ",
                                            type_id.value(),
                                            " not found in execution context"));
  }
  return it->second.get();
}

}  // namespace xla::ffi
