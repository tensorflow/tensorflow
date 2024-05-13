/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/ifrt/ifrt_executable_registry.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_executable.h"

namespace tensorflow {
namespace ifrt_serving {
ServingExecutableRegistry::Handle::Handle(Handle&& other) {
  *this = std::move(other);
}

ServingExecutableRegistry::Handle& ServingExecutableRegistry::Handle::operator=(
    Handle&& other) {
  if (this != &other) {
    program_id_ = std::move(other.program_id_);
    other.program_id_ = std::nullopt;
  }
  return *this;
}

ServingExecutableRegistry::Handle::~Handle() { Release(); }

absl::Status ServingExecutableRegistry::Handle::Freeze() {
  if (!program_id_.has_value()) {
    return absl::FailedPreconditionError("Program is not registered");
  }

  absl::MutexLock l(&ServingExecutableRegistry::mu_);

  const auto it = ServingExecutableRegistry::executables_->find(*program_id_);
  if (it == ServingExecutableRegistry::executables_->end()) {
    return absl::NotFoundError(
        absl::StrCat("Program ", *program_id_, " not found in the registry"));
  }

  VLOG(1) << "Freeze the program " << *program_id_ << " from signature '"
          << it->second->signature_name() << "' of model '"
          << it->second->model_name() << "'";

  it->second->Freeze();
  return absl::OkStatus();
}

void ServingExecutableRegistry::Handle::Release() {
  if (!program_id_.has_value()) {
    return;
  }

  absl::MutexLock l(&ServingExecutableRegistry::mu_);

  const auto it = ServingExecutableRegistry::executables_->find(*program_id_);
  if (it == ServingExecutableRegistry::executables_->end()) {
    LOG(ERROR) << "Program " << *program_id_ << " not found in the registry";
    return;
  }

  VLOG(1) << "Unregistering program " << *program_id_ << " from signature '"
          << it->second->signature_name() << "' of model '"
          << it->second->model_name() << "'";
  ServingExecutableRegistry::executables_->erase(it);

  program_id_ = std::nullopt;
}

ServingExecutableRegistry::Handle::Handle(int64_t program_id)
    : program_id_(program_id) {}

absl::StatusOr<ServingExecutableRegistry::Handle>
ServingExecutableRegistry::Register(
    int64_t program_id, std::unique_ptr<IfrtServingExecutable> executable) {
  absl::MutexLock l(&mu_);
  VLOG(1) << "Registering program " << program_id << " from signature '"
          << executable->signature_name() << "' of model '"
          << executable->model_name() << "'"
          << ", address is " << executable.get();
  if (!executables_->insert({program_id, std::move(executable)}).second) {
    return absl::AlreadyExistsError(absl::StrCat(
        "Program ", program_id, " already exists in the program registry"));
  }
  return Handle(program_id);
}

IfrtServingExecutable* ServingExecutableRegistry::Lookup(int64_t program_id) {
  absl::ReaderMutexLock l(&mu_);
  VLOG(1) << "Looking up program " << program_id;
  const auto it = executables_->find(program_id);
  return it != executables_->end() ? it->second.get() : nullptr;
}

ABSL_CONST_INIT absl::Mutex ServingExecutableRegistry::mu_(absl::kConstInit);

absl::flat_hash_map<int64_t, std::unique_ptr<IfrtServingExecutable>>* const
    ServingExecutableRegistry::executables_ =
        new absl::flat_hash_map<int64_t,
                                std::unique_ptr<IfrtServingExecutable>>();

}  // namespace ifrt_serving
}  // namespace tensorflow
