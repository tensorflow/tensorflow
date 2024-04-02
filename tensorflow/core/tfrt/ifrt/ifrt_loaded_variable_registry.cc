/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_registry.h"

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace ifrt_serving {

absl::Status IfrtLoadedVariableRegistry::TryRegisterLoadedVariable(
    absl::string_view name,
    LoadedVariableConstructor&& loaded_variable_constructor) {
  absl::MutexLock lock(&mutex_);
  auto& variable = loaded_variable_map_[name];
  if (variable.array.IsValid()) {
    // Already registered. This is rare.
    VLOG(1) << "Variable '" << name << "' already registered.";
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(variable, loaded_variable_constructor());
  return absl::OkStatus();
}

absl::StatusOr<IfrtLoadedVariableRegistry::LoadedVariable>
IfrtLoadedVariableRegistry::GetLoadedVariable(absl::string_view name) const {
  absl::MutexLock lock(&mutex_);
  auto it = loaded_variable_map_.find(name);
  if (it == loaded_variable_map_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Variable '", name, "' not found."));
  }
  return it->second;
}

}  // namespace ifrt_serving
}  // namespace tensorflow
