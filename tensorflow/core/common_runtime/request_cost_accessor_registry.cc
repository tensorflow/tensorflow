/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/request_cost_accessor_registry.h"

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

using RegistrationMap =
    absl::flat_hash_map<std::string, RequestCostAccessorRegistry::Creator>;

RegistrationMap* GetRegistrationMap() {
  static RegistrationMap* registered_request_cost_accessors =
      new RegistrationMap;
  return registered_request_cost_accessors;
}

}  // namespace

std::unique_ptr<RequestCostAccessor>
RequestCostAccessorRegistry::CreateByNameOrNull(absl::string_view name) {
  const auto it = GetRegistrationMap()->find(name);
  if (it == GetRegistrationMap()->end()) return nullptr;
  return std::unique_ptr<RequestCostAccessor>(it->second());
}

void RequestCostAccessorRegistry::RegisterRequestCostAccessor(
    absl::string_view name, Creator creator) {
  const auto it = GetRegistrationMap()->find(name);
  CHECK(it == GetRegistrationMap()->end())  // Crash OK
      << "RequestCostAccessor " << name << " is registered twice.";
  GetRegistrationMap()->emplace(name, std::move(creator));
}

}  // namespace tensorflow
