/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"

#include <string>
#include <unordered_map>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {

namespace {
typedef std::unordered_map<string, CustomGraphOptimizerRegistry::Creator>
    RegistrationMap;
RegistrationMap* registered_optimizers = nullptr;
RegistrationMap* GetRegistrationMap() {
  if (registered_optimizers == nullptr)
    registered_optimizers = new RegistrationMap;
  return registered_optimizers;
}
}  // namespace

std::unique_ptr<CustomGraphOptimizer>
CustomGraphOptimizerRegistry::CreateByNameOrNull(const string& name) {
  const auto it = GetRegistrationMap()->find(name);
  if (it == GetRegistrationMap()->end()) return nullptr;
  return std::unique_ptr<CustomGraphOptimizer>(it->second());
}

std::vector<string> CustomGraphOptimizerRegistry::GetRegisteredOptimizers() {
  std::vector<string> optimizer_names;
  optimizer_names.reserve(GetRegistrationMap()->size());
  for (const auto& opt : *GetRegistrationMap())
    optimizer_names.emplace_back(opt.first);
  return optimizer_names;
}

void CustomGraphOptimizerRegistry::RegisterOptimizerOrDie(
    const Creator& optimizer_creator, const string& name) {
  const auto it = GetRegistrationMap()->find(name);
  if (it != GetRegistrationMap()->end()) {
    LOG(FATAL) << "CustomGraphOptimizer is registered twice: " << name;
  }
  GetRegistrationMap()->insert({name, optimizer_creator});
}

}  // end namespace grappler
}  // end namespace tensorflow
