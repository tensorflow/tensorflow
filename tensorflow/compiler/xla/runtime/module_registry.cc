/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/runtime/module_registry.h"

#include <memory>
#include <utility>
#include <vector>

namespace xla {
namespace runtime {

static std::vector<std::unique_ptr<Module>>& RegisteredModules() {
  static auto* modules = new std::vector<std::unique_ptr<Module>>;
  return *modules;
}

void RegisterModule(std::unique_ptr<Module> module) {
  VLOG(1) << "Register XLA runtime module: " << module->name();
  RegisteredModules().push_back(std::move(module));
}

void ExportModules(DynamicCustomCallRegistry& registry) {
  for (auto& module : RegisteredModules()) {
    module->Export(registry);
  }
}

ModulesState::ModulesState(
    std::vector<std::pair<Module*, std::unique_ptr<Module::State>>> state)
    : state_(std::move(state)) {}

/*static*/ absl::StatusOr<ModulesState> ModulesState::Instantiate() {
  VLOG(1) << "Instantiate state for all registered XLA runtime modules";
  std::vector<std::pair<Module*, std::unique_ptr<Module::State>>> state_vec;

  for (auto& module : RegisteredModules()) {
    VLOG(2) << "Instantiate state for module: " << module->name();
    auto state = module->CreateState();
    if (!state.ok()) return state.status();
    state_vec.emplace_back(module.get(), std::move(*state));
  }

  return ModulesState(std::move(state_vec));
}

absl::StatusOr<std::vector<std::unique_ptr<Module::StateRef>>>
ModulesState::InitializeUserData(CustomCall::UserData& user_data) {
  VLOG(1) << "Initialize UserData for all XLA runtime modules";
  std::vector<std::unique_ptr<Module::StateRef>> ref_vec;
  ref_vec.reserve(state_.size());

  for (auto& [module, state] : state_) {
    VLOG(2) << "Initialize user data for module: " << module->name();
    auto ref = module->InitializeUserData(state.get(), user_data);
    if (!ref.ok()) return ref.status();
    ref_vec.push_back(std::move(*ref));
  }

  return {std::move(ref_vec)};
}

}  // namespace runtime
}  // namespace xla
