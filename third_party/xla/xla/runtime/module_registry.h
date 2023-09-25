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

#ifndef XLA_RUNTIME_MODULE_REGISTRY_H_
#define XLA_RUNTIME_MODULE_REGISTRY_H_

#include <memory>
#include <utility>
#include <vector>

#include "xla/runtime/module.h"

namespace xla {
namespace runtime {

// Registers Xla runtime module with a global modules registry.
//
// TODO(ezhulenev): We need to support modules restricted for different
// platforms, e.g. we should not instantiate state for GPU modules when
// compiling a CPU executable. Xla today uses a "platform name" (Host,
// CUDA, etc...) for this.
void RegisterModule(std::unique_ptr<Module> module);

// Exports registered modules to the given custom call registry.
//
// TODO(ezhulenev): We also need to support exporting direct custom calls.
void ExportModules(DynamicCustomCallRegistry& registry);

// Helper macro to define a static module registration.
#define XLA_REGISTER_RUNTIME_MODULE(FUNC) \
  XLA_REGISTER_RUNTIME_MODULE_IMPL(FUNC, __COUNTER__)

#define XLA_REGISTER_RUNTIME_MODULE_IMPL(FUNC, N)           \
  static bool xla_runtime_module_##N##_registered_ = []() { \
    ::xla::runtime::RegisterModule(FUNC);                   \
    return true;                                            \
  }()

// Container that owns the state of all registered modules.
class ModulesState {
 public:
  ModulesState() = default;

  // Instantiates `ModulesState` from the registered module.
  //
  // TODO(ezhulenev): Take module's platform.
  static absl::StatusOr<ModulesState> Instantiate();

  // Initializes `UserData` from the module's state.
  absl::StatusOr<std::vector<std::unique_ptr<Module::StateRef>>>
  InitializeUserData(CustomCall::UserData& user_data);

 private:
  explicit ModulesState(
      std::vector<std::pair<Module*, std::unique_ptr<Module::State>>> state);

  std::vector<std::pair<Module*, std::unique_ptr<Module::State>>> state_;
};

}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_MODULE_REGISTRY_H_
