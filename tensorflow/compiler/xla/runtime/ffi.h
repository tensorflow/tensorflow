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

#ifndef TENSORFLOW_COMPILER_XLA_RUNTIME_FFI_H_
#define TENSORFLOW_COMPILER_XLA_RUNTIME_FFI_H_

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/runtime/ffi/ffi_c_api.h"
#include "tensorflow/compiler/xla/runtime/module.h"

namespace xla {
namespace runtime {
namespace ffi {

// Returns FFI modules registered with an XLA runtime.
std::vector<const runtime::Module*> FfiModules();

// Exports registered FFI modules to the given custom call registry.
void ExportFfiModules(DynamicCustomCallRegistry& registry);

// XLA runtime wrapper around FFI module.
class FfiModule;

//===----------------------------------------------------------------------===//
// RAII helpers for the state instantiated by the FFI modules.
//===----------------------------------------------------------------------===//

// Deletes the FFI module `state` instantiated by the `module`.
struct FfiStateDeleter {
  void operator()(XLA_FFI_Module_State* state);
  const FfiModule* module;
};

// A smart pointer owning `state` instantiated by the `module`.
struct OwnedFfiState
    : public std::unique_ptr<XLA_FFI_Module_State, FfiStateDeleter> {
  using Base = std::unique_ptr<XLA_FFI_Module_State, FfiStateDeleter>;
  OwnedFfiState(const FfiModule* module, XLA_FFI_Module_State* state);
};

//===----------------------------------------------------------------------===//
// FFI modules <-> XLA runtime integration via UserData.
//===----------------------------------------------------------------------===//

// A vector of opaque pointers to FFI modules state that is passed around inside
// `UserData` and enables FFI functions to find their state.
struct FfiStateVector {
  std::vector<XLA_FFI_Module_State*> state;  // indexed by module id

  // If FFI module instantiates state for each execution, the state vector will
  // be the owner of that state for the duration of execution.
  std::vector<OwnedFfiState> per_execution_state;
};

// FfiModulesState is a container that owns the FFI modules state.
class FfiModulesState {
 public:
  FfiModulesState() = default;

  // Instantiates `FfiModulesState` from the registered FFI module.
  static absl::StatusOr<FfiModulesState> Instantiate();

  absl::StatusOr<FfiStateVector> state_vector() const;

 private:
  explicit FfiModulesState(std::vector<std::unique_ptr<Module::State>> state);

  std::vector<std::unique_ptr<Module::State>> state_;
};

}  // namespace ffi
}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_FFI_H_
