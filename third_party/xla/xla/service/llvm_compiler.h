/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_LLVM_COMPILER_H_
#define XLA_SERVICE_LLVM_COMPILER_H_

#include "llvm/IR/Module.h"
#include "xla/service/compiler.h"

namespace xla {

// Interface for an LLVM-based compiler. This provides the ability to register
// hooks to inspect the LLVM IR during compilation, both before and after
// optimizations are applied.
//
// Hooks get called once per HLO module being compiled. The following should not
// be relied on:
// * The order in which hooks get called.
// * Whether or not a hook gets called if a compilation exits with a non-OK
//   status.
class LLVMCompiler : public Compiler {
 public:
  ~LLVMCompiler() override {}

  // A callback of this type can be run before and/or after IR-level
  // optimization to e.g. dump out the generated IR to disk or gather some
  // statistics.
  using ModuleHook = std::function<void(const llvm::Module&)>;

  void SetPreOptimizationHook(ModuleHook hook) {
    CHECK(!user_pre_optimization_hook_)
        << "Pre-optimization hook is already set";
    CHECK(hook) << "hook cannot be null";
    user_pre_optimization_hook_ = hook;
  }

  void RemovePreOptimizationHook() { user_pre_optimization_hook_ = nullptr; }

  void SetPostOptimizationHook(ModuleHook hook) {
    CHECK(!user_post_optimization_hook_)
        << "Post-optimization hook is already set";
    CHECK(hook) << "hook cannot be null";
    user_post_optimization_hook_ = hook;
  }

  void RemovePostOptimizationHook() { user_post_optimization_hook_ = nullptr; }

  // Bring in
  //   absl::StatusOr<std::unique_ptr<Executable>> RunBackend(
  //       std::unique_ptr<HloModule> module,
  //       se::StreamExecutor* stream_exec,
  //       se::DeviceMemoryAllocator* device_allocator)
  //   absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
  //       std::unique_ptr<HloModule> module,
  //       se::StreamExecutor* stream_exec,
  //       se::DeviceMemoryAllocator* device_allocator)
  using Compiler::Compile;
  using Compiler::RunBackend;
  using Compiler::RunHloPasses;

  absl::StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_execs,
      const CompileOptions& options) override;

 protected:
  ModuleHook user_pre_optimization_hook_;
  ModuleHook user_post_optimization_hook_;
};

}  // namespace xla

#endif  // XLA_SERVICE_LLVM_COMPILER_H_
