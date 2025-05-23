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

#ifndef XLA_BACKENDS_CPU_CODEGEN_IR_COMPILER_H_
#define XLA_BACKENDS_CPU_CODEGEN_IR_COMPILER_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/hlo_module_config.h"
#include "tsl/platform/cpu_info.h"

namespace xla::cpu {

void SetXlaCpuBackendOptions(llvm::Module& llvm_module,
                             const LlvmKernelOptions& options);

// IrCompiler compiles LLVM modules to object files using LLVM compilation
// pipeline customized for XLA:CPU. Default LLVM compilation pipeline is
// optimized for compiling LLVM IR produced by Clang, and in XLA we are a lot
// more constrained and produce a very different IR.
class IrCompiler : public llvm::orc::IRCompileLayer::IRCompiler {
 public:
  // Returns an instance of `llvm::TargetMachine` for a compilation. It can be
  // a shared `llvm::TargetMachine` if compilation is single threaded, or must
  // be a unique instance of `llvm::TargetMachine` if compilation is multi
  // threaded (because `llvm::TargetMachine` is not thread safe).
  //
  // See `llvm::orc::ConcurrentIRCompiler` to see corresponding API in ORC.
  using TargetMachineBuilder =
      std::function<absl::StatusOr<std::unique_ptr<llvm::TargetMachine>>()>;

  // Options for configuring the LLVM compilation pipeline and optimizations.
  struct Options {
    llvm::CodeGenOptLevel opt_level = llvm::CodeGenOptLevel::None;
    bool optimize_for_size = false;

    // Maximum CPU instruction set for wich the compiler should generate code.
    // If instruction set is empty, compiler will generate code for all ISA
    // extensions detected on the current machine.
    std::optional<tsl::port::CPUFeature> max_cpu_feature;

    llvm::FastMathFlags fast_math_flags;

    bool disable_expensive_passes = false;
    bool disable_slp_vectorizer = false;

    bool disable_loop_unrolling = false;

    bool dfsan_enabled = false;
    std::vector<std::string> dfsan_abi_list_files;
  };

  // Compilation hooks for intercepting IR compilation stages.
  struct CompilationHooks {
    std::function<void(const llvm::Module&)> pre_optimization;
    std::function<void(const llvm::Module&)> post_optimization;
    std::function<void(const llvm::Module&, const llvm::object::ObjectFile&)>
        post_codegen;
  };

  static std::unique_ptr<IrCompiler> Create(llvm::TargetOptions target_options,
                                            Options options,
                                            CompilationHooks hooks);

  IrCompiler(TargetMachineBuilder target_machine_builder, Options options,
             CompilationHooks hooks);

  // Infers the `llvm::TargetMachine` for the current host. If `max_cpu_feature`
  // is provided, it will be used to constrain the set of features that LLVM
  // codegen (instruction selection) is allowed to use, e.g. it can be used to
  // explicitly disable certain AVX512 extensions, in case the compiled
  // executable will be serialized and later loaded on a different machine.
  static absl::StatusOr<std::unique_ptr<llvm::TargetMachine>>
  InferTargetMachine(const llvm::TargetOptions& target_options,
                     llvm::CodeGenOptLevel opt_level,
                     std::optional<tsl::port::CPUFeature> max_cpu_feature);

  // Returns a target machine builder that uses `InferTargetMachine` defined
  // above to infer the target machine for the given options.
  static TargetMachineBuilder InferTargetMachineBuilder(
      const llvm::TargetOptions& target_options,
      llvm::CodeGenOptLevel opt_level,
      std::optional<tsl::port::CPUFeature> max_cpu_feature);

  // Compiles a `module` to an ObjectFile.
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> operator()(
      llvm::Module& module) final;

  // Runs the IR passes on the given module.
  llvm::Error RunIrPasses(llvm::Module& module,
                          llvm::TargetMachine* target_machine) const;

  // Emits machine code for the given module.
  std::unique_ptr<llvm::MemoryBuffer> EmitMachineCode(
      llvm::Module& module, llvm::TargetMachine* target_machine) const;

  static llvm::CodeGenOptLevel GetCodeGenOptLevel(
      const HloModuleConfig& module_config);

  absl::StatusOr<std::unique_ptr<llvm::TargetMachine>> build_target_machine()
      const {
    return target_machine_builder_();
  }

 private:
  TargetMachineBuilder target_machine_builder_;
  Options options_;

  // IRCompiler can be called in concurrently when JitCompiler compiles multiple
  // modules concurrently, we need to make sure that we don't introduce data
  // races when calling user provided compilation hooks.
  absl::Mutex mutex_;
  CompilationHooks hooks_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_IR_COMPILER_H_
