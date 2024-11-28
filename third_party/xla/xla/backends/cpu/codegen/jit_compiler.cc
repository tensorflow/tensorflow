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

#include "xla/backends/cpu/codegen/jit_compiler.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "xla/backends/cpu/codegen/cpu_features.h"
#include "xla/backends/cpu/codegen/function_library.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/util.h"
#include "tsl/platform/cpu_info.h"

namespace xla::cpu {
namespace {

// XLA JIT compiler built on top of LLVM ORC APIs.
class LlvmOrcJitCompiler : public JitCompiler {
 public:
  LlvmOrcJitCompiler(llvm::TargetOptions target_options,
                     llvm::CodeGenOptLevel opt_level, const Options& options);

  absl::Status AddModule(llvm::orc::ThreadSafeModule module,
                         size_t dylib_index) final;

  absl::StatusOr<std::unique_ptr<FunctionLibrary>> Compile() && final;

 private:
};

// XLA function library compiled from LLVM module(s) using ORC APIs.
class LlvmOrcFunctionLibrary : public FunctionLibrary {
 public:
};

}  // namespace

absl::StatusOr<std::unique_ptr<llvm::TargetMachine>>
JitCompiler::InferTargetMachine(
    const llvm::TargetOptions& target_options, llvm::CodeGenOptLevel opt_level,
    std::optional<tsl::port::CPUFeature> max_cpu_feature) {
  // Detect machine attributes for the target CPU.
  auto result = DetectMachineAttributes(max_cpu_feature);
  llvm::SmallVector<std::string, 0> attrs(result.features.begin(),
                                          result.features.end());

  // If `max_cpu_feature` is newer than the host CPU, we should keep the host
  // CPU name, e.g., we don't want to set the target CPU to Skylake when we are
  // on a Broadwell host.
  std::string_view cpu = result.num_filtered_features
                             ? CpuTargetFromMaxFeature(*max_cpu_feature)
                             : std::string_view(llvm::sys::getHostCPUName());

  std::unique_ptr<llvm::TargetMachine> target_machine(
      llvm::EngineBuilder()
          .setTargetOptions(target_options)
          .setOptLevel(opt_level)
          .selectTarget(
              /*TargetTriple=*/llvm::Triple(), /*MArch=*/"",
              /*MCPU=*/cpu,
              /*MAttrs=*/attrs));

  if (target_machine == nullptr) {
    return Internal("Failed to create target machine for CPU %s", cpu);
  }

  return std::move(target_machine);
}

IrCompiler::TargetMachineBuilder JitCompiler::InferTargetMachineBuilder(
    const llvm::TargetOptions& target_options, llvm::CodeGenOptLevel opt_level,
    std::optional<tsl::port::CPUFeature> max_cpu_feature) {
  return [target_options, opt_level, max_cpu_feature] {
    return InferTargetMachine(target_options, opt_level, max_cpu_feature);
  };
}

absl::StatusOr<std::unique_ptr<JitCompiler>> JitCompiler::Create(
    llvm::TargetOptions target_options, llvm::CodeGenOptLevel opt_level,
    const Options& options) {
  return std::make_unique<LlvmOrcJitCompiler>(std::move(target_options),
                                              opt_level, options);
}

LlvmOrcJitCompiler::LlvmOrcJitCompiler(llvm::TargetOptions target_options,
                                       llvm::CodeGenOptLevel opt_level,
                                       const Options& options) {}

absl::Status LlvmOrcJitCompiler::AddModule(llvm::orc::ThreadSafeModule module,
                                           size_t dylib_index) {
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<FunctionLibrary>>
LlvmOrcJitCompiler::Compile() && {
  return Unimplemented("Not implemented yet");
}

}  // namespace xla::cpu
