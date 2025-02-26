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

#include "xla/backends/cpu/testlib/kernel_runner.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/backends/cpu/codegen/cpu_features.h"
#include "xla/backends/cpu/codegen/execution_engine.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/kernel.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_ir_kernel_source.h"
#include "xla/service/cpu/cpu_options.h"
#include "xla/service/cpu/runtime_symbol_generator.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {

absl::StatusOr<KernelRunner> KernelRunner::Create(
    KernelDefinition kernel_definition, JitCompiler compiler) {
  const auto [kernel_spec, kernel_source] =
      std::move(kernel_definition).release();

  // Use dynamic_cast rather than tsl::down_cast to allow for future
  // creation of KernelRunner from different kernel spec types.
  if (auto* llvm_kernel_source =
          dynamic_cast<LlvmIrKernelSource*>(kernel_source.get())) {
    return Create(kernel_spec, std::move(*llvm_kernel_source),
                  std::move(compiler));
  }

  return absl::InvalidArgumentError("Unrecognised kernel spec type");
}

absl::StatusOr<KernelRunner> KernelRunner::Create(
    const KernelSpec& kernel_spec, LlvmIrKernelSource llvm_ir_kernel_source,
    JitCompiler compiler) {
  TF_RETURN_IF_ERROR(compiler.AddModule(
      std::move(llvm_ir_kernel_source).thread_safe_module()));

  const std::string& kernel_name = kernel_spec.name();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<FunctionLibrary> library,
                      std::move(compiler).Compile(
                          {FunctionLibrary::Sym<XLA_CPU_Kernel>(kernel_name)}));

  TF_ASSIGN_OR_RETURN(XLA_CPU_Kernel * kernel_fn,
                      library->ResolveFunction<XLA_CPU_Kernel>(kernel_name));

  Kernel::ThreadDim thread_dim = kernel_spec.thread_dim();
  return KernelRunner(std::move(library), Kernel(1, kernel_fn), thread_dim);
}

KernelRunner::KernelRunner(std::unique_ptr<FunctionLibrary> library,
                           Kernel kernel, Kernel::ThreadDim thread_dim)
    : library_(std::move(library)),
      kernel_(std::move(kernel)),
      thread_dim_(thread_dim) {}

absl::Status KernelRunner::Call(absl::Span<const Argument> arguments) {
  std::vector<XLA_CPU_KernelArg> kernel_args;
  for (const Argument& arg : arguments) {
    kernel_args.push_back({arg.data(), arg.size()});
  }

  return kernel_.Launch(thread_dim_, kernel_args);
}

absl::StatusOr<JitCompiler> KernelRunner::CreateJitCompiler(
    const HloModuleConfig& config) {
  const DebugOptions& debug_options = config.debug_options();

  IrCompiler::Options ir_compiler_options{
      /*optimization_level=*/IrCompiler::GetCodeGenOptLevel(config),
      /*optimize_for_size=*/options::OptimizeForSizeRequested(config),
      /*fast_math_flags=*/llvm_ir::GetCpuFastMathFlags(config),
      /*disable_expensive_passes=*/
      debug_options.xla_llvm_disable_expensive_passes(),
      /*slp_vectorizer_disabled=*/options::SlpVectorizerDisabled(config),
      /*disable_loop_unrolling=*/options::DisableLoopUnrolling(config),
  };

  IrCompiler::CompilationHooks ir_compiler_hooks;

  // Needed to resolve symbols such as built in intrinsics (sin, cos etc).
  ExecutionEngine::DefinitionGenerator definition_generator =
      [](const llvm::DataLayout& data_layout) {
        return std::make_unique<RuntimeSymbolGenerator>(data_layout);
      };

  JitCompiler::Options jit_compiler_options{
      std::move(ir_compiler_options),
      std::move(ir_compiler_hooks),
      /*num_dylibs=*/1,
      /*definition_generator=*/std::move(definition_generator),
      /*max_cpu_isa=*/CpuFeatureFromString(debug_options.xla_cpu_max_isa()),
  };

  llvm::TargetOptions target_options;
  target_options.AllowFPOpFusion = llvm::FPOpFusion::Fast;

  return JitCompiler::Create(target_options, jit_compiler_options);
}

}  // namespace xla::cpu
