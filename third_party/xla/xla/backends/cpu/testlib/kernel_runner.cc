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
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/kernel.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/backends/cpu/testlib/llvm_ir_kernel_spec.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_ir_kernel_source.h"
#include "xla/service/cpu/runtime_symbol_generator.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {

absl::StatusOr<KernelRunner> KernelRunner::Create(
    std::unique_ptr<KernelSpec> kernel_spec) {
  // Use dynamic_cast rather than tsl::down_cast to allow for future
  // creation of KernelRunner from different kernel spec types.
  if (auto* llvm_kernel_spec =
          dynamic_cast<LlvmIrKernelSpec*>(kernel_spec.get())) {
    return Create(std::move(*llvm_kernel_spec));
  }

  return absl::InvalidArgumentError("Unrecognised kernel spec type");
}

absl::StatusOr<KernelRunner> KernelRunner::Create(
    LlvmIrKernelSpec kernel_spec) {
  LlvmIrKernelSource& kernel_source = kernel_spec.kernel_source();

  llvm::TargetOptions target_options;
  target_options.AllowFPOpFusion = llvm::FPOpFusion::Fast;

  // Needed to resolve symbols such as built in intrinsics (sin, cos etc).
  JitCompiler::Options jit_compiler_options;
  jit_compiler_options.definition_generator =
      [](llvm::TargetMachine* target_machine) {
        return std::make_unique<RuntimeSymbolGenerator>(
            target_machine->createDataLayout());
      };

  TF_ASSIGN_OR_RETURN(
      JitCompiler compiler,
      JitCompiler::Create(target_options, jit_compiler_options));

  // Intentional copy as we need to use the kernel name after consuming
  // (std::move) the kernel source.
  std::string kernel_name = kernel_source.kernel_name();

  TF_RETURN_IF_ERROR(
      compiler.AddModule(std::move(kernel_source).thread_safe_module()));

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

}  // namespace xla::cpu
