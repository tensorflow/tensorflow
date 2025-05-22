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

#ifndef XLA_BACKENDS_CPU_TESTLIB_KERNEL_RUNNER_H_
#define XLA_BACKENDS_CPU_TESTLIB_KERNEL_RUNNER_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/kernel.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_ir_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/codegen/testlib/kernel_runner.h"
#include "xla/runtime/work_group.h"
#include "xla/service/hlo_module_config.h"

namespace xla::cpu {

// Kernel runner for XLA:CPU backend.
class KernelRunner final : public xla::KernelRunner {
 public:
  // Create a KernelRunner from a KernelSpec, this factory takes care of the
  // downcasting to supported kernel source types, currently only
  // LLVM IR is supported.
  static absl::StatusOr<KernelRunner> Create(KernelDefinition kernel_definition,
                                             JitCompiler compiler);

  KernelRunner(KernelRunner&&) = default;
  KernelRunner& operator=(KernelRunner&&) = default;

  absl::Status Call(absl::Span<const Argument> arguments) final;

  static absl::StatusOr<JitCompiler> CreateJitCompiler(
      const HloModuleConfig& config);

 private:
  static absl::StatusOr<KernelRunner> Create(
      const KernelSpec& kernel_spec, LlvmIrKernelSource llvm_ir_kernel_source,
      JitCompiler compiler);
  static absl::StatusOr<KernelRunner> Create(
      const KernelSpec& kernel_spec, MlirKernelSource mlir_kernel_source,
      JitCompiler compiler);

  KernelRunner(std::unique_ptr<FunctionLibrary> library, Kernel kernel,
               NumWorkGroups num_workgroups);

  std::unique_ptr<FunctionLibrary> library_;
  Kernel kernel_;
  NumWorkGroups num_workgroups_;
};

absl::StatusOr<LlvmIrKernelSource> LowerToLlvm(
    MlirKernelSource& mlir_kernel_source);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_TESTLIB_KERNEL_RUNNER_H_
