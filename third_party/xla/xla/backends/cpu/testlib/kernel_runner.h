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
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/kernel.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/backends/cpu/testlib/llvm_ir_kernel_spec.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/testlib/kernel_runner.h"

namespace xla::cpu {

// Kernel runner for XLA:CPU backend.
class KernelRunner final : public xla::KernelRunner {
 public:
  // Create a KernelRunner from a KernelSpec, this factory takes care of the
  // downcasting to supported kernel spec types, currently only LlvmIrKernelSpec
  // is supported.
  static absl::StatusOr<KernelRunner> Create(
      std::unique_ptr<KernelSpec> kernel_spec);

  // Keep this llvm specific constructor for python bindings:
  // nanobind will do the downcasting for us and give the python specific
  // error if there is not a valid Create(...) call.
  static absl::StatusOr<KernelRunner> Create(LlvmIrKernelSpec kernel_spec);

  KernelRunner(KernelRunner&&) = default;
  KernelRunner& operator=(KernelRunner&&) = default;

  absl::Status Call(absl::Span<const Argument> arguments) final;

 private:
  KernelRunner(std::unique_ptr<FunctionLibrary> library, Kernel kernel,
               Kernel::ThreadDim thread_dim);

  std::unique_ptr<FunctionLibrary> library_;
  Kernel kernel_;
  Kernel::ThreadDim thread_dim_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_TESTLIB_KERNEL_RUNNER_H_
