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

#ifndef XLA_STREAM_EXECUTOR_CUDA_DEFER_RELOCATABLE_COMPILATION_COMPILATION_PROVIDER_H_
#define XLA_STREAM_EXECUTOR_CUDA_DEFER_RELOCATABLE_COMPILATION_COMPILATION_PROVIDER_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/device_description.h"

namespace stream_executor::cuda {

// Simulates support for CompileToRelocatableModule by deferring the actual
// compilation to the linking step.
//
// For parallel compilation of LLVM modules, we need support for both
// compilation into relocatable modules and linking. (Individual LLVM modules
// get compiled in parallel and then linked together in a single step.)
//
// Some compilation providers like libnvjitlink do not support compilation into
// relocatable modules. To be able to still benefit from the parallel
// compilation of LLVM modules, we can defer the PTX compilation to the linking
// step using this delegating compilation provider.
class DeferRelocatableCompilationCompilationProvider
    : public CompilationProvider {
 public:
  static absl::StatusOr<
      std::unique_ptr<DeferRelocatableCompilationCompilationProvider>>
  Create(std::unique_ptr<CompilationProvider> delegate);

  bool SupportsCompileToRelocatableModule() const override { return true; }
  bool SupportsCompileAndLink() const override { return true; }

  std::string name() const override {
    return absl::StrFormat(
        "DeferRelocatableCompilationCompilationProvider(delegate: %s)",
        delegate_->name());
  }

  absl::StatusOr<RelocatableModule> CompileToRelocatableModule(
      const CudaComputeCapability& cc, absl::string_view ptx,
      const CompilationOptions& options) const override;

  absl::StatusOr<Assembly> CompileAndLink(
      const CudaComputeCapability& cc,
      absl::Span<const RelocatableModuleOrPtx> inputs,
      const CompilationOptions& options) const override;

  absl::StatusOr<Assembly> Compile(
      const CudaComputeCapability& cc, absl::string_view ptx,
      const CompilationOptions& options) const override;

 private:
  explicit DeferRelocatableCompilationCompilationProvider(
      std::unique_ptr<CompilationProvider> delegate);

  std::unique_ptr<CompilationProvider> delegate_;
};
}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_DEFER_RELOCATABLE_COMPILATION_COMPILATION_PROVIDER_H_
