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

#ifndef XLA_STREAM_EXECUTOR_CUDA_COMPOSITE_COMPILATION_PROVIDER_H_
#define XLA_STREAM_EXECUTOR_CUDA_COMPOSITE_COMPILATION_PROVIDER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/device_description.h"

namespace stream_executor::cuda {

// Takes a list of CompilationProviders and delegates to the first one that
// supports the requested operation. Effectively this means there are never more
// than 3 providers in use - the first one which supports `Compile`, one that
// supports `CompileToRelocatableModule` and one that supports `CompileAndLink`.
//
// Note that it's up to the user to ensure that the providers are compatible
// with each other.
//
// A typical use case is to combine a provider that doesn't support relocatable
// compilation (e.g. NvJitLink or driver) with a driver that doesn't support
// linking (e.g. NvPtxCompiler).
class CompositeCompilationProvider : public CompilationProvider {
 public:
  static absl::StatusOr<std::unique_ptr<CompositeCompilationProvider>> Create(
      std::vector<std::unique_ptr<CompilationProvider>> providers);

  std::string name() const override;
  bool SupportsCompileToRelocatableModule() const override;
  bool SupportsCompileAndLink() const override;

  absl::StatusOr<Assembly> Compile(
      const CudaComputeCapability& cc, absl::string_view ptx,
      const CompilationOptions& options) const override;
  absl::StatusOr<RelocatableModule> CompileToRelocatableModule(
      const CudaComputeCapability& cc, absl::string_view ptx,
      const CompilationOptions& options) const override;
  absl::StatusOr<Assembly> CompileAndLink(
      const CudaComputeCapability& cc,
      absl::Span<const RelocatableModuleOrPtx> inputs,
      const CompilationOptions& options) const override;

 private:
  explicit CompositeCompilationProvider(
      std::vector<std::unique_ptr<CompilationProvider>> providers);

  std::vector<std::unique_ptr<CompilationProvider>> providers_;
  CompilationProvider* relocatable_compilation_provider_ = nullptr;
  CompilationProvider* compile_and_link_compilation_provider_ = nullptr;
};

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_COMPOSITE_COMPILATION_PROVIDER_H_
