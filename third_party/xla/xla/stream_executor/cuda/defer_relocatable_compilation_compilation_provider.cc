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

#include "xla/stream_executor/cuda/defer_relocatable_compilation_compilation_provider.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/device_description.h"

namespace stream_executor::cuda {

absl::StatusOr<std::unique_ptr<DeferRelocatableCompilationCompilationProvider>>
DeferRelocatableCompilationCompilationProvider::Create(
    std::unique_ptr<CompilationProvider> delegate) {
  if (!delegate->SupportsCompileAndLink()) {
    return absl::InvalidArgumentError(
        "The delegate compilation provider does not support CompileAndLink.");
  }
  if (delegate->SupportsCompileToRelocatableModule()) {
    return absl::InvalidArgumentError(
        "The delegate compilation provider supports "
        "CompileToRelocatableModule. Using "
        "DeferRelocatableCompilationCompilationProvider is not necessary.");
  }
  return absl::WrapUnique(
      new DeferRelocatableCompilationCompilationProvider(std::move(delegate)));
}
DeferRelocatableCompilationCompilationProvider::
    DeferRelocatableCompilationCompilationProvider(
        std::unique_ptr<CompilationProvider> delegate)
    : delegate_(std::move(delegate)) {}

constexpr const uint8_t kPtxPrefix[] = {'P', 'T', 'X', ':', ' '};

absl::StatusOr<RelocatableModule>
DeferRelocatableCompilationCompilationProvider::CompileToRelocatableModule(
    const CudaComputeCapability& cc, absl::string_view ptx,
    const CompilationOptions& options) const {
  if (ptx.empty()) return RelocatableModule{};

  // Instead of actually compiling the PTX to CUBIN, we just bundle the PTX
  // string into a fake CUBIN. In the `CompileAndLink` call we detect the prefix
  // and properly convert it back into a PTX input module.
  std::vector<uint8_t> cubin;
  cubin.reserve(sizeof(kPtxPrefix) + ptx.size() + 1);
  cubin.insert(cubin.end(), kPtxPrefix, kPtxPrefix + sizeof(kPtxPrefix));
  cubin.insert(cubin.end(), ptx.begin(), ptx.end());
  return RelocatableModule{std::move(cubin)};
}

absl::StatusOr<Assembly>
DeferRelocatableCompilationCompilationProvider::CompileAndLink(
    const CudaComputeCapability& cc,
    absl::Span<const RelocatableModuleOrPtx> inputs,
    const CompilationOptions& options) const {
  std::vector<RelocatableModuleOrPtx> deferred_inputs;
  deferred_inputs.reserve(inputs.size());
  for (const auto& input : inputs) {
    if (std::holds_alternative<Ptx>(input)) {
      deferred_inputs.push_back(std::get<Ptx>(input));
      continue;
    }
    const absl::Span<const uint8_t> cubin =
        std::get<RelocatableModule>(input).cubin;

    if (cubin.first(std::min(sizeof(kPtxPrefix), cubin.size())) == kPtxPrefix) {
      deferred_inputs.push_back(Ptx{std::string(
          reinterpret_cast<const char*>(cubin.data()) + sizeof(kPtxPrefix),
          cubin.size() - sizeof(kPtxPrefix))});
      continue;
    }

    deferred_inputs.push_back(std::get<RelocatableModule>(input));
  }
  return delegate_->CompileAndLink(cc, deferred_inputs, options);
}

absl::StatusOr<Assembly>
DeferRelocatableCompilationCompilationProvider::Compile(
    const CudaComputeCapability& cc, absl::string_view ptx,
    const CompilationOptions& options) const {
  return delegate_->Compile(cc, ptx, options);
}
}  // namespace stream_executor::cuda
