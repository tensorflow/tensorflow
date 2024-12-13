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

#include "xla/stream_executor/cuda/composite_compilation_provider.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/device_description.h"

namespace stream_executor::cuda {

std::string CompositeCompilationProvider::name() const {
  return absl::StrCat("CompositeCompilationProvider(",
                      absl::StrJoin(providers_, ", ",
                                    [](std::string* out, const auto& provider) {
                                      absl::StrAppend(out, provider->name());
                                    }),
                      ")");
}

bool CompositeCompilationProvider::SupportsCompileToRelocatableModule() const {
  return relocatable_compilation_provider_ != nullptr;
}

bool CompositeCompilationProvider::SupportsCompileAndLink() const {
  return compile_and_link_compilation_provider_ != nullptr;
}

CompositeCompilationProvider::CompositeCompilationProvider(
    std::vector<std::unique_ptr<CompilationProvider>> providers)
    : providers_(std::move(providers)) {
  for (const auto& provider : providers_) {
    if (provider->SupportsCompileToRelocatableModule()) {
      relocatable_compilation_provider_ = provider.get();
      break;
    }
  }
  for (const auto& provider : providers_) {
    if (provider->SupportsCompileAndLink()) {
      compile_and_link_compilation_provider_ = provider.get();
      break;
    }
  }
}

absl::StatusOr<std::unique_ptr<CompositeCompilationProvider>>
CompositeCompilationProvider::Create(
    std::vector<std::unique_ptr<CompilationProvider>> providers) {
  if (providers.empty()) {
    return absl::InvalidArgumentError("No providers provided");
  }
  return absl::WrapUnique(
      new CompositeCompilationProvider(std::move(providers)));
}

absl::StatusOr<Assembly> CompositeCompilationProvider::Compile(
    const CudaComputeCapability& cc, absl::string_view ptx,
    const CompilationOptions& options) const {
  return providers_.front()->Compile(cc, ptx, options);
}

absl::StatusOr<RelocatableModule>
CompositeCompilationProvider::CompileToRelocatableModule(
    const CudaComputeCapability& cc, absl::string_view ptx,
    const CompilationOptions& options) const {
  if (!relocatable_compilation_provider_) {
    return absl::UnavailableError(
        "Compilation provider doesn't support CompileToRelocatableModule");
  }
  return relocatable_compilation_provider_->CompileToRelocatableModule(cc, ptx,
                                                                       options);
}

absl::StatusOr<Assembly> CompositeCompilationProvider::CompileAndLink(
    const CudaComputeCapability& cc,
    absl::Span<const RelocatableModuleOrPtx> inputs,
    const CompilationOptions& options) const {
  if (!compile_and_link_compilation_provider_) {
    return absl::UnavailableError(
        "Compilation provider doesn't support CompileAndLink");
  }
  return compile_and_link_compilation_provider_->CompileAndLink(cc, inputs,
                                                                options);
}

}  // namespace stream_executor::cuda
