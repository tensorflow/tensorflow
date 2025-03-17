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

#include "xla/stream_executor/cuda/subprocess_compilation_provider.h"

#include <sys/types.h>

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/subprocess_compilation.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::cuda {

absl::StatusOr<std::vector<uint8_t>>
SubprocessCompilationProvider::CompileHelper(
    const CudaComputeCapability& cc, absl::string_view ptx,
    const CompilationOptions& options,
    bool compile_to_relocatable_module) const {
  GpuAsmOpts asm_opts{};
  asm_opts.disable_gpuasm_optimizations = options.disable_optimizations;
  if (compile_to_relocatable_module) {
    asm_opts.extra_flags.push_back("-c");
  }

  if (options.generate_line_info) {
    asm_opts.extra_flags.push_back("--generate-line-info");
  }
  if (options.generate_debug_info) {
    asm_opts.extra_flags.push_back("--device-debug");
  }

  return CompileGpuAsmUsingPtxAs(path_to_ptxas_, cc, ptx, asm_opts,
                                 options.cancel_if_reg_spill);
}

absl::StatusOr<Assembly> SubprocessCompilationProvider::Compile(
    const CudaComputeCapability& cc, absl::string_view ptx,
    const CompilationOptions& options) const {
  TF_ASSIGN_OR_RETURN(auto cubin,
                      CompileHelper(cc, ptx, options,
                                    /*compile_to_relocatable_module=*/false));
  return Assembly{std::move(cubin)};
}

absl::StatusOr<RelocatableModule>
SubprocessCompilationProvider::CompileToRelocatableModule(
    const CudaComputeCapability& cc, absl::string_view ptx,
    const CompilationOptions& options) const {
  TF_ASSIGN_OR_RETURN(auto cubin,
                      CompileHelper(cc, ptx, options,
                                    /*compile_to_relocatable_module=*/true));
  return RelocatableModule{std::move(cubin)};
}

absl::StatusOr<Assembly> SubprocessCompilationProvider::CompileAndLink(
    const CudaComputeCapability& cc,
    absl::Span<const RelocatableModuleOrPtx> inputs,
    const CompilationOptions& options) const {
  std::vector<std::vector<uint8_t>> images;
  for (const auto& input : inputs) {
    if (std::holds_alternative<RelocatableModule>(input)) {
      images.push_back(std::get<RelocatableModule>(input).cubin);
    } else {
      // If we have a PTX string, we need to compile it to CUBIN first.
      TF_ASSIGN_OR_RETURN(
          RelocatableModule module,
          CompileToRelocatableModule(cc, std::get<Ptx>(input).ptx, options));
      images.push_back(std::move(module.cubin));
    }
  }

  TF_ASSIGN_OR_RETURN(auto cubin, LinkUsingNvlink(path_to_nvlink_, cc, images));
  return Assembly{std::move(cubin)};
}

std::string SubprocessCompilationProvider::name() const {
  return absl::StrFormat("SubprocessCompilationProvider(ptxas: %s, nvlink: %s)",
                         path_to_ptxas_, path_to_nvlink_);
}

}  // namespace stream_executor::cuda
