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

#include "xla/stream_executor/cuda/nvjitlink_compilation_provider.h"

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/nvjitlink.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::cuda {

absl::StatusOr<Assembly>
stream_executor::cuda::NvJitLinkCompilationProvider::Compile(
    const CudaComputeCapability& cc, absl::string_view ptx,
    const CompilationOptions& options) const {
  return CompileAndLink(cc, {Ptx{std::string{ptx}}}, options);
}

absl::StatusOr<RelocatableModule>
stream_executor::cuda::NvJitLinkCompilationProvider::CompileToRelocatableModule(
    const CudaComputeCapability& cc, absl::string_view ptx,
    const CompilationOptions& options) const {
  return absl::UnavailableError(
      "Compilation to relocatable module is not supported.");
}

absl::StatusOr<Assembly>
stream_executor::cuda::NvJitLinkCompilationProvider::CompileAndLink(
    const CudaComputeCapability& cc,
    absl::Span<const RelocatableModuleOrPtx> inputs,
    const CompilationOptions& options) const {
  GpuAsmOpts asm_opts;
  asm_opts.disable_gpuasm_optimizations = options.disable_optimizations;

  if (options.generate_line_info) {
    asm_opts.extra_flags.push_back("--generate-line-info");
  }
  if (options.generate_debug_info) {
    asm_opts.extra_flags.push_back("--device-debug");
  }

  std::vector<NvJitLinkInput> nvjitlink_inputs;
  for (const auto& input : inputs) {
    if (std::holds_alternative<RelocatableModule>(input)) {
      nvjitlink_inputs.push_back({NvJitLinkInput::Type::kCubin,
                                  std::get<RelocatableModule>(input).cubin});
    } else {
      // The span needs to be null-terminated, that why we increase the size by
      // one.
      absl::Span<const uint8_t> ptx_bytes{
          reinterpret_cast<const uint8_t*>(std::get<Ptx>(input).ptx.data()),
          std::get<Ptx>(input).ptx.size() + 1};
      nvjitlink_inputs.push_back({NvJitLinkInput::Type::kPtx, ptx_bytes});
    }
  }

  TF_ASSIGN_OR_RETURN(auto cubin, CompileAndLinkUsingLibNvJitLink(
                                      cc, nvjitlink_inputs, asm_opts,
                                      options.cancel_if_reg_spill));
  return Assembly{std::move(cubin)};
}

}  // namespace stream_executor::cuda
