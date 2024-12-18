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

#include "xla/stream_executor/cuda/nvptxcompiler_compilation_provider.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/ptx_compiler.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::cuda {
absl::StatusOr<std::vector<uint8_t>>
NvptxcompilerCompilationProvider::CompileHelper(
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

  return CompileGpuAsmUsingLibNvPtxCompiler(cc, std::string(ptx), asm_opts,
                                            options.cancel_if_reg_spill);
}

absl::StatusOr<Assembly> NvptxcompilerCompilationProvider::Compile(
    const CudaComputeCapability& cc, absl::string_view ptx,
    const CompilationOptions& options) const {
  TF_ASSIGN_OR_RETURN(auto cubin,
                      CompileHelper(cc, ptx, options,
                                    /*compile_to_relocatable_module=*/false));
  return Assembly{std::move(cubin)};
}

absl::StatusOr<RelocatableModule>
NvptxcompilerCompilationProvider::CompileToRelocatableModule(
    const CudaComputeCapability& cc, absl::string_view ptx,
    const CompilationOptions& options) const {
  TF_ASSIGN_OR_RETURN(auto cubin,
                      CompileHelper(cc, ptx, options,
                                    /*compile_to_relocatable_module=*/true));
  return RelocatableModule{std::move(cubin)};
}

absl::StatusOr<Assembly> NvptxcompilerCompilationProvider::CompileAndLink(
    const CudaComputeCapability& cc,
    absl::Span<const RelocatableModuleOrPtx> inputs,
    const CompilationOptions& options) const {
  return absl::UnimplementedError(
      "Compilation and linking is not supported by nvptxcompiler.");
}
}  // namespace stream_executor::cuda
