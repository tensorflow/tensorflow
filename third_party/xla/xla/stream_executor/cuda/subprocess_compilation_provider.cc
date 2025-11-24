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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/ptx_compiler_helpers.h"
#include "xla/stream_executor/cuda/subprocess_compilation.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/subprocess.h"

namespace stream_executor::cuda {

namespace {

absl::StatusOr<Assembly> CompileHelper(absl::string_view ptxas_path,
                                       const CudaComputeCapability& cc,
                                       absl::string_view ptx,
                                       const CompilationOptions& options,
                                       bool compile_to_relocatable_module) {
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

  return CompileGpuAsmUsingPtxAs(ptxas_path, cc, ptx, asm_opts,
                                 options.cancel_if_reg_spill,
                                 options.dump_compilation_log);
}

}  // namespace

absl::StatusOr<Assembly> SubprocessCompilationProvider::Compile(
    const CudaComputeCapability& cc, absl::string_view ptx,
    const CompilationOptions& options) const {
  return CompileHelper(path_to_ptxas_, cc, ptx, options,
                       /*compile_to_relocatable_module=*/false);
}

absl::StatusOr<RelocatableModule>
SubprocessCompilationProvider::CompileToRelocatableModule(
    const CudaComputeCapability& cc, absl::string_view ptx,
    const CompilationOptions& options) const {
  TF_ASSIGN_OR_RETURN(auto assembly,
                      CompileHelper(path_to_ptxas_, cc, ptx, options,
                                    /*compile_to_relocatable_module=*/true));
  return RelocatableModule{std::move(assembly.cubin),
                           std::move(assembly.compilation_log)};
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

absl::StatusOr<int> SubprocessCompilationProvider::GetLatestPtxIsaVersion()
    const {
  std::vector<std::string> ptxas_args = {path_to_ptxas_, "--input-as-string",
                                         ".version 99.99"};
  tsl::SubProcess ptxas_info_dumper;
  ptxas_info_dumper.SetProgram(path_to_ptxas_, ptxas_args);
  ptxas_info_dumper.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
  if (!ptxas_info_dumper.Start()) {
    return absl::InternalError("Failed to launch ptxas");
  }
  std::string stderr_output;
  int exit_status = ptxas_info_dumper.Communicate(
      /*stdin_input=*/nullptr, /*stdout_output=*/nullptr, &stderr_output);
  if (exit_status == 0) {
    return absl::InternalError("ptxas succeeded where it was expected to fail");
  }

  return GetLatestPtxIsaVersionFromUnsupportedVersionErrorLog(stderr_output);
}

std::string SubprocessCompilationProvider::name() const {
  return absl::StrFormat("SubprocessCompilationProvider(ptxas: %s, nvlink: %s)",
                         path_to_ptxas_, path_to_nvlink_);
}

}  // namespace stream_executor::cuda
