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

#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/compileiq_db.pb.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/ptx_compiler.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/env.h"

namespace stream_executor::cuda {
absl::StatusOr<Assembly> CompileHelper(const CudaComputeCapability& cc,
                                       absl::string_view ptx,
                                       absl::string_view kernel_fingerprint,
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
  asm_opts.extra_flags.insert(asm_opts.extra_flags.end(),
                              options.additional_ptxas_flags.begin(),
                              options.additional_ptxas_flags.end());

  std::string acf_path;
  absl::Cleanup acf_cleaner = [&acf_path] {
    if (!acf_path.empty()) {
      auto status = tsl::Env::Default()->DeleteFile(acf_path);
      if (!status.ok()) {
        LOG(ERROR) << "Unable to delete " << acf_path << ":" << status;
      }
    }
  };
  if (!kernel_fingerprint.empty() && !options.compileiq_db.empty()) {
    ABSL_ASSIGN_OR_RETURN(SemanticVersion version, GetLibNvPtxCompilerVersion());
    if (version < SemanticVersion{13, 5, 0}) {
      return absl::UnimplementedError(
          "Current libnvptxcompiler does not support ACF.");
    }

    tsl::Env* env = tsl::Env::Default();
    xla::gpu::CompileIqDb db;
    ABSL_RETURN_IF_ERROR(ReadTextOrBinaryProto(env, options.compileiq_db, &db));

    if (db.format_version() != 0) {
      return absl::InvalidArgumentError("Unsupported CompileIq db version.");
    }
    if (auto it = db.control_files().find(kernel_fingerprint);
        it != db.control_files().end()) {
      for (const xla::gpu::AdvancedControlFile& acf : it->second.files()) {
        if (version.ToString() != acf.cuda_version()) {
          continue;
        }
        if (cc.GetPtxAsTargetName() != acf.gpu_arch()) {
          continue;
        }
        if (!env->LocalTempFilename(&acf_path)) {
          return absl::InternalError("Couldn't get temp file name for acf.");
        }
        ABSL_RETURN_IF_ERROR(tsl::WriteStringToFile(env, acf_path, acf.content()));
        VLOG(2) << "acf written to: " << acf_path;
        asm_opts.extra_flags.push_back(
            absl::StrCat("--apply-controls=", acf_path));
        break;
      }
    }
  }

  return CompileGpuAsmUsingLibNvPtxCompiler(cc, ptx, asm_opts,
                                            options.cancel_if_reg_spill,
                                            options.dump_compilation_log);
}

absl::StatusOr<Assembly> NvptxcompilerCompilationProvider::Compile(
    const CudaComputeCapability& cc, absl::string_view ptx,
    absl::string_view kernel_fingerprint,
    const CompilationOptions& options) const {
  return CompileHelper(cc, ptx, kernel_fingerprint, options,
                       /*compile_to_relocatable_module=*/false);
}

absl::StatusOr<RelocatableModule>
NvptxcompilerCompilationProvider::CompileToRelocatableModule(
    const CudaComputeCapability& cc, absl::string_view ptx,
    const CompilationOptions& options) const {
  ABSL_ASSIGN_OR_RETURN(Assembly assembly,
                   CompileHelper(cc, ptx, "", options,
                                 /*compile_to_relocatable_module=*/true));
  return RelocatableModule{std::move(assembly.cubin),
                           std::move(assembly.compilation_log),
                           std::move(assembly.module_stats)};
}

absl::StatusOr<Assembly> NvptxcompilerCompilationProvider::CompileAndLink(
    const CudaComputeCapability& cc,
    absl::Span<const RelocatableModuleOrPtx> inputs,
    const CompilationOptions& options) const {
  return absl::UnimplementedError(
      "Compilation and linking is not supported by nvptxcompiler.");
}

absl::StatusOr<int> NvptxcompilerCompilationProvider::GetLatestPtxIsaVersion()
    const {
  return GetLatestPtxIsaVersionForNvptxCompiler();
}

}  // namespace stream_executor::cuda
