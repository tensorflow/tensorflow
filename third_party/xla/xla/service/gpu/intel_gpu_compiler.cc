/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/intel_gpu_compiler.h"

#include "xla/service/dump.h"
#include "xla/service/gpu/llvm_gpu_backend/spirv_backend.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace xla {
namespace gpu {

IntelGpuCompiler::IntelGpuCompiler()
    : GpuCompiler(stream_executor::sycl::kSyclPlatformId, spir::TargetTriple(),
                  spir::DataLayout()) {}

absl::Status IntelGpuCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, const se::GpuComputeCapability& gpu_version,
    se::dnn::VersionInfo dnn_version,
    const se::SemanticVersion& toolkit_version,
    CompilationStats* compilation_stats) {
  // Return OkStatus as a stub.
  return absl::OkStatus();
}

absl::Status IntelGpuCompiler::AddConvAndGemmAutotuningPass(
    HloPassPipeline* pipeline, HloModule* hlo_module,
    const se::GpuComputeCapability& gpu_version, const CompileOptions& options,
    AutotuneConfig& autotune_config, tsl::thread::ThreadPool* thread_pool,
    se::StreamExecutor* stream_exec,
    const Compiler::GpuTargetConfig* target_config,
    const MultiProcessKeyValueStore& key_value_store,
    const se::SemanticVersion& toolkit_version, const AliasInfo* alias_info,
    const DebugOptions& debug_options, mlir::MLIRContext* mlir_context,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn) {
  // Return OkStatus as a stub.
  return absl::OkStatus();
}

absl::StatusOr<GpuCompiler::BackendCompileResult>
IntelGpuCompiler::CompileTargetBinary(
    const HloModuleConfig& module_config, llvm::Module* llvm_module,
    const stream_executor::DeviceDescription& device_description,
    bool relocatable, const HloModule* debug_module,
    const CompileOptions& options, std::optional<int> shard_number) {
  TF_ASSIGN_OR_RETURN(
      auto spirv_str,
      spirv::CompileToSPIRV(llvm_module,
                            device_description.gpu_compute_capability(),
                            module_config.debug_options()));
  if (DumpingEnabledForHloModule(debug_module ? debug_module->name() : "",
                                 module_config.debug_options())) {
    if (debug_module) {
      DumpToFileInDirOrStdout(*debug_module, "", "spv", spirv_str);
    } else {
      LOG(ERROR) << "Dumping is not implemented since the file name cannot be "
                    "inferred. Please implement (potentially MLIR) module -> "
                    "filename heuristic.";
    }
  }
  std::vector<uint8_t> spirv_bin(spirv_str.begin(), spirv_str.end());
  return BackendCompileResult{/*asm_text=*/"", std::move(spirv_bin)};
}

std::vector<std::string> IntelGpuCompiler::GetLLVMCommandLineOptions(
    const DebugOptions& debug_options) const {
  return spirv::GetSPIRVBackendOptions(debug_options);
}

}  // namespace gpu
}  // namespace xla
