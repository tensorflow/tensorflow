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

#include "xla/service/gpu/target_constants.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace xla {
namespace gpu {

IntelGpuCompiler::IntelGpuCompiler()
    : GpuCompiler(stream_executor::sycl::kSyclPlatformId, spir::TargetTriple(),
                  spir::DataLayout()) {}

absl::Status IntelGpuCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, se::GpuComputeCapability gpu_version,
    se::dnn::VersionInfo dnn_version,
    const se::SemanticVersion& toolkit_version) {
  // Note: this is a stub.
  return absl::OkStatus();
}

absl::StatusOr<GpuCompiler::BackendCompileResult>
IntelGpuCompiler::CompileTargetBinary(
    const HloModuleConfig& module_config, llvm::Module* llvm_module,
    const stream_executor::DeviceDescription& device_description,
    bool relocatable, const HloModule* debug_module,
    const CompileOptions& options, std::optional<int> shard_number) {
  // Note: this is a stub.
  return BackendCompileResult{};
}

std::vector<std::string> IntelGpuCompiler::GetLLVMCommandLineOptions(
    const DebugOptions& debug_options) const {
  return {};
}

}  // namespace gpu
}  // namespace xla
