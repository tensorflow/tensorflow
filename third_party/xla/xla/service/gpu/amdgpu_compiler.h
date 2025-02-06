/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_AMDGPU_COMPILER_H_
#define XLA_SERVICE_GPU_AMDGPU_COMPILER_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/IR/Module.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace gpu {

// AMDGPUCompiler generates efficient GPU executables for AMDGPU target.
class AMDGPUCompiler : public GpuCompiler {
 public:
  AMDGPUCompiler();

  absl::Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module, se::GpuComputeCapability gpu_version,
      se::dnn::VersionInfo dnn_version,
      const se::SemanticVersion& toolkit_version) override;

  absl::Status OptimizeHloPostLayoutAssignment(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      const CompileOptions& options, const TargetConfig& gpu_target_config,
      tsl::thread::ThreadPool* thread_pool) override;

  bool RequiresCollectiveScheduleLinearizer(
      const HloModule* module, se::StreamExecutor* stream_exec) override;

  absl::Status AddConvAndGemmAutotuningPasses(
      HloPassPipeline* pipeline, const se::GpuComputeCapability& gpu_version,
      const CompileOptions& options, HloModule* hlo_module,
      AutotuneConfig& autotune_config,
      tsl::thread::ThreadPool* thread_pool) override;

  absl::StatusOr<BackendCompileResult> CompileTargetBinary(
      const HloModuleConfig& module_config, llvm::Module* llvm_module,
      const se::DeviceDescription& device_description, bool relocatable,
      const HloModule* debug_module, const CompileOptions& options,
      std::optional<int> shard_number) override;

  absl::Status AddGemmFusionAutotuningPasses(
      HloPassPipeline* pipeline, HloModule* hlo_module,
      AutotuneConfig& autotune_config, tsl::thread::ThreadPool* thread_pool,
      const MultiProcessKeyValueStore& key_value_store,
      const se::SemanticVersion& toolkit_version) override;

 private:
  AMDGPUCompiler(const AMDGPUCompiler&) = delete;
  AMDGPUCompiler& operator=(const AMDGPUCompiler&) = delete;

  std::vector<std::string> GetLLVMCommandLineOptions(
      const DebugOptions& debug_options) const override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_AMDGPU_COMPILER_H_
