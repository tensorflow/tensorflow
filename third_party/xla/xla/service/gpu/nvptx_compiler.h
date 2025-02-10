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

#ifndef XLA_SERVICE_GPU_NVPTX_COMPILER_H_
#define XLA_SERVICE_GPU_NVPTX_COMPILER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "llvm/IR/Module.h"
#include "xla/autotune_results.pb.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/compilation_provider_options.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

void WarnIfBadDriverJITVersion();

// NVPTXCompiler generates efficient GPU executables for NVPTX target.
class NVPTXCompiler : public GpuCompiler {
 public:
  explicit NVPTXCompiler();

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

  absl::Status AddGemmFusionAutotuningPasses(
      HloPassPipeline* pipeline, HloModule* hlo_module,
      AutotuneConfig& autotune_config, tsl::thread::ThreadPool* thread_pool,
      const MultiProcessKeyValueStore& key_value_store,
      const se::SemanticVersion& toolkit_version) override;

  absl::Status RunCudnnCompilerPasses(HloModule* module,
                                      se::StreamExecutor* stream_exec,
                                      BinaryMap* dnn_compiled_graphs) override;

  HloDataflowAnalysis::CanShareBuffer GetCanShareBuffer(
      const se::DeviceDescription& device_description) const override;

  absl::StatusOr<BackendCompileResult> CompileTargetBinary(
      const HloModuleConfig& module_config, llvm::Module* llvm_module,
      const stream_executor::DeviceDescription& device_description,
      bool relocatable, const HloModule* debug_module,
      const CompileOptions& options, std::optional<int> shard_number) override;

  absl::StatusOr<bool> CanUseLinkModules(
      const HloModuleConfig& module_config,
      const stream_executor::DeviceDescription& device_description) override;

 private:
  absl::StatusOr<std::vector<uint8_t>> LinkModules(
      const stream_executor::DeviceDescription& device_description,
      se::StreamExecutor* stream_exec,
      std::vector<std::vector<uint8_t>> modules,
      const DebugOptions& debug_options) override;

  absl::Mutex compilation_providers_mutex_;
  absl::flat_hash_map<se::cuda::CompilationProviderOptions,
                      std::unique_ptr<se::cuda::CompilationProvider>>
      compilation_providers_ ABSL_GUARDED_BY(compilation_providers_mutex_);

  absl::StatusOr<const se::cuda::CompilationProvider*> GetCompilationProvider(
      const DebugOptions& debug_options);

  NVPTXCompiler(const NVPTXCompiler&) = delete;
  NVPTXCompiler& operator=(const NVPTXCompiler&) = delete;

  std::vector<std::string> GetLLVMCommandLineOptions(
      const DebugOptions& debug_options) const override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_NVPTX_COMPILER_H_
