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

#ifndef XLA_SERVICE_GPU_AUTOTUNING_AUTOTUNER_PASS_H_
#define XLA_SERVICE_GPU_AUTOTUNING_AUTOTUNER_PASS_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/codegen_orchestrator.h"
#include "xla/backends/autotuner/config_assigner.h"
#include "xla/backends/autotuner/hlo_extractor.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/compiler.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform_id.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

namespace xla {
class AliasInfo;
namespace gpu {

class GpuCompiler;

ConfigAssigner::Options GetConfigAssignerOptions(
    const DebugOptions& debug_options, bool is_deviceless = false);

CodegenOrchestrator::Options GetCodegenOrchestratorOptions(
    const DebugOptions& debug_options);

ProfileOptions GetProfileOptions(
    const DebugOptions& debug_options,
    const ConfigAssigner::Options& config_assigner_options);

// HloModulePass that runs the autotuner.
class AutotunerPass : public HloModulePass {
 public:
  using GetBackendsFn = std::function<
      absl::StatusOr<std::vector<std::unique_ptr<CodegenBackend>>>()>;

  static absl::StatusOr<std::vector<std::unique_ptr<CodegenBackend>>>
  GetGpuAutotunerBackends(se::StreamExecutor* stream_exec,
                          se::DeviceAddressAllocator* device_allocator,
                          const Compiler::GpuTargetConfig* target_config,
                          const AliasInfo* alias_info,
                          const DebugOptions& debug_options,
                          mlir::MLIRContext* mlir_context,
                          HloCostAnalysis::ShapeSizeFunction shape_size_fn,
                          Compiler* compiler, se::PlatformId platform_id);

  // Note: the target_config must outlive the pass.
  static absl::StatusOr<std::unique_ptr<AutotunerPass>> Create(
      GetBackendsFn get_backends_fn, const DebugOptions& debug_options,
      const se::GpuComputeCapability& gpu_version,
      se::StreamExecutor* stream_executor, tsl::thread::ThreadPool* thread_pool,
      const Compiler::GpuTargetConfig* target_config,
      const AliasInfo* alias_info, mlir::MLIRContext* mlir_context,
      HloCostAnalysis::ShapeSizeFunction shape_size_fn,
      se::DeviceAddressAllocator* allocator = nullptr,
      MultiProcessKeyValueStore key_value_store = MultiProcessKeyValueStore());

  absl::string_view name() const override { return "autotuner"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  explicit AutotunerPass(std::unique_ptr<ConfigAssigner> config_assigner,
                         InstructionFilterFn should_autotune,
                         MultiProcessKeyValueStore key_value_store,
                         bool enable_sharding)
      : config_assigner_(std::move(config_assigner)),
        should_autotune_(std::move(should_autotune)),
        key_value_store_(std::move(key_value_store)),
        enable_sharding_(enable_sharding) {}

  std::unique_ptr<ConfigAssigner> config_assigner_;
  InstructionFilterFn should_autotune_;
  MultiProcessKeyValueStore key_value_store_;
  bool enable_sharding_ = false;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_AUTOTUNING_AUTOTUNER_PASS_H_
