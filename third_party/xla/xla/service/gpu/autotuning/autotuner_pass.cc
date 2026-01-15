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

#include "xla/service/gpu/autotuning/autotuner_pass.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/autotuner.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/gpu/autotuner/gpu_profiler.h"
#include "xla/backends/gpu/autotuner/legacy_cache.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

namespace {

AutotuneConfig GetAutotuneConfig(const DebugOptions& debug_options,
                                 bool is_deviceless,
                                 bool optimize_scratch_bytes,
                                 bool allow_reg_spills) {
  AutotuneConfig autotune_config;
  autotune_config.check_buffers = debug_options.xla_gpu_autotune_level() >= 4;
  autotune_config.relative_tolerance =
      debug_options.xla_gpu_autotune_gemm_rtol();
  autotune_config.crash_on_check_failure =
      debug_options.xla_gpu_crash_on_verification_failures();
  autotune_config.dump_logs_to = debug_options.xla_gpu_dump_autotune_logs_to();
  autotune_config.exclude_cublas_config =
      !debug_options.xla_gpu_cublas_fallback();
  autotune_config.select_first_config =
      debug_options.xla_gpu_deterministic_ops() ||
      debug_options.xla_gpu_exclude_nondeterministic_ops() ||
      debug_options.xla_gpu_autotune_level() == 0;

  if (is_deviceless) {
    // If we are running on a deviceless target, we want to use default configs.
    autotune_config.use_default_config = true;
  }
  autotune_config.optimize_scratch_bytes = optimize_scratch_bytes;

  autotune_config.expect_all_instructions_in_cache =
      debug_options.xla_gpu_require_complete_aot_autotune_results();
  autotune_config.dump_hlos =
      debug_options.xla_gpu_dump_autotuned_gemm_fusions();
  if (!debug_options.xla_gpu_fail_ptx_compilation_on_register_spilling() &&
      allow_reg_spills) {
    autotune_config.allow_reg_spills = true;
  }
  // xla_gpu_filter_kernels_spilling_registers_on_autotuning is true by default,
  // but some autotuner passes need to set it to false explicitly as there
  // aren't enough configs to guarantee that no config will spill. So we allow
  // allow_reg_spills to override the autotuner config unless this flag is
  // explicitly set to false.
  if (!debug_options
           .xla_gpu_filter_kernels_spilling_registers_on_autotuning()) {
    autotune_config.allow_reg_spills = false;
  }

  return autotune_config;
}

ProfileOptions GetProfileOptions(const DebugOptions& debug_options,
                                 const AutotuneConfig& autotune_config) {
  ProfileOptions profile_options;
  profile_options.redzone_padding_bytes =
      debug_options.xla_gpu_redzone_padding_bytes();
  profile_options.should_init_buffers = autotune_config.check_buffers;
  return profile_options;
}

}  // namespace

absl::StatusOr<std::unique_ptr<AutotunerPass>> AutotunerPass::Create(
    std::vector<std::unique_ptr<CodegenBackend>> backends,
    const DebugOptions& debug_options,
    stream_executor::StreamExecutor* stream_executor,
    tsl::thread::ThreadPool* thread_pool, InstructionFilterFn should_autotune,
    const Compiler::GpuTargetConfig* target_config,
    se::DeviceAddressAllocator* allocator, bool optimize_scratch_bytes,
    MultiProcessKeyValueStore key_value_store, bool allow_reg_spills) {
  std::unique_ptr<Profiler> profiler = nullptr;
  bool is_deviceless = stream_executor == nullptr;
  AutotuneConfig autotune_config = GetAutotuneConfig(
      debug_options, is_deviceless, optimize_scratch_bytes, allow_reg_spills);
  VLOG(1) << "Autotune config: " << autotune_config.ToString();

  if (!is_deviceless) {
    profiler = GpuProfiler::Create(
        stream_executor, GetProfileOptions(debug_options, autotune_config),
        allocator);
  }

  std::unique_ptr<AutotunerCacheInterface> cache =
      std::make_unique<LegacyCache>(
          debug_options.xla_gpu_experimental_autotuner_cache_dir(),
          debug_options.xla_gpu_experimental_autotune_cache_mode(),
          target_config->device_description);

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Autotuner> autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler),
                        autotune_config, std::move(cache), thread_pool));
  return absl::WrapUnique(new AutotunerPass(
      std::move(autotuner), std::move(should_autotune),
      std::move(key_value_store), debug_options.xla_gpu_shard_autotuning()));
}

absl::StatusOr<bool> AutotunerPass::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_SCOPED_LOGGING_TIMER("AutotunerPass");
  VLOG(1) << "Running Autotuner Pass";

  bool shard_autotuning =
      enable_sharding_ && key_value_store_.process_count > 1;
  if (shard_autotuning) {
    TF_RETURN_IF_ERROR(
        autotuner_->Autotune(module, should_autotune_, key_value_store_));
  } else {
    TF_RETURN_IF_ERROR(autotuner_->Autotune(module, should_autotune_));
  }
  return true;
}

}  // namespace gpu
}  // namespace xla
