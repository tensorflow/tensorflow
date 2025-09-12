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
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

namespace {

AutotuneConfig GetAutotuneConfig(const DebugOptions& debug_options) {
  AutotuneConfig autotune_config;
  autotune_config.check_buffers = debug_options.xla_gpu_autotune_level() >= 4;
  autotune_config.relative_tolerance =
      debug_options.xla_gpu_autotune_gemm_rtol();
  autotune_config.crash_on_check_failure =
      debug_options.xla_gpu_crash_on_verification_failures();
  autotune_config.expect_all_instructions_in_cache =
      debug_options.xla_gpu_require_complete_aot_autotune_results();
  autotune_config.dump_logs_to = debug_options.xla_gpu_dump_autotune_logs_to();
  return autotune_config;
}

ProfileOptions GetProfileOptions(const DebugOptions& debug_options) {
  ProfileOptions profile_options;
  profile_options.redzone_padding_bytes =
      debug_options.xla_gpu_redzone_padding_bytes();
  return profile_options;
}

}  // namespace

absl::StatusOr<std::unique_ptr<AutotunerPass>> AutotunerPass::Create(
    std::vector<std::unique_ptr<CodegenBackend>> backends,
    const DebugOptions& debug_options,
    stream_executor::StreamExecutor* stream_executor,
    tsl::thread::ThreadPool* thread_pool, InstructionFilterFn should_autotune,
    se::DeviceMemoryAllocator* allocator) {
  // At least one of stream_executor or allocator must be provided.
  CHECK(stream_executor != nullptr || allocator != nullptr);

  std::unique_ptr<GpuProfiler> profiler = GpuProfiler::Create(
      stream_executor, GetProfileOptions(debug_options), allocator);

  std::unique_ptr<AutotunerCacheInterface> cache =
      std::make_unique<LegacyCache>(
          debug_options.xla_gpu_experimental_autotuner_cache_dir(),
          debug_options.xla_gpu_experimental_autotune_cache_mode(),
          stream_executor->GetDeviceDescription());

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Autotuner> autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler),
                        GetAutotuneConfig(debug_options), std::move(cache),
                        thread_pool));
  return absl::WrapUnique(
      new AutotunerPass(std::move(autotuner), should_autotune));
}

absl::StatusOr<bool> AutotunerPass::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Running Autotuner Pass";

  TF_RETURN_IF_ERROR(autotuner_->Autotune(module, should_autotune_));
  return true;
}

}  // namespace gpu
}  // namespace xla
