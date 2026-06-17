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

#include "xla/backends/autotuner/autotuner.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/codegen_orchestrator.h"
#include "xla/backends/autotuner/config_assigner.h"
#include "xla/backends/autotuner/hlo_extractor.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/autotuner/tuner.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {

absl::StatusOr<std::unique_ptr<Autotuner>> Autotuner::Create(
    std::vector<std::unique_ptr<CodegenBackend>> codegen_backends,
    std::unique_ptr<Profiler> profiler, AutotuneConfig autotune_config,
    std::unique_ptr<AutotunerCacheInterface> cache,
    tsl::thread::ThreadPool* thread_pool) {
  CodegenOrchestrator::Options orchestrator_options;
  orchestrator_options.allow_reg_spills_fn =
      autotune_config.allow_reg_spills_fn;
  orchestrator_options.exclude_cublas_config =
      autotune_config.exclude_cublas_config;

  ASSIGN_OR_RETURN(auto orchestrator, CodegenOrchestrator::Create(
                                          std::move(codegen_backends),
                                          orchestrator_options, thread_pool));

  if (cache == nullptr) {
    cache = std::make_unique<NoOpAutotunerCache>();
  }

  ConfigAssigner::Options assigner_options;
  assigner_options.use_default_config = autotune_config.use_default_config;
  assigner_options.select_first_config = autotune_config.select_first_config;
  assigner_options.expect_all_instructions_in_cache =
      autotune_config.expect_all_instructions_in_cache;
  assigner_options.dump_hlos = autotune_config.dump_hlos;

  assigner_options.check_buffers = autotune_config.check_buffers;
  assigner_options.relative_tolerance = autotune_config.relative_tolerance;
  assigner_options.crash_on_check_failure =
      autotune_config.crash_on_check_failure;
  assigner_options.scratch_bytes_window_size_us =
      autotune_config.scratch_bytes_window_size_us;
  assigner_options.dump_logs_to = autotune_config.dump_logs_to;

  ASSIGN_OR_RETURN(
      auto assigner,
      ConfigAssigner::Create(assigner_options, std::move(cache),
                             std::move(orchestrator), std::move(profiler)));

  return absl::WrapUnique(new Autotuner(std::move(assigner)));
}

absl::Status Autotuner::Autotune(HloInstruction* instr) {
  return assigner_->AssignConfig(instr);
}

absl::Status Autotuner::Autotune(HloModule* module,
                                 const InstructionFilterFn& should_autotune) {
  return assigner_->AssignConfigs(module, should_autotune);
}

absl::Status Autotuner::Autotune(HloModule* module,
                                 const InstructionFilterFn& should_autotune,
                                 MultiProcessKeyValueStore& sharding_kv_store) {
  return assigner_->AssignConfigs(module, should_autotune, sharding_kv_store);
}

AutotunerCacheInterface::CacheStats Autotuner::GetCacheStats() {
  return assigner_->GetCacheStats();
}

}  // namespace xla
