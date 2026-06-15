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

#ifndef XLA_BACKENDS_AUTOTUNER_AUTOTUNER_H_
#define XLA_BACKENDS_AUTOTUNER_AUTOTUNER_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/config_assigner.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {

// Autotuner class, now a thin wrapper over ConfigAssigner.
// To be deleted once the refactoring is complete.
class Autotuner {
 public:
  static absl::StatusOr<std::unique_ptr<Autotuner>> Create(
      std::vector<std::unique_ptr<CodegenBackend>> codegen_backends,
      std::unique_ptr<Profiler> profiler, AutotuneConfig autotune_config,
      std::unique_ptr<AutotunerCacheInterface> cache,
      tsl::thread::ThreadPool* thread_pool = nullptr);

  absl::Status Autotune(HloInstruction* instr);

  absl::Status Autotune(HloModule* module,
                        const InstructionFilterFn& should_autotune);

  absl::Status Autotune(HloModule* module,
                        const InstructionFilterFn& should_autotune,
                        MultiProcessKeyValueStore& sharding_kv_store);

  AutotunerCacheInterface::CacheStats GetCacheStats();

 private:
  explicit Autotuner(std::unique_ptr<ConfigAssigner> assigner)
      : assigner_(std::move(assigner)) {}

  std::unique_ptr<ConfigAssigner> assigner_;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_AUTOTUNER_H_
