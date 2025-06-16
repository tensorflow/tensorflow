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
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/stream_executor/stream_executor.h"

using InstructionFilterFn =
    absl::AnyInvocable<bool(const xla::HloInstruction&) const>;

namespace xla {

struct AutotuneConfig {
  // Whether to skip configs that failed to compile.
  bool skip_failing_configs = true;
  // TODO b/407495547 - Add and implement following options.
  // bool should_check_correctness;
  // bool should_skip_wrong_results;
  // bool should_crash_on_check_failure;
};

class Autotuner {
 public:
  static std::unique_ptr<Autotuner> Create(
      std::vector<std::unique_ptr<CodegenBackend>> codegen_backends,
      stream_executor::StreamExecutor* stream_executor,
      std::unique_ptr<Profiler> profiler, AutotuneConfig autotune_config);

  // Try all supported configs from the registered codegen backends for the
  // given HLO instruction and apply the best one.
  absl::Status Autotune(HloInstruction* instr);

  // Autotune all instructions in the module that matches the filter function.
  // If ignore_fusion_computations is true, the filter will ignore the
  // instructions that are inside fusion computations.
  absl::Status Autotune(HloModule* module, InstructionFilterFn should_autotune,
                        bool ignore_fusion_computations = false);

 private:
  Autotuner(std::vector<std::unique_ptr<CodegenBackend>> codegen_backends,
            stream_executor::StreamExecutor* stream_executor,
            std::unique_ptr<Profiler> profiler, AutotuneConfig autotune_config)
      : codegen_backends_(std::move(codegen_backends)),
        stream_executor_(stream_executor),
        profiler_(std::move(profiler)),
        autotune_config_(autotune_config) {}

  absl::StatusOr<std::pair<CodegenBackend*, std::unique_ptr<BackendConfig>>>
  GetBestConfig(HloInstruction* instr);

  absl::flat_hash_map<std::string, std::vector<HloInstruction*>>
  GetAutotuningCandidates(const HloModule* module,
                          InstructionFilterFn should_autotune,
                          bool ignore_fusion_computations);

  std::vector<std::unique_ptr<CodegenBackend>> codegen_backends_;
  se::StreamExecutor* stream_executor_;
  std::unique_ptr<Profiler> profiler_;
  AutotuneConfig autotune_config_;
};
}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_AUTOTUNER_H_
