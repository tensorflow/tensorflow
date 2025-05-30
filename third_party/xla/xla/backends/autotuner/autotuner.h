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
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/executor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {

struct AutotuneConfig {
  // TODO b/407495547 - Add and implement following options.
  bool skip_failing_configs = true;
  // bool should_check_correctness;
  // bool should_skip_wrong_results;
  // bool should_crash_on_check_failure;
};

class Autotuner {
 public:
  Autotuner(stream_executor::StreamExecutor* stream_executor,
            std::unique_ptr<Executor> executor, AutotuneConfig autotune_config)
      : stream_executor_(stream_executor),
        executor_(std::move(executor)),
        autotune_config_(autotune_config) {}

  void RegisterCodegenBackend(std::unique_ptr<CodegenBackend> codegen_backend) {
    codegen_backends_.push_back(std::move(codegen_backend));
  }

  // Try all supported configs from the registered codegen backends for the
  // given HLO instruction and apply the best one.
  absl::Status Autotune(HloInstruction* instr);

 private:
  std::vector<std::unique_ptr<CodegenBackend>> codegen_backends_;
  se::StreamExecutor* stream_executor_;
  std::unique_ptr<Executor> executor_;
  AutotuneConfig autotune_config_;
};
}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_AUTOTUNER_H_
