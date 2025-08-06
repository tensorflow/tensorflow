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
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/autotuner.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/gpu/autotuner/gpu_profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace gpu {

absl::StatusOr<std::unique_ptr<AutotunerPass>> AutotunerPass::Create(
    std::vector<std::unique_ptr<CodegenBackend>> backends,
    stream_executor::StreamExecutor* stream_executor,
    tsl::thread::ThreadPool* thread_pool) {
  std::unique_ptr<GpuProfiler> profiler =
      GpuProfiler::Create(stream_executor, ProfileOptions());
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Autotuner> autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler),
                        AutotuneConfig(), thread_pool));
  return absl::WrapUnique(new AutotunerPass(std::move(autotuner)));
}

absl::StatusOr<bool> AutotunerPass::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Running Autotuner Pass";

  auto should_autotune = [](const HloInstruction& instruction) -> bool {
    if ((instruction.opcode() == HloOpcode::kFusion &&
         instruction.fusion_kind() == HloInstruction::FusionKind::kCustom) ||
        instruction.opcode() == HloOpcode::kCustomCall) {
      return true;
    }
    return false;
  };

  TF_RETURN_IF_ERROR(autotuner_->Autotune(module, should_autotune));
  return true;
}

}  // namespace gpu
}  // namespace xla
