/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_COLLECTIVES_CONVERT_ASYNC_COLLECTIVES_TO_SYNC_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_COLLECTIVES_CONVERT_ASYNC_COLLECTIVES_TO_SYNC_H_

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/transforms/collectives/convert_async_collectives_to_sync.h"

namespace xla::gpu {

// Restores canonical synchronous HLO for GPU collectives after scheduling.
//
// Before scheduling, AsyncCollectiveAnnotator keeps regular collectives in
// async start/done form for the scheduler and marks collectives for which async
// execution is disabled with CollectiveBackendConfig::is_sync. This pass first
// restores all such marked pairs to their direct collective form, independent
// of whether useful work was scheduled between their start and done. It then
// also restores unmarked pairs that have no overlapping non-NOP work.
//
// For example, a pair marked by AsyncCollectiveAnnotator:
//
//   start = f32[8] all-reduce-start(p0), to_apply=add,
//       backend_config={"collective_backend_config":{"is_sync":true}}
//   done = f32[8] all-reduce-done(start)
//
// is restored even when other HLO is scheduled between start and done:
//
//   all-reduce = f32[8] all-reduce(p0), to_apply=add,
//       backend_config={"collective_backend_config":{"is_sync":true}}
//
// The inherited overlap analysis performs the same rewrite for an unmarked
// pair when only NOPs are scheduled between start and done:
//
//   start = f32[8] all-reduce-start(p0), to_apply=add
//   bitcast = f32[8] bitcast(p0)
//   done = f32[8] all-reduce-done(start)
//
// becomes:
//
//   bitcast = f32[8] bitcast(p0)
//   all-reduce = f32[8] all-reduce(p0), to_apply=add
//
// If an unmarked pair has non-NOP work between start and done, it remains in
// async form.
//
// Async execution scopes that wrap calls, including multi-operation collective
// groups, are not async collective pairs and are intentionally left unchanged.
class GpuConvertAsyncCollectivesToSync : public ConvertAsyncCollectivesToSync {
 public:
  GpuConvertAsyncCollectivesToSync();

  absl::string_view name() const override {
    return "gpu-convert-async-collectives-to-sync";
  }

  absl::Status ConvertAsyncInstructionsToSync(
      HloComputation* computation,
      absl::Span<const std::pair<HloInstruction*, HloInstruction*>> async_pairs)
      const override;

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_COLLECTIVES_CONVERT_ASYNC_COLLECTIVES_TO_SYNC_H_
