/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SCHEDULER_MEMORY_FENCING_H_
#define XLA_SERVICE_SCHEDULER_MEMORY_FENCING_H_

#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/hlo_cost_analysis.h"

namespace xla {

// Fences the live ranges of large buffers before a schedule-rewriting pass
// (typically LatencyHidingScheduler) runs, using the module's current
// (memory-minimizing) schedule as the reference.
//
// For every buffer of at least `size_threshold_bytes`, the pass finds the
// buffer's users and the async operation window W in which (or before which)
// the last use is scheduled in the reference schedule, and adds a control
// dependency from every user to the async start that opens window W +
// `slack_windows`. Any schedule that respects the dependency graph can then
// defer the buffer's release by at most `slack_windows` async windows relative
// to the reference schedule, which bounds how many such buffers a scheduler
// can keep simultaneously live by deferring their last users ("release before
// advance"). All users are fenced because the buffer stays live until every
// user has executed.
//
// Every added edge points forward in the reference schedule, which is a valid
// topological order, so the added edges can never create a cycle, and the
// existing schedule remains valid.
//
// The module must be scheduled. Buffers that escape their computation (module
// live-outs, computation root outputs) and parameter-defined buffers are not
// fenced.
class SchedulerMemoryFencing : public HloModulePass {
 public:
  SchedulerMemoryFencing(HloCostAnalysis::ShapeSizeFunction shape_size_bytes,
                         int64_t size_threshold_bytes, int32_t slack_windows,
                         const AliasInfo* alias_info)
      : shape_size_bytes_(std::move(shape_size_bytes)),
        size_threshold_bytes_(size_threshold_bytes),
        slack_windows_(slack_windows),
        alias_info_(alias_info) {}

  absl::string_view name() const override { return "scheduler-memory-fencing"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  HloCostAnalysis::ShapeSizeFunction shape_size_bytes_;
  int64_t size_threshold_bytes_;
  int32_t slack_windows_;
  const AliasInfo* alias_info_;
};

}  // namespace xla

#endif  // XLA_SERVICE_SCHEDULER_MEMORY_FENCING_H_
