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

#ifndef XLA_SERVICE_LIST_SCHEDULER_H_
#define XLA_SERVICE_LIST_SCHEDULER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// ListScheduler is an example scheduler that exists for pedagogical purposes.
//
// It schedules each computation separately. For each computation, it performs a
// topological traversal of the computation graph. At every step of the
// traversal, there is a frontier of instructions that can be scheduled next
// (i.e. the instructions with zero in-degree). The scheduler uses the following
// heuristics to pick the next instruction to schedule:
//
// 1. Always schedule an async start if possible.
// 2. Schedule an async done if there are no other instructions.
// 3. Schedule the operation that reduces memory pressure the most.
class ListScheduler : public HloModulePass {
 public:
  ListScheduler() = default;
  ~ListScheduler() override = default;
  absl::string_view name() const override { return "list-scheduler"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_LIST_SCHEDULER_H_
