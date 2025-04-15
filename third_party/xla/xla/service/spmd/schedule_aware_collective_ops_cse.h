/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SPMD_SCHEDULE_AWARE_COLLECTIVE_OPS_CSE_H_
#define XLA_SERVICE_SPMD_SCHEDULE_AWARE_COLLECTIVE_OPS_CSE_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Performs CSE for collectives if their users are within reasonable live range.
class ScheduleAwareCollectiveOpsCSE : public HloModulePass {
 public:
  // distance_threshold: maximum live range (in number of HLO instructions on
  //   the path) to consider CSE.
  // for_replicas: specifies if this pass is for cross-replica or
  //   cross-partition collectives.
  explicit ScheduleAwareCollectiveOpsCSE(int64_t distance_threshold,
                                         bool for_replicas)
      : distance_threshold_(distance_threshold), for_replicas_(for_replicas) {}

  ~ScheduleAwareCollectiveOpsCSE() override = default;
  absl::string_view name() const override {
    return "schedule-aware-collective-cse";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  int64_t distance_threshold_;
  bool for_replicas_;
};

}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SCHEDULE_AWARE_COLLECTIVE_OPS_CSE_H_
