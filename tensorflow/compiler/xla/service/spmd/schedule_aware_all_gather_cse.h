/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_SCHEDULE_AWARE_ALL_GATHER_CSE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_SCHEDULE_AWARE_ALL_GATHER_CSE_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// Performs CSE for all-gather if their users are within reasonable live range.
class ScheduleAwareAllGatherCSE : public HloModulePass {
 public:
  // distance_threshold: maximum live range (in number of HLO instructions on
  //   the path) to consider CSE.
  // for_replicas: specifies if this pass is for cross-replica or
  //   cross-partition all-gathers.
  explicit ScheduleAwareAllGatherCSE(int64 distance_threshold,
                                     bool for_replicas)
      : distance_threshold_(distance_threshold), for_replicas_(for_replicas) {}

  ~ScheduleAwareAllGatherCSE() override = default;
  absl::string_view name() const override {
    return "schedule-aware-all-gather-cse";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  int64 distance_threshold_;
  bool for_replicas_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_SCHEDULE_AWARE_ALL_GATHER_CSE_H_
