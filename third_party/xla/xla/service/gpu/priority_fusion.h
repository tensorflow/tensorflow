/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_PRIORITY_FUSION_H_
#define XLA_SERVICE_GPU_PRIORITY_FUSION_H_

#include <stdint.h>

#include <memory>
#include <optional>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/fusion_queue.h"
#include "xla/service/gpu/gpu_hlo_cost_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/service/instruction_fusion.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {

class GpuPriorityFusion : public InstructionFusion {
 public:
  explicit GpuPriorityFusion(
      const GpuDeviceInfo& d,
      const GpuHloCostAnalysis::Options& cost_analysis_options)
      : InstructionFusion(GpuPriorityFusion::IsExpensive),
        device_info_(d),
        cost_analysis_options_(cost_analysis_options) {}

  static bool IsExpensive(const HloInstruction& instruction);

 protected:
  std::unique_ptr<FusionQueue> GetFusionQueue(
      HloComputation* computation) override;
  FusionDecision ShouldFuse(HloInstruction* consumer,
                            int64_t operand_index) override;

  HloInstruction::FusionKind ChooseKind(
      const HloInstruction* producer, const HloInstruction* consumer) override;

 private:
  // This method is called by ShouldFuse() to do all the computationally
  // inexpensive checks whether we should fuse the operand into 'consumer'.
  FusionDecision ShouldFuseInexpensiveChecks(HloInstruction* consumer,
                                             int64_t operand_index);

  HloInstruction* FuseInstruction(HloInstruction* fusion_instruction,
                                  HloInstruction* producer) override;

  const GpuDeviceInfo device_info_;

  // Cost model options that defines priorities in the queue.
  GpuHloCostAnalysis::Options cost_analysis_options_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_PRIORITY_FUSION_H_
