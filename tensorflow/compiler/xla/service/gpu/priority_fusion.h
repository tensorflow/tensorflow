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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_PRIORITY_FUSION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_PRIORITY_FUSION_H_

#include <stdint.h>

#include <memory>
#include <optional>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/fusion_queue.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"
#include "tensorflow/compiler/xla/statusor.h"

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

  using HloPassInterface::Run;
  StatusOr<bool> Run(HloModule* module,
                     const absl::flat_hash_set<absl::string_view>&
                         execution_threads) override {
    cost_analysis_.emplace(cost_analysis_options_, &device_info_);
    return InstructionFusion::Run(module, execution_threads);
  }

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

  // Cost model that defines priorities in the queue.
  GpuHloCostAnalysis::Options cost_analysis_options_;
  std::optional<GpuHloCostAnalysis> cost_analysis_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_PRIORITY_FUSION_H_
