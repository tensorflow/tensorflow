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

#include "tensorflow/compiler/xla/service/cpu/parallel_task_assignment.h"

#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/cpu/shape_partition.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"

namespace xla {
namespace cpu {

class SimpleCostModel : public ParallelCostModel {
 public:
  SimpleCostModel(const int64 max_parallelism,
                  const HloCostAnalysis::ShapeSizeFunction& shape_size)
      : max_parallelism_(max_parallelism), shape_size_(shape_size) {}
  ~SimpleCostModel() override {}

  int64 GetParallelTaskCount(HloInstruction* instruction) override {
    // Simple cost model based on hlo size and typical L2 cache size.
    const int64 instruction_cost = shape_size_(instruction->shape());
    const int64 min_cost_per_thread = 256LL << 10;  // 256KB L2 Cache size.
    // Return target parallel task count in [1, max_parallelism_].
    return std::min(max_parallelism_,
                    std::max(1LL, instruction_cost / min_cost_per_thread));
  }

 private:
  const int64 max_parallelism_;
  const HloCostAnalysis::ShapeSizeFunction shape_size_;
};

class DefaultCostModel : public ParallelCostModel {
 public:
  DefaultCostModel(const int64 max_parallelism,
                   std::unique_ptr<HloCostAnalysis> cost_analysis)
      : max_parallelism_(max_parallelism),
        cost_analysis_(std::move(cost_analysis)) {}
  ~DefaultCostModel() override {}

  int64 GetParallelTaskCount(HloInstruction* instruction) override {
    // Calculate the instruction cost in cycles.
    // TODO(29630486) Improve on this linear cost model.
    // Consider making 'min_cost_per_thread' be a function of the target
    // bandwidth limit for instructions with low arithmetic complexity.
    const int64 instruction_cost =
        1 * cost_analysis_->flop_count(*instruction) +
        2 * cost_analysis_->transcendental_count(*instruction) +
        10 * cost_analysis_->bytes_accessed(*instruction);
    // Minimum per-thread cost is 100us of work on a 2GHz core.
    const int64 min_cost_per_thread = 100000;
    // Return target parallel task count in [1, max_parallelism_].
    return std::min(max_parallelism_,
                    std::max(1LL, instruction_cost / min_cost_per_thread));
  }

 private:
  const int64 max_parallelism_;
  const std::unique_ptr<HloCostAnalysis> cost_analysis_;
};


ParallelTaskAssignment::ParallelTaskAssignment(
    const int64 max_parallelism,
    const HloCostAnalysis::ShapeSizeFunction& shape_size,
    HloModule* module) {
  VLOG(1) << "ParallelTaskAssignment max_parallelism: " << max_parallelism;
  // Run cost analysis on 'module'.
  auto cost_analysis = MakeUnique<HloCostAnalysis>(shape_size);
  HloComputation* computation = module->entry_computation();
  Status status = computation->root_instruction()->Accept(cost_analysis.get());
  if (status.ok()) {
    // Set default cost model based on 'cost_analysis'.
    cost_model_.reset(new DefaultCostModel(max_parallelism,
                                           std::move(cost_analysis)));
  } else {
    // Fall back to a simple cost model based on hlo size and L2 cache size.
    // Note that HloCostAnalysis can returns an error status (likely because
    // HLOs like CustomCall are not yet implemented in the HloCostAnalysis).
    cost_model_.reset(new SimpleCostModel(max_parallelism, shape_size));
  }
}

int64 ParallelTaskAssignment::GetTargetParallelTaskCount(
    HloInstruction* instruction) {
  // Currently, we do not assign parallel tasks to instructions with at least
  // one of the following properties:
  // *) Internal threading (library calls to kConv, kDot, and kCustomCall).
  // *) Emit custom loops (kSelectAndScatter, FusionKind::kTransposeDot).
  // *) Tuple-shaped.
  // TODO(b/27458679) Parallelize instructions which are skipped here.
  if (instruction->opcode() == HloOpcode::kParameter ||
      instruction->opcode() == HloOpcode::kConstant ||
      instruction->opcode() == HloOpcode::kCall ||
      instruction->opcode() == HloOpcode::kCustomCall ||
      instruction->opcode() == HloOpcode::kSelectAndScatter ||
      (instruction->opcode() == HloOpcode::kConvolution &&
       PotentiallyImplementedAsEigenConvolution(*instruction)) ||
      PotentiallyImplementedAsEigenDot(*instruction) ||
      (instruction->opcode() == HloOpcode::kFusion &&
       instruction->fusion_kind() != HloInstruction::FusionKind::kLoop) ||
      ShapeUtil::IsTuple(instruction->shape())) {
    return 1;
  }
  // Consult 'cost_model_' to compute target parallel task count.
  return cost_model_->GetParallelTaskCount(instruction);
}

}  // namespace cpu
}  // namespace xla
