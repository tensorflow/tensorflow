/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/cpu/parallel_task_assignment.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/ir_emission_utils.h"
#include "xla/service/cpu/shape_partition.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/llvm_ir/dynamic_update_slice_util.h"
#include "xla/util.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/status.h"

namespace xla {
namespace cpu {

class SimpleCostModel : public ParallelCostModel {
 public:
  SimpleCostModel(const int64_t max_parallelism,
                  const HloCostAnalysis::ShapeSizeFunction& shape_size)
      : max_parallelism_(max_parallelism), shape_size_(shape_size) {}
  ~SimpleCostModel() override {}

  int64_t GetParallelTaskCount(HloInstruction* instruction) override {
    // Simple cost model based on hlo size and typical L2 cache size.
    const int64_t instruction_cost = shape_size_(instruction->shape());
    const int64_t min_cost_per_thread = 256LL << 10;  // 256KB L2 Cache size.
    // Return target parallel task count in [1, max_parallelism_].
    return std::min(
        max_parallelism_,
        std::max(int64_t{1}, instruction_cost / min_cost_per_thread));
  }

 private:
  const int64_t max_parallelism_;
  const HloCostAnalysis::ShapeSizeFunction shape_size_;
};

class DefaultCostModel : public ParallelCostModel {
 public:
  DefaultCostModel(const int64_t max_parallelism,
                   const HloCostAnalysis::ShapeSizeFunction& shape_size,
                   std::unique_ptr<HloCostAnalysis> cost_analysis)
      : max_parallelism_(max_parallelism),
        shape_size_(shape_size),
        cost_analysis_(std::move(cost_analysis)) {}
  ~DefaultCostModel() override {}

  int64_t GetParallelTaskCount(HloInstruction* instruction) override {
    // Parameters for parallel task count computation.
    int64_t instruction_cost;
    int64_t min_cost_per_thread;
    int64_t max_parallelism;
    // Calculate flops-to-bytes-ratio for 'instruction'.
    const int64_t bytes_accessed =
        std::max(int64_t{1}, cost_analysis_->bytes_accessed(*instruction));
    const float flops_to_bytes_ratio =
        cost_analysis_->flop_count(*instruction) /
        static_cast<float>(bytes_accessed);
    // Check for I/O bound instructions.
    if (flops_to_bytes_ratio <= 1.0) {
      // Limit max parallelism for I/O bound instructions by assuming a
      // sub-linear scaling function (fit based on empirical benchmark results).
      // TODO(b/29630486) Develop system bandwidth model.
      max_parallelism = std::min<int64_t>(
          max_parallelism_, std::ceil(std::sqrt(tsl::port::MaxParallelism())));
      // Use bytes accessed cost and L2 cache size min per-thread cost.
      instruction_cost = bytes_accessed;
      min_cost_per_thread = 256LL << 10;  // 256KB L2 Cache size.
    } else {
      // Use max parallelism for compute bound instructions.
      max_parallelism = max_parallelism_;
      // Calculate the instruction cost in cycles.
      // TODO(b/29630486) Improve on this linear cost model.
      // Consider making 'min_cost_per_thread' be a function of the target
      // bandwidth limit for instructions with low arithmetic complexity.
      instruction_cost =
          1 * cost_analysis_->flop_count(*instruction) +
          2 * cost_analysis_->transcendental_count(*instruction) +
          10 * cost_analysis_->bytes_accessed(*instruction);
      // Minimum per-thread cost is 100us of work on a 2GHz core.
      min_cost_per_thread = 100000;
    }
    // Return target parallel task count in [1, max_parallelism_].
    return std::min(
        max_parallelism,
        std::max(int64_t{1}, instruction_cost / min_cost_per_thread));
  }

 private:
  const int64_t max_parallelism_;
  const HloCostAnalysis::ShapeSizeFunction shape_size_;
  const std::unique_ptr<HloCostAnalysis> cost_analysis_;
};

ParallelTaskAssignment::ParallelTaskAssignment(
    const int64_t max_parallelism,
    const HloCostAnalysis::ShapeSizeFunction& shape_size, HloModule* module,
    const TargetMachineFeatures* target_machine_features)
    : target_machine_features_(*target_machine_features) {
  VLOG(1) << "ParallelTaskAssignment max_parallelism: " << max_parallelism;
  // Run cost analysis on 'module'.
  auto cost_analysis = std::make_unique<HloCostAnalysis>(shape_size);
  HloComputation* computation = module->entry_computation();
  absl::Status status =
      computation->root_instruction()->Accept(cost_analysis.get());
  if (status.ok()) {
    // Set default cost model based on 'cost_analysis'.
    cost_model_ = std::make_unique<DefaultCostModel>(
        max_parallelism, shape_size, std::move(cost_analysis));
  } else {
    // Fall back to a simple cost model based on hlo size and L2 cache size.
    // Note that HloCostAnalysis can returns an error status (likely because
    // HLOs like CustomCall are not yet implemented in the HloCostAnalysis).
    cost_model_ =
        std::make_unique<SimpleCostModel>(max_parallelism, shape_size);
  }
}

int64_t ParallelTaskAssignment::GetTargetParallelTaskCount(
    HloInstruction* instruction) {
  // Currently, we do not assign parallel tasks to instructions with at least
  // one of the following properties:
  // *) Internal threading (library calls to kConv, kDot, kFft, kCustomCall).
  // *) Emit custom loops (kSelectAndScatter).
  // *) Operations that are not thread safe (like infeed and rng).
  // *) Tuple-shaped.
  // *) Operations that might be implemented as an in-place
  //    dynamic-update-slice, because we can't know how many output elements
  //    they will write (out-of-place will touch the whole output buffer, while
  //    in-place will only touch the updated elements).
  // TODO(b/27458679) Parallelize instructions which are skipped here.
  auto opcode = instruction->opcode();
  if (llvm_ir::MayBeImplementedAsInPlaceDynamicUpdateSlice(instruction) ||
      instruction->shape().IsTuple() || opcode == HloOpcode::kRng ||
      opcode == HloOpcode::kConstant) {
    return 1;
  }

  // Only allow instructions that can be trivially parallelized (where all
  // outputs can be computed independently of each other).
  if (instruction->IsElementwise() || instruction->IsLoopFusion() ||
      opcode == HloOpcode::kBroadcast || opcode == HloOpcode::kConcatenate ||
      opcode == HloOpcode::kDynamicSlice ||
      opcode == HloOpcode::kDynamicUpdateSlice ||
      opcode == HloOpcode::kGather || opcode == HloOpcode::kIota ||
      opcode == HloOpcode::kPad || opcode == HloOpcode::kReduce ||
      opcode == HloOpcode::kReduceWindow || opcode == HloOpcode::kReshape ||
      opcode == HloOpcode::kReverse || opcode == HloOpcode::kSlice ||
      opcode == HloOpcode::kTranspose ||
      (opcode == HloOpcode::kConvolution &&
       !PotentiallyImplementedAsEigenConvolution(*instruction,
                                                 target_machine_features_))) {
    // Consult 'cost_model_' to compute target parallel task count.
    return cost_model_->GetParallelTaskCount(instruction);
  }

  return 1;
}

absl::StatusOr<bool> ParallelTaskAssigner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(2, "ParallelTaskAssigner ENTRY");
  XLA_VLOG_LINES(3, module->ToString());
  // Compute target parallel task counts for all instructions in 'module'.
  HloToParallelTasks hlo_to_parallel_tasks;
  ComputeTargetParallelTasks(module, &hlo_to_parallel_tasks);

  // Assign parallel tasks to target specific instructions in 'module'.
  // TODO(b/27458679) Support inter-op parallelism.
  bool changed = AssignParallelTasks(module, hlo_to_parallel_tasks);

  XLA_VLOG_LINES(2, "ParallelTaskAssigner EXIT");
  XLA_VLOG_LINES(3, module->ToString());
  return changed;
}

bool ParallelTaskAssigner::AssignParallelTasks(
    HloModule* module, const HloToParallelTasks& hlo_to_parallel_tasks) {
  return AssignParallelTasksHelper(module, module->entry_computation(),
                                   hlo_to_parallel_tasks);
}

bool ParallelTaskAssigner::AssignParallelTasksHelper(
    HloModule* module, HloComputation* computation,
    const HloToParallelTasks& hlo_to_parallel_tasks) {
  bool changed = false;
  // Snapshot set of instructions because outlining modifies the set below.
  std::vector<HloInstruction*> instructions(computation->instructions().begin(),
                                            computation->instructions().end());
  for (auto* instruction : instructions) {
    // Assign parallel tasks to sub-computations for While and Call HLOs.
    // TODO(b/27458679) Evaluate alternative intra-op parallelism placement,
    // and support other callable computations like reduce.
    if (instruction->opcode() == HloOpcode::kWhile) {
      changed |= AssignParallelTasksHelper(module, instruction->while_body(),
                                           hlo_to_parallel_tasks);
      continue;
    } else if (instruction->opcode() == HloOpcode::kCall) {
      changed |= AssignParallelTasksHelper(module, instruction->to_apply(),
                                           hlo_to_parallel_tasks);
      continue;
    }
    // Skip if no parallel tasks were computed in first pass.
    auto it = hlo_to_parallel_tasks.find(instruction);
    if (it == hlo_to_parallel_tasks.end()) {
      continue;
    }
    // Get target parallel task count computed for 'instruction'.
    const int64_t target_parallel_task_count = (*it).second;
    // Assign feasible dimension partitions (based on actual dimension sizes).
    auto dim_partition_counts = ShapePartitionAssigner(instruction->shape())
                                    .Run(target_parallel_task_count);
    const int64_t total_partition_count =
        ShapePartitionAssigner::GetTotalPartitionCount(dim_partition_counts);
    if (total_partition_count <= 1) {
      // Feasible partition calculation resulting in no partitioning, so skip.
      continue;
    }

    // Outline 'instruction' in 'computation' for parallel task assignment.
    auto* call = module->OutlineExpressionFromComputation(
        {instruction}, absl::StrCat("parallel_", instruction->name()),
        computation);

    // Set assigned dimension partitioning to 'instruction'.
    auto* new_root = call->to_apply()->root_instruction();
    BackendConfig backend_config;
    absl::c_copy(dim_partition_counts,
                 tsl::protobuf::RepeatedFieldBackInserter(
                     backend_config.mutable_outer_dimension_partitions()));
    TF_CHECK_OK(new_root->set_backend_config(backend_config));

    VLOG(2) << "Assigned parallel task count: " << total_partition_count
            << " to instruction: " << new_root->name()
            << " parent: " << new_root->parent()->name();
    changed = true;
  }
  return changed;
}

void ParallelTaskAssigner::ComputeTargetParallelTasks(
    HloModule* module, HloToParallelTasks* hlo_to_parallel_tasks) {
  ParallelTaskAssignment parallel_task_assignment(max_parallelism_,
                                                  shape_size_function_, module,
                                                  &target_machine_features_);

  // Compute parallel task counts for all instructions in 'module'.
  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto* instruction : computation->instructions()) {
      // Query ParallelTaskAssignment for target parallel task count.
      const int64_t target_parallel_task_count =
          parallel_task_assignment.GetTargetParallelTaskCount(instruction);
      if (target_parallel_task_count > 1) {
        hlo_to_parallel_tasks->insert(
            {instruction, target_parallel_task_count});
      }
    }
  }
}

}  // namespace cpu
}  // namespace xla
