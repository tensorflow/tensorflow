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

#ifndef XLA_SERVICE_CPU_PARALLEL_TASK_ASSIGNMENT_H_
#define XLA_SERVICE_CPU_PARALLEL_TASK_ASSIGNMENT_H_

#include <cstdint>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/cpu/target_machine_features.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/util.h"

namespace xla {
namespace cpu {

// Simple interface for different parallel cost model implementations.
class ParallelCostModel {
 public:
  virtual ~ParallelCostModel() = default;
  virtual int64_t GetParallelTaskCount(HloInstruction* instruction) = 0;
};

// ParallelTaskAssignment computes parallel task counts for HLOs in 'module'.
class ParallelTaskAssignment {
 public:
  // 'max_parallelism': the maximum parallel task count per instruction.
  // 'shape_size': shape size function used by HloCostAnalysis during parallel
  //               task assignment.
  // 'module': the containing HloModule.
  ParallelTaskAssignment(int64_t max_parallelism,
                         const HloCostAnalysis::ShapeSizeFunction& shape_size,
                         HloModule* module,
                         const TargetMachineFeatures* target_machine_features);
  ~ParallelTaskAssignment() {}

  // Computes and returns the target parallel task count for 'instruction'.
  int64_t GetTargetParallelTaskCount(HloInstruction* instruction);

 private:
  std::unique_ptr<ParallelCostModel> cost_model_;
  const TargetMachineFeatures& target_machine_features_;
};

// ParallelTaskAssigner computes target parallel task counts for all HLOs
// in the module, then assigns parallel task counts to HLOs in the entry
// computation, or to HLOs in embedded computations invoked by (potentially
// nested) kWhile or kCall instructions.
// Each HLO which is assigned parallel task counts is outlined into its
// own embedded computation, which is compiled as a parallel compute function,
// and which is invoked from a kCall instruction that is lowered in codegen to
// a runtime parallel fork/join call.
class ParallelTaskAssigner : public HloModulePass {
 public:
  // 'max_parallelism': the maximum parallel task count per instruction.
  // 'shape_size': shape size function used by HloCostAnalysis during parallel
  //               task assignment.
  ParallelTaskAssigner(const int64_t max_parallelism,
                       const HloCostAnalysis::ShapeSizeFunction& shape_size,
                       const TargetMachineFeatures* target_machine_features)
      : max_parallelism_(max_parallelism),
        shape_size_function_(shape_size),
        target_machine_features_(*target_machine_features) {}
  ~ParallelTaskAssigner() override {}

  absl::string_view name() const override {
    return "cpu-parallel-task-assigner";
  }

  // Run parallel task assigner on computations with specified
  // `execution_threads` in 'module'. By default, all `execution_threads` are
  // included. Returns true if the computation was changed, false otherwise.
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  using HloToParallelTasks =
      absl::flat_hash_map<const HloInstruction*, int64_t>;

  // Assigns target parallel tasks from 'hlo_to_parallel_tasks' to HLOs in
  // 'module'.
  // Returns true if the computation was changed, false otherwise.
  bool AssignParallelTasks(HloModule* module,
                           const HloToParallelTasks& hlo_to_parallel_tasks);
  bool AssignParallelTasksHelper(
      HloModule* module, HloComputation* computation,
      const HloToParallelTasks& hlo_to_parallel_tasks);

  // Computes target parallel task counts (returned in 'parallel_task_counts')
  // for parallelizable instructions in 'module'.
  void ComputeTargetParallelTasks(HloModule* module,
                                  HloToParallelTasks* hlo_to_parallel_tasks);

  int64_t max_parallelism_;
  HloCostAnalysis::ShapeSizeFunction shape_size_function_;
  const TargetMachineFeatures& target_machine_features_;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_PARALLEL_TASK_ASSIGNMENT_H_
