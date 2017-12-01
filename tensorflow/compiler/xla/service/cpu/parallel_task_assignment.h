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

#ifndef THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_CPU_PARALLEL_TASK_ASSIGNMENT_H_
#define THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_CPU_PARALLEL_TASK_ASSIGNMENT_H_

#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace cpu {

// Simple interface for different parallel cost model implementations.
class ParallelCostModel {
 public:
  virtual ~ParallelCostModel() = default;
  virtual int64 GetParallelTaskCount(HloInstruction* instruction) = 0;
};

// ParallelTaskAssignment computes parallel task counts for HLOs in 'module'.
class ParallelTaskAssignment {
 public:
  // 'max_parallelism': the maximum parallel task count per instruction.
  // 'shape_size': shape size function used by HloCostAnalysis during parallel
  //               task assignment.
  // 'module': the containing HloModule.
  ParallelTaskAssignment(
      const int64 max_parallelism,
      const HloCostAnalysis::ShapeSizeFunction& shape_size,
      HloModule* module);
  ~ParallelTaskAssignment() {}

  // Computes and returns the target parallel task count for 'instruction'.
  int64 GetTargetParallelTaskCount(HloInstruction* instruction);

 private:
  std::unique_ptr<ParallelCostModel> cost_model_;
};

// ParallelTaskAssigner computes target parallel task counts for all HLOs
// in the module, then assigns parallel task counts to HLOs in the entry
// computation, or to HLOs in embedded computations invoked by (potentially
// nested) kWhile or kCall instructions.
// Each HLO which is assigned parallel task counts is outlined into its
// own embedded computation, which is compiled as a parallel compute function,
// and which is invoked from a kCall instruction that is lowered in codegen to
// a runtime parallel fork/join call.
class ParallelTaskAssigner : public HloPassInterface {
 public:
  // 'max_parallelism': the maximum parallel task count per instruction.
  // 'shape_size': shape size function used by HloCostAnalysis during parallel
  //               task assignment.
  // 'module': the containing HloModule.
  ParallelTaskAssigner(const int64 max_parallelism,
                       const HloCostAnalysis::ShapeSizeFunction& shape_size,
                       HloModule* module)
      : parallel_task_assignment_(max_parallelism, shape_size, module) {}
  ~ParallelTaskAssigner() override {}

  tensorflow::StringPiece name() const override {
    return "cpu-parallel-task-assigner";
  }

  // Run parallel task assigner on 'module'.
  // Returns true if the computation was changed, false otherwise.
  StatusOr<bool> Run(HloModule* module) override;

 private:
  using HloToParallelTasks = std::unordered_map<const HloInstruction*, int64>;

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

  ParallelTaskAssignment parallel_task_assignment_;
};

}  // namespace cpu
}  // namespace xla

#endif  // THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_CPU_PARALLEL_TASK_ASSIGNMENT_H_
