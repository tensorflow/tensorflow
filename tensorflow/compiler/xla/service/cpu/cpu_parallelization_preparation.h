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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_PARALLELIZATION_PREPARATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_PARALLELIZATION_PREPARATION_H_

#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace cpu {

// This pass prepares an HLO module for parallel execution by transforming
// subgraphs of the top-level computation into embedded computations which can
// be executed in parallel.
// TODO(b/29630486): Currently, it is limited to turning all instructions (which
// are not constants or parameters) in the entry computation into embedded
// computations.  However, it could make sense to coarsen the parallelization to
// improve cache locality.  Also, we will need to do something to intelligently
// handle While constructs.
class ParallelizationPreparation : public HloPassInterface {
 public:
  // 'max_parallelism': the maximum parallel task count per instruction.
  // 'shape_size': shape size function used by HloCostAnalysis during parallel
  //               task assignment.
  ParallelizationPreparation(
      const int64 max_parallelism,
      const HloCostAnalysis::ShapeSizeFunction& shape_size)
      : max_parallelism_(max_parallelism), shape_size_(shape_size) {}
  ~ParallelizationPreparation() override {}

  tensorflow::StringPiece name() const override {
    return "cpu-parallel-prepare";
  }

  // Run parallel preparation on the given computation. Returns whether the
  // computation was changed.
  StatusOr<bool> Run(HloModule* module) override;

 private:
  // Assigns parallel task partitions to conformant instructions in 'module'.
  // Returns true on success or error status otherwise.
  StatusOr<bool> RunParallelTaskAssignment(HloModule* module);

  // Returns the target parallel task count for 'instruction'.
  // Utilizes 'cost_analysis' if non-null.
  // Otherwise defaults to a simple HLO output size-based cost model.
  int64 GetTargetParallelTaskCount(const HloCostAnalysis* cost_analysis,
                                   HloInstruction* instruction);

  // Outlines 'instruction' from entry computation, if it had
  // been assigned parallel tasks in an earlier pass through the computation.
  // Returns true if 'instruction' was successfully outlined, false otherwise.
  bool OutlineParallelizableInstruction(HloInstruction* instruction);

  // Returns true if 'instruction' can be outlined into the same sub-computation
  // with its single user (parallelizable instructions are not outlined with
  // each other). Returns false otherwise.
  bool CanOutlineWithUser(HloInstruction* instruction);

  // Returns true if 'instruction' (or the root of the sub-computation that
  // 'instruction' calls) has had parallel tasks assigned in earlier pass.
  // Returns false otherwise.
  bool AssignedParallelTasks(HloInstruction* instruction);

  const int64 max_parallelism_;
  const HloCostAnalysis::ShapeSizeFunction shape_size_;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_PARALLELIZATION_PREPARATION_H_
