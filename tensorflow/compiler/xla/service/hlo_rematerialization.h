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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_REMATERIALIZATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_REMATERIALIZATION_H_

#include "tensorflow/compiler/xla/service/buffer_liveness.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"

namespace xla {

class HloRematerialization {
 public:
  using ShapeSizeFunction = std::function<int64(const Shape&)>;

  // Helper struct that communicates the before / after sizes for the
  // rematerialization process.
  struct RematerializationSizes {
    int64 before_bytes;
    int64 after_bytes;
  };

  // Rematerialize HLO instructions in the given module to reduce peak memory
  // use below memory_limit_bytes where memory use is defined as the total size
  // of all live HLO instruction values. Parameters and constants are included
  // in memory use estimates. Method parameters:
  //
  //   size_function: Function which returns the size in bytes of the top-level
  //     buffer of the given shape.
  //
  //   memory_limit_bytes: The threshold number of bytes to reduce memory use to
  //     via rematerialization.
  //
  //   hlo_module: HLO module to rematerialize instructions in.
  //
  //   sequence: Should point to an empty HloModuleSequence. Upon return
  //     contains the HLO instruction order which was used for
  //     rematerialization. This is the order in which HLO instructions should
  //     be emitted to minimize memory use.
  //
  //   sizes: Optional outparam that indicates the peak memory usage of the HLO
  //     module before/after rematerialization.
  //
  // Returns whether any instructions were rematerialized. If memory use is
  // already below the given limit then no instructions are rematerialized and
  // false is returned.
  //
  // CSE will undo the effects of this optimization and should not be run after
  // this pass. In general, this pass should be run very late immediately before
  // code generation.
  static StatusOr<bool> RematerializeAndSchedule(
      const ShapeSizeFunction& size_function, int64 memory_limit_bytes,
      HloModule* hlo_module, SequentialHloOrdering::HloModuleSequence* sequence,
      RematerializationSizes* sizes = nullptr);

 protected:
  HloRematerialization(const ShapeSizeFunction& size_function)
      : size_function_(size_function) {}
  ~HloRematerialization() {}

  // Runs rematerialization on the given module. Returns whether the module was
  // changed. memory_limit is the target maximum peak memory usage by the
  // module. sequence should be an empty HloModuleSequence. Upon return sequence
  // contains the memory-minimizing order in which to emit the HLO instructions.
  StatusOr<bool> Run(HloModule* module,
                     SequentialHloOrdering::HloModuleSequence* sequence,
                     int64 memory_limit, RematerializationSizes* sizes);

  // Rematerializes instructions within the given computation. 'order' is the
  // order in which the computation's instructions will be emitted in the
  // backend. Rematerialized instructions will be added to the HLO computation
  // and inserted into 'order'.
  StatusOr<bool> RematerializeComputation(
      HloComputation* computation,
      SequentialHloOrdering::HloModuleSequence* sequence,
      int64 computation_memory_limit);

  // Computes and returns the peak memory used by the given computation. The
  // peak memory is the maximum total size of all live HLO instruction values at
  // any program point. 'order' is the order in which the HLO instructions will
  // be emitted which is used to determine lifespans of HLO values.
  StatusOr<int64> ComputePeakMemory(
      const HloComputation* computation,
      const std::vector<const HloInstruction*>& order) const;

  // Returns the peak memory usage of the called computations for the given
  // instruction. Zero is returned if the instruction calls no computations.
  StatusOr<int64> CalledComputationsMemoryUsage(
      const HloInstruction* instruction) const;

  // Function which computes the size of the top-level buffer of a shape.
  const ShapeSizeFunction size_function_;

  // Call graph of the hlo_module.
  std::unique_ptr<CallGraph> call_graph_;

  // The peak memory usage of each computation. The map contains only those
  // computations called from sequential context
  // (CallContext::kSequential). These values are updated as rematerialization
  // occurs.
  tensorflow::gtl::FlatMap<const HloComputation*, int64>
      computation_peak_memory_;

  std::unique_ptr<TuplePointsToAnalysis> points_to_analysis_;

  // Set of computations which have had rematerialization
  // applied. Rematerialization is only applied once per computation.
  tensorflow::gtl::FlatSet<const HloComputation*> rematerialized_computations_;

  // Count of the total instructions rematerialized.
  int64 instructions_rematerialized_ = 0;

  // Count of the net instructions added to the HLO module by
  // rematerialization. This can be different than instructions_rematerialized_
  // because some rematerializations are effectively moves in the HLO
  // schedule. In these cases, the rematerialization instruction replaces all
  // uses of the original instruction and the original instruction is
  // dead. Hence, no net instructions were added.
  int64 net_instructions_added_ = 0;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_REMATERIALIZATION_H_
