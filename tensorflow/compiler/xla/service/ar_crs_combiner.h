/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_AR_CRS_COMBINER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_AR_CRS_COMBINER_H_

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// When the HLO graph contains an AllReduce, followed by some simple linear
// operations, followed by a CrossReplicaSum, we can combine the AR and the CRS,
// to use an efficient CrossReplicaSum implementation that fully utilizes the
// interconnect bandwidth.
// Such sequences appear in spatially partitioned models.
// This pass must run right after spatial partitioning.
class ArCrsCombiner : public HloModulePass {
 public:
  ArCrsCombiner(int num_spatial_partitions)
      : num_spatial_partitions_(num_spatial_partitions) {}
  absl::string_view name() const override { return "ar-crs-combiner"; }
  StatusOr<bool> Run(HloModule* module) override;

  // Helper method to allow testing of InstructionsComputeSameValue.
  static bool TestInstructionsComputeSameValue(HloInstruction* i1,
                                               HloInstruction* i2);

 private:
  // If the passed instruction is a while parameter, and the while body is only
  // called by a single while instruction, return the while instruction.
  absl::optional<HloInstruction*> WhileFromBodyParameter(
      HloInstruction* instruction);

  // Returns a vector of tuple instructions.
  // If all instructions that flow to "instruction" are tuples, return them.
  // Otherwise, return an empty vector.
  std::vector<HloInstruction*> GetAllTuples(HloInstruction* instruction);

  // Checks whether two different elements in the same tuple compute the same
  // value.
  bool TupleElementsComputeSameValue(
      HloInstruction* tuple_shaped_instruction, int64 i1, int64 i2,
      absl::flat_hash_map<int64, int64>* visited_pairs);

  // Returns whether the instructions i1 and i2 can be shown to evaluate to the
  // same value. Handling WHILE requires recursion, which may cause us to visit
  // the same instruction again. To avoid infinite loops, we pass a cache of
  // visited instruction pairs.
  bool InstructionsComputeSameValue(
      HloInstruction* i1, HloInstruction* i2,
      absl::flat_hash_map<int64, int64>* visited_pairs);

  // Populates all_reduce_map_.
  void GroupAllReducesById(HloModule* module);

  // Looks at each AllReduce group in all_reduce_map_, and keeps only the
  // groups for which it's safe to move the AllReduce later in the HLO graph.
  void KeepProvablyEqualInstructionGroups();

  // Performs the graph rewrite that eliminates the early AllReduce and turns
  // the later CRS into an AllReduce.
  StatusOr<bool> RewriteGraph();

  int num_spatial_partitions_;

  // Map from all-reduce ids to the all reduce instructions.
  absl::flat_hash_map<int64, std::vector<HloInstruction*>> all_reduce_map_;

  std::unique_ptr<CallGraph> call_graph_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_AR_CRS_COMBINER_H_
