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

// When the HLO graph contains a cross-module AllReduce, followed by some simple
// linear operations, followed by a cross-replica AllReduce (also known as
// cross-replica sum, or CRS), we can combine the CMAR and the CRAR, to use an
// efficient AllReduce implementation that fully utilizes the interconnect
// bandwidth.
// Such sequences appear in spatially partitioned models.
// This pass must run right after spatial partitioning, when the code is still
// in a single HLO module.
//
// The steps are:
// 1) Find CMARs followed by simple ops followed by CRARs.
// 2) Group CMARs by all_reduce_id. They must all be rewritten.
// 3) Prove that the CMAR patterns in each core produce the same result.
// 4) Eliminate the CMAR, and if it feeds an addition/subtraction, divide the
//    other operand by the number of spatial partitions.
// 5) Turn the CRAR into an all-core AllReduce.
//
// The pass also handles the case where multiple CMARs lead to the same CRAR,
// and eliminates all CMARs. This graph:
//
//        Y
//        |
//  X   CMAR_2   Z
//  |      \    /
// CMAR_1     +
//    \     /
//       +
//       |
//     CRAR
//
// gets rewritten to:
//
//           Z   num_partitions
//            \  /
//       Y    div
//        \   /
//    X     +
//     \   /
//       +
//       |
//  all-core AR
//
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
  // We used this struct because multiple ARs could be paired with the same CRS.
  // In this case, we want to select the AR that is furthest from the CRS,
  // because it makes it easier to eliminate all ARs during RewriteGraph.
  struct ArCrsPair {
    HloInstruction* ar;
    HloInstruction* crs;
    // The length of the path from AR to CRS in the HLO graph.
    int64 distance;

    ArCrsPair(HloInstruction* all_reduce, HloInstruction* cross_replica_sum,
              int64 dist)
        : ar(all_reduce), crs(cross_replica_sum), distance(dist) {}

    string ToString() {
      std::vector<string> pieces;
      pieces.push_back("(");
      HloInstruction* instruction = ar;
      while (instruction != crs) {
        pieces.push_back(instruction->name());
        pieces.push_back(",");
        instruction = instruction->users()[0];
      }
      pieces.push_back(instruction->name());
      pieces.push_back(")[id:");
      pieces.push_back(std::to_string(*(ar->all_reduce_id())));
      pieces.push_back(",dist:");
      pieces.push_back(std::to_string(distance));
      pieces.push_back("]");
      return absl::StrJoin(pieces, "");
    }
  };

  absl::optional<ArCrsCombiner::ArCrsPair> MatchesArCrsPattern(
      HloInstruction* instruction);

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

  // Map from all-reduce ids to the AR/CRS pairs.
  absl::flat_hash_map<int64, std::vector<ArCrsPair>> all_reduce_map_;

  // Map from a CRS instruction to the all-reduce ID of the AR paired with the
  // CRS. Sometimes, several ARs in the code could be paired with the same CRS.
  // We use this map to pick a single AR/CRS path to rewrite.
  absl::flat_hash_map<HloInstruction*, int64> crs_reserved_map_;

  std::unique_ptr<CallGraph> call_graph_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_AR_CRS_COMBINER_H_
