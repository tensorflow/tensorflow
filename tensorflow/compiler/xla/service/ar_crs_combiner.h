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
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// When the HLO graph contains a cross-module AllReduce (N separate AllReduce
// ops that share the same channel_id for MPMD partitioning, or 1 AllReduce op
// for SPMD partitioning), followed by some simple linear operations, followed
// by a cross-replica AllReduce (also known as cross-replica sum, or CRS), we
// can combine the CMAR and the CRAR, to use an efficient AllReduce
// implementation that fully utilizes the interconnect bandwidth.
//
// Such sequences appear in spatially partitioned models (either MPMD or SPMD).
// This pass must run right after spatial partitioning, when the code is still
// in a single HLO module.
//
// The steps are:
// 1) Find CMARs followed by simple ops followed by CRARs.
// 2) Group CMARs by channel_id. They must all be rewritten. For SPMD
//    partitioning, there will only be a single CMAR for each channel_id.
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
  ArCrsCombiner(int num_spatial_partitions, bool spmd_partition)
      : num_spatial_partitions_(num_spatial_partitions),
        spmd_partition_(spmd_partition) {}
  absl::string_view name() const override { return "ar-crs-combiner"; }
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

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
    int64_t distance;

    ArCrsPair(HloInstruction* all_reduce, HloInstruction* cross_replica_sum,
              int64_t dist)
        : ar(all_reduce), crs(cross_replica_sum), distance(dist) {}

    std::string ToString() {
      std::string result;
      absl::StrAppend(&result, "(");
      HloInstruction* instruction = ar;
      while (instruction != crs) {
        absl::StrAppend(&result, instruction->name(), ",");
        instruction = instruction->users()[0];
      }
      absl::StrAppend(&result, instruction->name(),
                      ")[id:", *(ar->channel_id()), ",dist:", distance, "]");
      return result;
    }
  };

  std::optional<ArCrsCombiner::ArCrsPair> MatchesArCrsPattern(
      HloInstruction* instruction);

  // If the passed instruction is a while parameter, and the while body is only
  // called by a single while instruction, return the while instruction.
  std::optional<HloInstruction*> WhileFromBodyParameter(
      HloInstruction* instruction);

  // If the passed instruction is a parameter in one of the branch computations,
  // and the branch body is only called by a single instruction, return the
  // conditional instruction.
  std::optional<HloInstruction*> ConditionalFromBodyParameter(
      HloInstruction* instruction);

  // Returns a vector of tuple instructions.
  // If all instructions that flow to "instruction" are tuples, return them.
  // Otherwise, return std::nullopt. Returns an empty vector if the instruction
  // is already in the visited set.
  std::optional<std::vector<HloInstruction*>> GetAllTuples(
      HloInstruction* instruction,
      absl::flat_hash_set<HloInstruction*>* visited);

  // Checks whether two different elements in the same tuple compute the same
  // value.
  bool TupleElementsComputeSameValue(
      HloInstruction* tuple_shaped_instruction, int64_t i1, int64_t i2,
      absl::flat_hash_map<int64_t, int64_t>* visited_pairs);

  // Returns whether the instructions i1 and i2 can be shown to evaluate to the
  // same value. Handling WHILE requires recursion, which may cause us to visit
  // the same instruction again. To avoid infinite loops, we pass a cache of
  // visited instruction pairs.
  bool InstructionsComputeSameValue(
      HloInstruction* i1, HloInstruction* i2,
      absl::flat_hash_map<int64_t, int64_t>* visited_pairs);

  // Populates all_reduce_map_.
  void GroupAllReducesById(HloModule* module);

  // Looks at each AllReduce group in all_reduce_map_, and keeps only the
  // groups for which it's safe to move the AllReduce later in the HLO graph.
  Status KeepProvablyEqualInstructionGroupsMPMD();

  // Same as above, but runs on SPMD partitioned module instead of MPMD.
  Status KeepProvablyEqualInstructionGroupsSPMD(HloModule* module);

  // Performs the graph rewrite that eliminates the early AllReduce and turns
  // the later CRS into an AllReduce.
  StatusOr<bool> RewriteGraph();

  int num_spatial_partitions_;

  // Run this combiner pass assuming the input module is an SPMD partitioned
  // module (as opposed to MPMD partitioned).
  //
  // The main difference between the two w.r.t. this pass is that there would be
  // N all-reduce ops for each channel in MPMD mode, whereas there is only 1
  // for each channel in SPMD mode. Also we use HloReplicationAnalysis for HLO
  // equivalence check in SPMD mode.
  bool spmd_partition_;

  // Map from all-reduce ids to the AR/CRS pairs.
  absl::flat_hash_map<int64_t, std::vector<ArCrsPair>> all_reduce_map_;

  // Map from a CRS instruction to the all-reduce ID of the AR paired with the
  // CRS. Sometimes, several ARs in the code could be paired with the same CRS.
  // We use this map to pick a single AR/CRS path to rewrite.
  absl::flat_hash_map<HloInstruction*, int64_t> crs_reserved_map_;

  std::unique_ptr<CallGraph> call_graph_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_AR_CRS_COMBINER_H_
