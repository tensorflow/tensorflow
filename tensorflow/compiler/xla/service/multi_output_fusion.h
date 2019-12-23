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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MULTI_OUTPUT_FUSION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MULTI_OUTPUT_FUSION_H_

#include <queue>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// This class implements the fusing of sibling fusion instructions that sharing
// common operands.
// It constructs the following associated data structures.
//  (1) candidates_: stores the instruction and the set of instructions it can
//      fuse to.
//  (2) candidates_index_: maps instruction to id.
//  (3) reachability_: reachability map in this computation.
//  (4) all_fusion_candidates_: the vector of candidate instructions.
//  (5) worklist_: a priority queue that contains pairs of instructions to be
//      fused and their fusion profit scores.
//
//  Function Perform() applies the optimization. It picks up the most profitable
//  pair in the worklist_, checks if it's legal to fuse and fuses the pair.
//  After fusion, it updates the associated structures such as reachability_,
//  candidates_ and worklist_.
//  Note that the reachability map is updated based on the original computation.
//  This works because the reachability is monotonically increasing with
//  instruction fusion.
class MultiOutputFusion : public HloModulePass {
 public:
  MultiOutputFusion() = default;

  absl::string_view name() const override { return "multi_output_fusion"; }

  // Run multi-output fusion on the given module. Returns whether the module
  // was changed.
  StatusOr<bool> Run(HloModule* module) override;

 protected:
  // Main entry for the optimization. Returns true if the optimization happens.
  bool Perform();

  // Test if instr1 and instr2 have the compatible shapes that can be legally
  // fused.
  virtual bool ShapesCompatibleForFusion(HloInstruction* instr1,
                                         HloInstruction* instr2) = 0;

  // Whether the instruction is a candidate for fusion.
  virtual bool IsFusible(HloInstruction* instr) = 0;

  // This function estimates the savings by merging instr1 and instr2 into one
  // multi-output fusion instruction. It returns a result in kib. (The result
  // is intentionally not granules, because this method is not TPU-specific.)
  virtual int64 GetProfit(HloInstruction* instr1, HloInstruction* instr2) = 0;

  // Whether fusing the instruction can reduce memory reads.
  virtual bool IsProfitableOperand(HloInstruction* instr);

  // Test if it's legal to fuse instr1 and instr2 into one fusion instruction.
  virtual bool LegalToFuse(HloInstruction* instr1, HloInstruction* instr2);

  // Fuse HloInstruction instr1 and instr2 and return the fused instruction.
  // The other instruction is removed from its parent computation.
  virtual HloInstruction* Fuse(HloInstruction* instr1, HloInstruction* instr2);

  // Recompute reachability for the current computation.
  void RecomputeReachability();

  // Returns the reachability map for the current computation.
  HloReachabilityMap* reachability() const { return reachability_.get(); }

  // Returns the computation for the pass.
  HloComputation* computation() const { return computation_; }

  // Update the reachability map after fusing instr1 and instr2.
  void UpdateReachability(
      HloInstruction* instr1, HloInstruction* instr2,
      absl::Span<HloInstruction* const> instrs_to_update,
      const std::function<bool(HloInstruction*)>& skip = nullptr);

  // Hook for multi-output fusion along producer-consumer edges.
  // Returns whether any instructions were fused.
  //
  // TODO(b/80420762): Perform producer-consumer multi-output fusion in
  // InstructionFusion instead.
  virtual bool DoProducerConsumerMultiOutputFusion();

 private:
  // An internal data structure for each instruction in current computation.
  // When an instruction is removed, member 'hlo' is set to nullptr.
  struct FusionCandidate {
    HloInstruction* hlo;
    std::list<std::pair<HloInstruction*, int64>> fusibles;
    explicit FusionCandidate(HloInstruction* hlo) : hlo(hlo) {}
  };

  // The pair of candidates to be fused and the profit score.
  struct ToBeFused {
    HloInstruction* instr1;
    HloInstruction* instr2;
    int64 score;
    ToBeFused(HloInstruction* instr1, HloInstruction* instr2, int64 score)
        : instr1(instr1), instr2(instr2), score(score) {}
    bool operator<(const ToBeFused& rhs) const { return score < rhs.score; }
  };

  // Update the internal data structures after instr1 and instr2 are fused into
  // one fusion instruction.
  void Update(HloInstruction* instr1, HloInstruction* instr2);

  int64 get_candidate_id(HloInstruction* instr) {
    return FindOrDie(candidates_index_, instr);
  }

  bool is_fused(HloInstruction* instr) {
    return candidates_[get_candidate_id(instr)].hlo == nullptr;
  }

  void set_is_fused(HloInstruction* instr) {
    candidates_[get_candidate_id(instr)].hlo = nullptr;
  }

  bool is_connected(HloInstruction* instr1, HloInstruction* instr2) {
    return reachability_->IsConnected(instr1, instr2);
  }

  std::vector<FusionCandidate> candidates_;
  std::priority_queue<ToBeFused> worklist_;

  // A map that maps an instruction to the index_.
  absl::flat_hash_map<HloInstruction*, int> candidates_index_;

  // The reachability map of current computation.
  std::unique_ptr<HloReachabilityMap> reachability_;

  // This stores all the candidate instructions in current computation.
  std::vector<HloInstruction*> all_fusion_candidates_;

  // Computation for the pass.
  HloComputation* computation_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MULTI_OUTPUT_FUSION_H_
