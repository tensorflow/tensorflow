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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_INSTRUCTION_FUSION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_INSTRUCTION_FUSION_H_

#include <functional>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/fusion_queue.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// HLO pass which performs instruction fusion. Instructions are fused
// "vertically", meaning producing instructions are fused into their consumers
// with the intent that the loops which compute their values will be fused in
// code generation. Derived classes define ShouldFuse method to select which
// instructions to fuse.
class InstructionFusion : public HloModulePass {
 public:
  explicit InstructionFusion(
      std::function<bool(const HloInstruction& instruction)> is_expensive,
      bool may_duplicate = true,
      FusionConfigCollection config_collection_mode =
          FusionConfigCollection::kOff)
      : is_expensive_(is_expensive),
        may_duplicate_(may_duplicate),
        config_collection_mode_(config_collection_mode) {}
  ~InstructionFusion() override = default;
  absl::string_view name() const override { return "fusion"; }

  // Run instruction fusion on the given computation. Returns whether the
  // computation was changed (instructions were fused).
  StatusOr<bool> Run(HloModule* module) override;

  // Returns true if the computation of the given instruction is significantly
  // more expensive than just writing all the values of the instructions' result
  // array. Expensive operations will not be duplicated.
  static bool IsExpensive(const HloInstruction& instruction);

 protected:
  // Returns a list of computations on which Fusion is performed.
  virtual std::vector<HloComputation*> GetFusionComputations(HloModule* module);

  // Returns a FusionQueue that implements custom order of instructions being
  // fused. The default implementation processes consumers in reverse post
  // order.
  virtual std::unique_ptr<FusionQueue> GetFusionQueue(
      HloComputation* computation);

  // Returns whether the given producer instruction should be fused into the
  // given consumer instruction. producer is necessarily an operand of consumer.
  // Derived classes should define this method to specify which instructions
  // should be fused. `operand_index` is which operand of the consumer the
  // producer is.
  //
  // Instructions are traversed in reverse post order (computation root to
  // leaves). This method is called for each operand of the instruction (where
  // the operand is 'producer' and the instruction is 'consumer')
  //
  // Subtypes can override this with target-specific heuristics.
  virtual bool ShouldFuse(HloInstruction* consumer, int64 operand_index);

  // Returns whether multi-output fusion can be applied to fuse `producer` into
  // `consumer`. In contrast to "regular" fusion, the `producer` is not
  // duplicated by multi-output fusion.
  virtual bool ShouldFuseIntoMultiOutput(HloInstruction* consumer,
                                         int64 operand_index) {
    return false;
  }

  // Chooses a fusion kind for `producer` and `consumer`.
  // Default method chooses `kLoop`.
  virtual HloInstruction::FusionKind ChooseKind(const HloInstruction* producer,
                                                const HloInstruction* consumer);

  // Fuses 'producer' into 'fusion_instruction'. 'fusion_instruction' needs to
  // be a fusion instruction. Returns the newly created clone of 'producer'
  // which is part of the fusion computation.
  virtual HloInstruction* FuseInstruction(HloInstruction* fusion_instruction,
                                          HloInstruction* producer);

  // Fuses producer into consumer. Returns the fusion instruction.
  virtual HloInstruction* Fuse(HloInstruction* producer,
                               HloInstruction* consumer);

  // Creates a new fusion instruction containing `producer` and `consumer`. A
  // tuple is added as the fusion instruction's root, which consumes from both,
  // `producer` and `consumer`. This style of fusion is referred to as
  // multi-output fusion.
  virtual HloInstruction* FuseIntoMultiOutput(HloInstruction* producer,
                                              HloInstruction* consumer);

  // An "effectively unary" operation is one that has at most one "large"
  // input with the others being negligible in terms of memory usage.
  // We use "has a smaller true rank than the output" as a heuristic
  // for "negligible" memory usage.
  bool EffectivelyAtMostUnary(HloInstruction* hlo);

  // Returns true if fusing producer into consumer would cause producer to be
  // duplicated. This is the case if producer has uses other than consumer.
  bool FusionWouldDuplicate(const HloInstruction& producer,
                            const HloInstruction& consumer) {
    return !(producer.users().size() == 1 && consumer.IsUserOf(&producer));
  }

  bool is_expensive(const HloInstruction& instruction) {
    return is_expensive_(instruction);
  }

  // Whether multi-output fusion would introduce a cycle into the HLO graph.
  bool MultiOutputFusionCreatesCycle(HloInstruction* producer,
                                     HloInstruction* consumer);

  // Current HloComputation instance the loop fuser is traversing.
  HloComputation* computation_;
  HloModule* module_;
  // Reachability information for the current computation.
  std::unique_ptr<HloReachabilityMap> reachability_;

  FusionConfigCollection config_collection_mode() {
    return config_collection_mode_;
  }

  // Returns whether 'consumer' may reuse elements of its `operand_index`th
  // operand.
  bool ReusesOperandElements(const HloInstruction* consumer,
                             int64 operand_index);

  // The set of producers whose consumers we cannot fuse into.
  using HloInstructionSet = std::unordered_set<HloInstruction*>;

  // Computes the set of nodes that we do not want to fuse into any of their
  // consumers based on a global analysis of the HLO graph.
  virtual HloInstructionSet ComputeGloballyUnfusible(
      absl::Span<HloInstruction* const> post_order);

 private:
  HloInstruction* AddFusionInstruction(HloInstruction* producer,
                                       HloInstruction* consumer);

  // Whether or not we can fuse producer into consumer on all paths
  // from the producer to the consumer where nodes are HLOs and edges are uses.
  //
  // A map from <producer, consumer> to a bool is required as the result cache
  // to store and query the results of calls to this function, in order to avoid
  // repeated computations.
  bool CanFuseOnAllPaths(
      HloInstruction* producer, HloInstruction* consumer,
      const HloInstructionSet& do_not_fuse,
      absl::flat_hash_map<std::pair<HloInstruction*, HloInstruction*>, bool>*
          result_cache);

  // Used to determine if an HLO is expensive. Expensive operations will not be
  // duplicated.
  std::function<bool(const HloInstruction& instruction)> is_expensive_;

  // Returns whether we may duplicate an instruction if we want to fuse it.
  bool may_duplicate_;

  // Configuration mode.
  FusionConfigCollection config_collection_mode_;

  // Caches which operands are reused inside fusion computations.
  absl::flat_hash_map<const HloInstruction*,
                      absl::flat_hash_set<const HloInstruction*>>
      reused_fusion_operands_;

  TF_DISALLOW_COPY_AND_ASSIGN(InstructionFusion);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_INSTRUCTION_FUSION_H_
