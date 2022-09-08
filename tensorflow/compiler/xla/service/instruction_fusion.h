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
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/fusion_queue.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"

namespace xla {

struct NoFusionPossible;

// Propagating explanation of fusion decisions: if something could not be fused,
// explain the reason.
class FusionDecision {
 public:
  // Can not be fused: explain why. Implicit conversion due to optional-like
  // semantics: waiver granted in cl/419938611.
  FusionDecision(absl::string_view explanation)  // NOLINT
      : explanation_(explanation) {}

  // Same constructor as string_view, to allow implicit string conversion (can't
  // implicitly convert both char* to string_view and string_view to
  // FusionDecision).
  FusionDecision(const char* explanation)  // NOLINT
      : explanation_(explanation) {}

  // If condition is `true` means that we CAN fuse. In that case, explanation is
  // discarded.
  FusionDecision(bool condition, absl::string_view explanation) {
    if (!condition) {
      explanation_ = std::string(explanation);
    }
  }

  // Can be fused.
  FusionDecision() {}

  // A trick to declare and test fusion decision in a single statement (as TF
  // is still on C++14 and can't use if statement with explicit initializer).
  //
  // Cf. NoFusionPossible definition for sample usage.
  // TODO(b/157309856): Use conditional initializer instead.
  NoFusionPossible operator!();

  // Returns whether it can be fused.
  explicit operator bool() const { return CanFuse(); }

  // Whether the fusion decision is positive.
  bool CanFuse() const { return !explanation_.has_value(); }

  // Connects two decisions with a disjunction. This is different than just
  // picking one, as we also have to propagate both explanations if only one of
  // them is false to show why fusion wasn't performed.
  FusionDecision Or(const FusionDecision& decision) {
    if (CanFuse() || decision.CanFuse()) {
      return {};
    }
    return {absl::StrCat(explanation_.value_or(""), " ; ", decision.Explain())};
  }

  // Connects two fusion decision with a conjunction. Unlike disjunction,
  // propagates only one explanation (as it is enough to show that fusion could
  // not be done).
  FusionDecision And(const FusionDecision& decision) {
    if (CanFuse()) {
      return decision;
    }
    if (decision.CanFuse()) {
      return *this;
    }
    // Both conditions were violated: returning either is valid.
    return *this;
  }

  // Appends to explanation, or turns the decision negative.
  FusionDecision operator<<(absl::string_view explanation) {
    return {absl::StrCat(explanation_.value_or(""), explanation)};
  }

  // Appends to explanation, or turns the decision negative.
  FusionDecision operator<<(int64_t explanation) {
    return {absl::StrCat(explanation_.value_or(""), explanation)};
  }

  // Explains why the fusion could not be performed.
  std::string Explain() const { return *explanation_; }

 private:
  // Empty IFF fusion is possible (explanation provided for negative cases).
  std::optional<std::string> explanation_;
};

// Helper class: contextually convertible to "no fusion possible" unlike
// FusionDecision. This is a trick to declare and test fusion decision in a
// single statement (as we are still on C++14).
//
// Sample usage:
//
// if (NoFusionPossible fusible = !FusabilityRestriction(producer, consume)) {
//   return !fusible; // Note that negation converts it back to FusionDecision.
// }
struct NoFusionPossible {
  // Inverts the test value (true <=> not fusible) on wrapped FusionDecision.
  explicit operator bool() { return !static_cast<bool>(fusion_decision); }

  // Returns wrapped fusion decision.
  FusionDecision operator!() { return fusion_decision; }

  FusionDecision fusion_decision;
};

inline NoFusionPossible FusionDecision::operator!() { return {*this}; }

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
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // Returns true if the computation of the given instruction is significantly
  // more expensive than just writing all the values of the instructions' result
  // array. Expensive operations will not be duplicated.
  static bool IsExpensive(const HloInstruction& instruction);

  // Returns true if it's legal to fuse the producer instruction into consumer
  // with regard to in-place semantics of the consumer. For example, it is
  // illegal to fuse a slice into a dynamic-update-slice if the slice output is
  // used as the update and if slice and dynamic-update-slice indices cannot be
  // proven to be the same.
  static FusionDecision ShouldFuseInPlaceOp(const HloInstruction* producer,
                                            const HloInstruction* consumer);

 protected:
  // Returns a list of computations on which Fusion is performed.
  virtual std::vector<HloComputation*> GetFusionComputations(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

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
  virtual FusionDecision ShouldFuse(HloInstruction* consumer,
                                    int64_t operand_index);

  // Returns whether multi-output fusion can be applied to fuse `producer` into
  // `consumer`. In contrast to "regular" fusion, the `producer` is not
  // duplicated by multi-output fusion.
  virtual FusionDecision ShouldFuseIntoMultiOutput(HloInstruction* consumer,
                                                   int64_t operand_index) {
    return "multi-output fusion not supported by this pass";
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
                               HloInstruction* consumer,
                               HloComputation* computation);

  // Creates a new fusion instruction containing `producer` and `consumer`. A
  // tuple is added as the fusion instruction's root, which consumes from both,
  // `producer` and `consumer`. This style of fusion is referred to as
  // multi-output fusion.
  virtual HloInstruction* FuseIntoMultiOutput(HloInstruction* producer,
                                              HloInstruction* consumer,
                                              HloComputation* computation);

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

  // Overwrites the originally initialized is_expensive function.
  void set_is_expensive(
      std::function<bool(const HloInstruction& instruction)> is_expensive) {
    is_expensive_ = is_expensive;
  }

  // Whether multi-output fusion would introduce a cycle into the HLO graph.
  bool MultiOutputFusionCreatesCycle(HloInstruction* producer,
                                     HloInstruction* consumer,
                                     const HloReachabilityMap& reachability);

  FusionConfigCollection config_collection_mode() {
    return config_collection_mode_;
  }

  // Returns whether 'consumer' may reuse elements of its `operand_index`th
  // operand.
  bool ReusesOperandElements(const HloInstruction* consumer,
                             int64_t operand_index);

  // The set of producers whose consumers we cannot fuse into.
  using HloInstructionSet = absl::flat_hash_set<HloInstruction*>;

  // Computes the set of nodes that we do not want to fuse into any of their
  // consumers based on a global analysis of the HLO graph.
  virtual HloInstructionSet ComputeGloballyUnfusible(
      absl::Span<HloInstruction* const> post_order,
      const HloReachabilityMap& reachability);

 private:
  // Returns the reused operands of `instruction` from reused_fusion_operands_,
  // computing them if they have not previously been computed for that
  // instruction.
  // The returned value has pointer stability, assuming entries are not deleted
  // from reused_fusion_operands_.
  absl::flat_hash_set<const HloInstruction*>& ReusedOperandsOf(
      const HloInstruction* instruction);

  // Updates reused_fusion_operands_ for a fusion when we are about to fuse
  // `producer` into `fusion_instruction`.
  void UpdateReusedOperandsForFusion(HloInstruction* producer,
                                     HloInstruction* fusion_instruction);

  HloInstruction* AddFusionInstruction(HloInstruction* producer,
                                       HloInstruction* consumer,
                                       HloComputation* computation);

  // Whether or not we can fuse producer into consumer on all paths
  // from the producer to the consumer where nodes are HLOs and edges are uses.
  //
  // A map from <producer, consumer> to a bool is required as the result cache
  // to store and query the results of calls to this function, in order to avoid
  // repeated computations.
  bool CanFuseOnAllPaths(
      HloInstruction* producer, HloInstruction* consumer,
      const HloInstructionSet& do_not_fuse,
      const HloReachabilityMap& reachability,
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
  absl::flat_hash_map<
      const HloInstruction*,
      std::unique_ptr<absl::flat_hash_set<const HloInstruction*>>>
      reused_fusion_operands_;

  InstructionFusion(const InstructionFusion&) = delete;
  InstructionFusion& operator=(const InstructionFusion&) = delete;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_INSTRUCTION_FUSION_H_
