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

#ifndef XLA_SERVICE_INSTRUCTION_FUSION_H_
#define XLA_SERVICE_INSTRUCTION_FUSION_H_

#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/service/hlo_module_config.h"
#include "tsl/platform/macros.h"
// The source_location.h is not available in open source.
#if defined(PLATFORM_GOOGLE)
#include "absl/types/source_location.h"
#endif  // PLATFORM_GOOGLE
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/fusion_queue.h"

namespace xla {

// Propagating explanation of fusion decisions: if something could not be fused,
// explain the reason.
class FusionDecision {
 public:
  static FusionDecision Allow() { return FusionDecision(); }
  static FusionDecision Forbid(absl::string_view explanation) {
    return FusionDecision(explanation);
  }
  FusionDecision(const FusionDecision& decision) = default;

  // If condition is `true` means that we CAN fuse. In that case, explanation is
  // discarded.
  FusionDecision(bool condition, absl::string_view explanation) {
    if (!condition) {
      explanation_ = std::string(explanation);
    }
  }

#if defined(PLATFORM_GOOGLE)
  // We can fuse iff. the decision is `true`. The source location indicates
  // where an instance was created, making debugging easier without a need to
  // provide explicit explanation.
  FusionDecision(  // NOLINT
      bool decision,
      absl::SourceLocation source_location = absl::SourceLocation::current());
#endif  // PLATFORM_GOOGLE

  // Returns whether it can be fused.
  explicit operator bool() const { return CanFuse(); }

  // Whether the fusion decision is positive.
  bool CanFuse() const { return !explanation_.has_value(); }

  // Connects two decisions with a disjunction. This is different than just
  // picking one, as we also have to propagate both explanations if only one of
  // them is false to show why fusion wasn't performed.
  FusionDecision Or(const FusionDecision& decision) const {
    if (CanFuse() || decision.CanFuse()) {
      return Allow();
    }
    return Forbid(
        absl::StrCat(explanation_.value_or(""), " ; ", decision.Explain()));
  }

  // Connects two fusion decision with a conjunction. Unlike disjunction,
  // propagates only one explanation (as it is enough to show that fusion could
  // not be done).
  FusionDecision And(const FusionDecision& decision) const {
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
  FusionDecision operator<<(absl::string_view explanation) const {
    return Forbid(absl::StrCat(explanation_.value_or(""), explanation));
  }

  // Appends to explanation, or turns the decision negative.
  FusionDecision operator<<(int64_t explanation) const {
    return Forbid(absl::StrCat(explanation_.value_or(""), explanation));
  }

  // Explains why the fusion could not be performed.
  std::string Explain() const { return *explanation_; }

 private:
  // Empty IFF fusion is possible (explanation provided for negative cases).
  std::optional<std::string> explanation_;

  FusionDecision() = default;

  explicit FusionDecision(absl::string_view explanation)
      : explanation_(explanation) {}

  explicit FusionDecision(const char* explanation)
      : explanation_(explanation) {}
};

#define RETURN_IF_NOT_FUSIBLE(...)                   \
  do {                                               \
    ::xla::FusionDecision _decision = (__VA_ARGS__); \
    if (TF_PREDICT_FALSE(!_decision.CanFuse())) {    \
      return _decision;                              \
    }                                                \
  } while (0)

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
  absl::StatusOr<bool> Run(
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
  // Returns a list of computations that are not fusion computations. These
  // computations contain instructions which are candidates for fusions.
  virtual std::vector<HloComputation*> GetNonFusionComputations(
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

  // Returns whether a 'producer' at given operand index can be fused into the
  // consumer. It uses the provided function to check the legality of a possible
  // fusion when either the producer or the consumer contains an operation which
  // updates an operand in place.
  virtual FusionDecision ShouldFuse(
      HloInstruction* consumer, int64_t operand_index,
      std::function<FusionDecision(const HloInstruction*,
                                   const HloInstruction*)>
          inplace_op_fusion_decider);

  // Returns whether multi-output fusion can be applied to fuse `producer` into
  // `consumer`. In contrast to "regular" fusion, the `producer` is not
  // duplicated by multi-output fusion.
  virtual FusionDecision ShouldFuseIntoMultiOutput(HloInstruction* consumer,
                                                   int64_t operand_index) {
    return FusionDecision::Forbid(
        "multi-output fusion not supported by this pass");
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

  // Dumps the state of computation before fusion.
  void DumpPreFusionState(HloComputation* computation, HloInstruction* consumer,
                          HloInstruction* producer, bool is_mof = false);

  // Dumps the state of computation and the reason why the fusion was not
  // performed.
  void DumpNotFusingState(HloComputation* computation, HloInstruction* consumer,
                          HloInstruction* producer, FusionDecision decision);

  // Dumps the state of computation after fusion happened.
  void DumpStateAfterFusion(HloComputation* computation,
                            HloInstruction* fusion_instruction,
                            const std::string& producer_name);

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

#endif  // XLA_SERVICE_INSTRUCTION_FUSION_H_
