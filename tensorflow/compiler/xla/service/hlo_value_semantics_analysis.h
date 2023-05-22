/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_VALUE_SEMANTICS_ANALYSIS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_VALUE_SEMANTICS_ANALYSIS_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {

// The comment below explains where the labels could originate from. Once
// originated,  those labels are then propagated throughout the HLO module.
enum class HloValueSemanticLabel {
  // Values that are known or predictable at compile time, including constants,
  // iota, replica-id, and partition-id.
  kStatic,
  // Values that are not known or can't be predicated at compile time.
  kRandom,
  // HLO module parameters.
  kWeight,
  // Output of weight-weight or weight-activation matmuls.
  kActivation,
  // Output of weight-activation matmuls where the weight is a dependence of
  // that activation. Or output of weight-activation-gradient matmuls.
  kActivationGradient,
  // Output of activation-gradient-activation matmuls.
  kWeightGradient,
  kTupleOrToken,
};

std::string HloValueSemanticLabelToString(HloValueSemanticLabel label);

class HloValueSemantics {
 public:
  using Id = int64_t;
  HloValueSemantics(HloValueSemanticLabel label, const HloPosition& origin);
  HloValueSemantics(Id id, HloValueSemanticLabel label,
                    const HloPosition& origin);
  HloValueSemantics(const HloValueSemantics& other) = default;
  HloValueSemantics(HloValueSemantics&& other) = default;
  HloValueSemantics& operator=(const HloValueSemantics& other) = default;

  Id id() const { return id_; }
  HloValueSemanticLabel label() const { return label_; }
  const HloPosition& origin() const { return origin_; }
  std::string ToString() const;

 private:
  const Id id_;
  const HloValueSemanticLabel label_;
  const HloPosition origin_;
};

using HloValueSemanticsMap =
    absl::flat_hash_map<const HloInstruction*,
                        ShapeTree<const HloValueSemantics*>>;
class HloValueSemanticsPropagation;

class HloValueSemanticsAnalysis {
 public:
  static StatusOr<std::unique_ptr<HloValueSemanticsAnalysis>> Run(
      const HloModule& module);
  virtual ~HloValueSemanticsAnalysis() = default;
  const HloValueSemantics* GetSemantics(const HloInstruction* instruction,
                                        const ShapeIndex& index = {}) const;

  const HloValueSemanticsMap& GetSemanticsMap() const {
    return value_semantics_;
  }

 protected:
  friend class HloValueSemanticsPropagation;

  explicit HloValueSemanticsAnalysis(const HloModule& module);

  void AnnotateWeights();

  // Infer semantics for all instructions in the computation. Computation
  // parameters are assigned the semantics of the corresponding operand.
  Status RunOnComputation(const HloComputation& computation,
                          absl::Span<const HloInstruction* const> operands);
  // Same as the above RunOnComputation, but computation parameters have
  // already been assigned with semantics.
  virtual Status RunOnComputation(const HloComputation& computation);
  HloValueSemantics::Id NextId();
  const HloValueSemantics* NewHloValueSemantics(HloValueSemanticLabel label,
                                                const HloPosition& origin);
  const ShapeTree<const HloValueSemantics*>& GetInstructionSemantics(
      const HloInstruction* instruction) const;
  void DeepCopyHloValueSemantics(
      ShapeTree<const HloValueSemantics*>& copy_to,
      const ShapeTree<const HloValueSemantics*>& copy_from,
      const ShapeIndex& source_index, const ShapeIndex& destination_index);
  void DeepCopyHloValueSemantics(
      const HloInstruction* target,
      const ShapeTree<const HloValueSemantics*>& copy_from,
      const ShapeIndex& source_index = {});
  void SetHloValueSemantics(
      const HloInstruction* target,
      const ShapeTree<const HloValueSemantics*>& semantics);
  void DeleteHloValueSemantics(
      const ShapeTree<const HloValueSemantics*>& to_delete);
  void DeleteHloValueSemantics(const HloValueSemantics* to_delete);
  const HloModule& module_;
  HloValueSemanticsMap value_semantics_;
  absl::flat_hash_map<HloValueSemantics::Id, std::unique_ptr<HloValueSemantics>>
      value_semantics_map_;
  HloValueSemantics::Id next_id_;
};

class HloValueSemanticsPropagation : public DfsHloVisitorWithDefault {
 public:
  explicit HloValueSemanticsPropagation(HloValueSemanticsAnalysis* analysis);
  Status Run(const HloComputation& computation);
  // Infer the output semantics from all operands of the instruction.
  Status DefaultAction(HloInstruction* instruction) override;
  Status HandleParameter(HloInstruction* parameter) override;
  Status HandleConstant(HloInstruction* constant) override;
  Status HandleIota(HloInstruction* iota) override;
  Status HandlePartitionId(HloInstruction* partition_id) override;
  Status HandleReplicaId(HloInstruction* replica_id) override;
  Status HandleClamp(HloInstruction* clamp) override;
  Status HandleTuple(HloInstruction* tuple) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;
  Status HandleCall(HloInstruction* call) override;
  Status HandleFusion(HloInstruction* fusion) override;
  Status HandleCustomCall(HloInstruction* custom_call) override;
  Status HandleWhile(HloInstruction* xla_while) override;
  Status HandleConditional(HloInstruction* conditional) override;
  Status HandleSelect(HloInstruction* select) override;
  Status HandleConcatenate(HloInstruction* concatenate) override;
  Status HandleDynamicSlice(HloInstruction* dynamic_slice) override;
  Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override;
  Status HandleCopyStart(HloInstruction* copy_start) override;
  Status HandleCopyDone(HloInstruction* copy_done) override;
  Status HandleCollectivePermuteStart(
      HloInstruction* collective_permute_start) override;
  Status HandleCollectivePermuteDone(
      HloInstruction* collective_permute_done) override;
  Status HandleGather(HloInstruction* gather) override;
  Status HandleScatter(HloInstruction* scatter) override;
  Status HandleAfterAll(HloInstruction* after_all) override;
  Status HandleAsyncStart(HloInstruction* async_start) override;
  Status HandleAsyncDone(HloInstruction* async_done) override;
  Status HandleInfeed(HloInstruction* infeed) override;

 protected:
  HloValueSemantics CopySemantics(const HloValueSemantics& semantics);
  HloValueSemantics CopySemanticsWithNewOrigin(
      const HloValueSemantics& semantics, HloInstruction* new_origin,
      const ShapeIndex& index = {});
  const HloValueSemantics* AddSemantics(const HloValueSemantics& semantics);
  // Checks whether the given activation's origin depends on the given
  // origin_dependence. If recursive is true, recursively match
  // origin_dependence with operands, otherwise only match it with
  // activation_semantics' operands.
  bool IsActivationOriginDependentOn(
      const HloValueSemantics& activation_semantics,
      const HloPosition& origin_dependence, bool recursive = false) const;
  StatusOr<HloValueSemantics> ComputeSemanticsFromStaticAndOther(
      const HloValueSemantics& static_semantics,
      const HloValueSemantics& other_semantics, HloInstruction* instruction);
  StatusOr<HloValueSemantics> ComputeSemanticsFromRandomAndOther(
      const HloValueSemantics& random_semantics,
      const HloValueSemantics& other_semantics, HloInstruction* instruction);
  StatusOr<HloValueSemantics> ComputeSemanticsFromWeightAndOther(
      const HloValueSemantics& weight_semantics,
      const HloValueSemantics& other_semantics, HloInstruction* instruction);
  StatusOr<HloValueSemantics> ComputeSemanticsFromActivationAndOther(
      const HloValueSemantics& activation_semantics,
      const HloValueSemantics& other_semantics, HloInstruction* instruction);
  StatusOr<HloValueSemantics> ComputeSemanticsFromActivationGradientAndOther(
      const HloValueSemantics& activation_gradient_semantics,
      const HloValueSemantics& other_semantics, HloInstruction* instruction);
  StatusOr<HloValueSemantics> ComputeSemanticsFromWeightGradientAndOther(
      const HloValueSemantics& weight_gradient_semantics,
      const HloValueSemantics& other_semantics, HloInstruction* instruction);
  StatusOr<HloValueSemantics> ComputeSemanticsFromOperands(
      HloInstruction* instruction, absl::Span<const int64_t> operand_indices,
      absl::Span<const ShapeIndex> operand_shape_indices = {});
  HloValueSemanticsAnalysis* analysis_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_VALUE_SEMANTICS_ANALYSIS_H_
