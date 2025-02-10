/* Copyright 2023 The OpenXLA Authors.

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
#ifndef XLA_HLO_UTILS_HLO_TRAVERSAL_H_
#define XLA_HLO_UTILS_HLO_TRAVERSAL_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"

namespace xla {

class HloFusionAdaptor;

// Treats HloInstructions as if they were unfused.
class HloInstructionAdaptor {
 public:
  HloInstructionAdaptor() = delete;
  HloInstructionAdaptor(const HloInstruction& instruction,
                        const HloFusionAdaptor* parent);

  HloOpcode opcode() const { return instruction_->opcode(); }
  absl::string_view name() const { return instruction_->name(); }

  HloInstructionAdaptor GetOperand(int index) const;
  absl::InlinedVector<HloInstructionAdaptor, 2> GetOperands() const;
  absl::InlinedVector<HloInstructionAdaptor, 2> GetUsers() const;
  const xla::Shape& shape() const { return instruction_->shape(); }
  std::string ToString() const { return instruction_->ToString(); }

  friend bool operator==(const HloInstructionAdaptor& lhs,
                         const HloInstructionAdaptor& rhs);
  friend bool operator!=(const HloInstructionAdaptor& lhs,
                         const HloInstructionAdaptor& rhs);
  template <typename H>
  friend H AbslHashValue(H h, const HloInstructionAdaptor& m);

  // Use sparingly; prefer extending the interface.
  const HloInstruction& instruction() const { return *instruction_; }
  const HloFusionAdaptor& parent() const { return *parent_; }

 private:
  const HloInstruction* instruction_;

  // Pointer to the parent fusion adaptor. Is never null.
  const HloFusionAdaptor* parent_;
};

template <typename H>
H AbslHashValue(H h, const HloInstructionAdaptor& m) {
  return H::combine(std::move(h), m.instruction_->GetModule(),
                    m.instruction_->unique_id());
}

template <HloOpcode op, HloOpcode... rest>
bool IsOpcodeAnyOf(const HloInstruction* instr) {
  return (instr->opcode() == op) || ((instr->opcode() == rest) || ...);
}

class HloFusionInstructionAdaptor;

// Treats a set of HloInstructions as if they were fused.
class HloFusionAdaptor {
 public:
  ~HloFusionAdaptor();
  bool ContainsInstruction(HloInstructionAdaptor instruction) const;
  bool ContainsInstruction(const HloInstruction* instruction) const;
  absl::InlinedVector<HloInstructionAdaptor, 2> GetRoots() const;
  absl::InlinedVector<const HloInstruction*, 2> GetParameters() const;
  absl::InlinedVector<HloInstructionAdaptor, 2> MakeInstructionPostOrder()
      const;

  // Calls `fn` for each instruction in the fusion.
  void ForEach(const std::function<void(HloInstructionAdaptor)>& fn) const;

  std::string ToString() const;

  static std::unique_ptr<HloFusionAdaptor> ForInstruction(
      const HloInstruction* instruction);
  static std::unique_ptr<HloFusionAdaptor> ForProducerConsumer(
      const HloInstruction* producer, const HloInstruction* consumer,
      bool with_extra_outputs = false);
  static std::unique_ptr<HloFusionAdaptor> ForComputation(
      const HloComputation* computation);

 private:
  HloFusionAdaptor() = default;
  explicit HloFusionAdaptor(bool with_extra_outputs);
  HloFusionAdaptor(const HloFusionAdaptor&) = delete;
  HloFusionAdaptor& operator=(const HloFusionAdaptor&) = delete;

  void AddInstruction(const HloInstruction* instruction);
  void AddComputation(const HloComputation* computation);

  absl::InlinedVector<std::unique_ptr<HloFusionInstructionAdaptor>, 2>
      fusion_instructions_;
  // Whether extra fusion roots should be created for producer consumer fusions
  // where producer roots have extra usages outside the fusion.
  bool with_extra_outputs_ = false;
};

enum class TraversalResult {
  // Visit the operands/users of this node.
  kAdvance,
  // Do not visit any more nodes.
  kInterrupt,
  // Do not visit the operands/users of this node (but continue the traversal
  // otherwise).
  kSkip,
};

// Visit the HLO nodes starting from `roots` in BFS order (consumers before
// producers). Each node will be visited exactly once.
void HloBfsConsumersFirstTraversal(
    absl::Span<const HloInstructionAdaptor> roots,
    const HloFusionAdaptor& fusion,
    const std::function<TraversalResult(HloInstructionAdaptor node)>&
        visit_node);

// Visit the HLO nodes starting from `producers` in BFS order following the
// `user` edges. Each node will be visited exactly once.
void HloBfsProducersFirstTraversal(
    absl::Span<const HloInstructionAdaptor> producers,
    const HloFusionAdaptor& fusion,
    const std::function<TraversalResult(HloInstructionAdaptor node)>&
        visit_node);

// Visit the HLO nodes starting from `roots`, returning true if the return value
// of `visit` for any of nodes is true. Uses the same order as
// `HloBfsConsumersFirstTraversal` if `visit_operands` is true. Otherwise the
// same order as `HloBfsProducersFirstTraversal` is used.
bool HloBfsAnyOf(absl::Span<const HloInstructionAdaptor> roots,
                 const HloFusionAdaptor& fusion,
                 const std::function<bool(HloInstructionAdaptor node)>& visit,
                 bool visit_operands = true);

// Visit the HLO nodes starting from `roots`, returning true if the return value
// of `visit` for any of nodes is true. If `visit_operands` is true, the
// search is going towards the operands, otherwise towards the users. Doesn't
// require instruction and fusion adaptors.
bool HloBfsAnyOf(absl::Span<const HloInstruction* const> roots,
                 const std::function<bool(const HloInstruction* node)>& visit,
                 bool visit_operands = true);

// Visit the HLO nodes starting from `roots`, returning the first
// node for which `visit` returns true, or `nullopt` if no node matches. Uses
// the same order as `HloBfsConsumersFirstTraversal` if `visit_operands` is
// true. Otherwise the same order as `HloBfsProducersFirstTraversal` is used.
std::optional<HloInstructionAdaptor> HloBfsFindIf(
    absl::Span<const HloInstructionAdaptor> roots,
    const HloFusionAdaptor& fusion,
    const std::function<bool(HloInstructionAdaptor node)>& visit,
    bool visit_operands = true);

// Visit the HLO nodes starting from `roots`. If `visit_operands` is true, the
// search is going towards the operands, otherwise towards the users. Returns
// the first node for which `visit` returns true, or `nullopt` if no node
// matches.
std::optional<const HloInstruction*> HloBfsFindIf(
    absl::Span<const HloInstruction* const> roots,
    const std::function<bool(const HloInstruction* node)>& visit,
    bool visit_operands = true);

// Visit the HLO nodes starting from `roots`.  If `visit_operands` is true, the
// search is going towards the operands, otherwise towards the users. Returns
// all nodes for which `visit` returns true. If no node matches, returns an
// empty vector.
std::vector<const HloInstruction*> HloBfsFindAll(
    absl::Span<const HloInstruction* const> roots,
    const std::function<bool(const HloInstruction* node)>& visit,
    bool visit_operands = true);

// Returns true if any instruction in the fusion adaptor matches the predicate.
template <typename Pred>
bool HloAnyOf(const HloFusionAdaptor& fusion, Pred&& pred) {
  bool is_any = false;
  fusion.ForEach([&](HloInstructionAdaptor node) {
    if (pred(node)) {
      is_any = true;
    }
  });
  return is_any;
}

// Find a use chain from `parent` to `root`. Empty if no chain exists.
// `[parent]` if `parent` is `root`.
std::vector<HloInstructionAdaptor> HloFindUseChain(HloInstructionAdaptor parent,
                                                   HloInstructionAdaptor root);

}  // namespace xla

#endif  // XLA_HLO_UTILS_HLO_TRAVERSAL_H_
