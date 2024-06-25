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
#ifndef XLA_SERVICE_GPU_HLO_TRAVERSAL_H_
#define XLA_SERVICE_GPU_HLO_TRAVERSAL_H_

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
namespace gpu {

class HloFusionAdaptor;

// Treats HloInstructions as if they were unfused.
class HloInstructionAdaptor {
 public:
  HloInstructionAdaptor() = default;
  HloInstructionAdaptor(const HloInstruction& instruction,
                        const HloFusionAdaptor* parent)
      : instruction_(&instruction), parent_(parent) {}

  HloOpcode opcode() const { return instruction_->opcode(); }
  absl::string_view name() const { return instruction_->name(); }

  HloInstructionAdaptor GetOperand(int index) const;
  absl::InlinedVector<HloInstructionAdaptor, 2> GetOperands() const;
  absl::InlinedVector<HloInstructionAdaptor, 2> GetUsers() const;
  const xla::Shape& shape() const { return instruction_->shape(); }
  std::string ToString() const { return instruction_->ToString(); }

  friend bool operator==(const HloInstructionAdaptor& lhs,
                         const HloInstructionAdaptor& rhs);
  template <typename H>
  friend H AbslHashValue(H h, const HloInstructionAdaptor& m);

  // Use sparingly; prefer extending the interface.
  const HloInstruction& instruction() const { return *instruction_; }
  const HloFusionAdaptor& parent() const { return *parent_; }

 private:
  const HloInstruction* instruction_;

  // Pointer to the parent fusion adaptor. Can not be null.
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

namespace internal {

// An interface to abstract away the difference between single instruction
// fusion and fused computations.
class HloFusionInstructionAdaptor {
 public:
  virtual ~HloFusionInstructionAdaptor() = default;
  virtual bool ContainsInstruction(const HloInstruction* instruction) const = 0;
  // If it is a regular multi-output fusion, the order of the returned roots
  // matches the order of the tuple elements of the tuple root of the fusion
  // computation. We do not deduplicate fusion roots.
  virtual absl::InlinedVector<HloInstructionAdaptor, 2> GetRoots() const = 0;
  virtual absl::InlinedVector<const HloInstruction*, 2> GetParameters()
      const = 0;
  virtual const HloInstruction& FusionInstruction() const = 0;
  virtual absl::InlinedVector<HloInstructionAdaptor, 2>
  MakeInstructionPostOrder() const = 0;
  virtual std::string ToString() const = 0;
};

}  // namespace internal

class HloFusionAdaptor {
 public:
  bool ContainsInstruction(HloInstructionAdaptor instruction) const;
  bool ContainsInstruction(const HloInstruction* instruction) const;
  absl::InlinedVector<HloInstructionAdaptor, 2> GetRoots() const;
  absl::InlinedVector<const HloInstruction*, 2> GetParameters() const;
  absl::InlinedVector<HloInstructionAdaptor, 2> MakeInstructionPostOrder()
      const;
  std::string ToString() const;

  static std::unique_ptr<HloFusionAdaptor> ForInstruction(
      const HloInstruction* instruction);
  static std::unique_ptr<HloFusionAdaptor> ForProducerConsumer(
      const HloInstruction* producer, const HloInstruction* consumer);
  static std::unique_ptr<HloFusionAdaptor> ForComputation(
      const HloComputation* computation);

 private:
  void AddInstruction(const HloInstruction* instruction);
  void AddComputation(const HloComputation* computation);

  absl::InlinedVector<std::unique_ptr<internal::HloFusionInstructionAdaptor>, 2>
      fusion_instructions_;
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
        visit_node,
    const std::function<void(HloInstructionAdaptor producer)>& visit_arg =
        [](HloInstructionAdaptor) {});

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
bool HloAnyOf(absl::Span<const HloInstructionAdaptor> roots,
              const HloFusionAdaptor& fusion,
              const std::function<bool(HloInstructionAdaptor node)>& visit,
              bool visit_operands = true);

// Visit the HLO nodes starting from `roots`, returning true if the return value
// of `visit` for any of nodes is true. If `visit_operands` is true, the
// search is going towards the operands, otherwise towards the users. Doesn't
// require instruction and fusion adaptors.
bool HloAnyOf(absl::Span<const HloInstruction* const> roots,
              const std::function<bool(const HloInstruction* node)>& visit,
              bool visit_operands = true);

// Visit the HLO nodes starting from `roots`, returning the first
// node for which `visit` returns true, or `nullopt` if no node matches. Uses
// the same order as `HloBfsConsumersFirstTraversal` if `visit_operands` is
// true. Otherwise the same order as `HloBfsProducersFirstTraversal` is used.
std::optional<HloInstructionAdaptor> HloFindIf(
    absl::Span<const HloInstructionAdaptor> roots,
    const HloFusionAdaptor& fusion,
    const std::function<bool(HloInstructionAdaptor node)>& visit,
    bool visit_operands = true);

// Visit the HLO nodes starting from `roots`. If `visit_operands` is true, the
// search is going towards the operands, otherwise towards the users. Returns
// the first node for which `visit` returns true, or `nullopt` if no node
// matches.
std::optional<const HloInstruction*> HloFindIf(
    absl::Span<const HloInstruction* const> roots,
    const std::function<bool(const HloInstruction* node)>& visit,
    bool visit_operands = true);

// Visit the HLO nodes starting from `roots`.  If `visit_operands` is true, the
// search is going towards the operands, otherwise towards the users. Returns
// all nodes for which `visit` returns true. If no node matches, returns an
// empty vector.
std::vector<const HloInstruction*> HloFindAll(
    absl::Span<const HloInstruction* const> roots,
    const std::function<bool(const HloInstruction* node)>& visit,
    bool visit_operands = true);

// Find a use chain from `parent` to `root`. Empty if no chain exists.
// `[parent]` if `parent` is `root`.
std::vector<HloInstructionAdaptor> HloFindUseChain(HloInstructionAdaptor parent,
                                                   HloInstructionAdaptor root);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_HLO_TRAVERSAL_H_
