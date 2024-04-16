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
#include "xla/service/gpu/hlo_traversal.h"

#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {
namespace gpu {
namespace {

template <typename F>
void ResolveUsers(const HloInstruction* value, const HloInstruction* user,
                  F&& fn) {
  if (user->opcode() == HloOpcode::kFusion) {
    auto* param = user->fused_parameter(user->operand_index(value));
    for (const auto* param_user : param->users()) {
      fn(param_user);
    }
  } else if (user->opcode() == HloOpcode::kTuple && user->IsRoot()) {
    if (auto* fusion = user->parent()->FusionInstruction()) {
      // Skip through the tuple -> get-tuple-element ops and directly go to the
      // "real" users.
      for (const auto* gte : fusion->users()) {
        if (gte->opcode() != HloOpcode::kGetTupleElement) {
          fn(gte);
          continue;
        }
        for (const auto* gte_user : gte->users()) {
          ResolveUsers(gte, gte_user, fn);
        }
      }
    }
  } else {
    fn(user);
  }
}

const HloInstruction* ResolveOperand(const HloInstruction* operand) {
  if (operand->opcode() == HloOpcode::kFusion) {
    return operand->fused_expression_root();
  }
  // Deal with multi-output fusion operands, which are reached via a
  // get-tuple-element op.
  if (operand->opcode() == HloOpcode::kGetTupleElement &&
      operand->operand(0)->opcode() == HloOpcode::kFusion &&
      operand->operand(0)->fused_expression_root()->opcode() ==
          HloOpcode::kTuple) {
    return operand->operand(0)->fused_expression_root()->operand(
        operand->tuple_index());
  }
  if (operand->opcode() == HloOpcode::kParameter) {
    if (auto* fusion = operand->parent()->FusionInstruction()) {
      return ResolveOperand(fusion->operand(operand->parameter_number()));
    }
  }
  return operand;
}

class SingleInstructionFusion : public HloFusionAdaptor {
 public:
  explicit SingleInstructionFusion(const HloInstruction* instruction)
      : instruction_(*instruction) {
    CHECK_NE(instruction->opcode(), HloOpcode::kFusion)
        << "Use HloFusionFusion";
  }

  bool ContainsInstruction(HloInstructionAdaptor instruction) const override {
    return instruction == instruction_;
  }

  absl::InlinedVector<HloInstructionAdaptor, 2> GetRoots() const override {
    return {instruction_};
  }

  absl::InlinedVector<HloInstructionAdaptor, 2> MakeInstructionPostOrder()
      const override {
    return {instruction_};
  }

  std::string ToString() const override { return instruction_.ToString(); }

 private:
  HloInstructionAdaptor instruction_;
};

class HloComputationFusion : public HloFusionAdaptor {
 public:
  explicit HloComputationFusion(const HloComputation* computation)
      : computation_(computation) {
    // HloFusionAdaptor should only be created for fusion computations, that
    // usually have only a few roots, but there is a case when we can it for
    // non-fusion computations with thousands of roots. It happens inside
    // `FindNonTrivialHero` and it gets very expensive. Calling
    // `FindNonTrivialHero` also doesn't make sense on non-fusion computation,
    // but `InstructionFusion` and `FusionMerger` depend on this behavoiur in
    // `IsProducerConsumerFusible`.
    //
    // `FindNonTrivialHero` only call `ContainsInstruction` and doesn't use
    // information about roots, so we can skip looking for roots as performance
    // optimization.
    // TODO(shyshkov): Clean this up once priority fusion is fully launched.
    if (computation->IsFusionComputation()) {
      roots_ = FindRoots(computation);
    }
  }

  static absl::InlinedVector<HloInstructionAdaptor, 2> FindRoots(
      const HloComputation* computation) {
    absl::InlinedVector<HloInstructionAdaptor, 2> roots;

    std::function<void(const HloInstruction*)> get_roots;
    absl::flat_hash_set<HloInstructionAdaptor> roots_set;
    get_roots = [&](const HloInstruction* instr) {
      if (instr->opcode() == HloOpcode::kTuple) {
        for (const auto* operand : instr->operands()) {
          get_roots(operand);
        }
      } else {
        HloInstructionAdaptor wrapped{*instr};
        if (roots_set.insert(wrapped).second) {
          roots.push_back(wrapped);
        }
      }
    };
    get_roots(computation->root_instruction());

    return roots;
  }

  bool ContainsInstruction(HloInstructionAdaptor instruction) const override {
    return instruction.instruction().parent() == computation_;
  }

  absl::InlinedVector<HloInstructionAdaptor, 2> GetRoots() const override {
    CHECK(!roots_.empty())
        << "No roots found in the computation. HloFusionAdaptor was likely "
           "created for a non-fusion computation: "
        << computation_->ToString();

    return roots_;
  }

  absl::InlinedVector<HloInstructionAdaptor, 2> MakeInstructionPostOrder()
      const override {
    auto post_order = computation_->MakeInstructionPostOrder();

    absl::InlinedVector<HloInstructionAdaptor, 2> result;
    result.reserve(post_order.size() - computation_->num_parameters());

    for (auto* instr : post_order) {
      // Skip parameter and root tuple as FusionAdaptor hides their existence.
      // HloInstructionAdaptor will look through them and return operands
      // outside of the computation if necessary. We don't expect to see any
      // internal tuples, but the other logic only handles root tuples
      // explicitly.
      if (instr->opcode() == HloOpcode::kParameter ||
          (instr->opcode() == HloOpcode::kTuple && instr->IsRoot())) {
        continue;
      }
      result.emplace_back(*instr);
    }
    return result;
  }

  std::string ToString() const override { return computation_->ToString(); }

 private:
  const HloComputation* computation_;
  absl::InlinedVector<HloInstructionAdaptor, 2> roots_;
};

}  // namespace

std::unique_ptr<HloFusionAdaptor> HloFusionAdaptor::ForInstruction(
    const HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kFusion) {
    return ForComputation(instruction->fused_instructions_computation());
  }
  return std::make_unique<SingleInstructionFusion>(instruction);
}

std::unique_ptr<HloFusionAdaptor> HloFusionAdaptor::ForComputation(
    const HloComputation* computation) {
  return std::make_unique<HloComputationFusion>(computation);
}

absl::InlinedVector<HloInstructionAdaptor, 2>
HloInstructionAdaptor::GetOperands() const {
  absl::InlinedVector<HloInstructionAdaptor, 2> operands;
  if (instruction_->opcode() == HloOpcode::kParameter) {
    // The only time this should happen is when a fusion has a parameter
    // that is also a root. This probably never makes sense, but it technically
    // is valid HLO, so we support it by treating the parameter as an identity
    // function in this context.
    auto operand = ResolveOperand(instruction_);
    if (operand != instruction_) {
      operands.emplace_back(*operand);
    }
  } else {
    for (const auto* operand : instruction_->operands()) {
      operands.emplace_back(*ResolveOperand(operand));
    }
  }
  return operands;
}

HloInstructionAdaptor HloInstructionAdaptor::GetOperand(int index) const {
  return HloInstructionAdaptor{*ResolveOperand(instruction_->operand(index))};
}

absl::InlinedVector<HloInstructionAdaptor, 2> HloInstructionAdaptor::GetUsers()
    const {
  absl::InlinedVector<HloInstructionAdaptor, 2> users;
  auto add_user = [&](const HloInstruction* instr) {
    users.emplace_back(*instr);
  };

  if (instruction_->IsRoot()) {
    if (auto* fusion = instruction_->parent()->FusionInstruction()) {
      for (auto* user : fusion->users()) {
        ResolveUsers(fusion, user, add_user);
      }
    }
  }

  for (auto* user : instruction_->users()) {
    ResolveUsers(instruction_, user, add_user);
  }

  return users;
}

bool operator==(const HloInstructionAdaptor& lhs,
                const HloInstructionAdaptor& rhs) {
  return lhs.instruction_->GetModule() == rhs.instruction_->GetModule() &&
         lhs.instruction_->unique_id() == rhs.instruction_->unique_id();
}

namespace {
void HloBfsTraversal(
    absl::Span<const HloInstructionAdaptor> roots,
    const HloFusionAdaptor& fusion,
    const std::function<TraversalResult(HloInstructionAdaptor node)>&
        visit_node,
    const std::function<void(HloInstructionAdaptor producer)>& visit_arg,
    bool visit_operands) {
  absl::flat_hash_set<HloInstructionAdaptor> visited;
  std::queue<HloInstructionAdaptor> q;
  auto enqueue = [&](const HloInstructionAdaptor& node) {
    const auto& adjacent_nodes =
        visit_operands ? node.GetOperands() : node.GetUsers();
    for (const auto& node : adjacent_nodes) {
      if (visited.insert(node).second) {
        if (fusion.ContainsInstruction(node)) {
          q.push(node);
        } else {
          visit_arg(node);
        }
      }
    }
  };
  for (auto root : roots) {
    if (visited.insert(root).second) {
      q.push(root);
    }
  }
  while (!q.empty()) {
    HloInstructionAdaptor node = q.front();
    q.pop();
    switch (visit_node(node)) {
      case TraversalResult::kAdvance:
        enqueue(node);
        break;
      case TraversalResult::kInterrupt:
        return;
      case TraversalResult::kSkip:
        break;
    }
  }
}
}  // namespace

void HloBfsConsumersFirstTraversal(
    absl::Span<const HloInstructionAdaptor> roots,
    const HloFusionAdaptor& fusion,
    const std::function<TraversalResult(HloInstructionAdaptor node)>&
        visit_node,
    const std::function<void(HloInstructionAdaptor producer)>& visit_arg) {
  HloBfsTraversal(roots, fusion, visit_node, visit_arg,
                  /*visit_operands=*/true);
}

void HloBfsProducersFirstTraversal(
    absl::Span<const HloInstructionAdaptor> producers,
    const HloFusionAdaptor& fusion,
    const std::function<TraversalResult(HloInstructionAdaptor node)>&
        visit_node) {
  HloBfsTraversal(
      producers, fusion, visit_node, [](HloInstructionAdaptor) {},
      /*visit_operands=*/false);
}

void FindFusionArguments(
    const HloFusionAdaptor& fusion,
    const std::function<void(HloInstructionAdaptor param)>& visit) {
  HloBfsConsumersFirstTraversal(
      fusion.GetRoots(), fusion,
      [&](HloInstructionAdaptor) { return TraversalResult::kAdvance; }, visit);
}

bool HloAnyOf(absl::Span<const HloInstructionAdaptor> roots,
              const HloFusionAdaptor& fusion,
              const std::function<bool(HloInstructionAdaptor node)>& visit,
              bool visit_operands) {
  return HloFindIf(roots, fusion, visit, visit_operands).has_value();
}

std::optional<HloInstructionAdaptor> HloFindIf(
    absl::Span<const HloInstructionAdaptor> roots,
    const HloFusionAdaptor& fusion,
    const std::function<bool(HloInstructionAdaptor node)>& visit,
    bool visit_operands) {
  std::optional<HloInstructionAdaptor> result = std::nullopt;
  HloBfsTraversal(
      roots, fusion,
      [&](HloInstructionAdaptor node) {
        if (visit(node)) {
          result = node;
          return TraversalResult::kInterrupt;
        }
        return TraversalResult::kAdvance;
      },
      [](HloInstructionAdaptor) {}, visit_operands);
  return result;
}

}  // namespace gpu
}  // namespace xla
