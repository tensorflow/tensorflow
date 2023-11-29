/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <queue>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
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
  } else {
    fn(user);
  }
}

const HloInstruction* ResolveOperand(const HloInstruction* operand) {
  if (operand->opcode() == HloOpcode::kFusion) {
    return operand->fused_expression_root();
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

 private:
  HloInstructionAdaptor instruction_;
};

class HloComputationFusion : public HloFusionAdaptor {
 public:
  explicit HloComputationFusion(const HloComputation* computation)
      : computation_(computation) {
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
          roots_.push_back(wrapped);
        }
      }
    };
    get_roots(computation->root_instruction());
  }

  bool ContainsInstruction(HloInstructionAdaptor instruction) const override {
    return instruction.instruction().parent() == computation_;
  }

  absl::InlinedVector<HloInstructionAdaptor, 2> GetRoots() const override {
    return roots_;
  }

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

void HloBfsConsumersFirstTraversal(
    absl::Span<const HloInstructionAdaptor> roots,
    const HloFusionAdaptor& fusion,
    const std::function<TraversalResult(HloInstructionAdaptor node)>& visit,
    const std::function<void(HloInstructionAdaptor producer)>& visit_arg) {
  absl::flat_hash_set<HloInstructionAdaptor> visited;
  std::queue<HloInstructionAdaptor> q;
  auto enqueue_operands = [&](const HloInstructionAdaptor& node) {
    for (auto operand : node.GetOperands()) {
      if (visited.insert(operand).second) {
        if (fusion.ContainsInstruction(operand)) {
          q.push(operand);
        } else {
          visit_arg(operand);
        }
      }
    }
  };
  for (auto root : roots) {
    q.push(root);
  }
  while (!q.empty()) {
    HloInstructionAdaptor node = q.front();
    q.pop();
    switch (visit(node)) {
      case TraversalResult::kVisitOperands:
        enqueue_operands(node);
        break;
      case TraversalResult::kAbortTraversal:
        return;
      case TraversalResult::kDoNotVisitOperands:
        break;
    }
  }
}

void FindFusionArguments(
    const HloFusionAdaptor& fusion,
    const std::function<void(HloInstructionAdaptor param)>& visit) {
  HloBfsConsumersFirstTraversal(
      fusion.GetRoots(), fusion,
      [&](HloInstructionAdaptor) { return TraversalResult::kVisitOperands; },
      visit);
}

bool HloAnyOf(absl::Span<const HloInstructionAdaptor> roots,
              const HloFusionAdaptor& fusion,
              const std::function<bool(HloInstructionAdaptor node)>& visit) {
  return HloFindIf(roots, fusion, visit).has_value();
}

std::optional<HloInstructionAdaptor> HloFindIf(
    absl::Span<const HloInstructionAdaptor> roots,
    const HloFusionAdaptor& fusion,
    const std::function<bool(HloInstructionAdaptor node)>& visit) {
  std::optional<HloInstructionAdaptor> result = std::nullopt;
  HloBfsConsumersFirstTraversal(roots, fusion, [&](HloInstructionAdaptor node) {
    if (visit(node)) {
      result = node;
      return TraversalResult::kAbortTraversal;
    }
    return TraversalResult::kVisitOperands;
  });
  return result;
}

}  // namespace gpu
}  // namespace xla
