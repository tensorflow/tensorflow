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
#include "xla/hlo/utils/hlo_traversal.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {
namespace {

template <typename F>
void ResolveUsers(const HloInstruction* value, const HloInstruction* user,
                  const HloFusionAdaptor& fusion_adaptor, F&& add_user) {
  if (user->opcode() == HloOpcode::kTuple && user->IsRoot()) {
    if (auto* fusion = user->parent()->FusionInstruction()) {
      // Skip through the tuple -> get-tuple-element ops and directly go to the
      // "real" users.
      for (const auto* fusion_user : fusion->users()) {
        if (fusion_user->opcode() == HloOpcode::kGetTupleElement) {
          for (const auto* gte_user : fusion_user->users()) {
            ResolveUsers(fusion_user, gte_user, fusion_adaptor, add_user);
          }
        } else if (fusion_adaptor.ContainsInstruction(value)) {
          add_user(fusion_user);
        }
      }
    }
    return;
  }

  if (fusion_adaptor.ContainsInstruction(user)) {
    add_user(user);
    return;
  }

  if (user->opcode() == HloOpcode::kFusion &&  // Not a nested fusion.
      fusion_adaptor.ContainsInstruction(user->fused_expression_root())) {
    // Add users of the computation's parameter.
    auto* param = user->fused_parameter(user->operand_index(value));
    for (const auto* param_user : param->users()) {
      add_user(param_user);
    }
  }
}

const HloInstruction* ResolveOperand(const HloInstruction* operand,
                                     const HloFusionAdaptor& fusion_adaptor) {
  // Deal with multi-output fusion operands, which are reached via a
  // get-tuple-element op.
  if (operand->opcode() == HloOpcode::kGetTupleElement &&
      operand->operand(0)->opcode() == HloOpcode::kFusion) {
    HloInstruction* root = operand->operand(0)->fused_expression_root();
    if (root->opcode() == HloOpcode::kTuple &&
        fusion_adaptor.ContainsInstruction(root)) {
      return root->operand(operand->tuple_index());
    }
  }

  if (!fusion_adaptor.ContainsInstruction(operand)) {
    if (operand->opcode() == HloOpcode::kFusion &&  // Not a nested fusion.
        fusion_adaptor.ContainsInstruction(operand->fused_expression_root())) {
      return operand->fused_expression_root();
    }
    return operand;
  }

  if (operand->opcode() == HloOpcode::kParameter) {
    if (auto* fusion = operand->parent()->FusionInstruction()) {
      return ResolveOperand(fusion->operand(operand->parameter_number()),
                            fusion_adaptor);
    }
  }
  return operand;
}
}  // namespace

// An interface to abstract away the difference between a single instruction
// and a fusion instruction with all it's computations.
class HloFusionInstructionAdaptor {
 public:
  virtual ~HloFusionInstructionAdaptor() = default;
  // Returns true if the given 'instruction' is either the adapted instruction
  // or contained in its computation.
  virtual bool ContainsInstruction(const HloInstruction* instruction) const = 0;
  // If it is a regular multi-output fusion, the order of the returned roots
  // matches the order of the tuple elements of the tuple root of the fusion
  // computation. We do not deduplicate fusion roots.
  virtual absl::InlinedVector<HloInstructionAdaptor, 2> GetRoots() const = 0;
  // Returns the operands of the adapted instruction.
  virtual absl::InlinedVector<const HloInstruction*, 2> GetParameters()
      const = 0;
  // Returns the adapted instruction.
  virtual const HloInstruction& FusionInstruction() const = 0;
  // Returns the single instruction or the instructions of the computations, in
  // post order.
  virtual absl::InlinedVector<HloInstructionAdaptor, 2>
  MakeInstructionPostOrder() const = 0;
  // Calls 'fn' the single instruction or all instructions in the (potentially
  // nested) computations, in some order.
  virtual void ForEach(
      const std::function<void(HloInstructionAdaptor)>& fn) const = 0;
  virtual std::string ToString() const = 0;
};

namespace {

class SingleInstructionFusion : public HloFusionInstructionAdaptor {
 public:
  explicit SingleInstructionFusion(const HloInstruction* instruction,
                                   const HloFusionAdaptor* parent)
      : instruction_(instruction), parent_(parent) {
    CHECK_NE(instruction->opcode(), HloOpcode::kFusion)
        << "Use HloComputationFusion";
  }

  bool ContainsInstruction(const HloInstruction* instruction) const override {
    return instruction == instruction_;
  }

  absl::InlinedVector<HloInstructionAdaptor, 2> GetRoots() const override {
    return {HloInstructionAdaptor{*instruction_, parent_}};
  }

  absl::InlinedVector<const HloInstruction*, 2> GetParameters() const override {
    const auto& operands = instruction_->operands();
    return absl::InlinedVector<const HloInstruction*, 2>(operands.begin(),
                                                         operands.end());
  }

  const HloInstruction& FusionInstruction() const override {
    return *instruction_;
  }

  absl::InlinedVector<HloInstructionAdaptor, 2> MakeInstructionPostOrder()
      const override {
    return {HloInstructionAdaptor{*instruction_, parent_}};
  }

  void ForEach(
      const std::function<void(HloInstructionAdaptor)>& fn) const override {
    fn(HloInstructionAdaptor{*instruction_, parent_});
  }

  std::string ToString() const override { return instruction_->ToString(); }

 private:
  const HloInstruction* instruction_;
  const HloFusionAdaptor* parent_;
};

class HloComputationFusion : public HloFusionInstructionAdaptor {
 public:
  explicit HloComputationFusion(const HloComputation* computation,
                                const HloFusionAdaptor* parent)
      : computation_(computation), parent_(parent) {
    // `FindNonTrivialHero` only calls `ContainsInstruction` and doesn't use
    // information about roots, so we can skip looking for roots as performance
    // optimization.
    // TODO(shyshkov): Clean this up once priority fusion is fully launched.
    CHECK(computation->IsFusionComputation());
    roots_ = FindRoots(computation);
  }

  absl::InlinedVector<HloInstructionAdaptor, 2> FindRoots(
      const HloComputation* computation) {
    absl::InlinedVector<HloInstructionAdaptor, 2> roots;

    std::function<void(const HloInstruction*)> get_roots;
    get_roots = [&](const HloInstruction* instr) {
      if (instr->opcode() == HloOpcode::kTuple) {
        for (const auto* operand : instr->operands()) {
          get_roots(operand);
        }
      } else {
        HloInstructionAdaptor wrapped{*instr, parent_};
        roots.push_back(wrapped);
      }
    };
    get_roots(computation->root_instruction());

    return roots;
  }

  bool ContainsInstruction(const HloInstruction* instruction) const override {
    return instruction->parent() == computation_;
  }

  absl::InlinedVector<HloInstructionAdaptor, 2> GetRoots() const override {
    CHECK(!roots_.empty())
        << "No roots found in the computation. HloFusionAdaptor was likely "
           "created for a non-fusion computation: "
        << computation_->ToString();

    return roots_;
  }

  absl::InlinedVector<const HloInstruction*, 2> GetParameters() const override {
    const auto& operands = computation_->FusionInstruction()->operands();
    return absl::InlinedVector<const HloInstruction*, 2>(operands.begin(),
                                                         operands.end());
  }

  const HloInstruction& FusionInstruction() const override {
    return *computation_->FusionInstruction();
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
      result.emplace_back(*instr, parent_);
    }
    return result;
  }

  void ForEach(
      const std::function<void(HloInstructionAdaptor)>& fn) const override {
    for (const HloInstruction* instr : computation_->instructions()) {
      // HloFusionAdaptor hides existence of parameters, tuples and gte
      // instructions.
      if (instr->opcode() == HloOpcode::kParameter ||
          instr->opcode() == HloOpcode::kTuple ||
          instr->opcode() == HloOpcode::kGetTupleElement) {
        continue;
      }
      fn(HloInstructionAdaptor{*instr, parent_});
    }
  }

  std::string ToString() const override { return computation_->ToString(); }

 private:
  const HloComputation* computation_;
  absl::InlinedVector<HloInstructionAdaptor, 2> roots_;
  const HloFusionAdaptor* parent_;
};

}  // namespace

HloFusionAdaptor::~HloFusionAdaptor() = default;

HloFusionAdaptor::HloFusionAdaptor(bool with_extra_outputs)
    : with_extra_outputs_(with_extra_outputs) {}

/*static*/
std::unique_ptr<HloFusionAdaptor> HloFusionAdaptor::ForInstruction(
    const HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kFusion) {
    return ForComputation(instruction->fused_instructions_computation());
  }

  auto fusion_adaptor = absl::WrapUnique(new HloFusionAdaptor);
  fusion_adaptor->AddInstruction(instruction);
  return fusion_adaptor;
}

/*static*/
std::unique_ptr<HloFusionAdaptor> HloFusionAdaptor::ForProducerConsumer(
    const HloInstruction* producer, const HloInstruction* consumer,
    bool with_extra_outputs) {
  auto fusion_adaptor =
      absl::WrapUnique(new HloFusionAdaptor(with_extra_outputs));
  fusion_adaptor->AddInstruction(producer);
  fusion_adaptor->AddInstruction(consumer);
  return fusion_adaptor;
}

/*static*/
std::unique_ptr<HloFusionAdaptor> HloFusionAdaptor::ForComputation(
    const HloComputation* computation) {
  auto fusion_adaptor = absl::WrapUnique(new HloFusionAdaptor);
  fusion_adaptor->AddComputation(computation);
  return fusion_adaptor;
}

bool HloFusionAdaptor::ContainsInstruction(
    HloInstructionAdaptor instruction) const {
  return ContainsInstruction(&instruction.instruction());
}

bool HloFusionAdaptor::ContainsInstruction(
    const HloInstruction* instruction) const {
  return absl::c_any_of(fusion_instructions_, [&](const auto& adaptor) {
    return adaptor->ContainsInstruction(instruction);
  });
}

absl::InlinedVector<HloInstructionAdaptor, 2> HloFusionAdaptor::GetRoots()
    const {
  auto roots = fusion_instructions_.back()->GetRoots();
  if (fusion_instructions_.size() == 1) {
    return roots;
  }
  CHECK_EQ(fusion_instructions_.size(), 2);
  auto producer_roots = fusion_instructions_[0]->GetRoots();
  const HloInstruction& producer_fusion =
      fusion_instructions_[0]->FusionInstruction();
  const HloInstruction& consumer_fusion =
      fusion_instructions_.back()->FusionInstruction();

  // Check whether there are fusion roots that are parameters which will be
  // replaced by a producer fusion root.
  for (auto& root : roots) {
    if (root.opcode() != HloOpcode::kParameter) {
      continue;
    }
    const HloInstruction* operand =
        consumer_fusion.operand(root.instruction().parameter_number());
    int64_t root_index = 0;
    if (operand->opcode() == HloOpcode::kGetTupleElement) {
      root_index = operand->tuple_index();
      operand = operand->operand(0);
    }
    if (operand == &producer_fusion) {
      root = producer_roots[root_index];
    }
  }

  if (!with_extra_outputs_) {
    return roots;
  }

  // Also add the roots of the producer fusion if they are used outside of the
  // merged fusion computations. Skip roots that are parameters.
  absl::flat_hash_set<int64_t> root_indices_with_outside_usage;
  for (HloInstruction* instr : producer_fusion.users()) {
    bool has_outside_user = false;
    int64_t root_index = 0;
    if (instr->opcode() == HloOpcode::kGetTupleElement) {
      for (HloInstruction* user : instr->users()) {
        if (user != &consumer_fusion) {
          root_index = instr->tuple_index();
          has_outside_user = true;
          break;
        }
      }
    } else if (instr != &consumer_fusion) {
      has_outside_user = true;
    }
    if (has_outside_user) {
      root_indices_with_outside_usage.insert(root_index);
    }
  }
  for (int64_t i = 0; i < producer_roots.size(); ++i) {
    if (!root_indices_with_outside_usage.contains(i)) {
      continue;
    }
    // Also check the special case that the root is a parameter. We never fuse a
    // parameter, instead we would rewire users of such a root to the
    // corresponding fusion operand.
    if (producer_roots[i].opcode() != HloOpcode::kParameter) {
      roots.push_back(producer_roots[i]);
    }
  }
  return roots;
}

absl::InlinedVector<const HloInstruction*, 2> HloFusionAdaptor::GetParameters()
    const {
  if (fusion_instructions_.size() == 1) {
    return fusion_instructions_.back()->GetParameters();
  }
  CHECK_EQ(fusion_instructions_.size(), 2);
  absl::InlinedVector<const HloInstruction*, 2> combined_parameters;
  const HloInstruction& producer_fusion =
      fusion_instructions_[0]->FusionInstruction();
  for (const auto& param : fusion_instructions_.back()->GetParameters()) {
    const HloInstruction* operand = param;
    if (operand->opcode() == HloOpcode::kGetTupleElement) {
      operand = operand->operand(0);
    }
    // Check whether 'param' is a user of the producer fusion.
    if (operand != &producer_fusion) {
      combined_parameters.push_back(param);
    }
  }
  absl::flat_hash_set<const HloInstruction*> params(combined_parameters.begin(),
                                                    combined_parameters.end());
  auto producer_roots = fusion_instructions_[0]->GetRoots();
  absl::flat_hash_set<const HloInstruction*> parameters_to_skip;
  // Skip parameters that have just have a root user. Those will not be fused.
  for (const auto& root : producer_roots) {
    if (root.opcode() == HloOpcode::kParameter) {
      // If the root instruction is both a parameter and the fusion instruction,
      // then the producer fusion is a single instruction fusion of a parameter
      // instruction. In that case, the parameter number refers to a parameter
      // in the parent computation, and we mustn't query its operands.
      if (&root.instruction() == &producer_fusion) {
        parameters_to_skip.insert(&producer_fusion);
      } else if (root.instruction().user_count() <= 1) {
        parameters_to_skip.insert(
            producer_fusion.operand(root.instruction().parameter_number()));
      }
    }
  }
  for (auto param : fusion_instructions_[0]->GetParameters()) {
    if (!parameters_to_skip.contains(param) && params.insert(param).second) {
      combined_parameters.push_back(param);
    }
  }
  return combined_parameters;
}

absl::InlinedVector<HloInstructionAdaptor, 2>
HloFusionAdaptor::MakeInstructionPostOrder() const {
  absl::InlinedVector<HloInstructionAdaptor, 2> result_post_order;

  for (const auto& fusion_instruction : fusion_instructions_) {
    absl::c_move(fusion_instruction->MakeInstructionPostOrder(),
                 std::back_inserter(result_post_order));
  }

  return result_post_order;
}

void HloFusionAdaptor::ForEach(
    const std::function<void(HloInstructionAdaptor)>& fn) const {
  for (const auto& fusion_instruction : fusion_instructions_) {
    fusion_instruction->ForEach(fn);
  }
}

std::string HloFusionAdaptor::ToString() const {
  std::ostringstream ss;
  for (const auto& fusion_instruction : fusion_instructions_) {
    ss << fusion_instruction->ToString() << "\n";
  }
  return ss.str();
}

void HloFusionAdaptor::AddInstruction(const HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kFusion) {
    AddComputation(instruction->fused_instructions_computation());
  } else {
    fusion_instructions_.push_back(
        std::make_unique<SingleInstructionFusion>(instruction, this));
  }
}

void HloFusionAdaptor::AddComputation(const HloComputation* computation) {
  fusion_instructions_.push_back(
      std::make_unique<HloComputationFusion>(computation, this));
}

absl::InlinedVector<HloInstructionAdaptor, 2>
HloInstructionAdaptor::GetOperands() const {
  absl::InlinedVector<HloInstructionAdaptor, 2> operands;
  if (instruction_->opcode() == HloOpcode::kParameter) {
    // The only time this should happen is when a fusion has a parameter
    // that is also a root. This probably never makes sense, but it technically
    // is valid HLO, so we support it by treating the parameter as an identity
    // function in this context.
    auto operand = ResolveOperand(instruction_, *parent_);
    if (operand != instruction_) {
      operands.emplace_back(*operand, parent_);
    }
  } else {
    for (const auto* operand : instruction_->operands()) {
      operands.emplace_back(*ResolveOperand(operand, *parent_), parent_);
    }
  }
  return operands;
}

HloInstructionAdaptor::HloInstructionAdaptor(const HloInstruction& instruction,
                                             const HloFusionAdaptor* parent)
    : instruction_(&instruction), parent_(parent) {
  CHECK_NE(parent, nullptr) << "Parent fusion adaptor must not be null";
}

HloInstructionAdaptor HloInstructionAdaptor::GetOperand(int index) const {
  return HloInstructionAdaptor{
      *ResolveOperand(instruction_->operand(index), *parent_), parent_};
}

absl::InlinedVector<HloInstructionAdaptor, 2> HloInstructionAdaptor::GetUsers()
    const {
  absl::InlinedVector<HloInstructionAdaptor, 2> users;
  auto add_user = [&](const HloInstruction* instr) {
    users.emplace_back(*instr, parent_);
  };

  if (instruction_->IsRoot()) {
    if (auto* fusion = instruction_->parent()->FusionInstruction()) {
      for (auto* user : fusion->users()) {
        ResolveUsers(fusion, user, *parent_, add_user);
      }
    }
  }

  for (auto* user : instruction_->users()) {
    ResolveUsers(instruction_, user, *parent_, add_user);
  }

  return users;
}

bool operator==(const HloInstructionAdaptor& lhs,
                const HloInstructionAdaptor& rhs) {
  return lhs.instruction_->GetModule() == rhs.instruction_->GetModule() &&
         lhs.instruction_->unique_id() == rhs.instruction_->unique_id();
}

bool operator!=(const HloInstructionAdaptor& lhs,
                const HloInstructionAdaptor& rhs) {
  return !(lhs == rhs);
}

namespace {
void HloBfsTraversal(
    absl::Span<const HloInstructionAdaptor> roots,
    const HloFusionAdaptor& fusion,
    const std::function<TraversalResult(HloInstructionAdaptor node)>&
        visit_node,
    bool visit_operands) {
  absl::flat_hash_set<HloInstructionAdaptor> visited;
  std::queue<HloInstructionAdaptor> q;
  auto enqueue = [&](const HloInstructionAdaptor& node) {
    const auto& adjacent_nodes =
        visit_operands ? node.GetOperands() : node.GetUsers();
    for (const auto& node : adjacent_nodes) {
      if (fusion.ContainsInstruction(node) && visited.insert(node).second) {
        q.push(node);
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
        visit_node) {
  HloBfsTraversal(roots, fusion, visit_node,
                  /*visit_operands=*/true);
}

void HloBfsProducersFirstTraversal(
    absl::Span<const HloInstructionAdaptor> producers,
    const HloFusionAdaptor& fusion,
    const std::function<TraversalResult(HloInstructionAdaptor node)>&
        visit_node) {
  HloBfsTraversal(producers, fusion, visit_node,
                  /*visit_operands=*/false);
}

bool HloBfsAnyOf(absl::Span<const HloInstructionAdaptor> roots,
                 const HloFusionAdaptor& fusion,
                 const std::function<bool(HloInstructionAdaptor node)>& visit,
                 bool visit_operands) {
  return HloBfsFindIf(roots, fusion, visit, visit_operands).has_value();
}

bool HloBfsAnyOf(absl::Span<const HloInstruction* const> roots,
                 const std::function<bool(const HloInstruction* node)>& visit,
                 bool visit_operands) {
  return HloBfsFindIf(roots, visit, visit_operands).has_value();
}

std::optional<HloInstructionAdaptor> HloBfsFindIf(
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
      visit_operands);
  return result;
}

std::vector<const HloInstruction*> HloFindAllImpl(
    absl::Span<const HloInstruction* const> roots,
    const std::function<bool(const HloInstruction* node)>& visit,
    bool visit_operands, bool find_first_only = false) {
  std::vector<const HloInstruction*> result;
  absl::flat_hash_set<const HloInstruction*> visited;
  std::queue<const HloInstruction*> q;
  auto enqueue = [&](const HloInstruction* node) {
    if (visit_operands) {
      for (const HloInstruction* operand : node->operands()) {
        if (visited.insert(operand).second) {
          q.push(operand);
        }
      }
    } else {
      for (const HloInstruction* operand : node->users()) {
        if (visited.insert(operand).second) {
          q.push(operand);
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
    const HloInstruction* node = q.front();
    q.pop();
    if (visit(node)) {
      result.push_back(node);
      if (find_first_only) {
        return result;
      }
    }
    enqueue(node);
  }
  return result;
}

std::optional<const HloInstruction*> HloBfsFindIf(
    absl::Span<const HloInstruction* const> roots,
    const std::function<bool(const HloInstruction* node)>& visit,
    bool visit_operands) {
  auto result = HloFindAllImpl(roots, visit, visit_operands,
                               /*find_first_only=*/true);
  if (result.empty()) {
    return std::nullopt;
  }
  return result[0];
}

std::vector<const HloInstruction*> HloBfsFindAll(
    absl::Span<const HloInstruction* const> roots,
    const std::function<bool(const HloInstruction* node)>& visit,
    bool visit_operands) {
  std::vector<const HloInstruction*> result;
  return HloFindAllImpl(roots, visit, visit_operands);
}

std::vector<HloInstructionAdaptor> HloFindUseChain(HloInstructionAdaptor parent,
                                                   HloInstructionAdaptor root) {
  absl::flat_hash_set<HloInstructionAdaptor> visited;
  std::vector<HloInstructionAdaptor> result;
  std::function<bool(HloInstructionAdaptor)> visit;
  visit = [&](HloInstructionAdaptor node) {
    if (node == root) return true;
    for (const auto& user : node.GetUsers()) {
      if (visited.insert(user).second && visit(user)) {
        result.push_back(user);
        return true;
      }
    }
    return false;
  };
  if (visit(parent)) {
    result.push_back(parent);
    std::reverse(result.begin(), result.end());
  } else {
    result.clear();
  }
  return result;
}

}  // namespace xla
