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

#include "tensorflow/compiler/xla/service/hlo_computation.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <list>
#include <queue>
#include <set>
#include <sstream>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using absl::StrCat;

std::unique_ptr<HloComputation> HloComputation::Builder::Build(
    HloInstruction* root_instruction) {
  int parameter_count = 0;
  for (auto& instruction : instructions_) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      parameter_count++;
    }
  }
  // If root_instruction is not specified use the last added instruction.
  HloInstruction* root =
      root_instruction ? root_instruction : last_added_instruction_;
  CHECK_NE(nullptr, root);
  return absl::WrapUnique(new HloComputation(
      name_, parameter_count, &instructions_, root, fusion_instruction_));
}

HloComputation::HloComputation(
    const string& name, int parameter_count,
    std::vector<std::unique_ptr<HloInstruction>>* instructions,
    HloInstruction* root_instruction, HloInstruction* fusion_instruction)
    : name_(NameUniquer::GetSanitizedName(name)),
      unique_id_(-1),
      root_instruction_(root_instruction),
      fusion_instruction_(fusion_instruction) {
  param_instructions_.resize(parameter_count, nullptr);
  bool root_found = false;
  for (auto& instruction : *instructions) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      int64 param_no = instruction->parameter_number();
      CHECK(param_no >= 0 && param_no < parameter_count)
          << "\nERROR: invalid parameter number.  Expected [0, "
          << parameter_count << "), got " << param_no;
      CHECK(param_instructions_[param_no] == nullptr)
          << "\nERROR: parameter number " << param_no
          << " already allocated in this computation";
      param_instructions_[param_no] = instruction.get();
    }
    root_found |= instruction.get() == root_instruction_;
    AddInstructionInternal(std::move(instruction));
  }
  CHECK(root_found)
      << "\nERROR: root instruction is not present in computation.";
}

HloInstruction* HloComputation::AddInstruction(
    std::unique_ptr<HloInstruction> instruction) {
  CHECK(instruction->opcode() != HloOpcode::kParameter)
      << "Parameter instructions cannot be added to a computation after "
      << "it has been built";
  return AddInstructionInternal(std::move(instruction));
}

HloInstruction* HloComputation::AddInstructionInternal(
    std::unique_ptr<HloInstruction> instruction) {
  if (parent() != nullptr) {
    instruction->UniquifyName(&parent()->instruction_name_uniquer());
    instruction->SetUniqueId(parent()->NewUniqueInstructionId());
  }
  instruction->set_parent(this);
  HloInstruction* pinst = instruction.get();
  instruction_iterators_[pinst] =
      instructions_.insert(instructions_.end(), std::move(instruction));
  return pinst;
}

HloInstruction* HloComputation::AddParameter(
    std::unique_ptr<HloInstruction> instruction) {
  CHECK(instruction->opcode() == HloOpcode::kParameter);
  CHECK(IsFusionComputation());
  CHECK(fusion_instruction_->operand_count() == param_instructions_.size());
  instruction->set_parent(this);
  param_instructions_.push_back(instruction.get());
  AddInstructionInternal(std::move(instruction));
  return instructions_.back().get();
}

Status HloComputation::RemoveParameter(int64 param_no) {
  CHECK_GE(param_no, 0);
  CHECK_LT(param_no, param_instructions_.size());
  CHECK(IsFusionComputation());
  HloInstruction* param_instruction = param_instructions_[param_no];
  auto param_instruction_iterator = param_instructions_.begin() + param_no;
  param_instructions_.erase(param_instruction_iterator);
  // Throw removed fused parameter instruction away.
  TF_RETURN_IF_ERROR(RemoveInstruction(param_instruction));

  while (param_no < param_instructions_.size()) {
    param_instruction = param_instructions_[param_no];
    HloInstruction* new_instr =
        AddInstructionInternal(HloInstruction::CreateParameter(
            param_no, param_instruction->shape(), StrCat("param_", param_no)));
    TF_RETURN_IF_ERROR(param_instruction->ReplaceAllUsesWith(new_instr));
    param_instructions_[param_no] = new_instr;
    TF_RETURN_IF_ERROR(RemoveInstruction(param_instruction));
    param_no++;
  }

  return Status::OK();
}

Status HloComputation::RemoveUnusedParameters() {
  CHECK(IsFusionComputation());
  int64 removed = 0;
  for (int64 i = 0; i < param_instructions_.size(); ++i) {
    HloInstruction* param_instruction = param_instructions_[i];
    if (param_instruction->user_count() == 0 &&
        param_instruction != root_instruction()) {
      TF_RETURN_IF_ERROR(RemoveInstruction(param_instruction));
      ++removed;
      continue;
    }

    if (removed > 0) {
      const int64 param_no = i - removed;
      HloInstruction* new_instr = AddInstructionInternal(
          HloInstruction::CreateParameter(param_no, param_instruction->shape(),
                                          StrCat("param_", param_no)));
      TF_RETURN_IF_ERROR(param_instruction->ReplaceAllUsesWith(new_instr));
      param_instructions_[param_no] = new_instr;
      TF_RETURN_IF_ERROR(RemoveInstruction(param_instruction));
    }
  }
  param_instructions_.resize(param_instructions_.size() - removed);
  return Status::OK();
}

bool HloComputation::IsRemovable(const HloInstruction* instruction) {
  // If the instruction has control predecessors or successors then we cannot
  // remove the instruction without violating ordering constraints (added, for
  // example, to avert interference due to buffer aliasing).
  if (!instruction->control_predecessors().empty() ||
      !instruction->control_successors().empty()) {
    return false;
  }

  if (instruction->opcode() == HloOpcode::kParameter &&
      !IsFusionComputation()) {
    return false;
  }

  return true;
}

bool HloComputation::HasSideEffect() const {
  for (auto* instruction : instructions()) {
    if (instruction->HasSideEffect()) {
      return true;
    }
  }
  return false;
}

Status HloComputation::RemoveInstructionAndUnusedOperands(
    HloInstruction* instruction) {
  TF_RET_CHECK(root_instruction() != instruction);

  TF_RET_CHECK(instruction->user_count() == 0);
  TF_RET_CHECK(IsRemovable(instruction))
      << "Cannot remove instruction: " << instruction->ToString();
  std::unordered_set<HloInstruction*> removed;
  std::queue<HloInstruction*> worklist;
  worklist.push(instruction);
  while (!worklist.empty()) {
    HloInstruction* item = worklist.front();
    worklist.pop();

    if (removed.count(item) != 0 || item->user_count() != 0 ||
        item == root_instruction() || !IsRemovable(item) ||
        (item->HasSideEffect() && item != instruction)) {
      continue;
    }
    for (int i = 0; i < item->operand_count(); ++i) {
      worklist.push(item->mutable_operand(i));
    }

    TF_RETURN_IF_ERROR(RemoveInstruction(item));
    removed.insert(item);
  }
  return Status::OK();
}

Status HloComputation::RemoveInstruction(HloInstruction* instruction) {
  VLOG(2) << "Removing instruction " << instruction->name()
          << " from computation " << name();
  TF_RET_CHECK(IsRemovable(instruction))
      << "cannot remove instruction: " << instruction->ToString();
  TF_RET_CHECK(root_instruction() != instruction)
      << "cannot remove root instruction " << instruction->name();
  TF_RET_CHECK(instruction->user_count() == 0)
      << "instruction " << instruction->name()
      << " has users and cannot be removed";
  TF_RET_CHECK(instruction->control_predecessors().empty())
      << "instruction " << instruction->name()
      << " has control predecessors and cannot be removed";
  TF_RET_CHECK(instruction->control_successors().empty())
      << "instruction " << instruction->name()
      << " has control successors and cannot be removed";

  auto inst_it = instruction_iterators_.find(instruction);
  TF_RET_CHECK(inst_it != instruction_iterators_.end());
  (*inst_it->second)->set_parent(nullptr);
  instructions_.erase(inst_it->second);
  instruction_iterators_.erase(inst_it);
  return Status::OK();
}

void HloComputation::set_root_instruction(HloInstruction* new_root_instruction,
                                          bool accept_different_shape) {
  // The shape of the root (ignoring layout) is an invariant of the computation
  // for non-fusion cases.
  if (!IsFusionComputation() && !accept_different_shape) {
    CHECK(ShapeUtil::Compatible(new_root_instruction->shape(),
                                root_instruction_->shape()))
        << new_root_instruction->shape() << " is incompatible with "
        << root_instruction_->shape();
  }
  bool root_found = false;
  for (auto& instruction : instructions_) {
    if (new_root_instruction == instruction.get()) {
      root_found = true;
      break;
    }
  }
  DCHECK(root_found);

  root_instruction_ = new_root_instruction;
}

namespace {

// Helper which builds a post order of the HLO call graph.
void ComputeComputationPostOrder(HloComputation* computation,
                                 absl::flat_hash_set<HloComputation*>* visited,
                                 std::vector<HloComputation*>* post_order) {
  if (visited->insert(computation).second) {
    for (auto* instruction : computation->instructions()) {
      for (HloComputation* called_computation :
           instruction->called_computations()) {
        ComputeComputationPostOrder(called_computation, visited, post_order);
      }
    }
    post_order->push_back(computation);
  }
}

}  // namespace

void HloComputation::ComputeInstructionPostOrder(
    const HloComputation::ChannelDependencyMap& channel_dependency_map,
    std::vector<HloInstruction*>* post_order, HloInstruction* root,
    absl::flat_hash_map<HloInstruction*, VisitState>* visited) const {
  std::vector<HloInstruction*> dfs_stack;
  dfs_stack.push_back(root);
  while (!dfs_stack.empty()) {
    const auto current = dfs_stack.back();
    auto it = visited->find(current);
    if (it != visited->end()) {
      if (it->second == kVisited) {
        // Already visited.
        dfs_stack.pop_back();
        continue;
      }
      // Visit this node.
      CHECK_EQ(kVisiting, it->second);
      dfs_stack.pop_back();
      post_order->push_back(current);
      it->second = kVisited;
      continue;
    }

    visited->insert({current, kVisiting});

    // Add the operands to the stack in reverse order so the first operand is
    // processed first. This will produce a more natural ordering and a nicer
    // result for things like HLO stringification.
    const auto& operands = current->operands();
    for (int64 i = operands.size() - 1; i >= 0; --i) {
      dfs_stack.emplace_back(operands[i]);
    }

    for (HloInstruction* op : current->control_predecessors()) {
      dfs_stack.emplace_back(op);
    }

    // Add inputs for send->recv_done dependencies and cross-replica-sum
    // dependencies.
    switch (current->opcode()) {
      case HloOpcode::kRecvDone: {
        auto it = channel_dependency_map.find(current->channel_id());
        if (it != channel_dependency_map.end()) {
          for (HloInstruction* op : it->second) {
            dfs_stack.emplace_back(op);
          }
        }
        break;
      }
      case HloOpcode::kCrossReplicaSum: {
        auto all_reduce_id = current->all_reduce_id();
        if (all_reduce_id) {
          auto it = channel_dependency_map.find(all_reduce_id.value());
          if (it != channel_dependency_map.end()) {
            for (HloInstruction* op : it->second) {
              dfs_stack.emplace_back(op);
            }
          }
        }
        break;
      }
      default:
        break;
    }
  }
}

HloComputation::ChannelDependencyMap
HloComputation::ComputeChannelDependencies() const {
  ChannelDependencyMap channel_dependency_map;
  for (const auto& instruction : instructions_) {
    switch (instruction->opcode()) {
      case HloOpcode::kSend: {
        channel_dependency_map[instruction->channel_id()].push_back(
            instruction.get());
        break;
      }
      case HloOpcode::kCrossReplicaSum: {
        auto all_reduce_id = instruction->all_reduce_id();
        if (all_reduce_id) {
          auto& dependencies = channel_dependency_map[all_reduce_id.value()];
          absl::c_copy(instruction->operands(),
                       std::back_inserter(dependencies));
          absl::c_copy(instruction->control_predecessors(),
                       std::back_inserter(dependencies));
        }
        break;
      }
      default:
        break;
    }
  }
  return channel_dependency_map;
}

std::vector<HloInstruction*> HloComputation::MakeInstructionPostOrder() const {
  auto channel_dependency_map = ComputeChannelDependencies();
  std::vector<HloInstruction*> post_order;
  post_order.reserve(instruction_count());
  std::vector<HloInstruction*> trace_instructions;
  absl::flat_hash_map<HloInstruction*, VisitState> visited;
  visited.reserve(instruction_count());
  for (auto& instruction : instructions_) {
    if (instruction->opcode() == HloOpcode::kTrace) {
      // Trace instructions aren't handled by the DFS visitor. Add trace
      // instructions to the post order at the end (necessarily they have no
      // users).
      trace_instructions.push_back(instruction.get());
    } else if (instruction->users().empty()) {
      ComputeInstructionPostOrder(channel_dependency_map, &post_order,
                                  instruction.get(), &visited);
    }
  }
  post_order.insert(post_order.end(), trace_instructions.begin(),
                    trace_instructions.end());
  CHECK_EQ(instructions_.size(), post_order.size())
      << "number of instructions does not match post order size";
  return post_order;
}

std::vector<HloComputation*> HloComputation::MakeEmbeddedComputationsList()
    const {
  absl::flat_hash_set<HloComputation*> visited;
  std::vector<HloComputation*> post_order;

  // To avoid special handling of this computation, cast away const of
  // 'this'. 'this' is immediately removed from the post order after
  // construction.
  //
  // TODO(b/78350259): This violates const-correctness, since while the original
  // computation is not returned, we still retrieve non-const computations from
  // a const one. Consider also avoiding const for HloComputation, or review XLA
  // for const-correctness of non-HloInstruction* types like this.
  ComputeComputationPostOrder(const_cast<HloComputation*>(this), &visited,
                              &post_order);

  // We don't want to include this computation in the post order.
  CHECK_EQ(this, post_order.back());
  post_order.pop_back();

  return post_order;
}

string HloComputation::ToString(const HloPrintOptions& options) const {
  return ToString(options, MakeInstructionPostOrder());
}

string HloComputation::ToString(
    const HloPrintOptions& options,
    absl::Span<const HloInstruction* const> instruction_order) const {
  CHECK_EQ(instruction_order.size(), instruction_count());

  std::ostringstream s;
  for (int i = 0; i < options.indent_amount(); i++) {
    s << "  ";
  }

  if (!options.is_in_nested_computation()) {
    if (options.print_percent()) {
      s << "%";
    }
    s << name() << " ";
  }

  if (options.print_program_shape()) {
    s << ShapeUtil::HumanString(ComputeProgramShape()) << " ";
  }
  s << "{\n";
  {
    // Print the instructions in this computation.
    HloPrintOptions new_options = options;
    new_options.set_indent_amount(options.indent_amount() + 1)
        .set_is_in_nested_computation(true);
    CanonicalNameMap name_map;
    for (const HloInstruction* instruction : instruction_order) {
      CHECK_EQ(this, instruction->parent());

      for (int i = 0; i < new_options.indent_amount(); i++) {
        s << "  ";
      }
      s << (instruction == root_instruction_ ? "ROOT " : "")
        << instruction->ToStringWithCanonicalNameMap(new_options, &name_map)
        << "\n";
    }
  }

  for (int i = 0; i < options.indent_amount(); i++) {
    s << "  ";
  }
  s << "}";
  return s.str();
}

HloComputationProto HloComputation::ToProto() const {
  HloComputationProto proto;
  CHECK(unique_id_ != -1)
      << "This computation does not have a valid id. Please make sure the "
         "computation is inside a module before dumping it.";
  proto.set_id(unique_id_);
  proto.set_name(name_);
  for (const HloInstruction* instruction : MakeInstructionPostOrder()) {
    HloInstructionProto instruction_proto = instruction->ToProto();
    proto.add_instructions()->Swap(&instruction_proto);
  }
  proto.set_root_id(root_instruction()->unique_id());
  *proto.mutable_program_shape() = ComputeProgramShape().ToProto();
  return proto;
}

/* static */ StatusOr<std::unique_ptr<HloComputation>>
HloComputation::CreateFromProto(
    const HloComputationProto& proto,
    const absl::flat_hash_map<int64, HloComputation*>& computation_map) {
  absl::flat_hash_map<int64, HloInstruction*> instruction_map;
  absl::flat_hash_map<HloInstruction*, int64> to_proto_id;
  std::vector<std::unique_ptr<HloInstruction>> instructions;
  int64 parameter_count = 0;
  for (const HloInstructionProto& instruction_proto : proto.instructions()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloInstruction> instruction,
        HloInstruction::CreateFromProto(instruction_proto, instruction_map,
                                        computation_map));
    if (instruction->opcode() == HloOpcode::kParameter) {
      parameter_count++;
    }
    TF_RET_CHECK(!ContainsKey(instruction_map, instruction_proto.id()));
    instruction_map[instruction_proto.id()] = instruction.get();
    to_proto_id[instruction.get()] = instruction_proto.id();
    instructions.push_back(std::move(instruction));
  }

  TF_RET_CHECK(proto.root_id() != -1);
  TF_RET_CHECK(ContainsKey(instruction_map, proto.root_id()));
  HloInstruction* root = instruction_map.at(proto.root_id());

  // Sort the instructions in the proto id's order.
  std::sort(instructions.begin(), instructions.end(),
            [&](const std::unique_ptr<HloInstruction>& a,
                const std::unique_ptr<HloInstruction>& b) {
              return to_proto_id[a.get()] < to_proto_id[b.get()];
            });

  TF_RETURN_IF_ERROR([&]() -> Status {
    std::vector<bool> parameters_seen(parameter_count);
    int parameters_seen_count = 0;
    for (auto& instruction : instructions) {
      if (instruction->opcode() == HloOpcode::kParameter) {
        int64 param_no = instruction->parameter_number();
        TF_RET_CHECK(param_no >= 0 && param_no < parameter_count)
            << "Invalid parameter number.  Expected [0, " << parameter_count
            << "), got " << param_no;
        TF_RET_CHECK(!parameters_seen[param_no])
            << "Parameter number " << param_no
            << " already allocated in this computation";
        parameters_seen[param_no] = true;
        parameters_seen_count++;
      }
    }
    TF_RET_CHECK(parameters_seen_count == parameter_count)
        << "Not all parameters in range [0, " << parameter_count
        << ") were referenced";
    return Status::OK();
  }());

  auto computation = absl::WrapUnique(
      new HloComputation(proto.name(), parameter_count, &instructions, root,
                         /*fusion_instruction=*/nullptr));
  computation->unique_id_ = proto.id();
  return std::move(computation);
}

void HloComputation::FuseInstructionsInto(
    absl::Span<HloInstruction* const> instructions_to_fuse,
    HloInstruction* fusion_instruction) {
  CHECK_EQ(HloOpcode::kFusion, fusion_instruction->opcode());
  HloInstruction* root = instructions_to_fuse.front();
  TF_CHECK_OK(root->ReplaceAllUsesWith(fusion_instruction));
  if (root == root_instruction()) {
    set_root_instruction(fusion_instruction);
  }
  TF_CHECK_OK(RemoveInstruction(root));
  for (size_t i = 1; i < instructions_to_fuse.size(); ++i) {
    HloInstruction* instruction = instructions_to_fuse[i];
    fusion_instruction->FuseInstruction(instruction);
    if (instruction->user_count() == 0) {
      TF_CHECK_OK(RemoveInstruction(instruction));
    }
  }
}

HloInstruction* HloComputation::CreateFusionInstruction(
    absl::Span<HloInstruction* const> instructions_to_fuse,
    HloInstruction::FusionKind fusion_kind) {
  HloInstruction* root = instructions_to_fuse.front();
  HloInstruction* fusion_instruction = AddInstruction(
      HloInstruction::CreateFusion(root->shape(), fusion_kind, root));
  FuseInstructionsInto(instructions_to_fuse, fusion_instruction);
  return fusion_instruction;
}

StatusOr<HloInstruction*> HloComputation::DeepCopyHelper(
    HloInstruction* instruction, ShapeIndex* index,
    const std::function<
        HloInstruction*(HloInstruction* leaf, const ShapeIndex& leaf_index,
                        HloComputation* computation)>& copy_leaf) {
  if (ShapeUtil::IsTuple(instruction->shape())) {
    std::vector<HloInstruction*> elements;
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(instruction->shape());
         i++) {
      HloInstruction* gte =
          AddInstruction(HloInstruction::CreateGetTupleElement(
              ShapeUtil::GetTupleElementShape(instruction->shape(), i),
              instruction, i));

      index->push_back(i);
      TF_ASSIGN_OR_RETURN(HloInstruction * element,
                          DeepCopyHelper(gte, index, copy_leaf));
      elements.push_back(element);
      index->pop_back();
    }
    return AddInstruction(HloInstruction::CreateTuple(elements));
  }
  if (ShapeUtil::IsToken(instruction->shape())) {
    // Tokens have no on-device representation and cannot be copied. Pass
    // through transparently.
    return instruction;
  }

  // Array shape.
  TF_RET_CHECK(ShapeUtil::IsArray(instruction->shape()));
  return copy_leaf(instruction, *index, this);
}

StatusOr<HloInstruction*> HloComputation::DeepCopyInstruction(
    HloInstruction* instruction, const ShapeTree<bool>* indices_to_copy,
    ShapeTree<HloInstruction*>* copies_added) {
  if (instruction->parent() != this) {
    return FailedPrecondition(
        "Can't deep copy instruction %s: instruction is not in computation %s",
        instruction->name(), name());
  }
  if (indices_to_copy != nullptr &&
      !ShapeUtil::Compatible(instruction->shape(), indices_to_copy->shape())) {
    return FailedPrecondition(
        "Can't deep copy instruction %s: given shape tree of indices to copy "
        "has incompatible shapes: %s vs. %s",
        instruction->name(), ShapeUtil::HumanString(instruction->shape()),
        ShapeUtil::HumanString(indices_to_copy->shape()));
  }

  ShapeIndex index;
  auto copy_leaf = [indices_to_copy, copies_added](
                       HloInstruction* leaf, const ShapeIndex& leaf_index,
                       HloComputation* computation) {
    if (indices_to_copy == nullptr || indices_to_copy->element(leaf_index)) {
      HloInstruction* copy = computation->AddInstruction(
          HloInstruction::CreateUnary(leaf->shape(), HloOpcode::kCopy, leaf));
      if (copies_added != nullptr) {
        *copies_added->mutable_element(leaf_index) = copy;
      }
      return copy;
    }
    // Elements which are not to be copied are passed through
    // transparently.
    return leaf;
  };
  return DeepCopyHelper(instruction, &index, copy_leaf);
}

StatusOr<HloInstruction*> HloComputation::DeepCopyInstructionWithCustomCopier(
    HloInstruction* instruction,
    const std::function<
        HloInstruction*(HloInstruction* leaf, const ShapeIndex& leaf_index,
                        HloComputation* computation)>& copy_leaf) {
  if (instruction->parent() != this) {
    return FailedPrecondition(
        "Can't deep copy instruction %s: instruction is not in computation %s",
        instruction->name(), name());
  }
  ShapeIndex index;
  return DeepCopyHelper(instruction, &index, copy_leaf);
}

ProgramShape HloComputation::ComputeProgramShape() const {
  ProgramShape program_shape;

  for (auto* param_instruction : param_instructions_) {
    *program_shape.add_parameters() = param_instruction->shape();
    *program_shape.add_parameter_names() = param_instruction->name();
  }
  *program_shape.mutable_result() = root_instruction_->shape();

  return program_shape;
}

bool HloComputation::operator==(const HloComputation& other) const {
  if (this == &other) {
    return true;
  }
  std::set<std::pair<const HloInstruction*, const HloInstruction*>> visited;
  std::function<bool(const HloInstruction*, const HloInstruction*)> eq =
      [&visited, &eq](const HloInstruction* a, const HloInstruction* b) {
        // If <a,b> are visited but not identical, the recursion should have
        // been aborted. So, if <a,b> are visited at this point, they must be
        // identical.
        if (visited.count(std::make_pair(a, b)) > 0) {
          return true;
        }
        visited.emplace(a, b);
        return a->Identical(
            *b, eq, [](const HloComputation* a, const HloComputation* b) {
              return *a == *b;
            });
      };
  return eq(root_instruction(), other.root_instruction());
}

Status HloComputation::ReplaceWithNewInstruction(
    HloInstruction* old_instruction,
    std::unique_ptr<HloInstruction> new_instruction) {
  return ReplaceInstruction(old_instruction,
                            AddInstruction(std::move(new_instruction)));
}

Status HloComputation::ReplaceInstruction(HloInstruction* old_instruction,
                                          HloInstruction* new_instruction) {
  TF_RET_CHECK(
      ShapeUtil::Compatible(old_instruction->shape(), new_instruction->shape()))
      << ShapeUtil::HumanString(old_instruction->shape()) << " vs "
      << ShapeUtil::HumanString(new_instruction->shape());

  VLOG(10) << "transformed " << old_instruction->ToString() << " to "
           << new_instruction->ToString();
  // Try to add metadata for HLO instructions that are created to replace
  // existing HLO instructions (e.g. during optimizations). The assumption is
  // that the old instruction and the new instruction would perform the same
  // function, and that they would be correlated to the same TF op. This might
  // not always be correct since HLO optimizations can cross TF op boundaries.
  // But still this seems to be better than nothing.
  if (new_instruction->metadata().op_name().empty()) {
    new_instruction->set_metadata(old_instruction->metadata());
  }
  TF_RETURN_IF_ERROR(old_instruction->ReplaceAllUsesWith(new_instruction));
  return RemoveInstructionAndUnusedOperands(old_instruction);
}

std::vector<HloInstruction*> HloComputation::CollectUnreachableRoots() const {
  std::vector<HloInstruction*> unreachable_roots;
  for (auto* instruction : instructions()) {
    if (instruction->user_count() == 0 &&
        instruction->control_successors().empty() &&
        instruction != root_instruction()) {
      unreachable_roots.push_back(instruction);
    }
  }
  VLOG(3) << "Unreachable roots:"
          << absl::StrJoin(unreachable_roots, "\n\t",
                           [](string* out, const HloInstruction* hlo) {
                             absl::StrAppend(out, hlo->ToString());
                           });
  return unreachable_roots;
}

template <typename HloInstructionPtr>
Status HloComputation::Accept(
    DfsHloVisitorBase<HloInstructionPtr>* visitor) const {
  // Visit unreachable roots. Beware that the visitor might delete the currently
  // visited root, which would invalidate iterators if the unreachable roots
  // weren't computed ahead of time.
  for (HloInstruction* root : CollectUnreachableRoots()) {
    VLOG(3) << "Traversing unreachable root: " << root->ToString();
    // Call FinishVisit only at the end.
    TF_RETURN_IF_ERROR(root->Accept(visitor, /*call_finish_visit=*/false));
  }
  // Visit the computation root instruction last.
  return root_instruction()->Accept(visitor, /*call_finish_visit=*/true);
}

// Explicit instantiations.
template Status HloComputation::Accept(DfsHloVisitor* visitor) const;
template Status HloComputation::Accept(ConstDfsHloVisitor* visitor) const;

Status HloComputation::AcceptWithOperandOrder(
    DfsHloVisitor* visitor,
    const HloInstruction::CompareFunction& operand_order) const {
  // Visit unreachable roots. Beware that the visitor might delete the currently
  // visited root, which would invalidate iterators if the unreachable roots
  // weren't computed ahead of time.
  for (HloInstruction* root : CollectUnreachableRoots()) {
    TF_RETURN_IF_ERROR(
        root->AcceptWithOperandOrder(visitor, operand_order,
                                     /*call_finish_visit=*/false));
  }
  // Visit the computation root instruction last.
  return root_instruction()->AcceptWithOperandOrder(visitor, operand_order,
                                                    /*call_finish_visit=*/true);
}

template <typename HloInstructionPtr>
Status HloComputation::AcceptOrdered(
    DfsHloVisitorBase<HloInstructionPtr>* visitor,
    absl::Span<HloInstruction* const> order) const {
  VLOG(3) << "Accepting visitor with order.";
  for (HloInstruction* root : CollectUnreachableRoots()) {
    TF_RET_CHECK(std::find(order.begin(), order.end(), root) != order.end())
        << root->ToString();
  }
  TF_RET_CHECK(order.size() == instruction_count());
  std::unordered_set<const HloInstruction*> visited;
  for (const HloInstruction* instruction : order) {
    VLOG(3) << "Visiting ordered: " << instruction->ToString();
    TF_RET_CHECK(instruction_iterators_.count(instruction) == 1)
        << "Instruction " << instruction->name() << " is not in computation "
        << name();
    TF_RET_CHECK(visited.count(instruction) == 0)
        << "Instruction " << instruction->name()
        << " appears more than once in order";
    HloInstruction* mutable_instruction =
        const_cast<HloInstruction*>(instruction);
    TF_RETURN_IF_ERROR(visitor->Preprocess(mutable_instruction));
    TF_RETURN_IF_ERROR(mutable_instruction->Visit(visitor));
    visitor->SetVisited(*mutable_instruction);
    TF_RETURN_IF_ERROR(visitor->Postprocess(mutable_instruction));
    visited.insert(instruction);
  }
  TF_RETURN_IF_ERROR(visitor->FinishVisit(root_instruction()));
  return Status::OK();
}

// Explicit instantiations.
template Status HloComputation::AcceptOrdered(
    DfsHloVisitor*, absl::Span<HloInstruction* const>) const;
template Status HloComputation::AcceptOrdered(
    ConstDfsHloVisitor*, absl::Span<HloInstruction* const>) const;

Status HloComputation::Accept(
    const std::function<Status(HloInstruction*)>& visitor_func) {
  FunctionVisitor visitor(visitor_func);
  return this->Accept(&visitor);
}

Status HloComputation::Accept(
    const std::function<Status(const HloInstruction*)>& visitor_func) const {
  ConstFunctionVisitor visitor(visitor_func);
  return this->Accept(&visitor);
}

std::unique_ptr<HloComputation> HloComputation::Clone(
    const string& suffix, HloCloneContext* context) {
  return CloneWithReplacements(
      /*replacements=*/std::unordered_map<const HloInstruction*,
                                          std::unique_ptr<HloInstruction>>(),
      context, suffix);
}

std::unique_ptr<HloComputation> HloComputation::CloneWithReplacementPairs(
    std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r1,
    HloCloneContext* context, const string& suffix) {
  std::unordered_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;
  replacements.emplace(std::move(r1));
  return CloneWithReplacements(std::move(replacements), context, suffix);
}

std::unique_ptr<HloComputation> HloComputation::CloneWithReplacementPairs(
    std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r1,
    std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r2,
    HloCloneContext* context, const string& suffix) {
  std::unordered_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;
  replacements.emplace(std::move(r1));
  replacements.emplace(std::move(r2));
  return CloneWithReplacements(std::move(replacements), context, suffix);
}

std::unique_ptr<HloComputation> HloComputation::CloneWithReplacementPairs(
    std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r1,
    std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r2,
    std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r3,
    HloCloneContext* context, const string& suffix) {
  std::unordered_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;
  replacements.emplace(std::move(r1));
  replacements.emplace(std::move(r2));
  replacements.emplace(std::move(r3));
  return CloneWithReplacements(std::move(replacements), context, suffix);
}

std::unique_ptr<HloComputation> HloComputation::CloneWithReplacements(
    std::unordered_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
        replacements,
    HloCloneContext* context, const string& suffix) {
  std::unique_ptr<HloCloneContext> context_ptr;
  if (context == nullptr) {
    context_ptr = absl::make_unique<HloCloneContext>(parent(), suffix);
    context = context_ptr.get();
  }

  // Look up instr in the replacements map, and return either the replacement,
  // or instr, if the replacement isn't present.
  //
  // Note: This can return null, indicating that instr should not be present in
  // the new computation.
  auto replace = [&](HloInstruction* instr) {
    auto it = replacements.find(instr);
    if (it == replacements.end()) {
      return instr;
    }
    return it->second.get();
  };

  VLOG(1) << "Cloning " << name() << " --> " << suffix << "\n";

  // We want to do a postorder walk over [replace(i) for i in instructions_].
  // We can't reuse MakeInstructionPostOrder() for this, because that will
  // generate a postorder of plain instructions_, and our replacements may
  // change the postorder!
  //
  // The postorder we want here is simpler than what MakeInstructionPostOrder()
  // does -- we only care about operand dependencies -- so let's just do it
  // ourselves.
  std::vector<HloInstruction*> postorder;
  absl::flat_hash_map<HloInstruction*, VisitState> visited;
  for (const auto& instr : instructions_) {
    std::vector<HloInstruction*> dfs_stack;
    HloInstruction* new_instr = replace(instr.get());
    if (!new_instr) {
      continue;
    }
    dfs_stack.push_back(new_instr);

    while (!dfs_stack.empty()) {
      auto* cur = dfs_stack.back();
      auto it = visited.find(cur);
      if (it != visited.end()) {
        dfs_stack.pop_back();
        if (it->second == kVisited) {
          continue;
        }
        CHECK_EQ(it->second, kVisiting);
        postorder.push_back(cur);
        it->second = kVisited;
        continue;
      }

      visited.insert({cur, kVisiting});
      for (HloInstruction* operand : cur->operands()) {
        HloInstruction* new_operand = replace(operand);
        if (new_operand) {
          dfs_stack.emplace_back(new_operand);
        }
      }
    }
  }

  std::vector<std::unique_ptr<HloInstruction>> instructions;
  for (auto instr : postorder) {
    std::vector<HloInstruction*> new_operands;
    for (auto operand : instr->operands()) {
      auto replaced_operand = replace(operand);
      CHECK_NE(replaced_operand, nullptr)
          << "replacements map tried to eliminate a used instruction "
          << operand->ToString() << ", used by " << instr->ToString();
      new_operands.push_back(context->GetInstruction(replaced_operand));
    }
    instructions.push_back(
        instr->CloneWithNewOperands(instr->shape(), new_operands, context));
  }
  Builder builder(name() + "." + suffix);
  for (auto& instr : instructions) {
    builder.AddInstruction(std::move(instr));
  }
  auto result = builder.Build(
      /*root_instruction=*/context->GetInstruction(
          replace(root_instruction())));

  // Clone control dependencies.
  for (auto instr : postorder) {
    HloInstruction* new_instr = context->GetInstruction(instr);
    for (auto successor : instr->control_successors()) {
      auto replaced_successor = replace(successor);
      // successor may not have been remapped, because it might have been
      // removed by the replacements map.
      if (replaced_successor != nullptr) {
        TF_CHECK_OK(new_instr->AddControlDependencyTo(
            context->GetInstruction(replaced_successor)));
      }
    }
  }
  context->MapComputation(this, result.get());
  return result;
}

void HloComputation::UniquifyName(NameUniquer* name_uniquer) {
  name_ = name_uniquer->GetUniqueName(name_);
}

HloInstruction* HloComputation::GetInstructionWithName(absl::string_view name) {
  auto instructions_in_computation = instructions();
  auto it = absl::c_find_if(
      instructions_in_computation,
      [&](HloInstruction* instr) { return instr->name() == name; });
  return it == instructions_in_computation.end() ? nullptr : *it;
}

}  // namespace xla
