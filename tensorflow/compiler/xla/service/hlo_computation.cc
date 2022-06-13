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
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_clone_context.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/mapped_ptr_container_sorter.h"
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
    const std::string& name, int parameter_count,
    std::vector<std::unique_ptr<HloInstruction>>* instructions,
    HloInstruction* root_instruction, HloInstruction* fusion_instruction)
    : name_(NameUniquer::GetSanitizedName(name)),
      unique_id_(-1),
      root_instruction_(root_instruction),
      fusion_instruction_(fusion_instruction),
      is_fusion_computation_(fusion_instruction != nullptr),
      custom_call_instruction_(nullptr),
      is_custom_call_computation_(false) {
  param_instructions_.resize(parameter_count, nullptr);
  bool root_found = false;
  for (auto& instruction : *instructions) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      int64_t param_no = instruction->parameter_number();
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

HloComputation::~HloComputation() {
  if (fusion_instruction_ != nullptr) {
    CHECK(fusion_instruction_->fused_instructions_computation() == this);
    fusion_instruction_->ClearCalledComputations();
    fusion_instruction_ = nullptr;
  }
  if (IsAsyncComputation()) {
    for (auto* async_instr : async_instructions_) {
      CHECK(async_instr->async_wrapped_computation() == this);
      async_instr->ClearCalledComputations();
    }
    async_instructions_.clear();
  }
}

HloInstruction* HloComputation::AddInstruction(
    std::unique_ptr<HloInstruction> instruction, const std::string& new_name) {
  CHECK(instruction->opcode() != HloOpcode::kParameter)
      << "Parameter instructions cannot be added to a computation after "
      << "it has been built";
  if (!new_name.empty()) {
    instruction->SetAndSanitizeName(new_name);
  }
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

HloInstruction* HloComputation::AddEntryComputationParameter(
    std::unique_ptr<HloInstruction> instruction) {
  CHECK_EQ(instruction->opcode(), HloOpcode::kParameter);
  CHECK_EQ(instruction->parameter_number(), num_parameters());
  CHECK(parent()->entry_computation() == this);

  HloModuleConfig config = parent()->config();
  config.mutable_entry_computation_layout()->add_parameter_layout(
      ShapeLayout(instruction->shape()));
  parent()->set_config(config);

  instruction->set_parent(this);
  param_instructions_.push_back(instruction.get());
  AddInstructionInternal(std::move(instruction));

  return instructions_.back().get();
}

Status HloComputation::ReplaceEntryComputationParameter(
    int64_t param_no, HloInstruction* old_instruction,
    std::unique_ptr<HloInstruction> instruction) {
  CHECK_GE(param_no, 0);
  CHECK_LT(param_no, param_instructions_.size());
  CHECK_EQ(instruction->opcode(), HloOpcode::kParameter);
  CHECK(parent()->entry_computation() == this);

  HloModuleConfig config = parent()->config();
  *config.mutable_entry_computation_layout()->mutable_parameter_layout(
      param_no) = ShapeLayout(instruction->shape());
  parent()->set_config(config);

  instruction->set_parent(this);
  param_instructions_[param_no] = instruction.get();
  AddInstructionInternal(std::move(instruction));

  return ForceRemoveInstruction(old_instruction);
}

Status HloComputation::RemoveParameter(int64_t param_no) {
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

  return OkStatus();
}

HloInstruction* HloComputation::ReplaceParameter(
    int64_t param_no, std::unique_ptr<HloInstruction> instruction) {
  CHECK_GE(param_no, 0);
  CHECK_LT(param_no, param_instructions_.size());
  CHECK(instruction->opcode() == HloOpcode::kParameter);
  CHECK(IsFusionComputation());
  CHECK_EQ(fusion_instruction_->operand_count(), param_instructions_.size());

  instruction->set_parent(this);
  HloInstruction* new_instruction =
      AddInstructionInternal(std::move(instruction));
  HloInstruction* old_instruction = param_instructions_[param_no];
  CHECK(
      old_instruction->ReplaceAllUsesWithDifferentShape(new_instruction).ok());
  param_instructions_[param_no] = new_instruction;
  CHECK(RemoveInstruction(old_instruction).ok());
  return new_instruction;
}

Status HloComputation::RemoveUnusedParametersFromFusedComputation() {
  return RemoveUnusedParametersImpl(/*allow_non_fusion=*/false);
}

Status HloComputation::RemoveUnusedParametersFromAnyComputation() {
  return RemoveUnusedParametersImpl(/*allow_non_fusion=*/true);
}

Status HloComputation::RemoveUnusedParametersImpl(bool allow_non_fusion) {
  CHECK(allow_non_fusion || IsFusionComputation());
  int64_t removed = 0;
  for (int64_t i = 0; i < param_instructions_.size(); ++i) {
    HloInstruction* param_instruction = param_instructions_[i];
    if (param_instruction->IsDead()) {
      TF_RETURN_IF_ERROR(
          RemoveInstructionImpl(param_instruction, allow_non_fusion));
      ++removed;
      continue;
    }

    if (removed > 0) {
      const int64_t param_no = i - removed;
      HloInstruction* new_instr = AddInstructionInternal(
          HloInstruction::CreateParameter(param_no, param_instruction->shape(),
                                          StrCat("param_", param_no)));
      TF_RETURN_IF_ERROR(param_instruction->ReplaceAllUsesWith(new_instr));
      param_instructions_[param_no] = new_instr;
      TF_RETURN_IF_ERROR(
          RemoveInstructionImpl(param_instruction, allow_non_fusion));
    }
  }
  param_instructions_.resize(param_instructions_.size() - removed);
  return OkStatus();
}

bool HloComputation::IsSafelyRemovable(const HloInstruction* instruction) {
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

bool HloComputation::IsMarkedAsDead(const HloInstruction* inst) {
  return inst->IsMarkedAsDead();
}

Status HloComputation::RemoveInstructionAndUnusedOperands(
    HloInstruction* instruction, std::function<void(HloInstruction*)> cleanup) {
  TF_RET_CHECK(root_instruction() != instruction);

  TF_RET_CHECK(instruction->IsDead());
  TF_RET_CHECK(IsSafelyRemovable(instruction))
      << "Cannot remove instruction: " << instruction->ToString();
  absl::flat_hash_set<HloInstruction*> removed;
  std::queue<HloInstruction*> worklist;
  worklist.push(instruction);
  while (!worklist.empty()) {
    HloInstruction* item = worklist.front();
    worklist.pop();

    if (removed.contains(item) || !item->IsDead() || !IsSafelyRemovable(item) ||
        (item->HasSideEffect() && item != instruction)) {
      continue;
    }
    for (int i = 0; i < item->operand_count(); ++i) {
      worklist.push(item->mutable_operand(i));
    }

    if (cleanup) {
      cleanup(item);
    }
    TF_RETURN_IF_ERROR(RemoveInstruction(item));
    removed.insert(item);
  }
  return OkStatus();
}

Status HloComputation::RemoveInstruction(HloInstruction* instruction) {
  return RemoveInstructionImpl(instruction, /*ignore_safety_check=*/false);
}

Status HloComputation::ForceRemoveInstruction(HloInstruction* instruction) {
  return RemoveInstructionImpl(instruction, /*ignore_safety_check=*/true);
}

Status HloComputation::RemoveInstructionImpl(HloInstruction* instruction,
                                             bool ignore_safety_check) {
  VLOG(2) << "Removing instruction " << instruction->name()
          << " from computation " << name();
  TF_RET_CHECK(ignore_safety_check || IsSafelyRemovable(instruction))
      << "cannot remove instruction: " << instruction->ToString();
  TF_RET_CHECK(instruction->IsDead()) << "instruction " << instruction->name()
                                      << " is live and cannot be removed";
  TF_RET_CHECK(instruction->control_predecessors().empty())
      << "instruction " << instruction->name()
      << " has control predecessors and cannot be removed";
  TF_RET_CHECK(instruction->control_successors().empty())
      << "instruction " << instruction->name()
      << " has control successors and cannot be removed";

  auto inst_it = instruction_iterators_.find(instruction);
  TF_RET_CHECK(inst_it != instruction_iterators_.end());
  (*inst_it->second)->set_parent(nullptr);
  to_be_deleted_.emplace_back(inst_it->second->release());
  to_be_deleted_.back()->DetachFromOperandsAndUsers();
  // Clear all operands to avoid Null operands.
  to_be_deleted_.back()->RemoveAllOperands();
  to_be_deleted_.back()->ClearCalledComputations();
  to_be_deleted_.back()->MarkAsDead();
  instructions_.erase(inst_it->second);
  instruction_iterators_.erase(inst_it);
  return OkStatus();
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

  if (parent() && parent()->has_entry_computation() &&
      parent()->entry_computation() == this) {
    if (!Shape::Equal().IgnoreLayout()(new_root_instruction->shape(),
                                       root_instruction_->shape())) {
      // Rebuild input output alias config now that we have a new output shape.
      parent()->input_output_alias_config() =
          HloInputOutputAliasConfig(new_root_instruction->shape());
    }
  }

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

std::optional<int64_t> GetChannelId(const HloInstruction& inst) {
  // Note that we only include Send and RecvDone, as we want to create a
  // dependency between those, but not SendDone and Recv.
  switch (inst.opcode()) {
    case HloOpcode::kSend:
    case HloOpcode::kRecvDone:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kReduceScatter:
      return inst.channel_id();
    default:
      return std::nullopt;
  }
}

}  // namespace

void HloComputation::ComputeInstructionPostOrder(
    HloInstruction* root,
    HloComputation::ChannelDependencyGroup& channel_dependencies,
    absl::flat_hash_map<HloInstruction*, VisitState>& visited,
    std::vector<HloInstruction*>& post_order) const {
  std::vector<HloInstruction*> dfs_stack = {root};
  while (!dfs_stack.empty()) {
    HloInstruction& current = *dfs_stack.back();

    auto result = visited.insert({&current, kVisiting});
    if (!result.second) {  // We've already seen this instruction.
      dfs_stack.pop_back();
      if (result.first->second != kVisited) {
        CHECK_EQ(current.parent(), this)
            << "Instruction " << current.name()
            << " is not in the current computation (" << name() << ").";
        post_order.push_back(&current);
        result.first->second = kVisited;
      }
      continue;
    }

    // Add channel dependencies.
    // A RecvDone op must be preceded by the corresponding Send op.
    // Collectives with the same channel ID must be performed together, as these
    // represent MPMD-partitioned that will later be split into separate modules
    // and the order must be preserved.
    std::optional<int64_t> channel_id =
        ((&current != root) && (current.opcode() != HloOpcode::kSend))
            ? GetChannelId(current)
            : std::nullopt;
    if (channel_id) {
      auto it = channel_dependencies.find(*channel_id);
      if (it != channel_dependencies.end()) {
        dfs_stack.insert(dfs_stack.end(), it->second.begin(), it->second.end());
        channel_dependencies.erase(it);
      }
    }

    // Add the operands to the stack in reverse order so the first operand is
    // processed first. This will produce a more natural ordering and a nicer
    // result for things like HLO stringification.
    const HloInstruction::InstructionVector& operands = current.operands();
    dfs_stack.insert(dfs_stack.end(), operands.rbegin(), operands.rend());

    const std::vector<HloInstruction*>& predecessors =
        current.control_predecessors();
    dfs_stack.insert(dfs_stack.end(), predecessors.begin(), predecessors.end());
  }
}

HloComputation::ChannelDependencyGroup
HloComputation::ComputeChannelDependencies() const {
  if (parent() && parent()->config().has_static_device_assignment() &&
      (parent()->config().static_device_assignment().computation_count() == 1 ||
       parent()->config().use_spmd_partitioning())) {
    return {};
  }

  ChannelDependencyGroup channel_dependencies;
  for (const auto& instruction : instructions_) {
    std::optional<int64_t> channel_id = GetChannelId(*instruction);
    if (channel_id)
      channel_dependencies[*channel_id].push_back(instruction.get());
  }
  return channel_dependencies;
}

static inline bool HasOnlyTraceUsers(const HloInstruction* instruction) {
  return absl::c_all_of(instruction->users(),
                        [](HloInstruction* user) { return false; });
}

std::vector<HloInstruction*> HloComputation::MakeInstructionPostOrder() const {
  ChannelDependencyGroup channel_dependencies = ComputeChannelDependencies();
  std::vector<HloInstruction*> post_order;
  post_order.reserve(instruction_count());
  std::vector<HloInstruction*> trace_instructions;
  absl::flat_hash_map<HloInstruction*, VisitState> visited;
  visited.reserve(instruction_count());
  for (auto& instruction : instructions_) {
    if (HasOnlyTraceUsers(instruction.get())) {
      ComputeInstructionPostOrder(instruction.get(), channel_dependencies,
                                  visited, post_order);
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

std::string HloComputation::ToString(const HloPrintOptions& options) const {
  return std::string(ToCord(options));
}

std::string HloComputation::ToString(
    const HloPrintOptions& options,
    absl::Span<const HloInstruction* const> instruction_order) const {
  return std::string(ToCord(options, instruction_order));
}

absl::Cord HloComputation::ToCord(const HloPrintOptions& options) const {
  return ToCord(options, MakeInstructionPostOrder());
}

absl::Cord HloComputation::ToCord(
    const HloPrintOptions& options,
    absl::Span<const HloInstruction* const> instruction_order) const {
  CHECK_EQ(instruction_order.size(), instruction_count());
  const std::string tab(2 * options.indent_amount(), ' ');

  absl::Cord result;
  result.Append(tab);

  if (!options.is_in_nested_computation()) {
    if (options.print_percent()) {
      result.Append("%");
    }
    if (options.print_ids()) {
      // When print_ids() is false, exclude entry computation's name because it
      // includes and leads to non-deterministic fingerprint.
      result.Append(name());
      result.Append(" ");
    }
  }

  if (options.print_program_shape()) {
    result.Append(
        ShapeUtil::HumanString(ComputeProgramShape(options.print_ids())));
    result.Append(" ");
  }
  result.Append("{\n");

  {
    // Print the instructions in this computation.
    HloPrintOptions new_options =
        HloPrintOptions(options)
            .set_indent_amount(options.indent_amount() + 1)
            .set_is_in_nested_computation(true);

    const std::string new_tab(2 * new_options.indent_amount(), ' ');

    CanonicalNameMap name_map;
    for (const HloInstruction* const instruction : instruction_order) {
      DCHECK_EQ(this, instruction->parent());
      result.Append(new_tab);
      if (instruction == root_instruction_) {
        result.Append("ROOT ");
      }
      result.Append(
          instruction->ToStringWithCanonicalNameMap(new_options, &name_map));
      result.Append("\n");
    }
  }

  result.Append(tab);
  result.Append("}");
  return result;
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
  proto.set_is_fusion_computation(is_fusion_computation_);
  return proto;
}

/* static */ StatusOr<std::unique_ptr<HloComputation>>
HloComputation::CreateFromProto(
    const HloComputationProto& proto,
    const absl::flat_hash_map<int64_t, HloComputation*>& computation_map,
    bool prohibit_empty_literal) {
  absl::flat_hash_map<int64_t, HloInstruction*> instruction_map;
  absl::flat_hash_map<HloInstruction*, int64_t> to_proto_id;
  std::vector<std::unique_ptr<HloInstruction>> instructions;
  int64_t parameter_count = 0;
  for (const HloInstructionProto& instruction_proto : proto.instructions()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloInstruction> instruction,
                        HloInstruction::CreateFromProto(
                            instruction_proto, instruction_map, computation_map,
                            prohibit_empty_literal));
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
  absl::c_sort(instructions, [&](const std::unique_ptr<HloInstruction>& a,
                                 const std::unique_ptr<HloInstruction>& b) {
    return to_proto_id[a.get()] < to_proto_id[b.get()];
  });

  TF_RETURN_IF_ERROR([&]() -> Status {
    std::vector<bool> parameters_seen(parameter_count);
    int parameters_seen_count = 0;
    for (auto& instruction : instructions) {
      if (instruction->opcode() == HloOpcode::kParameter) {
        int64_t param_no = instruction->parameter_number();
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
    return OkStatus();
  }());

  auto computation = absl::WrapUnique(
      new HloComputation(proto.name(), parameter_count, &instructions, root,
                         /*fusion_instruction=*/nullptr));
  computation->unique_id_ = proto.id();
  computation->is_fusion_computation_ = proto.is_fusion_computation();
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
    if (instruction->IsDead()) {
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

StatusOr<HloInstruction*> HloComputation::CreateAsyncInstructions(
    HloInstruction* instruction, absl::Span<const Shape> context_shapes) {
  Builder builder("async_computation");
  std::vector<HloInstruction*> parameters(instruction->operand_count());
  std::vector<Shape> parameter_shapes(instruction->operand_count());
  for (int i = 0; i < instruction->operand_count(); ++i) {
    const Shape& parameter_shape = instruction->operand(i)->shape();
    parameters[i] = builder.AddInstruction(HloInstruction::CreateParameter(
        i, parameter_shape, absl::StrCat("param_", i)));
    parameter_shapes[i] = parameter_shape;
  }
  HloInstruction* root = builder.AddInstruction(
      instruction->CloneWithNewOperands(instruction->shape(), parameters));
  HloComputation* async_computation =
      parent_->AddEmbeddedComputation(builder.Build(root));
  std::vector<Shape> start_shapes = {
      ShapeUtil::MakeTupleShape(parameter_shapes), root->shape()};
  for (const Shape& context_shape : context_shapes) {
    start_shapes.push_back(context_shape);
  }
  HloInstruction* async_start = AddInstruction(HloInstruction::CreateAsyncStart(
      ShapeUtil::MakeTupleShape(start_shapes), instruction->operands(),
      async_computation));
  HloInstruction* async_done = AddInstruction(HloInstruction::CreateAsyncDone(
      root->shape(), async_start, async_computation));
  async_start->set_metadata(instruction->metadata());
  async_start->CopyBackendConfigFrom(instruction);
  async_done->set_metadata(instruction->metadata());
  async_done->CopyBackendConfigFrom(instruction);
  TF_RETURN_IF_ERROR(ReplaceInstruction(instruction, async_done));
  return async_done;
}

StatusOr<HloInstruction*> HloComputation::DeepCopyHelper(
    HloInstruction* instruction, ShapeIndex* index,
    const std::function<
        HloInstruction*(HloInstruction* leaf, const ShapeIndex& leaf_index,
                        HloComputation* computation)>& copy_leaf) {
  if (instruction->shape().IsTuple()) {
    std::vector<HloInstruction*> elements;
    for (int64_t i = 0; i < ShapeUtil::TupleElementCount(instruction->shape());
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
  if (instruction->shape().IsToken()) {
    // Tokens have no on-device representation and cannot be copied. Pass
    // through transparently.
    return instruction;
  }

  // Array shape.
  TF_RET_CHECK(instruction->shape().IsArray());
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

ProgramShape HloComputation::ComputeProgramShape(bool include_ids) const {
  ProgramShape program_shape;

  for (auto* param_instruction : param_instructions_) {
    *program_shape.add_parameters() = param_instruction->shape();
    *program_shape.add_parameter_names() =
        PrintName(param_instruction->name(), include_ids);
  }
  *program_shape.mutable_result() = root_instruction_->shape();

  return program_shape;
}

bool HloComputation::EqualInternal(const HloComputation& other,
                                   bool is_layout_sensitive,
                                   bool ignore_channel_id_values) const {
  if (this == &other) {
    return true;
  }
  absl::flat_hash_set<std::pair<const HloInstruction*, const HloInstruction*>>
      visited;
  std::vector<std::pair<const HloInstruction*, const HloInstruction*>> worklist;

  worklist.push_back({root_instruction(), other.root_instruction()});

  while (!worklist.empty()) {
    auto pair = worklist.back();
    worklist.pop_back();

    if (visited.contains(pair)) {
      continue;
    }
    visited.emplace(pair);
    // TODO(b/123082518): Avoid recursively invoking Equal because it may
    // cause a stack overflow with deeply nested subcomputations.
    auto operands_eq = [](const HloInstruction*, const HloInstruction*) {
      return true;
    };
    auto comp_eq = [&](const HloComputation* a, const HloComputation* b) {
      return a->EqualInternal(*b, is_layout_sensitive,
                              ignore_channel_id_values);
    };
    bool identical_ignoring_operands =
        ignore_channel_id_values
            ? pair.first->IdenticalIgnoringChannelIdValues(
                  *pair.second, operands_eq, comp_eq, is_layout_sensitive)
            : pair.first->Identical(*pair.second, operands_eq, comp_eq,
                                    is_layout_sensitive);
    if (!identical_ignoring_operands) {
      return false;
    }
    for (size_t i = 0; i < pair.first->operands().size(); ++i) {
      worklist.push_back({pair.first->operand(i), pair.second->operand(i)});
    }
  }
  return true;
}

Status HloComputation::ReplaceWithNewInstruction(
    HloInstruction* old_instruction,
    std::unique_ptr<HloInstruction> new_instruction) {
  return ReplaceInstruction(old_instruction,
                            AddInstruction(std::move(new_instruction)));
}

Status HloComputation::ReplaceWithNewEntryComputationParameter(
    HloInstruction* old_instruction,
    std::unique_ptr<HloInstruction> new_instruction) {
  return ReplaceInstruction(old_instruction, AddEntryComputationParameter(
                                                 std::move(new_instruction)));
}

StatusOr<bool> HloComputation::ReplaceInstruction(
    HloInstruction* old_instruction, HloInstruction* new_instruction,
    bool preserve_sharding) {
  TF_RET_CHECK(
      ShapeUtil::Compatible(old_instruction->shape(), new_instruction->shape()))
      << ShapeUtil::HumanString(old_instruction->shape()) << " vs "
      << ShapeUtil::HumanString(new_instruction->shape());
  return ReplaceInstructionWithDifferentShape(old_instruction, new_instruction,
                                              preserve_sharding);
}

Status HloComputation::ReplaceInstruction(HloInstruction* old_instruction,
                                          HloInstruction* new_instruction) {
  TF_ASSIGN_OR_RETURN(bool changed,
                      ReplaceInstruction(old_instruction, new_instruction,
                                         /*preserve_sharding=*/false));
  DCHECK(changed);
  return OkStatus();
}

StatusOr<bool> HloComputation::ReplaceInstructionWithDifferentShape(
    HloInstruction* old_instruction, HloInstruction* new_instruction,
    bool preserve_sharding) {
  if (preserve_sharding && new_instruction->has_sharding() &&
      old_instruction->has_sharding() &&
      !new_instruction->has_compatible_sharding(old_instruction)) {
    VLOG(10) << "Skipping replacement due to incompatible sharding";
    return false;
  }
  VLOG(10) << "transformed " << old_instruction->ToString() << " to "
           << new_instruction->ToString();
  // Try to add metadata for HLO instructions that are created to replace
  // existing HLO instructions (e.g. during optimizations). The assumption is
  // that the old instruction and the new instruction would perform the same
  // function, and that they would be correlated to the same TF op. This might
  // not always be correct since HLO optimizations can cross TF op boundaries.
  // But still this seems to be better than nothing.
  bool overwrite_op_name = new_instruction->metadata().op_name().empty() &&
                           !old_instruction->metadata().op_name().empty();
  bool overwrite_pass_id =
      new_instruction->metadata().op_name().empty() &&
      new_instruction->metadata().logical_creation_pass_id() == 0 &&
      old_instruction->metadata().logical_creation_pass_id() != 0;
  if (overwrite_op_name || overwrite_pass_id) {
    new_instruction->set_metadata(old_instruction->metadata());
  }
  if (new_instruction->frontend_attributes().map().empty()) {
    new_instruction->set_frontend_attributes(
        old_instruction->frontend_attributes());
  }

  // Like the metadata above, if the user didn't specify any sharding
  // information on the new instruction we should copy the old sharding
  // information (if any).
  if (!new_instruction->has_sharding()) {
    new_instruction->set_sharding(old_instruction->sharding_ptr());
  }

  TF_RETURN_IF_ERROR(
      old_instruction->ReplaceAllUsesWithDifferentShape(new_instruction));

  // Preserve the old instruction's name if the new and old instruction have the
  // same opcode.  This makes it easier to follow instructions as they're
  // mutated through passes.
  if (old_instruction->opcode() == new_instruction->opcode() &&
      (old_instruction->opcode() != HloOpcode::kCustomCall ||
       old_instruction->custom_call_target() ==
           new_instruction->custom_call_target())) {
    new_instruction->SetAndSanitizeName(old_instruction->name());
  }

  TF_RETURN_IF_ERROR(RemoveInstructionAndUnusedOperands(old_instruction));
  return true;
}

Status HloComputation::ReplaceInstructionWithDifferentShape(
    HloInstruction* old_instruction, HloInstruction* new_instruction) {
  TF_ASSIGN_OR_RETURN(bool changed, ReplaceInstructionWithDifferentShape(
                                        old_instruction, new_instruction,
                                        /*preserve_sharding=*/false));
  DCHECK(changed);
  return OkStatus();
}

std::vector<HloInstruction*> HloComputation::CollectUnreachableRoots() const {
  std::vector<HloInstruction*> unreachable_roots;
  for (auto* instruction : instructions()) {
    if (instruction->IsDead() && instruction->control_successors().empty()) {
      unreachable_roots.push_back(instruction);
    }
  }
  VLOG(3) << "Unreachable roots:"
          << absl::StrJoin(unreachable_roots, "\n\t",
                           [](std::string* out, const HloInstruction* hlo) {
                             absl::StrAppend(out, hlo->ToString());
                           });
  return unreachable_roots;
}

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

std::unique_ptr<HloComputation> HloComputation::Clone(
    const std::string& suffix, HloCloneContext* context) {
  return CloneWithReplacements(
      /*replacements=*/nullptr,
      /*extra_parameters=*/{}, context, suffix);
}

std::unique_ptr<HloComputation> HloComputation::CloneWithReplacementPairs(
    std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r1,
    HloCloneContext* context, const std::string& suffix) {
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;
  replacements.emplace(std::move(r1));
  return CloneWithReplacements(&replacements, /*extra_parameters=*/{}, context,
                               suffix);
}

std::unique_ptr<HloComputation> HloComputation::CloneWithReplacementPairs(
    std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r1,
    std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r2,
    HloCloneContext* context, const std::string& suffix) {
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;
  replacements.emplace(std::move(r1));
  replacements.emplace(std::move(r2));
  return CloneWithReplacements(&replacements, /*extra_parameters=*/{}, context,
                               suffix);
}

std::unique_ptr<HloComputation> HloComputation::CloneWithReplacementPairs(
    std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r1,
    std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r2,
    std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r3,
    HloCloneContext* context, const std::string& suffix) {
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;
  replacements.emplace(std::move(r1));
  replacements.emplace(std::move(r2));
  replacements.emplace(std::move(r3));
  return CloneWithReplacements(&replacements, /*extra_parameters=*/{}, context,
                               suffix);
}

namespace {

// Sorts unordered_instructions according to the order of ordered_instructions,
// using MappedPtrContainerSorter. context and replace are used to map
// instructions in ordered_instructions to instructions in
// unordered_instructions. Unmapped parameter instructions are placed just after
// the last parameter instruction in the sorted mapped instruction order. All
// other mapped instructions are placed at the end.
void SortClonedInstructions(
    const HloCloneContext& context,
    const std::function<const HloInstruction*(const HloInstruction*)>& replace,
    const HloComputation& computation,
    const HloComputation::InstructionList& ordered_instructions,
    std::vector<std::unique_ptr<HloInstruction>>& unordered_instructions) {
  using InstructionSorter = MappedPtrContainerSorter<HloInstruction>;
  InstructionSorter::MapPtrFn instruction_mapper =
      [&context, &replace](const HloInstruction* i) {
        return context.FindInstruction(replace(i));
      };
  size_t num_mapped_instructions = 0;
  size_t mapped_index_of_last_parameter_plus_one = 0;
  for (const auto& instruction : ordered_instructions) {
    if (!instruction_mapper(instruction.get())) {
      continue;
    }
    ++num_mapped_instructions;
    if (!dynamic_cast<const HloParameterInstruction*>(instruction.get())) {
      continue;
    }
    mapped_index_of_last_parameter_plus_one = num_mapped_instructions;
  }
  InstructionSorter::UnmappedPtrIndexFn unmapped_ptr_index =
      [num_mapped_instructions,
       mapped_index_of_last_parameter_plus_one](const HloInstruction* i) {
        if (dynamic_cast<const HloParameterInstruction*>(i)) {
          if (num_mapped_instructions > 0 &&
              mapped_index_of_last_parameter_plus_one > 0) {
            return mapped_index_of_last_parameter_plus_one - 1;
          }
          return InstructionSorter::IndexBeforeMappedElementsFn()(i);
        }
        return InstructionSorter::IndexAfterMappedElementsFn()(i);
      };
  auto status =
      InstructionSorter::Sort(instruction_mapper, unmapped_ptr_index,
                              ordered_instructions, unordered_instructions);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to reorder instructions while cloning computation: "
               << computation.name() << "; " << status;
  }
}

// For cloned instructions, sorts their users, control predecessors, and control
// successors, according to the orders of those lists in the original
// instructions, before cloning. context and replace help us to map original
// instructions to cloned instructions, in addition to creating a list of
// cloned instructions.
void SortClonedInstructionUsersAndControlLists(
    const HloCloneContext& context,
    const std::function<const HloInstruction*(const HloInstruction*)>& replace,
    const HloComputation::InstructionList& sorted_instructions) {
  using InstructionSorter = MappedPtrContainerSorter<HloInstruction>;
  InstructionSorter::MapPtrFn instruction_mapper =
      [&context, &replace](const HloInstruction* i) {
        return context.FindInstruction(replace(i));
      };
  for (const std::unique_ptr<HloInstruction>& instruction :
       sorted_instructions) {
    HloInstruction* cloned_instruction =
        context.FindInstruction(replace(instruction.get()));
    if (!cloned_instruction) {
      continue;
    }
    cloned_instruction->SortInstructionUsersAndControlLists(instruction_mapper,
                                                            *instruction);
  }
}

}  // namespace

std::unique_ptr<HloComputation> HloComputation::CloneWithReplacements(
    const absl::flat_hash_map<const HloInstruction*,
                              std::unique_ptr<HloInstruction>>* replacements,
    absl::Span<const HloInstruction* const> extra_parameters,
    HloCloneContext* context, const std::string& suffix,
    const HloInstruction* new_root) {
  std::unique_ptr<HloCloneContext> context_ptr;
  if (context == nullptr) {
    context_ptr = std::make_unique<HloCloneContext>(parent(), suffix);
    context = context_ptr.get();
  }
  if (new_root == nullptr) {
    new_root = root_instruction();
  }

  // Look up instr in the replacements map, and return either the replacement,
  // or instr, if the replacement isn't present.
  //
  // Note: This can return null, indicating that instr should not be present in
  // the new computation.
  auto replace = [&](const HloInstruction* instr) {
    if (!replacements) return instr;
    auto it = replacements->find(instr);
    return it != replacements->end() ? it->second.get() : instr;
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
  std::vector<const HloInstruction*> postorder;
  absl::flat_hash_map<const HloInstruction*, VisitState> visited;
  for (const auto& instr : instructions_) {
    std::vector<const HloInstruction*> dfs_stack;
    const HloInstruction* new_instr = replace(instr.get());
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
        const HloInstruction* new_operand = replace(operand);
        if (new_operand) {
          dfs_stack.emplace_back(new_operand);
        }
      }
    }
  }

  std::vector<std::unique_ptr<HloInstruction>> instructions;
  // First add the extra parameters to 'instructions'.
  for (const auto& instr : extra_parameters) {
    CHECK_EQ(instr->opcode(), HloOpcode::kParameter)
        << "Only parameter instructions are allowed in 'extra_parameters'";
    instructions.emplace_back(instr->Clone());
  }
  for (auto instr : postorder) {
    std::vector<HloInstruction*> new_operands;
    for (auto operand : instr->operands()) {
      auto replaced_operand = replace(operand);
      CHECK_NE(replaced_operand, nullptr)
          << "replacements map tried to eliminate a used instruction "
          << operand->ToString() << ", used by " << instr->ToString();
      new_operands.push_back(context->GetInstruction(replaced_operand));
    }
    std::unique_ptr<HloInstruction> new_instr =
        instr->CloneWithNewOperands(instr->shape(), new_operands, context);
    if (instr->opcode() == HloOpcode::kParameter &&
        instr->parameter_replicated_at_leaf_buffers().has_value()) {
      new_instr->set_parameter_replicated_at_leaf_buffers(
          instr->parameter_replicated_at_leaf_buffers().value());
    }
    instructions.push_back(std::move(new_instr));
  }

  // To make clone behavior match uncloned behavior, we reorder instructions to
  // match the order in instructions_.
  SortClonedInstructions(*context, replace, *this, instructions_, instructions);

  Builder builder(suffix.empty() ? name() : name() + "." + suffix);
  for (auto& instr : instructions) {
    builder.AddInstruction(std::move(instr));
  }
  auto result = builder.Build(
      /*root_instruction=*/context->GetInstruction(replace(new_root)));

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

  // To make clone behavior match uncloned behavior, we reorder the user and
  // control lists, kept by cloned instructions.
  SortClonedInstructionUsersAndControlLists(*context, replace, instructions_);

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

bool HloComputation::IsEntryComputation() const {
  return parent()->entry_computation() == this;
}
}  // namespace xla
