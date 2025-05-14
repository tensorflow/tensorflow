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

#include "xla/hlo/ir/hlo_computation.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <ostream>
#include <queue>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/ir/ptrvec.h"
#include "xla/literal.h"
#include "xla/map_util.h"
#include "xla/printer.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/mapped_ptr_container_sorter.h"
#include "xla/service/name_uniquer.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/lib/gtl/iterator_range.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

using absl::StrCat;

enum class VisitState { kNew = 0, kVisiting = 1, kVisited = 2 };

static std::ostream& operator<<(std::ostream& os, const VisitState& state) {
  switch (state) {
    case VisitState::kNew:
      os << "new";
      break;
    case VisitState::kVisiting:
      os << "visiting";
      break;
    case VisitState::kVisited:
      os << "visited";
      break;
  }
  return os;
}

class HloComputation::VisitMap {
 public:
  VisitMap() = default;
  explicit VisitMap(int capacity) : size_(capacity) {
    int num_words = (capacity + 31) / 32;
    bits_.resize(num_words);
    bit_ptr_ = bits_.empty() ? nullptr : bits_.data();
  }

  // A handle is a dense index used to identify a particular node.
  using Handle = uint32_t;

  // Returns the current VisitState for the instruction with handle "h"
  VisitState GetState(Handle h) const {
    DCHECK_LT(h, size_);
    uint32_t word = (h / 32);
    uint32_t shift = (h % 32) << 1;
    return static_cast<VisitState>((bit_ptr_[word] >> shift) & 0x3);
  }

  // Sets the VisitState for the instruction with Handle "h" to "new_state"
  void SetState(Handle h, VisitState new_state) {
    DCHECK_LT(h, size_);
    uint32_t word = (h / 32);
    uint32_t shift = (h % 32) << 1;
    uint64_t mask = ~(3ull << shift);
    uint64_t val = static_cast<uint64_t>(new_state);
    bit_ptr_[word] = (bit_ptr_[word] & mask) | (val << shift);
  }

 private:
  // bits_ stores VisitState entries (2 bits per entry, packed 32 entries per
  // 64-bit word)
  absl::InlinedVector<uint64_t, 1> bits_;
  uint64_t* bit_ptr_ = nullptr;  //
  int size_ = 0;  // Number of entries.  bits_ holds at least 2 * this many bits
};

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
      root_instruction ? root_instruction : last_added_instruction();
  CHECK_NE(nullptr, root);
  return absl::WrapUnique(
      new HloComputation(name_, parameter_count, &instructions_, root));
}

HloComputation::HloComputation(
    const std::string& name, int parameter_count,
    std::vector<std::unique_ptr<HloInstruction>>* instructions,
    HloInstruction* root_instruction)
    : unique_id_(-1),
      root_instruction_(root_instruction),
      instruction_count_(0),
      name_(NameUniquer::GetSanitizedName(name)) {
  param_instructions_.resize(parameter_count, nullptr);
  bool root_found = false;
  for (auto& instruction : *instructions) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      int64_t param_no = instruction->parameter_number();
      CHECK(param_no >= 0 && param_no < parameter_count)
          << "\nERROR: invalid parameter number. Expected [0, "
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
  root_instruction_->MarkAsRoot();
}

HloComputation::~HloComputation() {
  if (FusionInstruction() != nullptr) {
    CHECK(FusionInstruction()->fused_instructions_computation() == this);
    FusionInstruction()->ClearCalledComputations();
  }
  Cleanup();
  ClearCalledComputations();

  // We need to make sure there are no dangling references to this computation
  // from instructions in other computations.
  std::vector<HloComputation*> callers;
  for (const auto& [caller, count] : caller_computations_) {
    callers.push_back(caller);
  }
  for (HloComputation* caller : callers) {
    for (HloInstruction* inst : caller->instructions()) {
      if (inst->has_called_computations()) {
        for (int i = 0; i < inst->called_computations().size(); ++i) {
          if (inst->called_computations()[i] == this) {
            inst->set_called_computation(i, nullptr);
          }
        }
      }
    }
  }
  CHECK(caller_computations_.empty());

  // Delete the map from caller instructions to count, if it exists.
  delete GetCallersMap();

  for (const auto& i : instructions_) {
    delete i.inst();
  }
}

void HloComputation::ClearCalledComputations() {
  for (HloInstruction* i : instructions()) {
    i->ClearCalledComputations();
  }
  // Clearing the instructions should have removed all callee computations.
  CHECK(callee_computations_.empty());
}

void HloComputation::SetInstruction(HloInstruction* instruction,
                                    InstructionType type) {
  static_assert(alignof(HloInstruction) == kInstructionTypeMask + 1,
                "HloInstruction should be aligned as a QWORD");

  DCHECK(type != InstructionType::kUnset)
      << "Set instruction must be called with a valid type, not kUnset.";
  DCHECK(instruction_type() == InstructionType::kUnset ||
         instruction_type() == type)
      << "Unexpected instruction type. Current type is "
      << static_cast<int>(instruction_type()) << " and it cannot be reset to "
      << static_cast<int>(type);

  // If `instruction` is nullptr, we need to preserve the existing type.
  if (instruction == nullptr) {
    type = instruction_type();
  }

  instruction_and_type_ =
      reinterpret_cast<uintptr_t>(instruction) | static_cast<uintptr_t>(type);
}

HloInstruction* HloComputation::AddInstruction(
    std::unique_ptr<HloInstruction> instruction, absl::string_view new_name) {
  CHECK(instruction->opcode() != HloOpcode::kParameter)
      << "Parameter instructions cannot be added to a computation after "
      << "it has been built";
  if (!new_name.empty()) {
    instruction->SetAndSanitizeName(new_name);
  }
  return AddInstructionInternal(std::move(instruction));
}

HloInstruction* HloComputation::AddInstruction(
    std::unique_ptr<HloInstruction> instruction, const OpMetadata* metadata) {
  if (metadata != nullptr) {
    instruction->set_metadata(*metadata);
  }
  return AddInstruction(std::move(instruction));
}

HloInstruction* HloComputation::AddInstruction(
    std::unique_ptr<HloInstruction> instruction, const OpMetadata* metadata,
    const FrontendAttributes* frontend_attributes) {
  if (metadata != nullptr) {
    instruction->set_metadata(*metadata);
  }
  if (frontend_attributes != nullptr) {
    instruction->set_frontend_attributes(*frontend_attributes);
  }
  return AddInstruction(std::move(instruction));
}

static void IncrementCount(
    absl::btree_map<HloComputation*, int, HloComputation::UniqueIdComparator>&
        map,
    HloComputation* key) {
  ++map[key];
}

static void DecrementCount(
    absl::btree_map<HloComputation*, int, HloComputation::UniqueIdComparator>&
        map,
    HloComputation* key) {
  auto it = map.find(key);
  CHECK(it != map.end());
  CHECK_GT(it->second, 0);
  --it->second;
  if (it->second == 0) {
    map.erase(it);
  }
}

void HloComputation::AddCallee(HloInstruction* caller, HloComputation* callee) {
  IncrementCount(callee_computations_, callee);
  IncrementCount(callee->caller_computations_, this);

  if (auto* map = callee->GetCallersMap()) {
    ++(*map)[caller];
  } else if (callee->callers_ == 0) {
    callee->callers_ = reinterpret_cast<uintptr_t>(caller);
  } else {
    // Convert the single instruction to a map.
    auto* current_caller = reinterpret_cast<const HloInstruction*>(
        callee->callers_ & ~kCallerTypeMask);
    auto* map = new absl::flat_hash_map<const HloInstruction*, int>();
    (*map)[current_caller] = 1;
    ++(*map)[caller];
    callee->callers_ = reinterpret_cast<uintptr_t>(map) |
                       static_cast<uintptr_t>(CallersType::kCallerCountHashMap);
  }

  if (parent() != nullptr && callee->parent() == parent()) {
    parent()->topological_sort_.AddEdge(this, callee);
  }
}

void HloComputation::RemoveCallee(HloInstruction* caller,
                                  HloComputation* callee) {
  CHECK(caller);
  CHECK(callee);
  DecrementCount(callee_computations_, callee);
  DecrementCount(callee->caller_computations_, this);

  if (callee->callers_ == reinterpret_cast<uintptr_t>(caller)) {
    // The callee had just this single caller, so we reset it to 0 (no caller).
    callee->callers_ = 0;
  } else {
    auto* map = callee->GetCallersMap();
    CHECK(map) << "Attempted to remove a caller " << caller->name()
               << " that did not call the computation " << name() << "."
               << callee->callers_;
    auto it = map->find(caller);
    CHECK(it != map->end())
        << "Attempted to remove a caller " << caller->name()
        << " that did not call the computation " << name() << ".";
    --it->second;
    // We don't convert back to the inline representation, since this case
    // should be rare.
  }
}

absl::flat_hash_map<HloInstruction*, int>* HloComputation::GetCallersMap() {
  if (static_cast<CallersType>(callers_ & kCallerTypeMask) ==
      CallersType::kCallerCountHashMap) {
    return reinterpret_cast<absl::flat_hash_map<HloInstruction*, int>*>(
        callers_ & ~kCallerTypeMask);
  }
  return nullptr;
}

absl::flat_hash_map<HloInstruction*, int>* const HloComputation::GetCallersMap()
    const {
  if (static_cast<CallersType>(callers_ & kCallerTypeMask) ==
      CallersType::kCallerCountHashMap) {
    return reinterpret_cast<absl::flat_hash_map<HloInstruction*, int>* const>(
        callers_ & ~kCallerTypeMask);
  }
  return nullptr;
}

HloInstruction* HloComputation::AddInstructionInternal(
    std::unique_ptr<HloInstruction> instruction) {
  if (parent() != nullptr) {
    instruction->UniquifyName(parent());
    instruction->UniquifyId(parent());
  }
  instruction->set_parent(this);
  HloInstruction* pinst = instruction.release();  // Take ownership
  HloInstructionInfo info;
  info.opcode_ = pinst->opcode();
  info.inst_ = pinst;
  VLOG(2) << "Adding instruction " << pinst << " " << pinst->name()
          << " from computation " << name() << " opcode " << info.opcode();
  uint32_t index = instructions_.size();
  instruction_count_++;
  pinst->index_in_parent_ = index;
  instructions_.push_back(info);
  for (HloComputation* called_computation : pinst->called_computations()) {
    CHECK(called_computation);
    // TODO(b/399394039): Consider enforcing that
    // called_computation->parent() != nullptr.
    CHECK(parent() == nullptr || called_computation->parent() == parent())
        << "Called computation " << called_computation->name()
        << " is not in the same module as " << name();
    AddCallee(pinst, called_computation);
  }
  return pinst;
}

HloInstruction* HloComputation::AddParameter(
    std::unique_ptr<HloInstruction> instruction) {
  CHECK(instruction->opcode() == HloOpcode::kParameter);
  CHECK(!IsFusionComputation() ||
        FusionInstruction()->operand_count() == param_instructions_.size());
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

absl::Status HloComputation::ReplaceEntryComputationParameter(
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

absl::Status HloComputation::RemoveParameter(int64_t param_no) {
  CHECK_GE(param_no, 0);
  CHECK_LT(param_no, param_instructions_.size());
  HloInstruction* param_instruction = param_instructions_[param_no];
  auto param_instruction_iterator = param_instructions_.begin() + param_no;
  param_instructions_.erase(param_instruction_iterator);
  // Throw removed fused parameter instruction away.
  TF_RETURN_IF_ERROR(ForceRemoveInstruction(param_instruction));

  while (param_no < param_instructions_.size()) {
    param_instruction = param_instructions_[param_no];
    HloInstruction* new_instr =
        AddInstructionInternal(HloInstruction::CreateParameter(
            param_no, param_instruction->shape(), StrCat("param_", param_no)));
    param_instruction->SetupDerivedInstruction(new_instr);
    TF_RETURN_IF_ERROR(param_instruction->ReplaceAllUsesWith(new_instr));
    param_instructions_[param_no] = new_instr;
    TF_RETURN_IF_ERROR(ForceRemoveInstruction(param_instruction));
    param_no++;
  }

  return absl::OkStatus();
}

HloInstruction* HloComputation::ReplaceParameter(
    int64_t param_no, std::unique_ptr<HloInstruction> instruction) {
  CHECK_GE(param_no, 0);
  CHECK_LT(param_no, param_instructions_.size());
  CHECK(instruction->opcode() == HloOpcode::kParameter);
  CHECK(!IsFusionComputation() ||
        FusionInstruction()->operand_count() == param_instructions_.size());

  instruction->set_parent(this);
  HloInstruction* new_instruction =
      AddInstructionInternal(std::move(instruction));
  HloInstruction* old_instruction = param_instructions_[param_no];
  TF_CHECK_OK(
      old_instruction->ReplaceAllUsesWithDifferentShape(new_instruction));
  param_instructions_[param_no] = new_instruction;
  TF_CHECK_OK(ForceRemoveInstruction(old_instruction));
  return new_instruction;
}

absl::Status HloComputation::RemoveUnusedParametersFromFusedComputation() {
  return RemoveUnusedParametersImpl(/*allow_non_fusion=*/false);
}

absl::Status HloComputation::RemoveUnusedParametersFromAnyComputation() {
  return RemoveUnusedParametersImpl(/*allow_non_fusion=*/true);
}

absl::Status HloComputation::RemoveUnusedParametersImpl(bool allow_non_fusion) {
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
  return absl::OkStatus();
}

bool HloComputation::IsSafelyRemovable(
    const HloInstruction* instruction, bool ignore_control_dependency,
    std::optional<
        absl::FunctionRef<std::vector<HloInstruction*>(const HloComputation*)>>
        computation_callers) const {
  // If the instruction has control predecessors or successors then we cannot
  // remove the instruction without violating ordering constraints (added, for
  // example, to avert interference due to buffer aliasing).
  if (!ignore_control_dependency && instruction->HasControlDependencies()) {
    return false;
  }

  if (instruction->opcode() == HloOpcode::kParameter) {
    // If there is no parent, it is safe to remove the child.
    if (instruction->parent() == nullptr) {
      return true;
    }
    // Entry computation parameters can never be removed.
    if (instruction->parent()->IsEntryComputation()) {
      return false;
    }
    // We generally want to be using the call graph to determine who the caller
    // is, as this back pointer is very fragile, however its not reasonable to
    // expect every caller to be passing in the call graph.
    if (IsFusionComputation()) {
      return true;
    }
    // If we can't fixup the caller, then we can't remove the parameter.
    if (!computation_callers.has_value()) {
      return false;
    }
    std::vector<HloInstruction*> callers =
        (*computation_callers)(instruction->parent());
    if (callers.empty()) {
      return false;
    }
    for (HloInstruction* caller :
         (*computation_callers)(instruction->parent())) {
      if (caller->opcode() != HloOpcode::kFusion &&
          caller->opcode() != HloOpcode::kCall &&
          caller->opcode() != HloOpcode::kAsyncStart) {
        // We don't handle callers with non-trivial control flow today.
        return false;
      }
    }
  }

  // All instruction generally are safe to remove.
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

absl::Status HloComputation::RemoveInstructionAndUnusedOperands(
    HloInstruction* instruction,
    std::optional<absl::FunctionRef<void(HloInstruction*)>> cleanup,
    bool ignore_control_dependencies,
    std::optional<
        absl::FunctionRef<std::vector<HloInstruction*>(const HloComputation*)>>
        computation_callers) {
  TF_RET_CHECK(root_instruction() != instruction);

  TF_RET_CHECK(instruction->IsDead());
  TF_RET_CHECK(IsSafelyRemovable(instruction, ignore_control_dependencies,
                                 computation_callers))
      << "Cannot remove instruction: " << instruction->ToString();
  // Remember the parent, in case we lose all references to it, in order to
  // clean up the callers.
  HloComputation* parent = instruction->parent();
  absl::flat_hash_set<HloInstruction*> removed;
  std::queue<HloInstruction*> worklist;
  worklist.push(instruction);
  std::vector<HloInstruction*> parameters_to_be_removed;
  while (!worklist.empty()) {
    HloInstruction* item = worklist.front();
    worklist.pop();

    if (removed.contains(item) || !item->IsDead() ||
        !IsSafelyRemovable(item, ignore_control_dependencies,
                           computation_callers) ||
        (item->HasSideEffect() && item != instruction)) {
      continue;
    }
    if (ignore_control_dependencies) {
      TF_RETURN_IF_ERROR(item->SafelyDropAllControlDependencies());
    } else if (item->HasControlDependencies()) {
      continue;
    }

    for (int i = 0; i < item->operand_count(); ++i) {
      worklist.push(item->mutable_operand(i));
    }

    if (cleanup != std::nullopt) {
      (*cleanup)(item);
    }
    if (item->opcode() == HloOpcode::kParameter) {
      // We cannot remove a parameter directly, because it may cause a
      // renumbering of other parameters which may invalidate some of the
      // pointers in the worklist.
      parameters_to_be_removed.push_back(item);
    } else {
      TF_RETURN_IF_ERROR(RemoveInstruction(item));
    }
    removed.insert(item);
  }
  // Sort into decreasing order by parameter number, otherwise the renumbering
  // of parameters when one parameter is deleted will cause issues.
  std::sort(parameters_to_be_removed.begin(), parameters_to_be_removed.end(),
            [](HloInstruction* a, HloInstruction* b) {
              return a->parameter_number() > b->parameter_number();
            });
  std::vector<HloInstruction*> callers;
  if (!parameters_to_be_removed.empty()) {
    if (parent != nullptr && computation_callers.has_value()) {
      callers = (*computation_callers)(parent);
    }
    // We generally want to be using the call graph to determine who the caller
    // is, as this back pointer is very fragile, however its not reasonable to
    // expect every caller to be passing in the call graph.
    if (callers.empty() && FusionInstruction() != nullptr) {
      callers = {FusionInstruction()};
    }
  }
  // Only attempt to remove parameters if we can fixup the caller.
  if (callers.empty()) {
    return absl::OkStatus();
  }
  for (HloInstruction* param : parameters_to_be_removed) {
    int64_t parameter_number = param->parameter_number();
    TF_RETURN_IF_ERROR(RemoveParameter(parameter_number));
    for (HloInstruction* caller : callers) {
      // The caller could have been eagerly removed.
      if (caller->IsDead()) {
        continue;
      }
      auto operand = caller->mutable_operand(parameter_number);
      caller->RemoveOperandAt(parameter_number);
      caller->DetachFrom(operand);
      // Cleanup operand shape embedded into the async-start shape.
      if (caller->opcode() == HloOpcode::kAsyncStart) {
        std::vector<Shape>* operand_shapes = caller->mutable_shape()
                                                 ->mutable_tuple_shapes(0)
                                                 ->mutable_tuple_shapes();
        operand_shapes->erase(operand_shapes->begin() + parameter_number);
      }
      if (operand->IsDead() &&
          operand->parent()->IsSafelyRemovable(
              operand, ignore_control_dependencies, computation_callers)) {
        TF_RETURN_IF_ERROR(
            operand->parent()->RemoveInstructionAndUnusedOperands(
                operand, cleanup, ignore_control_dependencies,
                computation_callers));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status HloComputation::RemoveInstruction(HloInstruction* instruction) {
  return RemoveInstructionImpl(instruction, /*ignore_safety_check=*/false);
}

absl::Status HloComputation::ForceRemoveInstruction(
    HloInstruction* instruction) {
  return RemoveInstructionImpl(instruction, /*ignore_safety_check=*/true);
}

absl::Status HloComputation::RemoveInstructionImpl(HloInstruction* instruction,
                                                   bool ignore_safety_check) {
  VLOG(2) << "Removing instruction " << instruction << " "
          << instruction->name() << " from computation " << name();
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

  HloInstructionInfo* info = &instructions_[instruction->index_in_parent_];
  DCHECK_EQ(info->inst(), instruction);
  to_be_deleted_.push_back(info->inst());  // Takes ownership
  to_be_deleted_.back()->DetachFromOperandsAndUsers();
  // Clear all operands to avoid Null operands.
  to_be_deleted_.back()->RemoveAllOperands();
  to_be_deleted_.back()->ClearCalledComputations();
  to_be_deleted_.back()->MarkAsDead();
  info->inst()->set_parent(nullptr);

  // If this instruction is a constant, clear the literal eagerly instead of
  // waiting for the instruction to be deleted in Cleanup(). This greatly
  // reduces the peak heap memory during constant folding.
  if (auto constant = DynCast<HloConstantInstruction>(to_be_deleted_.back())) {
    *constant->mutable_literal() = Literal();
  }
  // TODO(jeff): should we set info->opcode to something?
  info->inst_ =
      nullptr;  // Leave a hole: this is no longer part of "instructions()"
  instruction->index_in_parent_ = ~0u;
  instruction_count_--;
  DCHECK_EQ(instructions_.size() - to_be_deleted_.size(), instruction_count())
      << "instructions_.size(): " << instructions_.size()
      << ", to_be_deleted_.size(): " << to_be_deleted_.size();
  return absl::OkStatus();
}

void HloComputation::Cleanup() {
  if (to_be_deleted_.empty()) return;

  // Given that there are instructions to be deleted, there must be at least one
  // instruction not marked for deletion. Otherwise we have deleted *all*
  // instructions, which is probably a bug.
  DCHECK_GT(instruction_count(), 0);

  // Perform a stable compaction with the erase-remove idiom. We have to open
  // code it (instead of using std::erase(std::remove_if)) because we must
  // update the reverse mapping.
  auto is_marked_for_removal = [](const HloInstructionInfo& info) {
    return info.inst() == nullptr;
  };
  auto marked_it = absl::c_find_if(instructions_, is_marked_for_removal);
  DCHECK(marked_it < instructions_.end());
  for (auto it = marked_it + 1; it < instructions_.end(); ++it) {
    if (is_marked_for_removal(*it)) continue;
    // Update reverse mapping and overwrite the 'marked' entry.
    HloInstruction* unmarked_instruction = it->inst();
    unmarked_instruction->index_in_parent_ =
        std::distance(instructions_.begin(), marked_it);
    *marked_it++ = std::move(*it);
  }

  DCHECK(marked_it < instructions_.end());
  DCHECK_EQ(std::distance(marked_it, instructions_.end()),
            to_be_deleted_.size());
  DCHECK_EQ(instructions_.size() - to_be_deleted_.size(), instruction_count())
      << "instructions_.size(): " << instructions_.size()
      << ", to_be_deleted_.size(): " << to_be_deleted_.size();
  for (HloInstruction* marked_instruction : to_be_deleted_) {
    delete marked_instruction;
  }
  to_be_deleted_.clear();
  instructions_.resize(instruction_count());
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

  // `root_instruction_` can be equal to `new_root_instruction` and so it is
  // important that we call MarkAsNonRoot before calling MarkAsRoot.
  root_instruction_->MarkAsNonRoot();
  new_root_instruction->MarkAsRoot();
  root_instruction_ = new_root_instruction;
}

void HloComputation::ComputeInstructionPostOrder(
    HloInstruction* root, const ChannelDependencies& channel_dependencies,
    VisitMap& visited, std::vector<HloInstruction*>& post_order,
    std::vector<HloInstruction*>* dfs_stack_scratch) const {
  ForEachInstructionPostOrderImpl(
      [&post_order](HloInstruction* hlo) { post_order.push_back(hlo); }, root,
      channel_dependencies, visited, dfs_stack_scratch);
}

void HloComputation::ForEachInstructionPostOrderImpl(
    absl::FunctionRef<void(HloInstruction*)> func, HloInstruction* root,
    const ChannelDependencies& channel_dependencies, VisitMap& visited,
    std::vector<HloInstruction*>* dfs_stack_scratch) const {
  bool has_channel_dependencies = !channel_dependencies.empty();
  auto* dfs_stack = dfs_stack_scratch;
  dfs_stack->clear();

  // Pushes instruction to dfs stack only if it was not already processed.
  auto dfs_stack_push = [&](HloInstruction* instr) {
    VisitState state = visited.GetState(instr->index_in_parent_);
    if (state != VisitState::kVisited) dfs_stack->push_back(instr);
  };

  dfs_stack_push(root);
  while (!dfs_stack->empty()) {
    HloInstruction* current = dfs_stack->back();
    DCHECK_EQ(current->parent(), this)
        << "Instruction " << current->name()
        << " is not in the current computation (" << name() << ").";

    VisitMap::Handle h = current->index_in_parent_;
    VisitState state = visited.GetState(h);
    if (state == VisitState::kNew) {
      visited.SetState(h, VisitState::kVisiting);
    } else {
      dfs_stack->pop_back();
      if (state != VisitState::kVisited) {
        visited.SetState(h, VisitState::kVisited);
        func(current);
      }
      continue;
    }

    // Add channel dependencies.
    // Collectives with the same channel ID must be performed together, as these
    // represent MPMD-partitioned that will later be split into separate modules
    // and the order must be preserved.
    if (has_channel_dependencies && current != root) {
      auto it = channel_dependencies.find(current);
      if (it != channel_dependencies.end()) {
        absl::c_for_each(it->second, dfs_stack_push);
      }
    }

    // Add the operands to the stack in reverse order so the first operand is
    // processed first. This will produce a more natural ordering and a nicer
    // result for things like HLO stringification.
    const HloInstruction::InstructionVector& operands = current->operands();
    absl::c_for_each(tsl::gtl::make_range(operands.rbegin(), operands.rend()),
                     dfs_stack_push);

    // Add control predecessors to the stack.
    absl::c_for_each(current->control_predecessors(), dfs_stack_push);
  }
}

HloComputation::ChannelDependencies HloComputation::ComputeChannelDependencies()
    const {
  if (parent() && parent()->config().has_static_device_assignment() &&
      (parent()->config().static_device_assignment().computation_count() == 1 ||
       parent()->config().use_spmd_partitioning())) {
    return {};
  }

  using Instructions = absl::InlinedVector<HloInstruction*, 1>;
  absl::flat_hash_map<int64_t, Instructions> channel_groups;

  // Create dependencies between partitioned collectives.
  ChannelDependencies dependencies;
  for (const auto& inst : instructions_with_info()) {
    switch (inst.opcode()) {
      case HloOpcode::kAllReduce:
      case HloOpcode::kAllGather:
      case HloOpcode::kAllToAll:
      case HloOpcode::kCollectiveBroadcast:
      case HloOpcode::kCollectivePermute:
      case HloOpcode::kRaggedAllToAll:
      case HloOpcode::kReduceScatter: {
        HloInstruction* instruction = inst.inst();
        std::optional<int64_t> channel_id = instruction->channel_id();
        if (channel_id) {
          Instructions& group = channel_groups[*channel_id];
          dependencies[instruction] = group;
          group.push_back(instruction);
        }
        break;
      }
      default:
        break;
    }
  }
  return dependencies;
}

std::vector<HloInstruction*> HloComputation::MakeInstructionPostOrderFrom(
    HloInstruction& postorder_root) const {
  std::vector<HloInstruction*> post_order;
  VisitMap visited(instructions_.size());

  std::vector<HloInstruction*> dfs_stack_scratch;
  ComputeInstructionPostOrder(&postorder_root, ComputeChannelDependencies(),
                              visited, post_order, &dfs_stack_scratch);
  return post_order;
}

std::vector<HloInstruction*> HloComputation::MakeInstructionPostOrder() const {
  return MakeInstructionPostOrder(ComputeChannelDependencies());
}

std::vector<HloInstruction*> HloComputation::MakeInstructionPostOrder(
    const ChannelDependencies& channel_dependencies) const {
  std::vector<HloInstruction*> post_order;
  post_order.reserve(instruction_count());
  VisitMap visited(instructions_.size());
  std::vector<HloInstruction*> dfs_stack_scratch;
  dfs_stack_scratch.reserve(instruction_count());

  for (const auto& instruction : instructions()) {
    // We don't consider users outside any computation as real users. This can
    // happen when creating new instructions for replacement when cloning
    // computations.
    if (absl::c_all_of(instruction->users(), [](const HloInstruction* user) {
          return user->parent() == nullptr;
        })) {
      ComputeInstructionPostOrder(instruction, channel_dependencies, visited,
                                  post_order, &dfs_stack_scratch);
    }
  }
  CHECK_EQ(instruction_count(), post_order.size())
      << "number of instructions does not match post order size";
  return post_order;
}

std::vector<HloInstruction*>
HloComputation::MakeInstructionPostOrderWithReshapeFirst() const {
  std::vector<HloInstruction*> frontier_std;
  std::vector<HloInstruction*> frontier_reshapes;
  std::vector<HloInstruction*> sorted;
  absl::flat_hash_map<int, uint32_t> visitations;
  sorted.reserve(instruction_count());
  visitations.reserve(instruction_count());

  auto pop_frontier_element = [&frontier_std, &frontier_reshapes]() mutable {
    // Because the result of this sort is going to be reverse, check for
    // Reshapes later, which we want to occur earlier in the final result
    if (!frontier_std.empty()) {
      HloInstruction* const to_return = frontier_std.back();
      frontier_std.pop_back();
      return to_return;
    }
    if (!frontier_reshapes.empty()) {
      HloInstruction* const to_return = frontier_reshapes.back();
      frontier_reshapes.pop_back();
      return to_return;
    }
    return static_cast<HloInstruction*>(nullptr);
  };

  auto add_to_frontier = [&frontier_std, &frontier_reshapes](
                             HloInstruction* const instruction_to_add) mutable {
    if (instruction_to_add->opcode() == HloOpcode::kReshape) {
      frontier_reshapes.push_back(instruction_to_add);
    } else {
      frontier_std.push_back(instruction_to_add);
    }
  };

  // Add all instructions with no users inside the computation, including the
  // root instruction
  bool found_root_instruction = false;
  for (HloInstruction* const inst : instructions()) {
    if (inst->user_count() == 0) {
      if (inst == root_instruction()) {
        found_root_instruction = true;
      }
      add_to_frontier(inst);
    }
  }
  CHECK(found_root_instruction);

  while (HloInstruction* const inst = pop_frontier_element()) {
    sorted.push_back(inst);
    for (HloInstruction* const child : inst->operands()) {
      // Will increment, or set to 1 if not present
      visitations[child->unique_id()]++;
      if (child->user_count() == visitations[child->unique_id()]) {
        add_to_frontier(child);
      }
    }
  }

  std::reverse(sorted.begin(), sorted.end());
  CHECK_EQ(sorted.size(), instruction_count());
  return sorted;
}

void HloComputation::ForEachInstructionPostOrder(
    absl::FunctionRef<void(HloInstruction*)> func) const {
  VisitMap visited(instructions_.size());
  std::vector<HloInstruction*> dfs_stack_scratch;
  dfs_stack_scratch.reserve(instruction_count());
  auto channel_dependencies = ComputeChannelDependencies();
  for (const auto& instruction : instructions()) {
    // We don't consider users outside any computation as real users. This can
    // happen when creating new instructions for replacement when cloning
    // computations.
    if (absl::c_all_of(instruction->users(), [](const HloInstruction* user) {
          return user->parent() == nullptr;
        })) {
      ForEachInstructionPostOrderImpl(func, instruction, channel_dependencies,
                                      visited, &dfs_stack_scratch);
    }
  }
}

std::vector<HloComputation*> HloComputation::MakeEmbeddedComputationsList()
    const {
  absl::flat_hash_set<HloComputation*> visited;
  std::vector<HloComputation*> post_order;
  // The first element of the pair is the currently processed computation, the
  // second is iterator inside the instructions list of the computation that is
  // currently being processed.
  using ComputationIter =
      std::pair<HloComputation*, InstructionList::const_iterator>;
  std::stack<ComputationIter, absl::InlinedVector<ComputationIter, 8>> st;

  // We cannot directly push (this, instructions_.cbegin()) to the stack, as the
  // stack should contain only mutable computations. Also, we don't want to
  // include the computation itself in the list of embedded computations.
  for (const HloInstructionInfo& instruction : instructions_with_info()) {
    using PtrVec = PtrVec<HloComputation*>;
    auto process_called_computations = [&](const PtrVec& called_computations) {
      if (called_computations.empty()) return;
      // Put the called computations in reverse order onto the stack.
      // Otherwise we don't match the recursive enumeration of
      // computations, which processes the first called computation first.
      std::reverse_iterator<PtrVec::const_iterator> i(
          called_computations.end());
      std::reverse_iterator<PtrVec::const_iterator> rend(
          called_computations.begin());
      for (; i != rend; ++i) {
        HloComputation* called_computation = *i;
        if (visited.insert(called_computation).second) {
          st.emplace(called_computation,
                     called_computation->instructions_.cbegin());
        }
      }
    };
    process_called_computations(instruction->called_computations());
    while (!st.empty()) {
      auto& cur = st.top();
      HloComputation* computation = cur.first;
      if (cur.second == computation->instructions_.cend()) {
        st.pop();
        post_order.push_back(computation);
      } else {
        if (cur.second->inst() == nullptr) {
          ++cur.second;
        } else {
          HloOpcode opcode = cur.second->opcode();
          HloInstruction* next_instruction = cur.second->get();
          ++cur.second;
          if (HloInstruction::MightHaveCalledComputations(opcode)) {
            process_called_computations(
                next_instruction->called_computations());
          } else {
            DCHECK(next_instruction->called_computations().empty());
          }
        }
      }
    }
  }

  return post_order;
}

void HloComputation::Print(Printer* printer,
                           const HloPrintOptions& options) const {
  // Use post-order if order is not specified.
  Print(printer, options, /*instruction_order=*/{});
}

void HloComputation::Print(
    Printer* printer, const HloPrintOptions& options,
    absl::Span<const HloInstruction* const> instruction_order) const {
  const std::string tab(2 * options.indent_amount(), ' ');

  printer->Append(tab);

  if (!options.is_in_nested_computation()) {
    if (options.print_percent()) {
      printer->Append("%");
    }
    if (options.print_ids()) {
      // When print_ids() is false, exclude entry computation's name because it
      // includes and leads to non-deterministic fingerprint.
      printer->Append(name());
      printer->Append(" ");
    }
  }

  if (options.print_program_shape()) {
    ShapeUtil::PrintHumanString(printer,
                                ComputeProgramShape(options.print_ids()));
    printer->Append(" ");
  }
  printer->Append("{\n");

  {
    // Print the instructions in this computation.
    HloPrintOptions new_options =
        HloPrintOptions(options)
            .set_indent_amount(options.indent_amount() + 1)
            .set_is_in_nested_computation(true);

    CanonicalNameMap name_map;
    name_map.Reserve(instruction_count());
    auto print_one = [&](const HloInstruction* instruction) {
      DCHECK_EQ(this, instruction->parent());
      // 2 more spaces than just 'tab' due to indent_amount()+1 above
      printer->Append(tab);
      printer->Append("  ");
      if (instruction == root_instruction_) {
        printer->Append("ROOT ");
      }
      instruction->PrintWithCanonicalNameMap(printer, new_options, &name_map);
      printer->Append("\n");
    };
    // Use post-order if order is not specified.
    if (instruction_order.empty()) {
      ForEachInstructionPostOrder(print_one);
    } else {
      for (const HloInstruction* const instruction : instruction_order) {
        print_one(instruction);
      }
    }
  }

  printer->Append(tab);
  printer->Append("}");
  if (options.print_ids() && !IsMainThread()) {
    // When print_ids() is false, exclude entry computation's thread name
    // because it includes and leads to non-deterministic fingerprint.
    printer->Append(", execution_thread=\"");
    printer->Append(execution_thread());
    printer->Append("\"");
  }
  if (options.print_name_after_closing_brace() && instruction_count() > 5) {
    printer->Append(" // ");
    printer->Append(name());
  }
}

std::string HloComputation::ToString() const {
  return ToString(HloPrintOptions::Default());
}

std::string HloComputation::ToString(const HloPrintOptions& options) const {
  return ToString(options, MakeInstructionPostOrder());
}

std::string HloComputation::ToString(
    const HloPrintOptions& options,
    absl::Span<const HloInstruction* const> instruction_order) const {
  StringPrinter printer;
  Print(&printer, options, instruction_order);
  return std::move(printer).ToString();
}

absl::Cord HloComputation::ToCord(
    const HloPrintOptions& options,
    absl::Span<const HloInstruction* const> instruction_order) const {
  CordPrinter printer;
  Print(&printer, options, instruction_order);
  return std::move(printer).ToCord();
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
  proto.set_is_fusion_computation(IsFusionComputation());
  proto.set_execution_thread(IsMainThread() ? ""
                                            : std::string(execution_thread()));
  return proto;
}

/* static */ absl::StatusOr<std::unique_ptr<HloComputation>>
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

  TF_RETURN_IF_ERROR([&]() -> absl::Status {
    std::vector<bool> parameters_seen(parameter_count);
    int parameters_seen_count = 0;
    for (auto& instruction : instructions) {
      if (instruction->opcode() == HloOpcode::kParameter) {
        int64_t param_no = instruction->parameter_number();
        TF_RET_CHECK(param_no >= 0 && param_no < parameter_count)
            << "Invalid parameter number. Expected [0, " << parameter_count
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
    return absl::OkStatus();
  }());

  auto computation = absl::WrapUnique(
      new HloComputation(proto.name(), parameter_count, &instructions, root));
  computation->SetUniqueIdHelper(proto.id());
  if (proto.is_fusion_computation()) {
    computation->instruction_and_type_ =
        static_cast<uintptr_t>(InstructionType::kFusion);
  }
  if (!proto.execution_thread().empty()) {
    computation->SetExecutionThread(proto.execution_thread());
  }
  return computation;
}

void HloComputation::AppendInstructionsIntoCalledComputation(
    absl::Span<HloInstruction* const> instructions_to_append,
    HloInstruction* caller) {
  HloInstruction* root = instructions_to_append.front();
  TF_CHECK_OK(caller->CopyAllControlDepsFrom(root));
  TF_CHECK_OK(root->DropAllControlDeps());
  TF_CHECK_OK(root->ReplaceAllUsesWith(caller));
  if (root == root_instruction()) {
    set_root_instruction(caller);
  }
  TF_CHECK_OK(RemoveInstruction(root));
  for (size_t i = 1; i < instructions_to_append.size(); ++i) {
    HloInstruction* instruction = instructions_to_append[i];
    caller->AppendInstructionIntoCalledComputation(instruction);
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
  AppendInstructionsIntoCalledComputation(instructions_to_fuse,
                                          fusion_instruction);
  return fusion_instruction;
}

HloInstruction* HloComputation::CreateCallInstruction(
    absl::Span<HloInstruction* const> instructions_to_call) {
  HloInstruction* root = instructions_to_call.front();
  HloInstruction* call_instruction = AddInstruction(
      HloInstruction::CreateCall(root->shape(), root), root->name());
  AppendInstructionsIntoCalledComputation(instructions_to_call,
                                          call_instruction);
  return call_instruction;
}

HloInstruction* HloComputation::CreateCompositeCallInstruction(
    absl::Span<HloInstruction* const> instructions_to_call,
    const std::string& name, const std::string& attributes, int64_t version) {
  HloInstruction* root = instructions_to_call.front();
  HloInstruction* call_instruction =
      AddInstruction(HloInstruction::CreateCompositeCall(
                         root->shape(), root, name, attributes, version),
                     root->name());
  AppendInstructionsIntoCalledComputation(instructions_to_call,
                                          call_instruction);
  return call_instruction;
}

absl::StatusOr<HloInstruction*> HloComputation::CreateAsyncInstructions(
    HloInstruction* instruction, absl::Span<const Shape> context_shapes,
    absl::string_view async_execution_thread, bool replace,
    bool override_names) {
  HloInstruction* async_start;
  HloInstruction* async_done;
  if (instruction->opcode() == HloOpcode::kCopy) {
    // Until the async ops are unified, add specialized support for copy here.
    // TODO(b/319466176): Remove this special case once this bug is complete.
    // Note that CopyStart/CopyDone uses (dest_shape, src_shape, context)
    // convention while async-start/async-done uses ((src_shapes), dest_shape,
    // context).
    std::vector<Shape> context_shapes_tuple;
    context_shapes_tuple.reserve(context_shapes.size() + 2);
    Shape instruction_shape_destination = instruction->shape();
    context_shapes_tuple.push_back(instruction_shape_destination);
    Shape instruction_shape_source = instruction->operand(0)->shape();
    context_shapes_tuple.push_back(instruction_shape_source);
    context_shapes_tuple.insert(context_shapes_tuple.end(),
                                context_shapes.begin(), context_shapes.end());

    async_start = AddInstruction(HloInstruction::CreateCopyStart(
        ShapeUtil::MakeTupleShape(context_shapes_tuple),
        instruction->mutable_operand(0)));
    async_done = AddInstruction(HloInstruction::CreateUnary(
        instruction_shape_destination, HloOpcode::kCopyDone, async_start));
  } else {
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
    if (override_names) {
      parent()->SetAndUniquifyInstrName(
          root, absl::StrCat(instruction->name(), ".cloned"));
    }
    HloComputation* async_computation =
        parent_->AddEmbeddedComputation(builder.Build(root));
    std::vector<Shape> start_shapes = {
        ShapeUtil::MakeTupleShape(parameter_shapes), root->shape()};
    for (const Shape& context_shape : context_shapes) {
      start_shapes.push_back(context_shape);
    }
    async_start = AddInstruction(HloInstruction::CreateAsyncStart(
        ShapeUtil::MakeTupleShape(start_shapes), instruction->operands(),
        async_computation, async_execution_thread));
    async_done = AddInstruction(
        HloInstruction::CreateAsyncDone(root->shape(), async_start));
    if (override_names) {
      parent()->SetAndUniquifyInstrName(
          async_start, absl::StrCat(root->name(), ".call-start"));
      parent()->SetAndUniquifyInstrName(
          async_done, absl::StrCat(root->name(), ".call-done"));
    }
  }
  async_start->set_metadata(instruction->metadata());
  async_start->CopyBackendConfigFrom(instruction);
  async_done->set_metadata(instruction->metadata());
  async_done->CopyBackendConfigFrom(instruction);
  for (HloInstruction* control_pred : instruction->control_predecessors()) {
    TF_RETURN_IF_ERROR(control_pred->AddControlDependencyTo(async_start));
  }
  for (HloInstruction* control_successor : instruction->control_successors()) {
    TF_RETURN_IF_ERROR(async_done->AddControlDependencyTo(control_successor));
  }

  if (replace) {
    TF_RETURN_IF_ERROR(instruction->DropAllControlDeps());
    TF_RETURN_IF_ERROR(ReplaceInstruction(instruction, async_done));
  }
  return async_done;
}

absl::StatusOr<HloInstruction*> HloComputation::DeepCopyHelper(
    HloInstruction* instruction, ShapeIndex* index,
    absl::FunctionRef<HloInstruction*(HloInstruction* leaf,
                                      const ShapeIndex& leaf_index,
                                      HloComputation* computation)>
        copy_leaf) {
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

absl::StatusOr<HloInstruction*> HloComputation::DeepCopyInstruction(
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

absl::StatusOr<HloInstruction*>
HloComputation::DeepCopyInstructionWithCustomCopier(
    HloInstruction* instruction,
    absl::FunctionRef<HloInstruction*(HloInstruction* leaf,
                                      const ShapeIndex& leaf_index,
                                      HloComputation* computation)>
        copy_leaf) {
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
    program_shape.AddParameter(
        param_instruction->shape(),
        std::string(PrintName(param_instruction->name(), include_ids)));
  }
  *program_shape.mutable_result() = root_instruction_->shape();

  return program_shape;
}

bool HloComputation::EqualInternal(
    const HloComputation& other, bool is_layout_sensitive,
    std::optional<
        absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>>
        computations_comparator,
    bool ignore_channel_id_values, bool ignore_execution_thread) const {
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
      return a->EqualInternal(*b, is_layout_sensitive, computations_comparator,
                              ignore_channel_id_values,
                              ignore_execution_thread);
    };

    bool identical_ignoring_operands =
        ignore_channel_id_values
            ? pair.first->IdenticalIgnoringChannelIdValues(
                  *pair.second, operands_eq,
                  (computations_comparator ? *computations_comparator
                                           : comp_eq),
                  is_layout_sensitive)
            : pair.first->Identical(
                  *pair.second, operands_eq,
                  (computations_comparator ? *computations_comparator
                                           : comp_eq),
                  is_layout_sensitive);
    if (!identical_ignoring_operands) {
      return false;
    }
    for (size_t i = 0; i < pair.first->operands().size(); ++i) {
      worklist.push_back({pair.first->operand(i), pair.second->operand(i)});
    }
  }

  if (!ignore_execution_thread) {
    return execution_thread() == other.execution_thread();
  }
  return true;
}

absl::Status HloComputation::ReplaceWithNewInstruction(
    HloInstruction* old_instruction,
    std::unique_ptr<HloInstruction> new_instruction) {
  return ReplaceInstruction(old_instruction,
                            AddInstruction(std::move(new_instruction)));
}

absl::Status HloComputation::ReplaceWithNewEntryComputationParameter(
    HloInstruction* old_instruction,
    std::unique_ptr<HloInstruction> new_instruction) {
  return ReplaceInstruction(old_instruction, AddEntryComputationParameter(
                                                 std::move(new_instruction)));
}

absl::StatusOr<bool> HloComputation::ReplaceInstruction(
    HloInstruction* old_instruction, HloInstruction* new_instruction,
    bool preserve_sharding, bool relay_control_dependency,
    bool remove_unused_operands) {
  TF_RET_CHECK(
      ShapeUtil::Compatible(old_instruction->shape(), new_instruction->shape()))
      << absl::StreamFormat(
             "\"%s\" (%s) vs \"%s\" (%s)", old_instruction->name(),
             old_instruction->shape().ToString(/*print_layout=*/true),
             new_instruction->name(),
             new_instruction->shape().ToString(/*print_layout=*/true));
  return ReplaceInstructionWithDifferentShape(
      old_instruction, new_instruction, preserve_sharding,
      relay_control_dependency, remove_unused_operands);
}

absl::Status HloComputation::ReplaceInstruction(
    HloInstruction* old_instruction, HloInstruction* new_instruction) {
  TF_ASSIGN_OR_RETURN(bool changed,
                      ReplaceInstruction(old_instruction, new_instruction,
                                         /*preserve_sharding=*/false));
  DCHECK(changed);
  return absl::OkStatus();
}

absl::StatusOr<bool> HloComputation::ReplaceInstructionWithDifferentShape(
    HloInstruction* old_instruction, HloInstruction* new_instruction,
    bool preserve_sharding, bool relay_control_dependency,
    bool remove_unused_operands) {
  if (preserve_sharding && new_instruction->has_sharding() &&
      old_instruction->has_sharding() &&
      !new_instruction->has_compatible_sharding(old_instruction)) {
    VLOG(10) << "Skipping replacement due to incompatible sharding";
    return false;
  }
  if (relay_control_dependency) {
    TF_RETURN_IF_ERROR(
        new_instruction->CopyAllControlDepsFrom(old_instruction));
    TF_RETURN_IF_ERROR(old_instruction->DropAllControlDeps());
  } else if (old_instruction->HasControlDependencies()) {
    VLOG(10) << "Skipping replacement because old instruction has "
                "control dependencies";
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
  if (overwrite_op_name) {
    new_instruction->set_metadata(old_instruction->metadata());
  }
  if (new_instruction->frontend_attributes().map().empty()) {
    new_instruction->set_frontend_attributes(
        old_instruction->frontend_attributes());
  }
  MoveOriginalValue(old_instruction, new_instruction);

  // Like the metadata above, if the user didn't specify any sharding
  // information on the new instruction we should copy the old sharding
  // information (if any).
  if (!new_instruction->has_sharding()) {
    new_instruction->copy_sharding(old_instruction);
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
  if (remove_unused_operands) {
    TF_RETURN_IF_ERROR(RemoveInstructionAndUnusedOperands(
        old_instruction, /*cleanup=*/std::nullopt,
        /*ignore_control_dependencies=*/relay_control_dependency));
  } else {
    TF_RETURN_IF_ERROR(RemoveInstruction(old_instruction));
  }
  return true;
}

absl::Status HloComputation::ReplaceInstructionWithDifferentShape(
    HloInstruction* old_instruction, HloInstruction* new_instruction) {
  TF_ASSIGN_OR_RETURN(bool changed, ReplaceInstructionWithDifferentShape(
                                        old_instruction, new_instruction,
                                        /*preserve_sharding=*/false));
  DCHECK(changed);
  return absl::OkStatus();
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

absl::Status HloComputation::AcceptWithOperandOrder(
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
    absl::FunctionRef<const HloInstruction*(const HloInstruction*)> replace,
    const HloComputation& computation,
    const HloComputation::InstructionList& ordered_instructions,
    std::vector<std::unique_ptr<HloInstruction>>& unordered_instructions) {
  using InstructionSorter = MappedPtrContainerSorter<HloInstruction>;
  auto instruction_mapper = [&context, replace](const HloInstruction* i) {
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
  auto unmapped_ptr_index =
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
    absl::FunctionRef<const HloInstruction*(const HloInstruction*)> replace,
    const HloComputation::InstructionList& sorted_instructions) {
  auto instruction_mapper = [&context, replace](const HloInstruction* i) {
    return context.FindInstruction(replace(i));
  };
  for (const HloInstructionInfo& instruction : sorted_instructions) {
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
  return CloneInContext(*context, replacements, extra_parameters, suffix,
                        new_root);
}

std::unique_ptr<HloComputation> HloComputation::CloneInContext(
    HloCloneContext& context,
    const absl::flat_hash_map<const HloInstruction*,
                              std::unique_ptr<HloInstruction>>* replacements,
    absl::Span<const HloInstruction* const> extra_parameters,
    const std::string& suffix, const HloInstruction* new_root) const {
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
  std::vector<const HloInstruction*> dfs_stack;
  for (const auto& instr : instructions()) {
    const HloInstruction* new_instr = replace(instr);
    if (!new_instr) {
      continue;
    }
    dfs_stack.clear();
    dfs_stack.push_back(new_instr);

    while (!dfs_stack.empty()) {
      auto* cur = dfs_stack.back();
      auto it = visited.find(cur);
      if (it != visited.end()) {
        dfs_stack.pop_back();
        if (it->second == VisitState::kVisited) {
          continue;
        }
        CHECK_EQ(it->second, VisitState::kVisiting);
        postorder.push_back(cur);
        it->second = VisitState::kVisited;
        continue;
      }

      visited.insert({cur, VisitState::kVisiting});
      for (HloInstruction* operand : cur->operands()) {
        const HloInstruction* new_operand = replace(operand);
        if (new_operand) {
          dfs_stack.push_back(new_operand);
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
      new_operands.push_back(context.GetInstruction(replaced_operand));
    }
    std::unique_ptr<HloInstruction> new_instr =
        instr->CloneWithNewOperands(instr->shape(), new_operands, &context);
    if (instr->opcode() == HloOpcode::kParameter &&
        instr->parameter_replicated_at_leaf_buffers().has_value()) {
      new_instr->set_parameter_replicated_at_leaf_buffers(
          instr->parameter_replicated_at_leaf_buffers().value());
    }
    instructions.push_back(std::move(new_instr));
  }

  // To make clone behavior match uncloned behavior, we reorder instructions to
  // match the order in instructions_.
  SortClonedInstructions(context, replace, *this, instructions_, instructions);

  Builder builder(suffix.empty() ? std::string(name())
                                 : absl::StrCat(name(), ".", suffix));
  for (auto& instr : instructions) {
    builder.AddInstruction(std::move(instr));
  }
  auto result = builder.Build(
      /*root_instruction=*/context.GetInstruction(replace(new_root)));

  // Clone control dependencies.
  for (auto instr : postorder) {
    HloInstruction* new_instr = context.GetInstruction(instr);
    for (auto successor : instr->control_successors()) {
      auto replaced_successor = replace(successor);
      // successor may not have been remapped, because it might have been
      // removed by the replacements map.
      if (replaced_successor != nullptr) {
        TF_CHECK_OK(new_instr->AddControlDependencyTo(
            context.GetInstruction(replaced_successor)));
      }
    }
  }

  // To make clone behavior match uncloned behavior, we reorder the user and
  // control lists, kept by cloned instructions.
  SortClonedInstructionUsersAndControlLists(context, replace, instructions_);

  context.MapComputation(this, result.get());
  result->SetExecutionThread(execution_thread());
  return result;
}

void HloComputation::UniquifyName(NameUniquer* name_uniquer) {
  name_ = name_uniquer->GetUniqueName(name_);
}

void HloComputation::UniquifyName(HloModule* module) {
  UniquifyName(&module->computation_name_uniquer());
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

bool HloComputation::CanExpandIntoSingleInstruction() const {
  return absl::c_all_of(
      instructions(), [root = root_instruction()](const HloInstruction* instr) {
        return root == instr || instr->opcode() == HloOpcode::kParameter;
      });
}

void HloComputation::ClearUniqueIdInternal() { SetUniqueIdHelper(-1); }

void HloComputation::SetUniqueId(int64_t id) {
  CHECK_EQ(unique_id_, -1);
  CHECK_GE(id, 0);
  SetUniqueIdHelper(id);
}

void HloComputation::SetUniqueIdHelper(int64_t id) {
  // The caller/callee computations are ordered by unique ID, so we need to
  // remove and readd them to our neighbor's data structures.
  for (auto& [computation, count] : caller_computations_) {
    auto it = computation->callee_computations_.find(this);
    CHECK(it != computation->callee_computations_.end());
    CHECK_EQ(it->second, count);
    computation->callee_computations_.erase(it);
  }
  for (auto& [computation, count] : callee_computations_) {
    auto it = computation->caller_computations_.find(this);
    CHECK(it != computation->caller_computations_.end());
    CHECK_EQ(it->second, count);
    computation->caller_computations_.erase(it);
  }
  unique_id_ = id;
  for (auto& [computation, count] : caller_computations_) {
    computation->callee_computations_[this] = count;
  }
  for (auto& [computation, count] : callee_computations_) {
    computation->caller_computations_[this] = count;
  }
}

}  // namespace xla
