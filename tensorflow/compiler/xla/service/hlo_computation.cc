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

#include <stddef.h>
#include <algorithm>
#include <functional>
#include <list>
#include <queue>
#include <set>
#include <sstream>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using ::tensorflow::strings::StrCat;

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
  return WrapUnique(new HloComputation(name_, parameter_count, &instructions_,
                                       root, fusion_instruction_));
}

HloComputation::HloComputation(
    const string& name, int parameter_count,
    std::vector<std::unique_ptr<HloInstruction>>* instructions,
    HloInstruction* root_instruction, HloInstruction* fusion_instruction)
    : name_(name),
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
  Reparent(instruction.get());
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
    string param_name = param_instruction->parameter_name();
    // Fusion parameters are named foo.param_1, bar.param_2, etc. We are
    // renumbering the parameters so replace the final number in the name with
    // the updated value.
    const string param_underscore = ".param_";
    size_t index = param_name.rfind(param_underscore);
    if (index == string::npos) {
      string after_param = name().substr(index + param_underscore.size());
      int64 numeric_suffix;
      if (tensorflow::strings::safe_strto64(after_param, &numeric_suffix)) {
        param_name =
            StrCat(param_name.substr(0, index), param_underscore, param_no);
      }
    }

    HloInstruction* new_instr =
        AddInstructionInternal(HloInstruction::CreateParameter(
            param_no, param_instruction->shape(), param_name));
    TF_RETURN_IF_ERROR(param_instruction->ReplaceAllUsesWith(new_instr));
    param_instructions_[param_no] = new_instr;
    TF_RETURN_IF_ERROR(RemoveInstruction(param_instruction));
    param_no++;
  }

  return Status::OK();
}

void HloComputation::Reparent(HloInstruction* instruction) {
  instruction->set_parent(this);
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

  if (instruction->HasSideEffect()) {
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
        item == root_instruction() || !IsRemovable(item)) {
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

  TF_RET_CHECK(instruction_iterators_.count(instruction) != 0);
  auto inst_it = instruction_iterators_.at(instruction);
  (*inst_it)->set_parent(nullptr);
  instruction->DetachFromOperands();
  instructions_.erase(inst_it);
  return Status::OK();
}

void HloComputation::set_root_instruction(
    HloInstruction* new_root_instruction) {
  // The shape of the root (ignoring layout) is an invariant of the computation
  // for non-fusion cases.
  if (!IsFusionComputation()) {
    CHECK(ShapeUtil::Compatible(new_root_instruction->shape(),
                                root_instruction_->shape()))
        << new_root_instruction->shape().ShortDebugString()
        << " is incompatible with "
        << root_instruction_->shape().ShortDebugString();
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

// Helper class which computes the post order of an expression rooted at a
// particular instruction.
class InstructionPostOrderer : public DfsHloVisitorWithDefault {
 public:
  // added_instructions is the set of instructions which have already been
  // accounted for in the post order in previous invocations of
  // GetOrder. Without this mechanism, instructions which are predecessors of
  // multiple root instructions of the computation can be added to the post
  // order more than once.
  static std::list<HloInstruction*> GetOrder(
      HloInstruction* root,
      tensorflow::gtl::FlatSet<HloInstruction*>* added_instructions) {
    InstructionPostOrderer orderer(added_instructions);
    TF_CHECK_OK(root->Accept(&orderer));
    return std::move(orderer.post_order_);
  }

 private:
  explicit InstructionPostOrderer(
      tensorflow::gtl::FlatSet<HloInstruction*>* added_instructions)
      : added_instructions_(added_instructions) {}
  ~InstructionPostOrderer() override {}

  Status DefaultAction(HloInstruction* hlo_instruction) override {
    if (added_instructions_->count(hlo_instruction) == 0) {
      post_order_.push_back(hlo_instruction);
      added_instructions_->insert(hlo_instruction);
    }
    return Status::OK();
  }

  std::list<HloInstruction*> post_order_;
  tensorflow::gtl::FlatSet<HloInstruction*>* added_instructions_;
};

// Helper which builds a post order of the HLO call graph.
void ComputeComputationPostOrder(
    HloComputation* computation,
    tensorflow::gtl::FlatSet<HloComputation*>* visited,
    std::list<HloComputation*>* post_order) {
  if (visited->count(computation) > 0) {
    return;
  }

  for (auto* instruction : computation->instructions()) {
    for (HloComputation* called_computation :
         instruction->called_computations()) {
      ComputeComputationPostOrder(called_computation, visited, post_order);
    }
  }

  visited->insert(computation);
  post_order->push_back(computation);
}

}  // namespace

std::list<HloInstruction*> HloComputation::MakeInstructionPostOrder() const {
  std::list<HloInstruction*> post_order;
  std::list<HloInstruction*> trace_instructions;
  tensorflow::gtl::FlatSet<HloInstruction*> added_instructions;
  for (auto& instruction : instructions_) {
    if (instruction->opcode() == HloOpcode::kTrace) {
      // Trace instructions aren't handled by the DFS visitor. Add trace
      // instructions to the post order at the end (necessarily they have no
      // users).
      trace_instructions.push_back(instruction.get());
    } else if (instruction->users().empty()) {
      post_order.splice(post_order.end(),
                        InstructionPostOrderer::GetOrder(instruction.get(),
                                                         &added_instructions));
    }
  }
  post_order.splice(post_order.end(), trace_instructions);
  CHECK_EQ(instructions_.size(), post_order.size())
      << "number of instructions does not match post order size";
  return post_order;
}

std::list<HloComputation*> HloComputation::MakeEmbeddedComputationsList()
    const {
  tensorflow::gtl::FlatSet<HloComputation*> visited;
  std::list<HloComputation*> post_order;

  // To avoid special handling of this computation, cast away const of
  // 'this'. 'this' is immediately removed from the post order after
  // construction.
  ComputeComputationPostOrder(const_cast<HloComputation*>(this), &visited,
                              &post_order);

  // We don't want to include this computation in the post order.
  CHECK_EQ(this, post_order.back());
  post_order.pop_back();

  return post_order;
}

string HloComputation::ToString(int nested_level,
                                bool include_large_constants) const {
  std::ostringstream s;
  for (int i = 0; i < nested_level; i++) {
    s << "    ";
  }
  s << "%" << name() << " " << ShapeUtil::HumanString(ComputeProgramShape())
    << " {\n";
  for (const HloInstruction* instruction : MakeInstructionPostOrder()) {
    for (int i = 0; i < nested_level; i++) {
      s << "    ";
    }
    s << "  " << (instruction == root_instruction_ ? "ROOT " : "")
      << instruction->ToString(
             /*compact_operands=*/false,
             /*include_metadata=*/true,
             /*include_large_constants=*/include_large_constants)
      << "\n";
  }
  for (int i = 0; i < nested_level; i++) {
    s << "    ";
  }
  s << "}";
  return s.str();
}

HloComputationProto HloComputation::ToProto() const {
  HloComputationProto proto;
  proto.set_name(name_);
  for (const HloInstruction* instruction : MakeInstructionPostOrder()) {
    HloInstructionProto instruction_proto = instruction->ToProto();
    proto.add_instructions()->Swap(&instruction_proto);
  }
  proto.set_root_name(root_instruction()->name());
  return proto;
}

/* static */ StatusOr<std::unique_ptr<HloComputation>>
HloComputation::CreateFromProto(
    HloModule* module, const HloComputationProto& proto,
    const tensorflow::gtl::FlatMap<string, HloComputation*>& computation_map,
    const std::function<void(std::unique_ptr<HloComputation>)>&
        add_fused_computation,
    HloInstruction* fusion_instruction) {
  std::vector<std::unique_ptr<HloInstruction>> instructions;
  tensorflow::gtl::FlatMap<string, HloInstruction*> instruction_map;
  int64 parameter_count = 0;
  for (const HloInstructionProto& instruction_proto : proto.instructions()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloInstruction> instruction,
                        HloInstruction::CreateFromProto(
                            module, instruction_proto, instruction_map,
                            computation_map, add_fused_computation));
    if (instruction->opcode() == HloOpcode::kParameter) {
      parameter_count++;
    }
    TF_RET_CHECK(!ContainsKey(instruction_map, instruction->name()));
    instruction_map[instruction->name()] = instruction.get();
    instructions.push_back(std::move(instruction));
  }

  TF_RET_CHECK(!proto.root_name().empty());
  TF_RET_CHECK(ContainsKey(instruction_map, proto.root_name()));
  HloInstruction* root = instruction_map.at(proto.root_name());
  return WrapUnique(new HloComputation(
      proto.name(), parameter_count, &instructions, root, fusion_instruction));
}

void HloComputation::FuseInstructionsInto(
    tensorflow::gtl::ArraySlice<HloInstruction*> instructions_to_fuse,
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
    tensorflow::gtl::ArraySlice<HloInstruction*> instructions_to_fuse,
    HloInstruction::FusionKind fusion_kind) {
  HloInstruction* root = instructions_to_fuse.front();
  HloInstruction* fusion_instruction = AddInstruction(
      HloInstruction::CreateFusion(root->shape(), fusion_kind, root));
  FuseInstructionsInto(instructions_to_fuse, fusion_instruction);
  return fusion_instruction;
}

HloInstruction* HloComputation::CreateFusionInstructionForBackwardConvolution(
    tensorflow::gtl::ArraySlice<HloInstruction*> instructions_to_fuse,
    HloInstruction::FusionKind fusion_kind, const Window& window,
    const ConvolutionDimensionNumbers& conv_dnums) {
  CHECK(HloInstruction::FusionKind::kConvBackwardFilter == fusion_kind ||
        HloInstruction::FusionKind::kConvBackwardInput == fusion_kind);
  HloInstruction* root = instructions_to_fuse.front();
  HloInstruction* fusion_instruction =
      AddInstruction(HloInstruction::CreateFusionForBackwardConvolution(
          root->shape(), fusion_kind, window, conv_dnums, root));
  FuseInstructionsInto(instructions_to_fuse, fusion_instruction);
  return fusion_instruction;
}

StatusOr<HloInstruction*> HloComputation::DeepCopyHelper(
    HloInstruction* instruction, const ShapeTree<bool>* indices_to_copy,
    ShapeTree<HloInstruction*>* copies_added, ShapeIndex* index) {
  if (ShapeUtil::IsArray(instruction->shape())) {
    if (indices_to_copy == nullptr || indices_to_copy->element(*index)) {
      // Use kCopy to copy array elements
      HloInstruction* copy = AddInstruction(HloInstruction::CreateUnary(
          instruction->shape(), HloOpcode::kCopy, instruction));
      if (copies_added != nullptr) {
        *copies_added->mutable_element(*index) = copy;
      }
      return copy;
    } else {
      // Array elements which are not to be copied are passed through
      // transparently.
      return instruction;
    }
  } else if (ShapeUtil::IsTuple(instruction->shape())) {
    std::vector<HloInstruction*> elements;
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(instruction->shape());
         i++) {
      HloInstruction* gte =
          AddInstruction(HloInstruction::CreateGetTupleElement(
              ShapeUtil::GetTupleElementShape(instruction->shape(), i),
              instruction, i));

      index->push_back(i);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * element,
          DeepCopyHelper(gte, indices_to_copy, copies_added, index));
      elements.push_back(element);
      index->pop_back();
    }
    return AddInstruction(HloInstruction::CreateTuple(elements));
  } else {
    return FailedPrecondition(
        "Can only copy array and tuple shaped instructions");
  }
}

StatusOr<HloInstruction*> HloComputation::DeepCopyInstruction(
    HloInstruction* instruction, const ShapeTree<bool>* indices_to_copy,
    ShapeTree<HloInstruction*>* copies_added) {
  if (instruction->parent() != this) {
    return FailedPrecondition(
        "Can't deep copy instruction %s: instruction is not in computation %s",
        instruction->name().c_str(), name().c_str());
  }

  if (indices_to_copy != nullptr &&
      !ShapeUtil::Compatible(instruction->shape(), indices_to_copy->shape())) {
    return FailedPrecondition(
        "Can't deep copy instruction %s: given shape tree of indices to copy "
        "has incompatible shape",
        instruction->name().c_str());
  }

  ShapeIndex index;
  return DeepCopyHelper(instruction, indices_to_copy, copies_added, &index);
}

ProgramShape HloComputation::ComputeProgramShape() const {
  ProgramShape program_shape;

  for (auto* param_instruction : param_instructions_) {
    *program_shape.add_parameters() = param_instruction->shape();
    *program_shape.add_parameter_names() = param_instruction->parameter_name();
  }
  *program_shape.mutable_result() = root_instruction_->shape();

  LayoutUtil::ClearLayout(&program_shape);
  return program_shape;
}

bool HloComputation::operator==(const HloComputation& other) const {
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
  TF_RET_CHECK(ShapeUtil::Compatible(old_instruction->shape(),
                                     new_instruction->shape()));
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

std::unique_ptr<HloReachabilityMap> HloComputation::ComputeReachability()
    const {
  const std::list<HloInstruction*> all = MakeInstructionPostOrder();
  auto result = MakeUnique<HloReachabilityMap>(all);

  std::vector<HloInstruction*> inputs;
  for (const HloInstruction* hlo : all) {
    inputs.assign(hlo->operands().begin(), hlo->operands().end());
    inputs.insert(inputs.end(), hlo->control_predecessors().begin(),
                  hlo->control_predecessors().end());
    result->SetReachabilityToUnion(inputs, hlo);
  }
  return result;
}

void HloComputation::UpdateReachabilityThroughInstruction(
    const HloInstruction* instruction, HloReachabilityMap* reachability_map) {
  std::queue<const HloInstruction*> worklist;
  worklist.push(instruction);

  std::vector<HloInstruction*> inputs;

  while (!worklist.empty()) {
    const HloInstruction* item = worklist.front();
    worklist.pop();

    inputs.assign(item->operands().begin(), item->operands().end());
    inputs.insert(inputs.end(), item->control_predecessors().begin(),
                  item->control_predecessors().end());

    if (reachability_map->SetReachabilityToUnion(inputs, item)) {
      // Add immediate successors to worklist.
      for (const HloInstruction* user : item->users()) {
        worklist.push(user);
      }
      for (const HloInstruction* succ : item->control_successors()) {
        worklist.push(succ);
      }
    }
  }
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
          << tensorflow::str_util::Join(
                 unreachable_roots, "\n\t",
                 [](string* out, const HloInstruction* hlo) {
                   tensorflow::strings::StrAppend(out, hlo->ToString());
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
    const std::vector<const HloInstruction*>& order) const {
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
    DfsHloVisitor*, const std::vector<const HloInstruction*>&) const;
template Status HloComputation::AcceptOrdered(
    ConstDfsHloVisitor*, const std::vector<const HloInstruction*>&) const;

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

std::unique_ptr<HloComputation> HloComputation::Clone(const string& suffix,
                                                      HloModule* module) {
  return CloneWithReplacements(
      /*replacements=*/std::unordered_map<const HloInstruction*,
                                          std::unique_ptr<HloInstruction>>(),
      module, suffix);
}

std::unique_ptr<HloComputation> HloComputation::CloneWithReplacements(
    std::unordered_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
        replacements,
    HloModule* module, const string& suffix) {
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
  std::vector<HloInstruction*> postorder;
  for (HloInstruction* instr : MakeInstructionPostOrder()) {
    if (HloInstruction* replacement = replace(instr)) {
      postorder.push_back(replacement);
    }
  }

  std::unordered_map<HloInstruction*, HloInstruction*> clone_map;
  std::vector<std::unique_ptr<HloInstruction>> instructions;
  std::unique_ptr<HloInstruction> new_instr = nullptr;
  for (auto instr : postorder) {
    std::vector<HloInstruction*> new_operands;
    for (auto operand : instr->operands()) {
      auto replaced_operand = replace(operand);
      // If replaced_operand is null, that means 'replacements' asked us not to
      // include operand in the new computation.  But we can't do that, because
      // operand is used by instr.
      CHECK_NE(replaced_operand, nullptr)
          << "replacements map tried to eliminate a used instruction "
          << operand->ToString() << ", used by " << instr->ToString();
      new_operands.push_back(FindOrDie(clone_map, replaced_operand));
    }
    new_instr =
        instr->CloneWithNewOperands(instr->shape(), new_operands, module);
    InsertOrDie(&clone_map, instr, new_instr.get());
    instructions.push_back(std::move(new_instr));
  }
  Builder builder(name() + "." + suffix);
  for (auto& instr : instructions) {
    builder.AddInstruction(std::move(instr));
  }
  auto result = builder.Build(
      /*root_instruction=*/FindOrDie(clone_map, replace(root_instruction())));

  // Clone control dependencies.
  for (auto instr : postorder) {
    HloInstruction* new_instr = FindOrDie(clone_map, instr);
    for (auto successor : instr->control_successors()) {
      auto replaced_successor = replace(successor);

      // successor may not be in clone_map, because it might have been
      // removed by the replacements map.
      if (replaced_successor == nullptr) {
        continue;
      }

      TF_CHECK_OK(new_instr->AddControlDependencyTo(
          FindOrDie(clone_map, replaced_successor)));
    }
  }

  // We cloned the elements of 'replacements', so they're all going to be
  // destroyed.  HloInstructions need to be detached from their operands before
  // they're destroyed, otherwise they stick around in the operands' users lists
  // and cause use-after-frees.
  for (auto& kv : replacements) {
    if (std::unique_ptr<HloInstruction>& new_instr = kv.second) {
      new_instr->DetachFromOperands();
    }
  }

  return result;
}

void HloComputation::UniquifyName(NameUniquer* name_uniquer) {
  name_ = name_uniquer->GetUniqueName(name_);
}

}  // namespace xla
