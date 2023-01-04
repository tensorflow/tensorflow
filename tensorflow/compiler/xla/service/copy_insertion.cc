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

#include "tensorflow/compiler/xla/service/copy_insertion.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <sstream>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/any.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/compile_time_cap.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_buffer.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {
namespace {

using absl::StrAppend;

bool IsReadonlyEntryParameterValue(const HloValue& value) {
  const HloComputation* computation = value.defining_instruction()->parent();
  return value.defining_instruction()->opcode() == HloOpcode::kParameter &&
         computation == computation->parent()->entry_computation() &&
         !computation->parent()->input_output_alias_config().ParameterHasAlias(
             value.defining_instruction()->parameter_number(), value.index());
}

bool IsConstantValue(const HloValue& value) {
  return value.defining_instruction()->opcode() == HloOpcode::kConstant;
}

bool ValueIsReadOnly(const HloValue& value) {
  return IsConstantValue(value) || IsReadonlyEntryParameterValue(value);
}

// Data structure describing the action which should be taken on parts of a
// computation buffers, with respect to the adding of special case copies.
struct SpecialCaseCopyPolicy {
  // Insert a copy if the same buffer is found at multiple indices within the
  // output tuple.
  bool copy_root_replicated_buffers = false;
  // If true, insert a copy if a buffer coming from a constant or a parameter
  // is found within the output tuple.
  bool copy_parameters_and_constants = false;
};

SpecialCaseCopyPolicy GetSpecialCaseCopyPolicy(const CallGraphNode& node,
                                               HloModule* module,
                                               HloComputation* computation) {
  SpecialCaseCopyPolicy policy;
  if (computation == module->entry_computation()) {
    policy.copy_parameters_and_constants = true;
    policy.copy_root_replicated_buffers = true;
  }
  return policy;
}

bool ShouldCopyRootValue(const HloValue& value,
                         const SpecialCaseCopyPolicy& policy) {
  if (policy.copy_parameters_and_constants) {
    return ValueIsReadOnly(value);
  }
  return false;
}

// Deep copy the given instructions 'from' and 'to' at the ShapeIndexes given in
// 'indices_to_copy'. Add control edges from the respective kCopy instructions
// in deep copy of 'from' to the respective kCopy instruction in the deep copy
// of 'to'.
//
// Requirements: 'from' and 'to' must have compatible shapes.
//
// For example, suppose 'from' and 'to' are two-element tuples where index 0 is
// the only index to copy. Prior to deep-copying we have:
//
//
//      'from'
//         |
//        ...
//         |
//       'to'
//
// DeepCopyAndAddControlEdges produces:
//
//       'from'
//        /   \
//      GTE   GTE
//       |     |
//     Copy    |
//    /   \   /
//   |    Tuple
//   |      |
//  ctrl   ...
//  edge    |
//   |      |
//   |    'to'
//   |    /   \
//   |  GTE   GTE
//    \  |     |
//     Copy    |
//        \   /
//        Tuple
//
StatusOr<std::pair<HloInstruction*, HloInstruction*>>
DeepCopyAndAddControlEdges(HloInstruction* from, HloInstruction* to,
                           const ShapeTree<bool>& indices_to_copy) {
  DCHECK(ShapeUtil::Compatible(from->shape(), to->shape()));
  // to/from_copy_tree hold the kCopy instruction produces by the deep
  // copies. Elements which are not copied (indices_to_copy.element(index) ==
  // false) have nullptr at that index.
  ShapeTree<HloInstruction*> from_copy_tree(from->shape(),
                                            /*init_value=*/nullptr);
  TF_ASSIGN_OR_RETURN(HloInstruction * from_deep_copy,
                      from->parent()->DeepCopyInstruction(
                          from, &indices_to_copy, &from_copy_tree));

  ShapeTree<HloInstruction*> to_copy_tree(to->shape(), /*init_value=*/nullptr);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * to_deep_copy,
      to->parent()->DeepCopyInstruction(to, &indices_to_copy, &to_copy_tree));

  // Add control edges between the respective kCopy instructions.
  for (const auto& pair : from_copy_tree) {
    const ShapeIndex& index = pair.first;
    HloInstruction* from_copy = pair.second;
    HloInstruction* to_copy = to_copy_tree.element(index);
    if (from_copy == nullptr) {
      TF_RET_CHECK(to_copy == nullptr);
      continue;
    }
    TF_RET_CHECK(to_copy != nullptr);
    TF_RETURN_IF_ERROR(from_copy->AddControlDependencyTo(to_copy));
  }

  return std::make_pair(from_deep_copy, to_deep_copy);
}

// Compute the indices of the loop state which need copies in order to avoid
// live range interference. Generally, an element in the loop state does not
// need to be copied if the element is passed through transparently through the
// body.
//
// Returns whether any indices need to be copied.
bool IndicesToCopyForWhile(const HloDataflowAnalysis& dataflow,
                           const HloInstruction* xla_while,
                           ShapeTree<bool>* indices_to_copy) {
  DCHECK(ShapeUtil::Compatible(indices_to_copy->shape(), xla_while->shape()));

  bool any_copies = false;
  const HloInstruction* init = xla_while->operand(0);
  for (auto& pair : *indices_to_copy) {
    const ShapeIndex& index = pair.first;
    bool& should_copy = pair.second;
    // If there is any ambiguity, then loop state must be copied.
    if (dataflow.GetValueSet(init, index).values().size() > 1 ||
        dataflow.GetValueSet(xla_while, index).values().size() > 1) {
      should_copy = true;
    } else {
      // If the output of the while instruction is not the same as the init
      // value of the while, then this element is not passed through the body
      // transparently and must be copied.
      should_copy = dataflow.GetUniqueValueAt(xla_while, index) !=
                    dataflow.GetUniqueValueAt(init, index);
    }
    any_copies |= should_copy;
  }
  return any_copies;
}

// Compute the indices of the conditional outputs which need copies. Umambiguous
// buffers(buffer with only one value) don't need copies.
bool IndicesToCopyForConditional(const HloDataflowAnalysis& dataflow,
                                 const HloInstruction* xla_conditional,
                                 ShapeTree<bool>* indices_to_copy) {
  DCHECK(ShapeUtil::Compatible(indices_to_copy->shape(),
                               xla_conditional->shape()));

  bool any_copies = false;
  for (auto& pair : *indices_to_copy) {
    const ShapeIndex& index = pair.first;
    bool& should_copy = pair.second;

    CHECK_EQ(dataflow.GetValueSet(xla_conditional, index).values().size(), 1);

    auto value = dataflow.GetValueSet(xla_conditional, index).values()[0];
    // The conditional must be copied if the value is a phi.
    should_copy =
        value->is_phi() && value->defining_instruction() == xla_conditional;
    any_copies |= should_copy;
  }
  return any_copies;
}

// Add kCopy instructions around the given kWhile instruction to eliminate any
// possible live range interference of HLO values assuming a dependency-based
// ordering. Copies are added conservatively. There  likely are copies which are
// not strictly necessary, but they are removed later in the pass via
// RemoveUnnecessaryCopies.
//
// Elements (each ShapeIndex) in the loop state are considered independently.  A
// copy is added to each element of the loop state which is modified in the
// while body. For each such element, a total of three kCopy instructions are
// added at following locations:
//
//   (1) The init value is copied before the kWhile instruction. Before:
//
//           (Init)
//             |
//           kWhile
//             |
//            ...
//
//       After:
//
//           (Init)
//             |
//           kCopy
//             |
//           kWhile
//             |
//            ...
//
//       This copy is necessary in case the init value is simultaneously live
//       with the kWhile.
//
//   (2) Copies are added to the parameter and root of the while body
//       computation. Before:
//
//           kParameter
//               |
//              ...
//               |
//           (body root)
//
//       After:
//
//           kParameter
//               |
//             kCopy ----------+
//               |             |
//              ...           ctrl
//               |            edge
//           (body root)       |
//               |             |
//             kCopy <---------+
//
//       The root kCopy becomes the new root of the computation. Both copies are
//       necessary to any potential interference between the parameter value and
//       the root value. The control edge prevents potential interference
//       between the copies themselves.
//
// If the loop state is a tuple then the above kCopy instructions are a deep
// copy constructed of kCopy, kGetTupleElement, and kTuple instruction as
// constructed by HloInstruction::DeepCopyInstruction.
Status AddCopiesForWhile(const HloAliasAnalysis& alias_analysis,
                         HloInstruction* xla_while) {
  VLOG(2) << "Adding copies for kWhile instruction " << xla_while->name();
  TF_RET_CHECK(xla_while->opcode() == HloOpcode::kWhile);

  ShapeTree<bool> indices_to_copy(xla_while->shape());
  if (!IndicesToCopyForWhile(alias_analysis.dataflow_analysis(), xla_while,
                             &indices_to_copy)) {
    VLOG(2) << "No copies necessary for kWhile instruction "
            << xla_while->name();
    return OkStatus();
  }

  VLOG(2) << "Adding copies for " << xla_while->name() << " at indices:";
  for (auto& pair : indices_to_copy) {
    if (pair.second) {
      VLOG(2) << "  " << pair.first;
    }
  }

  // Deep copy init.
  HloInstruction* while_init = xla_while->mutable_operand(0);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * while_init_copy,
      xla_while->parent()->DeepCopyInstruction(while_init, &indices_to_copy));
  TF_RETURN_IF_ERROR(while_init->ReplaceUseWith(xla_while, while_init_copy));

  // Deep copy the parameter and the root. Extend a control edge from the copy
  // of the parameter value to the corresponding copy value of the root.
  HloComputation* body = xla_while->while_body();
  HloInstruction* param = body->parameter_instruction(0);
  HloInstruction* root = body->root_instruction();

  // If param is the root then all indices should have been passed through the
  // while body and we should have returned early above.
  TF_RET_CHECK(param != root);

  // Copy users before making a deep copy of the parameter as the deep copy
  // will create new users of the parameter (eg, the GTE instructions of the
  // deep copy).
  std::vector<HloInstruction*> param_users = param->users();

  TF_ASSIGN_OR_RETURN(auto pair,
                      DeepCopyAndAddControlEdges(param, root, indices_to_copy));

  HloInstruction* param_copy = pair.first;
  HloInstruction* root_copy = pair.second;

  for (HloInstruction* user : param_users) {
    TF_RETURN_IF_ERROR(param->ReplaceUseWith(user, param_copy));
  }

  body->set_root_instruction(root_copy);
  return OkStatus();
}

// Add copies for the operands of in-place operations. RemoveUnnecessaryCopies
// will remove the unnecessary copies.
Status AddCopiesForInPlaceOperation(const HloAliasAnalysis& alias_analysis,
                                    HloInstruction* in_place_op,
                                    int64_t operand_number) {
  VLOG(2) << "Adding copies for in-place operation " << in_place_op->name();
  HloInstruction* operand = in_place_op->mutable_operand(operand_number);
  TF_ASSIGN_OR_RETURN(HloInstruction * deep_copy,
                      in_place_op->parent()->DeepCopyInstruction(operand));
  TF_RETURN_IF_ERROR(
      operand->ReplaceUseWith(in_place_op, operand_number, deep_copy));
  return OkStatus();
}

// Conservatively adds copies before root instruction of entry computation and
// each aliased parameter to resolve interference of aliased input and output
// buffer. We later rely on RemoveUnnecessaryCopies to drop the unnecessary
// ones.
Status AddCopiesForAliasedInputOutputs(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HloComputation* entry = module->entry_computation();
  if (!HloInstruction::IsThreadIncluded(entry->execution_thread(),
                                        execution_threads)) {
    return OkStatus();
  }
  HloInstruction* root = entry->root_instruction();

  ShapeTree<bool> output_indices_to_copy(root->shape());
  std::vector<std::optional<ShapeTree<HloInstruction*>>> copied_parameters(
      entry->num_parameters());
  bool has_alias = false;
  for (auto* param : entry->parameter_instructions()) {
    bool param_has_alias = false;
    ShapeTree<bool> param_indices_to_copy(param->shape());

    module->input_output_alias_config().ForEachAlias(
        [&](const ShapeIndex& output_index,
            const HloInputOutputAliasConfig::Alias& alias) {
          if (alias.parameter_number == param->parameter_number()) {
            param_has_alias = true;
            *(param_indices_to_copy.mutable_element(alias.parameter_index)) =
                true;
            *(output_indices_to_copy.mutable_element(output_index)) = true;
          }
        });

    if (!param_has_alias) {
      continue;
    }

    TF_RET_CHECK(param->parameter_number() < entry->num_parameters());
    TF_RET_CHECK(!copied_parameters[param->parameter_number()]);

    has_alias = true;
    // Store a snapshot of users before DeepCopyInstruction, as
    // DeepCopyInstruction introduces new users of the instruction.
    std::vector<HloInstruction*> users = param->users();
    ShapeTree<HloInstruction*> param_copy_tree(param->shape(),
                                               /*init_value=*/nullptr);
    TF_ASSIGN_OR_RETURN(HloInstruction * copied,
                        entry->DeepCopyInstruction(
                            param, &param_indices_to_copy, &param_copy_tree));
    if (param == root) {
      entry->set_root_instruction(copied);
      root = copied;
    }
    for (HloInstruction* user : users) {
      TF_RETURN_IF_ERROR(param->ReplaceUseWith(user, copied));
    }

    copied_parameters[param->parameter_number()] = param_copy_tree;
  }

  if (!has_alias) {
    return OkStatus();
  }

  // Add copies before root instruction.
  ShapeTree<HloInstruction*> output_copy_tree(root->shape(),
                                              /*init_value=*/nullptr);

  TF_ASSIGN_OR_RETURN(HloInstruction * root_copied,
                      root->parent()->DeepCopyInstruction(
                          root, &output_indices_to_copy, &output_copy_tree));

  // Add control dependencies between the input/output copies.
  TF_RETURN_IF_ERROR(module->input_output_alias_config().ForEachAliasWithStatus(
      [&](const ShapeIndex& output_index,
          const HloInputOutputAliasConfig::Alias& alias) -> Status {
        if (!copied_parameters[alias.parameter_number]) {
          return OkStatus();
        }
        HloInstruction* from =
            copied_parameters[alias.parameter_number]->element(
                alias.parameter_index);
        HloInstruction* to = output_copy_tree.element(output_index);

        TF_RET_CHECK(from != nullptr);
        TF_RET_CHECK(to != nullptr);
        TF_RETURN_IF_ERROR(from->AddControlDependencyTo(to));
        return OkStatus();
      }));

  entry->set_root_instruction(root_copied);

  return OkStatus();
}

// Removes any control dependencies to or from the given instruction.
Status StripControlDependenciesFrom(HloInstruction* instruction) {
  while (!instruction->control_successors().empty()) {
    TF_RETURN_IF_ERROR(instruction->RemoveControlDependencyTo(
        instruction->control_successors().front()));
  }

  while (!instruction->control_predecessors().empty()) {
    TF_RETURN_IF_ERROR(
        instruction->control_predecessors().front()->RemoveControlDependencyTo(
            instruction));
  }

  return OkStatus();
}

class LiveRangeRegions {
 public:
  struct InstructionInfo {
    InstructionInfo() : value_definition(nullptr), is_definition(false) {}

    // The instruction that defines the value being used. It basically saves
    // the defining instruction of each HloValue.
    HloInstruction* value_definition;
    // Whether the instruction defines a new value (or merely uses one). This
    // basically remembers whether the instruction actually creates an HloValue
    // or merely uses one, from a collection of given HloValues. Note that if
    // is_definition = true, it merely says the instruction creates a new
    // HloValue with or without defining a new one. For example, kAdd create a
    // new HloValue (can be value_definition), but tuples or get-tuple-element,
    // create a new HloValue aliasing without defining a new value (cannot be
    // value_definition).
    bool is_definition;
  };
  // Map instructions that use a value to the defining instruction of the value.
  // Because all values must belong to the same live range, an instruction can
  // have at most a single value-defining instruction; otherwise the multiple
  // incoming active values would share a single buffer, which is not allowed.
  // The value-defining and value-use instructions do not have to belong to the
  // same computation, but the value use needs to be nested within the defining
  // computation.
  typedef HloInstructionMap<InstructionInfo> InstructionMap;
  typedef std::pair<HloInstruction*, InstructionInfo> InstructionEntry;
  // Map each computation to its immediately contained instructions.
  typedef absl::flat_hash_map<const HloComputation*, InstructionMap>
      ComputationMap;

  InstructionMap& operator[](const HloComputation* computation) {
    if (computation_map_.find(computation) == computation_map_.end()) {
      computation_vector_.push_back(computation);
    }
    return computation_map_[computation];
  }

  const InstructionMap& operator[](const HloComputation* computation) const {
    ComputationMap::const_iterator p = computation_map_.find(computation);
    CHECK(p != computation_map_.end());
    return p->second;
  }
  absl::InlinedVector<const HloComputation*, 5>::const_iterator begin() const {
    return computation_vector_.begin();
  }
  absl::InlinedVector<const HloComputation*, 5>::const_iterator end() const {
    return computation_vector_.end();
  }
  int64_t size() const {
    CHECK_EQ(computation_vector_.size(), computation_map_.size());
    return computation_vector_.size();
  }
  bool empty() const { return size() == 0; }
  const HloComputation* Computation(int64_t index) const {
    return computation_vector_[index];
  }
  bool contains(HloInstruction* instr) const {
    CHECK_NE(instr, nullptr);
    auto* computation = instr->parent();
    auto p = computation_map_.find(computation);
    if (p == computation_map_.end()) {
      return false;
    }
    auto instr_map = (*p).second;
    return instr_map.find(instr) != instr_map.end();
  }

 private:
  ComputationMap computation_map_;
  absl::InlinedVector<const HloComputation*, 5> computation_vector_;
};

namespace {
// Represent relations between the locations of two regions of instructions,
// each region can include 0-n instructions.
class Relation {
 public:
  enum RuntimeOrder {
    // Indicate that there is no overlap whatsoever between the two regions.
    kNoOverlap = 0,
    // Indicate that the first region includes the same set of instructions as
    // the second region.
    kSameInstr = 1,
    // Indicate that the first region is entirely before the second region
    // starts.
    kBeforeStart = 2,
    // Indicate that the first region is before the second region ends.
    kBeforeStartOrSameInstr = kBeforeStart | kSameInstr,
    // Indicate that the first region is entirely after the second region ends.
    kAfterEnd = 4,
    // Indicate that the first region is after the second region
    // starts, with some instructions before the second region ends.
    kAfterEndOrSameInstr = kAfterEnd | kSameInstr,
    // Indicate that the first region overlaps with the second one, but share no
    // common instructions.
    kBeforeStartOrAfterEnd = kBeforeStart | kAfterEnd,
    // Indicate that the first region overlaps with the second one, and have
    // some common instructions.
    kBeforeOrAfterOrOverlap = kBeforeStart | kAfterEnd | kSameInstr,
  };
  Relation() : intercept_def_use_(false) {}
  explicit Relation(RuntimeOrder order, bool intercept_def_use = false)
      : intercept_def_use_(intercept_def_use) {
    orders_.push_back(order);
  }
  Relation(const Relation& that)
      : intercept_def_use_(that.intercept_def_use_), orders_(that.orders_) {}
  bool operator==(const Relation& that) const {
    return intercept_def_use_ == that.intercept_def_use_ &&
           absl::c_equal(orders_, that.orders_);
  }

  // Return whether the runtime ordering may imply interception, assuming it
  // models the relation between a modifying and a use instruction.
  bool UseImpliesInterception() const {
    CHECK_EQ(orders_.size(), 1);
    return UseImpliesInterception(orders_[0]);
  }
  // Return whether the runtime ordering may imply interception, assuming it
  // models the relation between a modifying and a definition instruction.
  bool DefinitionImpliesInterception() const {
    CHECK_EQ(orders_.size(), 1);
    return DefinitionImpliesInterception(orders_[0]);
  }
  // Return whether the current relation models a modifying instruction that
  // intercepts the dataflow of another live range region.
  bool InterceptDefUse() const { return intercept_def_use_; }
  // Update interception state to the given value.
  void UpdateInterception(bool value) {
    CHECK_EQ(orders_.size(), 1);
    intercept_def_use_ = value;
  }
  Relation::RuntimeOrder GetRuntimeOrder() const {
    if (orders_.empty()) {
      return Relation::kNoOverlap;
    }
    CHECK_EQ(orders_.size(), 1);
    return orders_[0];
  }
  // Return whether the current relation implies two overlapping regions.
  bool RuntimeOrderOverlap() const {
    return absl::c_any_of(orders_, ImpliesOverlap);
  }
  bool RuntimeOrderIsUnordered() const {
    return orders_.size() == 1 && orders_[0] == kBeforeStartOrAfterEnd;
  }
  bool RuntimeOrderIsNoOverlap() const {
    return orders_.empty() || (orders_.size() == 1 && orders_[0] == kNoOverlap);
  }
  bool RuntimeOrderIsRunBefore() const {
    return orders_.size() == 1 && orders_[0] == kBeforeStart;
  }
  bool RuntimeOrderIsRunAfter() const {
    return orders_.size() == 1 && orders_[0] == kAfterEnd;
  }
  std::string ToString() const {
    return absl::StrCat("Interception = ", intercept_def_use_, ";",
                        absl::StrJoin(orders_, ","));
  }

  static bool DefinitionImpliesInterception(RuntimeOrder definition) {
    return (definition == kAfterEnd || definition == kBeforeStartOrAfterEnd);
  }
  static bool UseImpliesInterception(RuntimeOrder use) {
    return (use == kBeforeStart || use == kBeforeStartOrAfterEnd);
  }

  // Summarize additional relations into a single runtime ordering, assuming
  // both relations are modeling constraints of the same source instruction.
  void UnionRelationFromSameSource(const Relation& rel) {
    CHECK_LE(orders_.size(), 1);
    CHECK_EQ(rel.orders_.size(), 1);
    if (orders_.empty()) {
      orders_.push_back(rel.orders_[0]);
    } else {
      orders_[0] = Union(orders_[0], rel.orders_[0]);
    }
    intercept_def_use_ = intercept_def_use_ || rel.intercept_def_use_;
  }

  // Summarize additional relations into disjoint runtime orderings, assuming
  // the relations are modeling constraints of different source instructions.
  void UnionRelationFromDifferentSource(const Relation& rel) {
    if (rel.orders_.empty()) {
      return;
    }
    CHECK_EQ(rel.orders_.size(), 1);
    intercept_def_use_ = intercept_def_use_ || rel.intercept_def_use_;
    for (auto& local_order : orders_) {
      if (OverwriteIfSubsume(rel.orders_[0], &local_order)) {
        return;
      }
    }
    orders_.push_back(rel.orders_[0]);
  }

  static Relation::RuntimeOrder ReverseRuntimeOrder(RuntimeOrder order) {
    switch (order) {
      case kNoOverlap:
      case kSameInstr:
      case kBeforeStartOrAfterEnd:
      case kBeforeOrAfterOrOverlap:
        return order;
      case kBeforeStart:
        return kAfterEnd;
      case kBeforeStartOrSameInstr:
        return kAfterEndOrSameInstr;
      case kAfterEnd:
        return kBeforeStart;
      case kAfterEndOrSameInstr:
        return kBeforeStartOrSameInstr;
    }
  }

 private:
  // Indicate that the second region may intercept the def-use dataflow of the
  // first region, if their buffers are combined.
  bool intercept_def_use_;
  // Remember the different runtime orderings of different instructions.
  absl::InlinedVector<RuntimeOrder, 4> orders_;

  static RuntimeOrder Union(RuntimeOrder o1, RuntimeOrder o2) {
    return static_cast<Relation::RuntimeOrder>(o1 | o2);
  }
  static bool ImpliesOverlap(RuntimeOrder o) {
    return o >= RuntimeOrder::kBeforeStartOrAfterEnd;
  }
  // Returns whether ordering constraint o1 includes o2 as a subset, when they
  // represent runtime orderings (interleavings) of two different regions.
  static bool Subsume(RuntimeOrder o1, RuntimeOrder o2) {
    return Union(o1, o2) == o1;
  }
  // Overwrites o1 with o2 if o2 subsumes o1 (as defined above by the Subsume
  // function). Return whether o2 is subsumed by the new value in o1.
  static bool OverwriteIfSubsume(RuntimeOrder o2, RuntimeOrder* o1) {
    if (*o1 == o2) {
      return true;
    }
    CHECK_NE(o1, nullptr);
    // Overwrite o1 with o2 if it is subsumed by o2.
    if (Subsume(o2, *o1)) {
      *o1 = o2;
      return true;
    } else if (Subsume(*o1, o2)) {
      // If o2 is already subsumed by o1, do nothing.
      return true;
    }
    // If neither o1 nor o2 is subsumed by the other, return false, so that o2
    // will be inserted as a separate entry representing all possible orderings.
    return false;
  }
};

class ComputeRelativeLocation {
 public:
  typedef LiveRangeRegions::InstructionEntry InstructionEntry;
  explicit ComputeRelativeLocation(HloOrdering* ordering)
      : ordering_(ordering) {
    VLOG(3) << "New analysis\n";
  }

  // Compute locationing constraints between two instructions. Here entry2 is
  // the source instruction, in that the returned value describes the relation
  // of entry2 in terms of whether it is before or after entry1, and whether it
  // can intercept the def-use data flow of entry1.
  Relation Compute(const InstructionEntry& entry1,
                   const InstructionEntry& entry2, bool instr2_can_modify) {
    auto def = entry1.second.value_definition;
    auto use = entry1.first;
    Relation::RuntimeOrder order =
        ComputeRuntimeOrdering(entry2.first, entry1.first);
    if (order == Relation::kSameInstr &&
        entry1.second.is_definition != entry2.second.is_definition) {
      if (entry1.second.is_definition) {
        order = Relation::kBeforeStart;
      } else {
        order = Relation::kAfterEnd;
      }
    }
    bool intercept = AlwaysForceInterception(entry2.first);
    if (def == nullptr || !instr2_can_modify) {
      return Relation(order, intercept);
    }
    // If the definition and use are parameter and return (root) of the parent
    // computation, then any modification is considered intercepting.
    if (def->opcode() == HloOpcode::kParameter &&
        use == use->parent()->root_instruction()) {
      VLOG(3) << "Setting interception due to parameter/root relation\n";
      return Relation(order, true);
    }
    if (Relation::UseImpliesInterception(order)) {
      auto order2 = ComputeRuntimeOrdering(entry2.first, def);
      if (Relation::DefinitionImpliesInterception(order2)) {
        VLOG(3) << "Setting interception for " << def->ToString()
                << " with use:" << entry1.first->ToString() << "\n";
        intercept = true;
      }
    }
    return Relation(order, intercept);
  }

  // Return the relative locations (defined above) of range2 in relation to
  // instructions in range1. Return kNoOverlap if range2 is outside of range1.
  Relation Compute(const LiveRangeRegions& range1,
                   const LiveRangeRegions& range2) {
    Relation dir_src_dest;
    for (int64_t index = 0; index < range1.size(); index++) {
      auto* computation1 = range1.Computation(index);
      for (const auto* computation2 : range2) {
        for (auto instr_entry2 : range2[computation2]) {
          if (!ordering_->call_graph().Dominates(computation1, computation2)) {
            continue;
          }
          VLOG(3) << "Locationing " << instr_entry2.first->ToString();
          // Saves relations between instr2 and other instructions in range1.
          bool instr2_can_modify =
              InstructionCanIntercept(instr_entry2, range1);
          Relation instr2_relation;
          std::vector<InstructionEntry> unordered_ops;
          bool unordered_intercept = false;
          for (auto instr_entry1 : range1[computation1]) {
            auto rel = Compute(instr_entry1, instr_entry2, instr2_can_modify);
            VLOG(3) << "new relation with:" << instr_entry1.first->ToString()
                    << " = " << rel.ToString() << "\n";
            if (!rel.RuntimeOrderIsUnordered()) {
              instr2_relation.UnionRelationFromSameSource(rel);
            } else {
              unordered_ops.push_back(instr_entry1);
              unordered_intercept |= rel.InterceptDefUse();
            }
            VLOG(3) << "instr2 relation:" << instr2_relation.ToString() << "\n";
          }
          // Here instru2_relation is guaranteed to have at most a single entry,
          // because it was initialized to be empty, and has been updated only
          // via instr2_relation.UnionRelationFromSameSource(rel), which
          // maintains that the updated result has only a single entry.
          if (!ForceRuntimeOrder(unordered_ops, instr_entry2,
                                 instr2_relation.GetRuntimeOrder())) {
            VLOG(3) << "Unable to force ordering of unordered ops\n";
            instr2_relation.UnionRelationFromSameSource(Relation(
                Relation::kBeforeStartOrAfterEnd, unordered_intercept));
          }
          dir_src_dest.UnionRelationFromDifferentSource(instr2_relation);
          VLOG(3) << "Resulting relation : " << dir_src_dest.ToString() << "\n";
        }
      }
    }
    return dir_src_dest;
  }

  // Return whether control dependences, if exist, are added successfully.
  bool AddControlDependenceForUnorderedOps() {
    if (ctrl_deps_.empty()) {
      return true;
    }
    PredecessorHloOrdering* ordering =
        dynamic_cast<PredecessorHloOrdering*>(ordering_);
    if (ordering == nullptr) {
      // Support force ordering of unordered-ops only when using predecssor
      // ordering.
      return false;
    }
    for (const auto& comp_it : ctrl_deps_) {
      HloComputation* parent = comp_it.first;
      HloReachabilityMap& reachability_map = ordering->reachability_map(parent);
      for (const auto& instr_it : comp_it.second) {
        HloInstruction* entry1 = instr_it.first;
        for (HloInstruction* entry2 : instr_it.second) {
          VLOG(3) << "Add control dependence between " << entry2->ToString();
          VLOG(3) << "\n vs " << entry1->ToString() << "\n";
          TF_CHECK_OK(entry2->AddControlDependencyTo(entry1));
        }
        reachability_map.UpdateReachabilityThroughInstruction(entry1);
        for (HloInstruction* entry2 : instr_it.second) {
          DCHECK(ordering_->GetExecutionConstraint(entry1, entry2) ==
                 HloOrdering::ExecutionConstraint::kRunAfter);
        }
      }
    }
    return true;
  }

 private:
  enum ComputeStatus {
    kFullyComputed,
    kPartiallyComputed,
    kNotComputed,
  };
  typedef std::pair<ComputeStatus, Relation::RuntimeOrder> SavedRelation;

  // Returns whether it is safe to force the desired_relation ordering between
  // all operations in unordered_ops and entry2. If safe, save the new enforced
  // ordering relations.
  bool ForceRuntimeOrder(absl::Span<const InstructionEntry> unordered_ops,
                         const InstructionEntry entry2,
                         Relation::RuntimeOrder desired_relation) {
    if (unordered_ops.empty()) {
      return true;
    }
    if (desired_relation != Relation::kBeforeStart &&
        desired_relation != Relation::kAfterEnd) {
      return false;
    }
    auto ModifiesNonCopy = [](HloInstruction* instr, const HloInstruction* op) {
      auto in_place = HloDataflowAnalysis::GetInPlaceInputOutputPairs(instr);
      if (in_place.empty()) {
        return false;
      }
      return absl::c_any_of(
          in_place, [&](const std::pair<HloOperandIndex, ShapeIndex>&
                            operand_and_output_index) {
            auto* op2 =
                instr->operand(operand_and_output_index.first.operand_number);
            return (op == nullptr) ? (op2->opcode() == HloOpcode::kCopy)
                                   : (op2 == op);
          });
    };
    for (const InstructionEntry& entry1 : unordered_ops) {
      // Only consider instructions in the same computation.
      if (entry1.first->parent() != entry2.first->parent()) {
        return false;
      }
      HloInstruction* pred = (desired_relation == Relation::kBeforeStart)
                                 ? entry2.first
                                 : entry1.first;
      HloInstruction* succ = (desired_relation == Relation::kBeforeStart)
                                 ? entry1.first
                                 : entry2.first;
      if (pred == pred->parent()->root_instruction()) {
        return false;
      }
      if (succ->opcode() == HloOpcode::kCopy &&
          ModifiesNonCopy(pred, succ->operand(0))) {
        VLOG(3) << "Failed to force unordered op ordering due to copy ordering "
                << " between " << pred->ToString() << "\n";
        VLOG(3) << " vs. " << succ->ToString() << "\n";
        return false;
      }
    }
    for (const InstructionEntry& entry1 : unordered_ops) {
      Save(entry2.first, entry1.first, desired_relation, true);
    }
    return true;
  }

  static bool AlwaysForceInterception(HloInstruction* instr) {
    // The following communication operations can have some unexpected side
    // effects, when synchronizing across processes. Therefore, we
    // conservatively try provide dedicated buffers to these operations instead
    // of allowing them to share buffers with other operations, as the reuse may
    // cause unexpected interferences.
    if (HloDataflowAnalysis::IsAsynchronousOperationStart(instr->opcode()) ||
        HloDataflowAnalysis::IsAsynchronousOperationDone(instr->opcode())) {
      return true;
    }
    switch (instr->opcode()) {
      // TODO(b/190903339): It appears that collectivePermute needs to be
      // followed by a copy when escaping through a computation root.
      case HloOpcode::kCollectivePermute:
        return true;
      default:
        return false;
    }
  }

  // Returns whether the given instr may intercept the def-use flow of another
  // ongoing live range if its buffer is combined with the other live range.
  // The function should return true if instr creates a new HloValue that could
  // overwrite an existing HloValue in the combined buffer.
  // More specifically, here we are looking for operations that create new
  // values, e.g., add, subtract, in contrast to HLOs that merely create
  // aliasings among existing values, e.g., tuple, get-tuple-element. Any of the
  // new values created by operations such as add or subtract, when included as
  // definition operations in a live range, are aliases of the buffer to be
  // allocated to the live range and so are treated as they may be modifying the
  // targeting buffer.
  bool InstructionCanIntercept(const InstructionEntry& entry,
                               const LiveRangeRegions& region) {
    auto instr = entry.first;
    if (!entry.second.is_definition) {
      // If the instruction only uses the value, it can intercept only if it
      // modifies the buffer in place.
      return !HloDataflowAnalysis::GetInPlaceInputOutputPairs(instr).empty();
    }
    switch (instr->opcode()) {
      // If the copy instruction is used to connect two live range regions,
      // it does not overwrite the combined buffer with new values.
      case HloOpcode::kCopy:
        // Checking the copy simply copies from the other live range with no
        // layout conflicts.
        if (region.contains(instr->mutable_operand(0)) &&
            ShapeUtil::Equal(instr->shape(), instr->operand(0)->shape())) {
          return false;  // Cannot intercept.
        }
        return true;
      // The following operations merely create aliases among the HloValues.
      case HloOpcode::kParameter:
      case HloOpcode::kTuple:
      case HloOpcode::kGetTupleElement:
      // Here we consider all the compound operations (e.g., conditionals and
      // while loops) as if they do not modify any HloValue, with the argument
      // being that any value modifying operation contained inside will be
      // considered separately to make sure the kIntercept relation being
      // recorded as appropriate. Since the compound operations may or may not
      // modify, not treating them as value modifying would make the algorithm
      // less conservative.
      case HloOpcode::kWhile:
      case HloOpcode::kCall:
      case HloOpcode::kConditional:
        return false;
      default:
        return true;
    }
    return true;
  }

  SavedRelation AlreadyComputed(HloInstruction* op1, HloInstruction* op2) {
    auto p2 = saved_relations_.find(op2);
    if (p2 != saved_relations_.end()) {
      auto p1 = (*p2).second.find(op1);
      if (p1 != (*p2).second.end()) {
        return SavedRelation(kFullyComputed, (*p1).second);
      }
    }
    p2 = saved_relations_.find(op1);
    if (p2 != saved_relations_.end()) {
      auto p1 = (*p2).second.find(op2);
      if (p1 != (*p2).second.end()) {
        return SavedRelation(kPartiallyComputed,
                             Relation::ReverseRuntimeOrder((*p1).second));
      }
    }
    return SavedRelation(kNotComputed, Relation::kNoOverlap);
  }

  Relation::RuntimeOrder Save(HloInstruction* entry1, HloInstruction* entry2,
                              const Relation::RuntimeOrder relation,
                              bool is_unordered_originally = false) {
    CHECK_EQ(AlreadyComputed(entry1, entry2).first, kNotComputed);
    // Do not save unordered relations.
    CHECK_NE(relation, Relation::kBeforeStartOrAfterEnd);
    saved_relations_[entry2][entry1] = relation;
    if (is_unordered_originally) {
      CHECK(relation == Relation::kBeforeStart ||
            relation == Relation::kAfterEnd)
          << relation;
      HloInstruction* pred =
          (relation == Relation::kBeforeStart) ? entry1 : entry2;
      HloInstruction* succ =
          (relation == Relation::kBeforeStart) ? entry2 : entry1;
      VLOG(3) << "Save unordered relation: " << pred->ToString() << "\n";
      VLOG(3) << " vs " << succ->ToString() << "\n";
      CHECK_EQ(succ->parent(), pred->parent());
      auto& dep_vec = ctrl_deps_[succ->parent()][succ];
      for (HloInstruction*& op : dep_vec) {
        auto rel = AlreadyComputed(pred, op);
        if (rel.first != kNotComputed) {
          if (rel.second == Relation::kAfterEnd) {
            op = pred;
          } else {
            CHECK(rel.second == Relation::kBeforeStart);
          }
          return relation;
        }
      }
      VLOG(2) << "Forcing unordered:" << pred->ToString() << "\n";
      VLOG(2) << " vs " << succ->ToString() << "\n";
      dep_vec.push_back(pred);
    }
    return relation;
  }

  // Compute the runtime ordering constraints between two instructions.
  Relation::RuntimeOrder ComputeRuntimeOrdering(HloInstruction* instr1,
                                                HloInstruction* instr2) {
    auto saved_relation = AlreadyComputed(instr1, instr2);
    if (saved_relation.first != kNotComputed) {
      VLOG(3) << "Already computed between " << instr1->ToString() << "\n vs "
              << instr2->ToString() << "\n";
      return saved_relation.second;
    }
    auto constraint = ordering_->GetExecutionConstraint(instr1, instr2);
    switch (constraint) {
      case HloOrdering::ExecutionConstraint::kIsSame:
        return Save(instr1, instr2, Relation::kSameInstr);
      case HloOrdering::ExecutionConstraint::kRunBeforeEnd:
        return Save(instr1, instr2, Relation::kBeforeStartOrSameInstr);
      case HloOrdering::ExecutionConstraint::kRunBeforeStart:
        return Save(instr1, instr2, Relation::kBeforeStart);
      case HloOrdering::ExecutionConstraint::kRunAfter:
        return Save(instr1, instr2, Relation::kAfterEnd);
      case HloOrdering::ExecutionConstraint::kRunExclusiveBefore:
      case HloOrdering::ExecutionConstraint::kRunExclusiveAfter:
        return Save(instr1, instr2, Relation::kNoOverlap);
      case HloOrdering::ExecutionConstraint::kUnordered: {
        if (instr1->parent() != instr2->parent()) {
          return Relation::kBeforeStartOrAfterEnd;
        }
        auto ControlDependenceBefore = [&](HloInstruction* op1,
                                           HloInstruction* op2) {
          auto constraint = ComputeRuntimeOrdering(op1, op2);
          if (constraint == Relation::kBeforeStart ||
              constraint == Relation::kSameInstr ||
              constraint == Relation::kBeforeStartOrSameInstr) {
            return true;
          } else {
            return false;
          }
        };
        if (!ctrl_deps_.empty()) {
          auto ctrl_deps = ctrl_deps_[instr1->parent()];
          if (absl::c_any_of(ctrl_deps[instr2], [&](HloInstruction* pred2) {
                return ControlDependenceBefore(instr1, pred2);
              })) {
            VLOG(2) << "control-dependent: " << instr1->ToString() << "\n";
            VLOG(2) << "vs " << instr2->ToString() << "\n";
            return Save(instr1, instr2, Relation::kBeforeStart);
          } else if (absl::c_any_of(
                         ctrl_deps[instr1], [&](HloInstruction* pred1) {
                           return ControlDependenceBefore(instr2, pred1);
                         })) {
            VLOG(2) << "control-dependent: " << instr2->ToString() << "\n";
            VLOG(2) << "vs " << instr1->ToString() << "\n";
            return Save(instr1, instr2, Relation::kAfterEnd);
          }
        }
        // Don't save the result for unordered operations, so they can be
        // refined later.
        return Relation::kBeforeStartOrAfterEnd;
      }
    }
  }

  HloOrdering* ordering_;
  absl::flat_hash_map<
      HloInstruction*,
      absl::flat_hash_map<HloInstruction*, Relation::RuntimeOrder>>
      saved_relations_;
  absl::flat_hash_map<
      HloComputation*,
      absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>>
      ctrl_deps_;
};
}  // namespace

// Class which tracks the HLO values within each HLO buffer in the module
// during copy removal.
//
// The values are held in a linked list where there is one list for each
// buffer. Removing a copy instruction merges together the values in the
// source buffer of the copy to the destination buffer of the copy. This class
// tracks these value lists as copies are removed from the graph (and value
// lists are merged).
//
// The CopyRemover object is initialized to match the state of
// HloAliasAnalysis. However, as copies are removed this state diverges. The
// values-to-buffer mapping is maintained outside of HloAliasAnalysis because
// a fully updatable alias analysis is very slow.
class CopyRemover {
 public:
  // The values held in a single HLO buffer are represented using a linked
  // list. An element type in this list is ValueNode.
  //
  // This linked list is hand-rolled to enable efficient splicing of lists
  // using only references to list elements without knowing which lists are
  // being spliced. std::list requires a reference to the list object to
  // splice.
  struct ValueNode {
    explicit ValueNode(const HloValue* v) : value(v) {}

    const HloValue* value;

    // The uses are maintained outside of HloValue::uses() because
    // HloValue::uses() is not updatable (a fully updatable dataflow analysis
    // is slow).
    std::vector<const HloUse*> uses;

    // next/prev elements in the linked list. The list is circularly linked so
    // these values are never null for elements in the list.
    ValueNode* prev = nullptr;
    ValueNode* next = nullptr;
  };

  CopyRemover(const HloModule& module, const HloAliasAnalysis& alias_analysis,
              HloOrdering* ordering, bool check_live_range_ordering)
      : dataflow_(alias_analysis.dataflow_analysis()), ordering_(ordering) {
    // Construct a list for each HLO buffer in the alias analysis. Maintain a
    // map from HloValue to the respective list element representing that
    // value. The map is used to construct the copy info map below.
    absl::flat_hash_map<const HloValue*, ValueNode*> value_to_node;
    // Perform check only if the default dependence-based ordering is used.
    for (const HloBuffer& buffer : alias_analysis.buffers()) {
      // No copies should have been inserted within fused computations, so no
      // need to remove them. HloOrdering isn't compatible with HloValues inside
      // fusions, so skip copy removal for them.
      if (buffer.values().at(0)->defining_instruction()->IsFused()) {
        continue;
      }
      if (check_live_range_ordering) {
        // Verify values contained in the buffer are strictly ordered. This
        // should always be the case after adding copies to eliminate
        // interference. Specifically, the addition of the control flow edges
        // between copies added around aliased operations (kWhile) guarantees
        // this strict order.
        for (const HloValue* value_a : buffer.values()) {
          if (value_a->shape().IsToken()) {
            // Token values have no representation and cannot interfere.
            continue;
          }
          for (const HloValue* value_b : buffer.values()) {
            if (value_a != value_b) {
              DCHECK(ordering_->LiveRangeStrictlyBefore(
                         *value_a, *value_b, dataflow_,
                         /*use_is_always_before_def_in_same_instr=*/true) ||
                     ordering_->LiveRangeStrictlyBefore(
                         *value_b, *value_a, dataflow_,
                         /*use_is_always_before_def_in_same_instr=*/true))
                  << value_a->ToString() << " and " << value_b->ToString()
                  << " are not ordered";
            }
          }
        }
      }

      std::vector<const HloValue*> values = buffer.values();
      absl::c_sort(values, [this](const HloValue* a, const HloValue* b) {
        return ordering_->IsDefinedBefore(*a, *b);
      });

      // Create a list containing all of the values in the buffer.
      AddValueList(values, &value_to_node);
    }

    // Create copy_map_ which contains the source and destination values
    // of all copies.
    CreateCopyMap(module, value_to_node);

    XLA_VLOG_LINES(3, ToString());
    TF_DCHECK_OK(Verify());
  }

  // Add a list containing the given values to CopyRemover. This
  // represents the values contained in a single buffer. For each value in
  // 'values' an entry is created in value_to_node which indicates the
  // respective ValueNode representing that value.
  void AddValueList(
      absl::Span<const HloValue* const> values,
      absl::flat_hash_map<const HloValue*, ValueNode*>* value_to_node) {
    ValueNode* tail = nullptr;
    ValueNode* head = nullptr;
    for (const HloValue* value : values) {
      auto new_node = new ValueNode(value);
      (*value_to_node)[value] = new_node;

      // Copy the HLO values's uses into the ValueNode for the value. These
      // uses in ValueNode are updated as copies are removed.
      new_node->uses.reserve(value->GetUses().size());
      for (const HloUse& use : value->GetUses()) {
        new_node->uses.push_back(&use);
      }

      // Connect the new node into the linked list.
      if (tail == nullptr) {
        head = new_node;
      } else {
        tail->next = new_node;
        new_node->prev = tail;
      }
      tail = new_node;
    }

    // The linked list is circular so connect the head and tail.
    tail->next = head;
    head->prev = tail;
    value_lists_.insert(head);
  }

  // This method also fills in copy_map_ which indicates which nodes
  // in the value lists corresponding to the source and destination values of
  // kCopy instructions. value_to_node should map each HloValue to its
  // respective ValueNode.
  void CreateCopyMap(
      const HloModule& module,
      const absl::flat_hash_map<const HloValue*, ValueNode*>& value_to_node) {
    for (HloComputation* computation : module.MakeNonfusionComputations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        // Add copies with unambiguous source values to the map. Copies with
        // ambiguous sources are not removable.
        if (instruction->opcode() == HloOpcode::kCopy) {
          const HloValueSet& src_value_set =
              dataflow_.GetValueSet(instruction->operand(0));
          if (src_value_set.values().size() == 1) {
            CopyNodes& copy_node = copy_map_[instruction];
            copy_node.dest =
                value_to_node.at(&dataflow_.GetUniqueValueAt(instruction));
            copy_node.src = value_to_node.at(&src_value_set.GetUniqueValue());
          }
        }
      }
    }
  }

  ~CopyRemover() {
    for (const ValueNode* head : value_lists_) {
      const ValueNode* p = head;
      do {
        const ValueNode* tmp = p->next;
        delete p;
        p = tmp;
      } while (p != head);
    }
  }

  // Verify invariants within the linked lists.
  Status Verify() const {
    for (const ValueNode* head : value_lists_) {
      const ValueNode* p = head;
      do {
        // Verify links between elements are consistent.
        TF_RET_CHECK(p->prev->next == p);
        TF_RET_CHECK(p->next->prev == p);

        const HloInstruction* def = p->value->defining_instruction();
        if (def->opcode() == HloOpcode::kCopy && ContainsKey(copy_map_, def)) {
          TF_RET_CHECK(copy_map_.at(def).dest == p);
        }
        for (const HloUse* use : p->uses) {
          if (use->instruction->opcode() == HloOpcode::kCopy &&
              ContainsKey(copy_map_, use->instruction)) {
            TF_RET_CHECK(copy_map_.at(use->instruction).src == p);
          }
        }

        p = p->next;
      } while (p != head);
    }
    return OkStatus();
  }

  // Compute the set of instructions where values are alive and organize these
  // instructions by separating them into their respective computations.
  LiveRangeRegions ComputeLiveRangeRegions(const ValueNode* head) {
    LiveRangeRegions live_range;

    auto VisitValueNode = [&](const ValueNode* node) {
      HloInstruction* def_op = node->value->instruction();
      HloComputation* def_parent = def_op->parent();
      live_range[def_parent][def_op].is_definition = true;
      for (const auto& use : node->uses) {
        auto* use_op = use->instruction;
        HloComputation* use_parent = use_op->parent();
        live_range[use_parent][use_op].value_definition = def_op;
      }
    };
    ForEachValueInRange(head, VisitValueNode);
    return live_range;
  }

  // Try to elide the given copy. Elision of a copy is possible only if no
  // live range interference is introduced by the copy's elimination. If
  // elision is possible, then the internal state (value lists) are updated,
  // and true is returned. Returns false otherwise.
  bool TryElideCopy(const HloInstruction* copy,
                    int64_t* region_analysis_limit) {
    VLOG(2) << "Trying to remove " << copy->name();
    CHECK_NE(region_analysis_limit, nullptr);

    if (!ContainsKey(copy_map_, copy)) {
      VLOG(2) << copy->name() << " is not removable";
      return false;
    }
    if (!ShapeUtil::Equal(copy->shape(), copy->operand(0)->shape())) {
      VLOG(2) << copy->name() << " is not removable (shape mismatch)";
      return false;
    }
    const CopyNodes& copy_node = copy_map_.at(copy);
    DCHECK(copy_node.src != nullptr);
    DCHECK(copy_node.dest != nullptr);

    int64_t live_range_size1 = 0, live_range_size2 = 0;
    ForEachValueInRange(copy_node.src, [&](const ValueNode* node) {
      live_range_size1 += 1 + node->uses.size();
    });
    ForEachValueInRange(copy_node.dest, [&](const ValueNode* node) {
      live_range_size2 += 1 + node->uses.size();
    });
    // Use the more accurate region-based live range interference analysis if
    // the live range size is within a given limit (or if no limit is given).
    // Also don't use the new analysis for copies of broadcasts as these copies
    // are cheap and are later removed by replicating the broadcasts.
    bool use_region_analysis =
        copy->operand(0)->opcode() != HloOpcode::kBroadcast &&
        (*region_analysis_limit < 0 ||
         live_range_size1 * live_range_size2 <= *region_analysis_limit);
    *region_analysis_limit = 0;
    VLOG(3) << copy->name() << " copies value "
            << copy_node.src->value->ToShortString();
    VLOG(3) << "Source buffer values: " << ValueListToString(copy_node.src);
    VLOG(3) << "Dest buffer values: " << ValueListToString(copy_node.dest);
    // Checks whether the live range at src is before that defined by dest.
    auto CheckLiveRangeBefore = [&](ValueNode* src, ValueNode* dest) {
      for (ValueNode* next_dest = dest; next_dest != nullptr;
           next_dest = Next(*next_dest)) {
        for (ValueNode* prev_src = src; prev_src != nullptr;
             prev_src = Prev(*prev_src)) {
          if (!LiveRangeBefore(*prev_src, *next_dest)) {
            VLOG(2) << "Live range of " << prev_src->value->ToShortString()
                    << " is not before " << next_dest->value->ToShortString();
            return false;
          }
        }
      }
      return true;
    };
    auto CheckLiveRangeInterference = [&](ValueNode* src, ValueNode* dest,
                                          const CombineLiveRangeOption option) {
      CHECK_NE(src, nullptr);
      CHECK_NE(dest, nullptr);
      if (!use_region_analysis) {
        VLOG(2) << "Configured to not use region-based analysis.\n";
        return true;
      }
      *region_analysis_limit += live_range_size1 * live_range_size2;
      if (ValuesInterfere(src, dest, option)) {
        VLOG(2) << "Region-based interference is true. \n";
        return true;
      }
      VLOG(2) << "Region-based interference is false. \n";
      return false;
    };

    // A kCopy instruction copies an HLO value from a source buffer and
    // defines an HLO value in a destination buffer. Most generally, the
    // source and destination buffers may each hold more than one value at
    // different points in the computation so we define the following:
    //
    //   Values in source buffer:      {s_0, ..., s_n}
    //   Values in destination buffer: {d_0, ..., d_m}
    //
    // A kCopy instruction between these buffers copies a value s_x in the
    // source buffer and defines a value d_y in the destination buffer. The
    // elision of a copy merges the source and destination buffers together,
    // so the list of values for the source and destination buffers are
    // merged.
    //
    // We handle two different cases for copy elision:
    //
    //  (1) the kCopy defines the first value in the destination buffer (d_0).
    //
    //  (2) the kCopy copies the last value in the source buffer (s_n).
    //
    // For the remaining case where the kCopy copies a not-last value from the
    // source buffer to a not-first value of the destination buffer, the kCopy
    // instruction cannot be removed. This case is generated, for example, if
    // the kCopy copies a while body parameter of the loop state at one tuple
    // index to a different tuple index in the while body root. Removal of the
    // copy necessarily results in live range interference of values in the
    // loop state at the two different tuple indices.
    //
    //  We can only perform copy elision if the resulting merged values have
    //  totally ordered live ranges; otherwise the merged buffer would have
    //  live range interference.
    if (copy_node.src->next == copy_node.dest) {
      // In the process of eliding copies, its possible for a copy to have the
      // same source and destination buffer. In this case, the copy can be
      // safely removed.
      VLOG(2) << copy->name() << " source and destination buffers are same.";
    } else if (IsHead(*copy_node.dest)) {
      // The copy copies an arbitrary value in the source buffer (call it s_x)
      // and defines d_0, the first value in the destination buffer. After
      // merging, the values in the combined buffer must be strictly ordered
      // as follows** to elide the copy:
      //
      // {s_0, ..., s_x, d_1, ..., d_m, s_{x+1}, ..., s_n}
      //
      // Removing the copy eliminates d_0, and uses of d_0 become uses of
      // s_x. In the above ordering, the live range of d_m will be ordered
      // before the live range of s_{x+1} and the definition and all uses of
      // s_x will be ordered before the definition of d_1. To make sure the
      // copy elision is safe, the following code checks that this ordering is
      // valid --- in particular we check it is safe to order d_m ahead of all
      // the liverages at and after x_{x+1}, and it is safe to order all uses
      // of s_x before the definition of d_1, by checking the live range
      // constraints for each pair --- we cannot skip the later checks because
      // the live range ordering is not guranteed to be transitive --- while it
      // may be ok to have lr_1 before lr_2, and lr_2 before lv_3 while merging
      // their buffers, it may not be ok to merge the buffers of lr_1 and lv_3,
      // because the exclusiveness relation of non-overlapping computations is
      // not transitive.
      //
      // ** Technically it might be possible to have a non-interfering
      //    non-trivial interleaving of the values of the source and
      //    destination buffers in the resulting order. This can be potentially
      //    supported in the ValuesInterfere function, which performs
      //    interference analysis at a more global scope than the alternative
      //    LiveRangeBefore analysis which requires strict ordering of all live
      //    ranges. Currently, however, this is not yet supported, as
      //    we simply check for the case where *all* values of the destination
      //    buffer (d_1 through d_m) are spliced into the point where the copy
      //    used to be.
      VLOG(2) << copy->name() << " defines the first value in its buffer";
      bool live_range_before =
          // Live range of (s_x, s_{x-1},...) must be before 'next_dest' (d_1);
          CheckLiveRangeBefore(copy_node.src, Next(*copy_node.dest)) &&
          // Live range of 'last_dest' (d_m) must be before 'next_src' s_{x+1}.
          CheckLiveRangeBefore(copy_node.dest->prev, Next(*copy_node.src));
      VLOG(2) << "LiveRangeBefore result: " << live_range_before << "\n";
      if (!live_range_before &&
          CheckLiveRangeInterference(copy_node.src, copy_node.dest,
                                     kMergeFirstDestInSource)) {
        return false;
      }
      VLOG(2) << "Splice dest after source.";
      // Splice in destination buffer values list right after 'src'.
      SpliceAfter(copy_node.dest, copy_node.src);
    } else if (IsTail(*copy_node.src)) {
      // The copy copies the last value in the source buffer, s_n, and defines
      // an arbitrary value in the destination buffer, d_y.  After
      // merging, the values in the combined buffer must be strictly ordered
      // as follows** to elide the copy:
      //
      // {d_0, ..., d_{y-1}, s_0, ..., s_n, d_{y+1}, ..., d_m}
      //
      // Removing the copy eliminates d_y, and uses of d_y become uses of
      // s_n. To enforce the above order, the live range of d_{y-1} must be
      // before the live range of s_0, and the live range of s_n must be
      // before the live range of d_{y+1}.
      //
      // ** See comment above in the code handling Case (1).
      VLOG(2) << copy->name() << " copies the last value ("
              << copy_node.src->value->ToShortString() << ") in its buffer";
      bool live_range_before =
          // Live range of d_0, ..., d_{y-1} must be before s_0;
          CheckLiveRangeBefore(Prev(*copy_node.dest), copy_node.src->next) &&
          // Live range of 'last_src' must be before next_dest d_{y+1}.
          CheckLiveRangeBefore(copy_node.src, Next(*copy_node.dest));
      VLOG(2) << "LiveRangeBefore result: " << live_range_before << "\n";
      if (!live_range_before &&
          CheckLiveRangeInterference(copy_node.src, copy_node.dest,
                                     kMergeLastSourceInDest)) {
        VLOG(2) << "Region-based analysis concludes interference.\n";
        return false;
      }
      VLOG(2) << "Splice src after prev of dest.";
      // Splice source buffer values list right after 'prev_dest'.
      SpliceAfter(copy_node.src->next, Prev(*copy_node.dest));
    } else {
      VLOG(2) << copy->name()
              << " copies value in middle of source buffer to value in middle "
                 "of destination buffer";
      return false;
    }

    RemoveCopyValue(copy_node.dest);

    XLA_VLOG_LINES(4, ToString());
    TF_DCHECK_OK(Verify());

    return true;
  }

  // Delete the given ValueNode associated with a elided kCopy
  // instruction. This should be called after splicing the value lists of the
  // source and destination buffers together.
  void RemoveCopyValue(ValueNode* copy_value_node) {
    CHECK_EQ(copy_value_node->value->defining_instruction()->opcode(),
             HloOpcode::kCopy);
    ValueNode* operand_node = copy_value_node->prev;
    CHECK(operand_node != copy_value_node);

    VLOG(2) << "Removing copy " << operand_node->value->ToShortString()
            << " => " << copy_value_node->value->ToShortString();

    // Splice out the copy value node.
    operand_node->next = copy_value_node->next;
    copy_value_node->next->prev = operand_node;

    // Patch up uses. Remove use of copy from operand_node uses.
    auto it = absl::c_find_if(operand_node->uses, [copy_value_node](
                                                      const HloUse* use) {
      return use->instruction == copy_value_node->value->defining_instruction();
    });
    CHECK(it != operand_node->uses.end());
    operand_node->uses.erase(it);

    // If the elided copy has any uses which are themselves kCopy instructions
    // then patch up the copy info to reflect the that this kCopy instruction
    // has a different operand (the operand of the elided copy).
    for (const HloUse* copy_use : copy_value_node->uses) {
      operand_node->uses.push_back(copy_use);
      if (copy_use->instruction->opcode() == HloOpcode::kCopy &&
          ContainsKey(copy_map_, copy_use->instruction)) {
        copy_map_.at(copy_use->instruction).src = operand_node;
      }
    }

    // Delete the copy info and the value node.
    copy_map_.erase(copy_value_node->value->defining_instruction());
    delete copy_value_node;
  }

  // Returns true if the live range of given value 'a' is before the live
  // range of 'b'.
  //
  // We cannot use LiveRangeStrictlyBefore because HloValue::uses() is not
  // updated as copies are removed. Also here because the result is used
  // to directly drive copy elision, use_is_always_before_def_in_same_instr is
  // set to false.
  bool LiveRangeBefore(const ValueNode& a, const ValueNode& b) {
    if (a.uses.empty()) {
      VLOG(2) << "Empty uses for " << *a.value;
      return ordering_->IsDefinedBefore(*a.value, *b.value);
    }
    VLOG(3) << "Checking live ranges before :" << ValueListToString(&a)
            << " vs " << ValueListToString(&b) << "\n";
    // If any of the positions of the "a" value is a root of the same
    // computation as "b", "a"'s live range cannot be before "b"'s. This catches
    // the cases where the root may not be the last instruction in the
    // computation.
    if (a.value->IsRootOf(b.value->defining_instruction()->parent())) {
      VLOG(3) << "Value is root of the same computation";
      return false;
    }
    return ordering_->UsesBeforeValueDefinition(
        a.uses, *b.value, dataflow_,
        /* use_is_always_before_def_in_same_instr=*/false);
  }

  // Returns whether 'node' is the last node in its list.
  bool IsTail(const ValueNode& node) const {
    return ContainsKey(value_lists_, node.next);
  }

  // Returns whether 'node' is the first node in its list.
  bool IsHead(const ValueNode& node) const {
    return ContainsKey(value_lists_, &node);
  }

  // Returns the next node in the list after 'node'. If 'node' is the
  // tail, then nullptr is returned.
  ValueNode* Next(const ValueNode& node) const {
    if (IsTail(node)) {
      return nullptr;
    } else {
      return node.next;
    }
  }

  // Returns the previous node in the list before 'node'. If 'node'
  // is the head, then nullptr is returned.
  ValueNode* Prev(const ValueNode& node) const {
    if (IsHead(node)) {
      return nullptr;
    } else {
      return node.prev;
    }
  }

  // Splices the entire linked list with 'head' as its head right after the
  // node 'insert_after' in another linked list.
  void SpliceAfter(ValueNode* head, ValueNode* insert_after) {
    DCHECK(IsHead(*head));
    value_lists_.erase(head);

    ValueNode* tail = head->prev;
    tail->next = insert_after->next;
    insert_after->next->prev = tail;

    insert_after->next = head;
    head->prev = insert_after;
  }

  enum CombineLiveRangeOption {
    kMergeFirstDestInSource = 1,
    kMergeLastSourceInDest = 2
  };
  // This function analyzes all the HloValues that have been grouped together
  // with src to share a single buffer, and all the HloValues that have been
  // similarly grouped together with dest, to determine whether these two groups
  // can be combined, by removing the operation in dest, which makes a copy of
  // the buffer in src.
  bool ValuesInterfere(const ValueNode* src, const ValueNode* dest,
                       CombineLiveRangeOption merge_location) {
    // Get the entire range of values sharing the buffers in src and dest.
    auto src_live_range = ComputeLiveRangeRegions(src);
    auto dest_live_range = ComputeLiveRangeRegions(dest);
    ComputeRelativeLocation relative_location_analysis(ordering_);
    auto rel1 =
        relative_location_analysis.Compute(src_live_range, dest_live_range);
    VLOG(3) << "Location of dest in relation to src:" << rel1.ToString()
            << " with interception set to " << rel1.InterceptDefUse() << "\n";
    auto rel2 =
        relative_location_analysis.Compute(dest_live_range, src_live_range);
    VLOG(3) << "Location of src in relation to dest:" << rel2.ToString()
            << " with interception set to " << rel1.InterceptDefUse() << "\n";
    // If src and dest are interleaved with each other, they interfere.
    if (rel1.RuntimeOrderOverlap() && rel2.RuntimeOrderOverlap()) {
      VLOG(3) << "Both relations are overlap.\n";
      return true;
    }
    // If src and dest belong to the same group of computations and do not
    // overlap, they do not interfere.
    if (rel1.RuntimeOrderOverlap() || rel2.RuntimeOrderOverlap()) {
      VLOG(3) << "At least one relation is overlap.\n";
      if (rel1.RuntimeOrderOverlap()) {
        VLOG(3) << "rel1 is overlap, with interception = "
                << rel1.InterceptDefUse() << "\n";
        if (rel1.InterceptDefUse() ||
            (merge_location != kMergeFirstDestInSource &&
             rel2.InterceptDefUse())) {
          return true;
        }
      } else {
        VLOG(3) << "rel2 is overlap, with interception = "
                << rel2.InterceptDefUse() << "\n";
        // Here src is at the end of a nested computation inside dest.
        if (rel2.InterceptDefUse() ||
            (merge_location != kMergeLastSourceInDest &&
             rel1.InterceptDefUse())) {
          return true;
        }
      }
    }
    if (relative_location_analysis.AddControlDependenceForUnorderedOps()) {
      return false;
    } else {
      // Disallow removing of copy if control deps cannot be added.
      return true;
    }
  }

  // Calls `visitor` on each item in the sequence of HloValues starting from
  // `element`.
  //
  // If element is not head, traverse from element to tail, then wrap
  // around. The ordering is important for live range region analysis.
  void ForEachValueInRange(const ValueNode* element,
                           absl::FunctionRef<void(const ValueNode*)> visitor) {
    const ValueNode* head = element;
    for (const ValueNode* p = head; p != nullptr; p = Next(*p)) {
      visitor(p);
    }
    while (!IsHead(*head)) {
      head = Prev(*head);
    }
    for (const ValueNode* p = head; p != element; p = Next(*p)) {
      visitor(p);
    }
  }

  std::string ValueListToString(const ValueNode* element) {
    std::string result = "{";
    auto VisitValueNode = [&](const ValueNode* node) {
      if (result == "{") {
        result = node->value->ToShortString();
      } else {
        StrAppend(&result, ", ");
        StrAppend(&result, node->value->ToShortString());
      }
    };
    VisitValueNode(element);
    StrAppend(&result, "}");
    return result;
  }

  std::string ToString() const {
    std::string out = absl::StrCat("CopyRemover:\n");
    StrAppend(&out, "  Def-use chains in each buffer:\n");
    for (const ValueNode* head : value_lists_) {
      StrAppend(&out, "    Buffer defined by ", head->value->ToShortString(),
                ":\n");
      const ValueNode* p = head;
      do {
        StrAppend(&out, "      ", p->value->ToShortString(), ", uses: ",
                  absl::StrJoin(p->uses, "; ",
                                [](std::string* s, const HloUse* use) {
                                  StrAppend(s, use->ToString());
                                }),
                  "\n");

        p = p->next;
      } while (p != head);
    }
    StrAppend(&out, "  Potentially removable copies:\n");
    for (const auto& pair : copy_map_) {
      const HloInstruction* copy = pair.first;
      const CopyNodes& copy_info = pair.second;

      StrAppend(&out, "    ", copy->name(), " : ",
                copy_info.src->value->ToShortString(), " => ",
                copy_info.dest->value->ToShortString(), "\n");
    }
    return out;
  }

 private:
  const HloDataflowAnalysis& dataflow_;
  HloOrdering* ordering_;

  // The heads of all the value lists. Each value list represents the HLO
  // values contained in a particular HLO buffer. The values in the list are
  // in dependency order.
  absl::flat_hash_set<const ValueNode*> value_lists_;

  // Copy removal requires fast access to the value list elements
  // corresponding to the source and destination values of the kCopy
  // instruction. This data structure holds pointers to these elements for
  // each kCopy instruction in the graph.
  struct CopyNodes {
    // The source and destinations values of the kCopy instruction.
    ValueNode* src = nullptr;
    ValueNode* dest = nullptr;
  };
  absl::flat_hash_map<const HloInstruction*, CopyNodes> copy_map_;
};

}  // namespace

// We add copies for all non-phi indices of the true and false computation
// roots, in order to resolve interference. We later rely on
// RemoveUnnecessaryCopies to drop the unnecessary ones.
Status CopyInsertion::AddCopiesForConditional(
    const HloAliasAnalysis& alias_analysis, HloInstruction* conditional) {
  VLOG(2) << "Adding copies for kConditional instruction "
          << conditional->name();
  ShapeTree<bool> indices_to_copy(conditional->shape());
  TF_RET_CHECK(conditional->opcode() == HloOpcode::kConditional);
  if (!IndicesToCopyForConditional(alias_analysis.dataflow_analysis(),
                                   conditional, &indices_to_copy)) {
    VLOG(2) << "No copies necessary for kWhile instruction "
            << conditional->name();
    return OkStatus();
  }

  for (HloComputation* computation : conditional->branch_computations()) {
    HloInstruction* root = computation->root_instruction();
    std::vector<HloInstruction*> users = root->users();
    TF_ASSIGN_OR_RETURN(
        HloInstruction * deep_copy,
        computation->DeepCopyInstruction(root, &indices_to_copy));
    for (HloInstruction* user : users) {
      TF_RETURN_IF_ERROR(root->ReplaceUseWith(user, deep_copy));
    }
    computation->set_root_instruction(deep_copy);
  }
  return OkStatus();
}

// Add kCopy instructions to the given module to guarantee there is no
// live-range interference. Generally interference can only occur around kWhile
// instructions which have update-in-place semantics.
Status CopyInsertion::AddCopiesToResolveInterference(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module, can_share_buffer_));
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        TF_RETURN_IF_ERROR(AddCopiesForWhile(*alias_analysis, instruction));
      } else if (instruction->opcode() == HloOpcode::kConditional) {
        TF_RETURN_IF_ERROR(
            AddCopiesForConditional(*alias_analysis, instruction));
      } else {
        // When an operand is a tuple, we avoid copying the operand multiple
        // times by recording and checking the operand number of operands that
        // have been copied.
        absl::flat_hash_set<int64_t> copied_operands;
        for (const auto& operand_and_output_index :
             HloDataflowAnalysis::GetInPlaceInputOutputPairs(instruction)) {
          const HloOperandIndex& operand_index = operand_and_output_index.first;
          if (copied_operands.contains(operand_index.operand_number)) {
            continue;
          }
          copied_operands.insert(operand_index.operand_number);
          TF_RETURN_IF_ERROR(AddCopiesForInPlaceOperation(
              *alias_analysis, instruction, operand_index.operand_number));
        }
      }
    }
  }

  TF_RETURN_IF_ERROR(
      AddCopiesForAliasedInputOutputs(module, execution_threads));
  return OkStatus();
}

Status CopyInsertion::AddSpecialCaseCopies(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  return AddSpecialCaseCopies(*call_graph, execution_threads, module);
}

Status CopyInsertion::AddSpecialCaseCopies(
    const CallGraph& call_graph,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    HloModule* module) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module, can_share_buffer_));

  // Identify which shape indices of which instructions need to be copied. Store
  // these results in 'instructions_to_copy'.
  HloInstructionMap<ShapeTree<bool>> instructions_to_copy;
  auto add_index_to_copy = [&instructions_to_copy](HloInstruction* instruction,
                                                   const ShapeIndex& index) {
    auto it = instructions_to_copy.find(instruction);
    if (it == instructions_to_copy.end()) {
      auto it_added = instructions_to_copy.emplace(
          std::piecewise_construct, std::forward_as_tuple(instruction),
          std::forward_as_tuple(instruction->shape(), /*init_value=*/false));
      it = it_added.first;
    }
    *it->second.mutable_element(index) = true;
  };

  // Iterate through values of all constants and entry parameters. These values
  // are special because they are held in read-only buffers. If any of these
  // values share a buffer with other values (for example, the init value of a
  // while is a constant) then copy the value at its definition and replace all
  // its uses with the copy.
  // Also, locate all input-output aliasing violations for operations that
  // cannot be done in place. Such aliasing can be created when some copies are
  // removed too aggressively by CopyRemoval.
  for (const HloValue* value : alias_analysis->dataflow_analysis().values()) {
    HloBuffer& buffer = alias_analysis->GetBufferContainingValue(*value);
    if (buffer.values().size() > 1 && ValueIsReadOnly(*value)) {
      VLOG(2) << "Value " << value->ToShortString()
              << " is read only, but its buffer contains more than one value. "
                 "Copying.";
      add_index_to_copy(value->defining_instruction(), value->defining_index());
    }
    for (const HloValue* value2 : buffer.values()) {
      // Find HloValues that share a position and use, which would cause the use
      // and operand to share buffers. Check if this is allowed and insert a
      // copy if it isn't.
      if (value2 == value) {
        continue;
      }
      HloPosition position = value2->defining_position();
      for (const HloUse& use : value->GetUses()) {
        if (use.instruction == position.instruction) {
          VLOG(3) << "Same instruction: " << position.instruction->ToString();
          if (!alias_analysis->dataflow_analysis()
                   .CanShareOperandBufferWithUser(
                       /*operand=*/use.instruction->mutable_operand(
                           use.operand_number),
                       /*operand_index=*/use.operand_index,
                       /*user=*/position.instruction,
                       /*user_index=*/position.index)) {
            VLOG(2) << "Adding back copy: "
                    << use.instruction->operand(use.operand_number)->ToString()
                    << "@" << use.operand_index.ToString()
                    << " instr: " << position.instruction->ToString() << "@"
                    << position.index;
            add_index_to_copy(
                use.instruction->mutable_operand(use.operand_number),
                use.operand_index);
          }
        }
      }
    }
  }

  // Identify copies which must be added at root instructions
  for (HloComputation* computation : module->computations(execution_threads)) {
    const CallGraphNode& node = call_graph.GetNode(computation);
    if (node.context() == CallContext::kEmbedded) {
      continue;
    }
    TF_RET_CHECK(node.context() == CallContext::kControlFlow);

    SpecialCaseCopyPolicy policy =
        GetSpecialCaseCopyPolicy(node, module, computation);
    HloInstruction* root = computation->root_instruction();

    // Mark nondistinct/ambiguous indices.
    absl::flat_hash_map<const HloBuffer*, ShapeIndex> seen;
    ShapeUtil::ForEachSubshape(
        root->shape(), [&](const Shape& /*subshape*/, const ShapeIndex& index) {
          std::vector<const HloBuffer*> buffers_at_index =
              alias_analysis->ComputeBuffersAt(root, index);
          bool buffer_seen_before = false;
          for (const HloBuffer* buffer : buffers_at_index) {
            buffer_seen_before |= !seen.emplace(buffer, index).second;
          }

          if (buffer_seen_before && policy.copy_root_replicated_buffers &&
              computation == module->entry_computation() &&
              module->input_output_alias_config().OutputHasAlias(index) &&
              buffers_at_index.size() == 1) {
            std::optional<HloInputOutputAliasConfig::Alias> alias =
                module->input_output_alias_config().GetAliasedParameter(index);
            CHECK(alias) << "Alias does not exist";
            const ShapeIndex& other_index = seen[buffers_at_index[0]];
            VLOG(2) << "Output indices " << index.ToString() << " and "
                    << other_index.ToString() << " are both aliased to "
                    << alias->parameter_number << " copying " << other_index;
            add_index_to_copy(root, other_index);
            return;
          }

          if (buffers_at_index.size() > 1 ||
              (buffer_seen_before && policy.copy_root_replicated_buffers)) {
            VLOG(2) << "Index " << index << " of computation "
                    << computation->name() << " (" << root->name()
                    << ") has ambiguous or non-distinct buffer. Copying.";
            add_index_to_copy(root, index);
          }
        });

    for (const auto& pair :
         alias_analysis->dataflow_analysis().GetInstructionValueSet(root)) {
      const ShapeIndex& index = pair.first;
      const HloValueSet& value_set = pair.second;
      for (const HloValue* value : value_set.values()) {
        if (ShouldCopyRootValue(*value, policy)) {
          VLOG(2) << "Root of (" << root->name() << ") of computation("
                  << computation->name()
                  << ") has constant or parameter value at index " << index
                  << ". Copying.";
          add_index_to_copy(root, index);
        }
      }
    }
  }

  // Add copy instructions indicated in 'instructions_to_copy' to the module.
  for (const auto& pair : instructions_to_copy) {
    HloInstruction* instruction = pair.first;
    const ShapeTree<bool>& indices_to_copy = pair.second;

    ShapeTree<HloInstruction*> copies_added(indices_to_copy.shape());
    std::vector<HloInstruction*> users = instruction->users();
    TF_ASSIGN_OR_RETURN(HloInstruction * deep_copy,
                        instruction->parent()->DeepCopyInstruction(
                            instruction, &indices_to_copy, &copies_added));
    for (HloInstruction* user : users) {
      TF_RETURN_IF_ERROR(instruction->ReplaceUseWith(user, deep_copy));
    }
    if (instruction == instruction->parent()->root_instruction()) {
      instruction->parent()->set_root_instruction(deep_copy);
    }
  }
  return OkStatus();
}

static int64_t GetNumExistingCopies(
    const HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  int64_t num_existing_copies = 0;
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCopy) {
        ++num_existing_copies;
      }
    }
  }
  return num_existing_copies;
}

Status CopyInsertion::RemoveUnnecessaryCopies(
    HloOrdering* ordering, HloModule* module, bool check_live_range_ordering,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(4, module->ToString());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module, can_share_buffer_));
  CopyRemover copy_remover(*module, *alias_analysis, ordering,
                           check_live_range_ordering);
  if (VLOG_IS_ON(3)) {
    LOG(INFO) << "Removing unnecessary copies in " << module->name();
    LOG(INFO) << "Buffer values, in dependency order: ";
    for (const HloBuffer& buffer : alias_analysis->buffers()) {
      LOG(INFO) << "    HloBuffer " << buffer.id();
    }
  }

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);

  int64_t num_existing_copies = GetNumExistingCopies(module, execution_threads);
  bool changed = true;
  int64_t num_iterations = -1;
  VLOG(6) << "Copy Insertion analyzing module with instruction count = "
          << module->instruction_count() << "\n";
  BoundNonLinearCompilerAnalysis allowance(module, name(), 10);
  while (changed) {
    CHECK_LE(++num_iterations, num_existing_copies);
    changed = false;
    VLOG(2) << "Running fixpoint iteration " << num_iterations
            << " of copy elision";
    for (HloComputation* computation :
         module->computations(execution_threads)) {
      VLOG(2) << "computation:" << computation->name() << "\n";
      for (HloInstruction* instruction : computation->instructions()) {
        VLOG(2) << instruction->ToString() << "\n";
        // The region_analysis_cost_now is always set to
        // use_region_based_live_range_analysis_ if it is < 0, in which case the
        // analysis is always performed.
        int64_t region_analysis_cost_now =
            (use_region_based_live_range_analysis_ == 0)
                ? 0
                : std::min(allowance.analysis_allowance(),
                           use_region_based_live_range_analysis_);
        if (instruction->opcode() == HloOpcode::kCopy) {
          if (copy_remover.TryElideCopy(instruction,
                                        &region_analysis_cost_now)) {
            changed = true;
            TF_RETURN_IF_ERROR(StripControlDependenciesFrom(instruction));
            TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(
                instruction->mutable_operand(0)));
            VLOG(6) << "succeeded in eliminating copy.\n";
          }
          if (allowance.ContinueAnalysis() && region_analysis_cost_now > 0) {
            VLOG(6) << "Copy Insertion analyzing module cost: "
                    << region_analysis_cost_now << "\n";
            VLOG(6) << "instruction:" << instruction->ToString() << "\n";
            allowance.DeductCost(region_analysis_cost_now);
            VLOG(6) << "allowance:" << allowance.analysis_allowance() << "\n";
          }
        }
      }
    }
  }
  return OkStatus();
}

StatusOr<bool> CopyInsertion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Copy insertion is performed in three steps:
  //
  // (1) Add copies conservatively to guarantee that there is no live-range
  //     interference. This is done simplistically and usually results in more
  //     copies than is strictly necessary.
  //
  // (2) Using a more fine-grained analysis, remove as many copies that were
  //     added in (1) as possible while ensuring no live-range interference.
  //
  // (3) Add copies to resolve issues not related to live range interference
  //     such as parameters and constants live out of the entry computation.
  //
  // We add copies then remove them (step (1) then (2)) rather than simply
  // adding only the copies that are necessary because, in general, it is
  // difficult to figure out the minimal set of copies to add once there is
  // interference. On the other hand, it is easy to determine if removing a copy
  // will introduce interference.
  //
  // The final copy insertion in (3) is done separately to simplify the
  // implementation of copy removal in (2) which is the most complicated part of
  // the pass. As is, copy removal only has to reason about live range
  // interference. If all copies were added in step (1) then copy removal would
  // also have to reason about things like constants and parameters live out of
  // the computation.
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  if (!call_graph->IsFlattened()) {
    return FailedPrecondition(
        "Call graph must be flattened before copy insertion.");
  }

  int64_t num_copies_before = GetNumExistingCopies(module, execution_threads);

  TF_RETURN_IF_ERROR(AddCopiesToResolveInterference(module, execution_threads));

  // Simplify the tuple structures introduced by the deep copies. This should be
  // done before removing copies (RemoveUnnecessaryCopies) because tuple
  // simplification changes dependencies in the graph which changes live range
  // interference in the graph. Also run DCE to remove the dead Tuple/GTE
  // instructions introduced by tuple simplification.
  TupleSimplifier tuple_simplifier;
  HloDCE dce;
  TF_RETURN_IF_ERROR(tuple_simplifier.Run(module, execution_threads).status());
  TF_RETURN_IF_ERROR(dce.Run(module, execution_threads).status());
  DumpHloModuleDuringPassIfEnabled(
      name(), "after adding copies to resolve interference", *module);

  DependencyHloOrdering ordering(module);
  TF_RETURN_IF_ERROR(RemoveUnnecessaryCopies(&ordering, module,
                                             /*check_live_range_ordering=*/true,
                                             execution_threads));
  DumpHloModuleDuringPassIfEnabled(name(), "after removing unnecessary copies",
                                   *module);
  TF_RETURN_IF_ERROR(
      AddSpecialCaseCopies(*call_graph, execution_threads, module));
  DumpHloModuleDuringPassIfEnabled(name(), "after adding special-case copies",
                                   *module);

  TF_RETURN_IF_ERROR(tuple_simplifier.Run(module, execution_threads).status());
  TF_RETURN_IF_ERROR(dce.Run(module, execution_threads).status());

  VLOG(1) << "Num copies before copy-insertion: " << num_copies_before;
  VLOG(1) << "Num copies after copy-insertion: "
          << GetNumExistingCopies(module, execution_threads);

  return true;
}
}  // namespace xla
