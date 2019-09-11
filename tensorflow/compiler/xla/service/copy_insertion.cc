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

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

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

// Add kCopy instructions around the given kWhile instruction to eliminate any
// possible live range interference of HLO values assuming a dependency-based
// ordering (HloDependencyOrdering). Copies are added conservatively. There
// likely are copies which are not strictly necessary, but they are removed
// later in the pass via RemoveUnnecessaryCopies.
//
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
// copy constructed of kCopy, KGetTupleElement, and kTuple instruction as
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
    return Status::OK();
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

  return Status::OK();
}

// We add copies for all the indices of the true and false computation roots, in
// order to resolve interference. We later rely on RemoveUnnecessaryCopies to
// drop the unnecessary ones.
Status AddCopiesForConditional(const HloAliasAnalysis& alias_analysis,
                               HloInstruction* conditional) {
  VLOG(2) << "Adding copies for kConditional instruction "
          << conditional->name();
  TF_RET_CHECK(conditional->opcode() == HloOpcode::kConditional);

  for (HloComputation* computation : conditional->branch_computations()) {
    HloInstruction* root = computation->root_instruction();
    std::vector<HloInstruction*> users = root->users();
    TF_ASSIGN_OR_RETURN(HloInstruction * deep_copy,
                        computation->DeepCopyInstruction(root));
    for (HloInstruction* user : users) {
      TF_RETURN_IF_ERROR(root->ReplaceUseWith(user, deep_copy));
    }
    computation->set_root_instruction(deep_copy);
  }
  return Status::OK();
}

// Conservatively adds copies before root instruction of entry computation and
// each aliased parameter to resolve interference of aliased input and output
// buffer. We later rely on RemoveUnnecessaryCopies to drop the unnecessary
// ones.
Status AddCopiesForAliasedInputOutputs(HloModule* module) {
  HloComputation* entry = module->entry_computation();
  HloInstruction* root = entry->root_instruction();

  ShapeTree<bool> output_indices_to_copy(root->shape());
  std::vector<absl::optional<ShapeTree<HloInstruction*>>> copied_parameters(
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
    for (HloInstruction* user : users) {
      TF_RETURN_IF_ERROR(param->ReplaceUseWith(user, copied));
    }

    copied_parameters[param->parameter_number()] = param_copy_tree;
  }

  if (!has_alias) {
    return Status::OK();
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
          return Status::OK();
        }
        HloInstruction* from =
            copied_parameters[alias.parameter_number]->element(
                alias.parameter_index);
        HloInstruction* to = output_copy_tree.element(output_index);

        TF_RET_CHECK(from != nullptr);
        TF_RET_CHECK(to != nullptr);
        TF_RETURN_IF_ERROR(from->AddControlDependencyTo(to));
        return Status::OK();
      }));

  entry->set_root_instruction(root_copied);

  return Status::OK();
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

  return Status::OK();
}

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
              const HloOrdering& ordering)
      : dataflow_(alias_analysis.dataflow_analysis()), ordering_(ordering) {
    // Construct a list for each HLO buffer in the alias analysis. Maintain a
    // map from HloValue to the respective list element representing that
    // value. The map is used to construct the copy info map below.
    absl::flat_hash_map<const HloValue*, ValueNode*> value_to_node;
    for (const HloBuffer& buffer : alias_analysis.buffers()) {
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
            DCHECK(ordering_.LiveRangeStrictlyBefore(*value_a, *value_b,
                                                     dataflow_) ||
                   ordering_.LiveRangeStrictlyBefore(*value_b, *value_a,
                                                     dataflow_))
                << value_a->ToShortString() << " and "
                << value_b->ToShortString() << " are not ordered";
          }
        }
      }

      std::vector<const HloValue*> values = buffer.values();
      absl::c_sort(values, [this](const HloValue* a, const HloValue* b) {
        return ordering_.IsDefinedBefore(*a, *b);
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
      new_node->uses.reserve(value->uses().size());
      for (const HloUse& use : value->uses()) {
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
    for (HloComputation* computation : module.computations()) {
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
    return Status::OK();
  }

  // Try to elide the given copy. Elision of a copy is possible only if no
  // live range interference is introduced by the copy's elimination. If
  // elision is possible, then the internal state (value lists) are updated,
  // and true is returned. Returns false otherwise.
  bool TryElideCopy(const HloInstruction* copy) {
    VLOG(2) << "Trying to remove " << copy->name();

    if (!ContainsKey(copy_map_, copy)) {
      VLOG(2) << copy->name() << " is not removable";
      return false;
    }
    if (!ShapeUtil::Equal(copy->shape(), copy->operand(0)->shape())) {
      VLOG(2) << copy->name() << " is not removable (shape mismatch)";
      return false;
    }
    const CopyNodes& copy_node = copy_map_.at(copy);
    ValueNode* src = copy_node.src;
    ValueNode* dest = copy_node.dest;
    DCHECK(src != nullptr);
    DCHECK(dest != nullptr);

    VLOG(3) << copy->name() << " copies value " << src->value->ToShortString();
    VLOG(3) << "Source buffer values: " << ValueListToString(src);
    VLOG(3) << "Dest buffer values: " << ValueListToString(dest);

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
    if (src->next == dest) {
      // In the process of eliding copies, its possible for a copy to have the
      // same source and destination buffer. In this case, the copy can be
      // safely removed.
      VLOG(2) << copy->name() << " source and destination buffers are same.";
    } else if (IsHead(*dest)) {
      // The copy copies an arbitrary value in the source buffer (call it s_x)
      // and defines d_0, the first value in the destination buffer. After
      // merging, the values in the combined buffer must be strictly ordered
      // as follows** to elide the copy:
      //
      // {s_0, ..., s_x, d_1, ..., d_m, s_{x+1}, ..., s_n}
      //
      // Removing the copy eliminates d_0, and uses of d_0 become uses of
      // s_x. In the above ordering, the live range of d_m must be ordered
      // before the live range of s_{x+1} and the definition and all uses of
      // s_x must be ordered before the definition of d_1. These conditions
      // are checked below prior to elision.
      //
      // ** Technically it might be possible to have a non-interfering
      //    non-trivial interleaving of the values of the source and
      //    destination buffers in the resulting order. However, this case is
      //    slow and complicated to check and likely not worth it. So instead
      //    we simply check for the case where *all* values of the destination
      //    buffer (d_1 through d_m) are spliced into the point where the copy
      //    used to be.
      VLOG(2) << copy->name() << " defines the first value in its buffer";
      ValueNode* next_dest = Next(*dest);
      if (next_dest != nullptr) {
        // Live range of 'from' value (s_x) must be before 'next_dest' (d_1);
        if (!LiveRangeBefore(*src, *next_dest)) {
          return false;
        }
      }
      ValueNode* next_src = Next(*src);

      if (next_src != nullptr) {
        // Live range of 'last_dest' (d_m) must be before 'next_src' s_{x+1}.
        ValueNode* last_dest = dest->prev;
        DCHECK(IsTail(*last_dest));
        if (!LiveRangeBefore(*last_dest, *next_src)) {
          return false;
        }
      }

      // Splice in destination buffer values list right after 'src'.
      SpliceAfter(dest, src);
    } else if (IsTail(*src)) {
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
              << src->value->ToShortString() << ") in its buffer";

      ValueNode* prev_dest = Prev(*dest);
      // nullptr condition handled above in the first 'if' case.
      DCHECK(prev_dest != nullptr);
      ValueNode* first_src = src->next;
      DCHECK(IsHead(*first_src));
      if (!LiveRangeBefore(*prev_dest, *first_src)) {
        // Live range of value d_{y-1} is not before s_0.
        return false;
      }
      ValueNode* next_dest = Next(*dest);
      if (next_dest != nullptr) {
        if (!LiveRangeBefore(*src, *next_dest)) {
          // Live range of value s_n is not before d_{y+1}.
          return false;
        }
      }

      // Splice source buffer values list right after 'prev_dest'.
      SpliceAfter(first_src, prev_dest);
    } else {
      VLOG(2) << copy->name()
              << " copies value in middle of source buffer to value in middle "
                 "of destination buffer";
      return false;
    }

    RemoveCopyValue(dest);

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
  // updated as copies are removed.
  bool LiveRangeBefore(const ValueNode& a, const ValueNode& b) {
    VLOG(3) << "Checking live range of " << *a.value << " WRT " << *b.value;
    bool is_live_range_before = [&] {
      if (a.uses.empty()) {
        VLOG(2) << "Empty uses for " << *a.value;
        return ordering_.IsDefinedBefore(*a.value, *b.value);
      }
      for (const HloUse* use : a.uses) {
        VLOG(3) << "Checking use " << *use << " against " << *b.value;
        if (!ordering_.UseIsBeforeValueDefinition(*use, *b.value, dataflow_)) {
          VLOG(2) << "Use " << *use << " is NOT before " << *b.value;
          return false;
        }
        VLOG(3) << "Use " << *use << " is before " << *b.value;
      }
      return true;
    }();
    if (is_live_range_before) {
      VLOG(2) << "  Live range of " << a.value->ToShortString() << " is before "
              << b.value->ToShortString();
    } else {
      VLOG(2) << "  Live range of " << a.value->ToShortString()
              << " is not before " << b.value->ToShortString();
    }
    return is_live_range_before;
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

  string ValueListToString(const ValueNode* element) {
    const ValueNode* head = element;
    while (!IsHead(*head)) {
      head = Prev(*head);
    }
    std::vector<const HloValue*> values;
    for (const ValueNode* p = head; p != nullptr; p = Next(*p)) {
      values.push_back(p->value);
    }
    return absl::StrCat("{",
                        absl::StrJoin(values, ", ",
                                      [](string* s, const HloValue* value) {
                                        StrAppend(s, value->ToShortString());
                                      }),
                        "}");
  }

  string ToString() const {
    string out = absl::StrCat("CopyRemover:\n");
    StrAppend(&out, "  Def-use chains in each buffer:\n");
    for (const ValueNode* head : value_lists_) {
      StrAppend(&out, "    Buffer defined by ", head->value->ToShortString(),
                ":\n");
      const ValueNode* p = head;
      do {
        StrAppend(&out, "      ", p->value->ToShortString(), ", uses: ",
                  absl::StrJoin(p->uses, "; ",
                                [](string* s, const HloUse* use) {
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
  const HloOrdering& ordering_;

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

// Add kCopy instructions to the given module to guarantee there is no
// live-range interference. Generally interference can only occur around kWhile
// instructions which have update-in-place semantics.
Status CopyInsertion::AddCopiesToResolveInterference(HloModule* module) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module, can_share_buffer_));

  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        TF_RETURN_IF_ERROR(AddCopiesForWhile(*alias_analysis, instruction));
      } else if (instruction->opcode() == HloOpcode::kConditional) {
        TF_RETURN_IF_ERROR(
            AddCopiesForConditional(*alias_analysis, instruction));
      }
    }
  }

  TF_RETURN_IF_ERROR(AddCopiesForAliasedInputOutputs(module));
  return Status::OK();
}

Status CopyInsertion::AddSpecialCaseCopies(HloModule* module) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  return AddSpecialCaseCopies(*call_graph, module);
}

Status CopyInsertion::AddSpecialCaseCopies(const CallGraph& call_graph,
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
  for (const HloValue* value : alias_analysis->dataflow_analysis().values()) {
    if (ValueIsReadOnly(*value) &&
        alias_analysis->GetBufferContainingValue(*value).values().size() > 1) {
      VLOG(2) << "Value " << value->ToShortString()
              << " is read only, but its buffer contains more than one value. "
                 "Copying.";
      add_index_to_copy(value->defining_instruction(), value->defining_index());
    }
  }

  // Identify copies which must be added at root instructions
  for (HloComputation* computation : module->computations()) {
    const CallGraphNode& node = call_graph.GetNode(computation);
    if (node.context() == CallContext::kParallel) {
      continue;
    }
    TF_RET_CHECK(node.context() == CallContext::kSequential);

    SpecialCaseCopyPolicy policy =
        GetSpecialCaseCopyPolicy(node, module, computation);
    HloInstruction* root = computation->root_instruction();

    // Mark nondistinct/ambiguous indices.
    absl::flat_hash_set<const HloBuffer*> seen;
    ShapeUtil::ForEachSubshape(
        root->shape(), [&](const Shape& /*subshape*/, const ShapeIndex& index) {
          std::vector<const HloBuffer*> buffers_at_index =
              alias_analysis->ComputeBuffersAt(root, index);
          bool buffer_seen_before = false;
          for (const HloBuffer* buffer : buffers_at_index) {
            buffer_seen_before |= !seen.insert(buffer).second;
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
  return Status::OK();
}

Status CopyInsertion::RemoveUnnecessaryCopies(const HloOrdering& ordering,
                                              HloModule* module) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module, can_share_buffer_));

  CopyRemover copy_remover(*module, *alias_analysis, ordering);
  if (VLOG_IS_ON(3)) {
    LOG(INFO) << "Removing unnecessary copies in " << module->name();
    LOG(INFO) << "Buffer values, in dependency order: ";
    for (const HloBuffer& buffer : alias_analysis->buffers()) {
      LOG(INFO) << "    HloBuffer " << buffer.id();
    }
  }

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCopy &&
          copy_remover.TryElideCopy(instruction)) {
        TF_RETURN_IF_ERROR(StripControlDependenciesFrom(instruction));
        TF_RETURN_IF_ERROR(
            instruction->ReplaceAllUsesWith(instruction->mutable_operand(0)));
      }
    }
  }
  return Status::OK();
}

StatusOr<bool> CopyInsertion::Run(HloModule* module) {
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

  int64 num_existing_copies = 0;
  if (VLOG_IS_ON(1)) {
    for (HloComputation* computation : module->computations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->opcode() == HloOpcode::kCopy) {
          ++num_existing_copies;
        }
      }
    }
  }

  TF_RETURN_IF_ERROR(AddCopiesToResolveInterference(module));

  // Simplify the tuple structures introduced by the deep copies. This should be
  // done before removing copies (RemoveUnnecessaryCopies) because tuple
  // simplification changes dependencies in the graph which changes live range
  // interference in the graph. Also run DCE to remove the dead Tuple/GTE
  // instructions introduced by tuple simplification.
  TupleSimplifier tuple_simplifier;
  HloDCE dce;
  TF_RETURN_IF_ERROR(tuple_simplifier.Run(module).status());
  TF_RETURN_IF_ERROR(dce.Run(module).status());
  DumpHloModuleDuringPassIfEnabled(
      name(), "after adding copies to resolve interference", *module);

  TF_RETURN_IF_ERROR(
      RemoveUnnecessaryCopies(DependencyHloOrdering(module), module));
  DumpHloModuleDuringPassIfEnabled(name(), "after removing unnecessary copies",
                                   *module);

  TF_RETURN_IF_ERROR(AddSpecialCaseCopies(*call_graph, module));
  DumpHloModuleDuringPassIfEnabled(name(), "after adding special-case copies",
                                   *module);

  TF_RETURN_IF_ERROR(tuple_simplifier.Run(module).status());
  TF_RETURN_IF_ERROR(dce.Run(module).status());

  if (VLOG_IS_ON(1)) {
    int64 num_total_copies = 0;
    for (HloComputation* computation : module->computations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->opcode() == HloOpcode::kCopy) {
          num_total_copies++;
        }
      }
    }
    VLOG(1) << "Num copies before copy-insertion: " << num_existing_copies;
    VLOG(1) << "Num copies after copy-insertion: " << num_total_copies;
  }

  return true;
}
}  // namespace xla
