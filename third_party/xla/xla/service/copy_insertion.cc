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

#include "xla/service/copy_insertion.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/frontend_attributes.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/ir/ptrvec.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/service/call_graph.h"
#include "xla/service/compile_time_cap.h"
#include "xla/service/copy_removal.h"
#include "xla/service/dump.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace {

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
// of 'to'. These control edges are necessary to prevent live range interference
// between the kCopy instructions in the deep copy of 'from' and the kCopy
// instructions in the deep copy of 'to'.
//
// Requirements: 'from' and 'to' must have compatible shapes.
//
// For example, suppose 'from' and 'to' are two-element tuples where index 0 is
// the only index to copy. Prior to deep-copying we have:
//
//
//       'from'
//          |
//         ...
//          |
//        'to'
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
absl::StatusOr<std::pair<HloInstruction*, HloInstruction*>>
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

// Returns true if the instruction produces non-copyable results.
//
// Currently, only asynchronous start ops produce non-copyable results and the
// the whole result is non-copyable.
bool IsNonCopyable(const HloInstruction* instruction) {
  // Currently, the verifier only allows the pipelining of Send/Recv. As such,
  // here we only handle to the ops allowed by
  // HloDataflowAnalysis::IsAsynchronousOperationStart that pass through its
  // operand for now. For the ops that don't pass through its operand, we need
  // to add a copy of its operand for the straight line case in order to allow
  // all ops in HloDataflowAnalysis::IsAsynchronousOperationStart.
  HloOpcode opcode = instruction->opcode();
  return opcode == HloOpcode::kSend || opcode == HloOpcode::kRecv ||
         opcode == HloOpcode::kCopyStart;
}

// Returns true if the value at the given index in the while init is
// non-copyable.
bool IsNonCopyableInWhileInit(const HloInstruction* while_init,
                              const ShapeIndex& index) {
  if (index.empty()) {
    return false;
  }
  int64_t i = index.front();
  return i < while_init->operand_count() &&
         IsNonCopyable(while_init->operand(i));
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
    if (IsNonCopyableInWhileInit(init, index)) {
      // Do not copy non-copyable values, instead, we will add copies for
      // transitioning into and out of non-copyable values.
      should_copy = false;
      continue;
    }
    if (dataflow.GetValueSet(init, index).values().size() > 1 ||
        dataflow.GetValueSet(xla_while, index).values().size() > 1) {
      // If there is any ambiguity, then loop state must be copied.
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
absl::Status AddCopiesForWhile(const HloAliasAnalysis& alias_analysis,
                               HloInstruction* xla_while) {
  VLOG(2) << "Adding copies for kWhile instruction " << xla_while->name();
  TF_RET_CHECK(xla_while->opcode() == HloOpcode::kWhile);

  ShapeTree<bool> indices_to_copy(xla_while->shape());
  if (!IndicesToCopyForWhile(alias_analysis.dataflow_analysis(), xla_while,
                             &indices_to_copy)) {
    VLOG(2) << "No copies necessary for kWhile instruction "
            << xla_while->name();
    return absl::OkStatus();
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
  return absl::OkStatus();
}

// Add copies for the operands of in-place operations. RemoveUnnecessaryCopies
// will remove the unnecessary copies.
absl::Status AddCopiesForInPlaceOperation(
    const HloAliasAnalysis& alias_analysis, HloInstruction* in_place_op,
    int64_t operand_number) {
  VLOG(2) << "Adding copies for in-place operation " << in_place_op->name();
  HloInstruction* operand = in_place_op->mutable_operand(operand_number);
  TF_ASSIGN_OR_RETURN(HloInstruction * deep_copy,
                      in_place_op->parent()->DeepCopyInstruction(operand));
  TF_RETURN_IF_ERROR(
      operand->ReplaceUseWith(in_place_op, operand_number, deep_copy));
  return absl::OkStatus();
}

// Conservatively adds copies before root instruction of entry computation and
// each aliased parameter to resolve interference of aliased input and output
// buffer. We later rely on RemoveUnnecessaryCopies to drop the unnecessary
// ones.
absl::Status AddCopiesForAliasedInputOutputs(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HloComputation* entry = module->entry_computation();
  if (!HloInstruction::IsThreadIncluded(entry->execution_thread(),
                                        execution_threads)) {
    return absl::OkStatus();
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
    return absl::OkStatus();
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
          const HloInputOutputAliasConfig::Alias& alias) -> absl::Status {
        if (!copied_parameters[alias.parameter_number]) {
          return absl::OkStatus();
        }
        HloInstruction* from =
            copied_parameters[alias.parameter_number]->element(
                alias.parameter_index);
        HloInstruction* to = output_copy_tree.element(output_index);

        TF_RET_CHECK(from != nullptr);
        TF_RET_CHECK(to != nullptr);
        TF_RETURN_IF_ERROR(from->AddControlDependencyTo(to));
        return absl::OkStatus();
      }));

  entry->set_root_instruction(root_copied);

  return absl::OkStatus();
}

// Removes any control dependencies to or from the given instruction.
absl::Status StripControlDependenciesFrom(HloInstruction* instruction) {
  while (!instruction->control_successors().empty()) {
    TF_RETURN_IF_ERROR(instruction->RemoveControlDependencyTo(
        instruction->control_successors().front()));
  }

  while (!instruction->control_predecessors().empty()) {
    TF_RETURN_IF_ERROR(
        instruction->control_predecessors().front()->RemoveControlDependencyTo(
            instruction));
  }

  return absl::OkStatus();
}
}  // namespace

// We add copies for all phi indices of the true and false computation
// roots, in order to resolve interference. We later rely on
// RemoveUnnecessaryCopies to drop the unnecessary ones.
absl::Status CopyInsertion::AddCopiesForConditional(
    const HloAliasAnalysis& alias_analysis, HloInstruction* conditional) {
  VLOG(2) << "Adding copies for kConditional instruction "
          << conditional->name();
  ShapeTree<bool> indices_to_copy(conditional->shape());
  TF_RET_CHECK(conditional->opcode() == HloOpcode::kConditional);
  if (!IndicesToCopyForConditional(alias_analysis.dataflow_analysis(),
                                   conditional, &indices_to_copy)) {
    VLOG(2) << "No copies necessary for kConditional instruction "
            << conditional->name();
    return absl::OkStatus();
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
  return absl::OkStatus();
}

// If `chain_start` is the head of a chain of non-copyable ops inside a while
// loop, and part of the chain is rotated to the next iteration, returns the
// chain end in the rotated part. Otherwise, returns nullptr.
HloInstruction* FindEndOpForRotatedNonCopyableChain(
    const HloComputation* while_body, const HloInstruction* chain_start) {
  // Non-copyable op must have a single user.
  if (chain_start->user_count() != 1) {
    return nullptr;
  }
  HloInstruction* unique_user = chain_start->users().front();
  if (unique_user->opcode() != HloOpcode::kTuple || !unique_user->IsRoot()) {
    return nullptr;
  }
  int64_t index = unique_user->operand_index(chain_start);
  for (const HloInstruction* it :
       while_body->parameter_instruction(0)->users()) {
    const auto* gte = DynCast<HloGetTupleElementInstruction>(it);
    if (gte->tuple_index() == index) {
      CHECK_EQ(gte->user_count(), 1)
          << "non-copyable value in next loop iteration must "
             "be consumed by unique instruction.";
      HloInstruction* next_unique_user = gte->users().front();
      if (HloDataflowAnalysis::IsAsynchronousOperationDone(
              next_unique_user->opcode())) {
        return next_unique_user;
      }
      break;
    }
  }
  return nullptr;
}

// Adds copies for non-copyable transitioning between copyable and non-copyable
// for a chain start with `chain_start` and part of the chain is rotated to the
// next iteration that ends with `chain_end`.
absl::Status AddCopiesForNonCopyableTransitionsRotatedCase(
    HloInstruction* chain_start, HloInstruction* chain_end) {
  HloComputation* while_body = chain_start->parent();
  // Handle aliasing input for the op, where we transition from copyable to
  // non-copyable.
  if (!chain_start->operands().empty()) {
    // A chain_start may have multiple operands, but we assume only the first
    // operand is a buffer aliasing with the output, which is true currently.
    HloInstruction* operand = chain_start->mutable_operand(0);
    HloInstruction* copied_operand =
        while_body->AddInstruction(HloInstruction::CreateUnary(
            operand->shape(), HloOpcode::kCopy, operand));
    TF_RETURN_IF_ERROR(operand->ReplaceUseWith(chain_start, copied_operand));
    TF_RETURN_IF_ERROR(chain_end->AddControlDependencyTo(copied_operand));
  }

  // The chain_end is rotated and semantically paired with the chain_start of
  // the previous iteration. We add a control dependency from the chain_end to
  // the chain_start to in the same lexical iteration guarantee disjoint live
  // times of the buffers involved.
  TF_RETURN_IF_ERROR(chain_end->AddControlDependencyTo(chain_start));

  // If chain_end has users, insert copies for the result produced by the
  // chain_end with aliasing input and output buffers, where we transition from
  // non-copyable to copyable.
  PtrVec<HloInstruction*> users = chain_end->users();
  if (users.empty()) {
    return absl::OkStatus();
  }
  ShapeTree<HloInstruction*> copies_added(chain_end->shape());
  TF_ASSIGN_OR_RETURN(
      HloInstruction * copy,
      while_body->DeepCopyInstruction(chain_end, /*indices_to_copy=*/nullptr,
                                      &copies_added));
  for (auto [shape_index, instr] : copies_added) {
    if (instr != nullptr) {
      TF_RETURN_IF_ERROR(instr->AddControlDependencyTo(chain_start));
    }
  }
  for (HloInstruction* it : users) {
    TF_RETURN_IF_ERROR(chain_end->ReplaceUseWith(it, copy));
  }
  return absl::OkStatus();
}

// Adds the needed copies for transitioning into and out of non-copyable values,
// to prevent overlapping live times of buffers. This is needed when the unique
// user of the non-copyable op is rotated (also called pipelined) in a
// while-loop. In particlar, if a non-copyable op has an input aliasing with its
// output, such as async Send, we make a copy of its input to transition from
// copyable to non-copyable. If a non-copyable op's unique user produces an
// output aliasing with its input, such as async Recv, we make a copy of the
// output produced by the unique user, to transition out of non-copyable to
// copyable. We also add control-flow edges between the copies and the
// non-copyable op to guarantee disjoint live times of the buffers invovled.
//
// Using async Send and Recv as examples, here is the transformation:
//
// Before:
//
//      kParameter               kParameter
//          |                        |
//      kSendDone                kRecvDone (end of a non-copyable chain)
//                                   |
//         ...                    consumer
//
//       producer                   ...
//          |
//        kSend                    kRecv   (start of a non-copyable op)
//          |                        |
//     (body root)              (body root)
//
//
// After:
//
//      kParameter                kParameter
//          |                         |
//      kSendDone ----+           kRecvDone
//                    |               |
//                   ctrl           kCopy ----+
//       producer    edge             |       |
//          |         |            consumer  ctrl
//        kCopy <-----+                      edge
//          |                                 |
//        kSend                     kRecv <---+
//          |                         |
//     (body root)               (body root)
//
absl::Status CopyInsertion::AddCopiesForNonCopyableTransitions(
    const HloAliasAnalysis& alias_analysis, HloInstruction* chain_start) {
  if (chain_start->users().empty()) {
    return absl::OkStatus();
  }

  // Currently non-copyable ops can have at most one user.
  if (chain_start->users().size() != 1) {
    return absl::InvalidArgumentError(
        "Non-copyable op must have a single user.");
  }

  HloInstruction* unique_user = chain_start->users().front();
  // If start feeds directly into done, the live time is contained and we don't
  // need to add any copies.
  if (HloDataflowAnalysis::IsAsynchronousOperationDone(unique_user->opcode())) {
    return absl::OkStatus();
  }

  HloComputation* parent = chain_start->parent();
  // If a start op with an operand is fed into a pipelined while-loop, we
  // need to make a copy of the operand and use the copy in the start op.
  if (chain_start->operand_count() > 0 &&
      unique_user->opcode() == HloOpcode::kTuple &&
      unique_user->users().size() == 1 &&
      unique_user->users().front()->opcode() == HloOpcode::kWhile) {
    HloInstruction* operand = chain_start->mutable_operand(0);
    HloInstruction* copied_operand =
        parent->AddInstruction(HloInstruction::CreateUnary(
            operand->shape(), HloOpcode::kCopy, operand));
    TF_RETURN_IF_ERROR(operand->ReplaceUseWith(chain_start, copied_operand));
    return absl::OkStatus();
  }

  // For other cases where a non-copyable chain is outside of the while loop,
  // live times are disjoint. No copies are needed.
  if (parent->caller_instructions(HloOpcode::kWhile).empty()) {
    return absl::OkStatus();
  }

  // For async start ops, the end of the chain is the async done op.
  HloInstruction* chain_end =
      FindEndOpForRotatedNonCopyableChain(parent, chain_start);
  if (chain_end) {
    return AddCopiesForNonCopyableTransitionsRotatedCase(chain_start,
                                                         chain_end);
  }
  return absl::OkStatus();
}

// Add kCopy instructions to the given module to guarantee there is no
// live-range interference. Generally interference can only occur around kWhile
// instructions which have update-in-place semantics.
absl::Status CopyInsertion::AddCopiesToResolveInterference(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module, can_share_buffer_));
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (computation->IsAsyncComputation()) {
      continue;
    }
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        TF_RETURN_IF_ERROR(AddCopiesForWhile(*alias_analysis, instruction));
      } else if (instruction->opcode() == HloOpcode::kConditional) {
        TF_RETURN_IF_ERROR(
            AddCopiesForConditional(*alias_analysis, instruction));
      } else if (IsNonCopyable(instruction)) {
        TF_RETURN_IF_ERROR(
            AddCopiesForNonCopyableTransitions(*alias_analysis, instruction));
      } else {
        // When an operand is a tuple, we avoid copying the operand multiple
        // times by recording and checking the operand number of operands that
        // have been copied.
        absl::flat_hash_set<int64_t> copied_operands;
        for (const auto& operand_and_output_index :
             HloDataflowAnalysis::GetInPlaceInputOutputPairs(
                 // Input/output buffer aliasing analysis needs to be done
                 // directly with the wrapped instruction when the compiler sees
                 // an async box.
                 instruction->opcode() == HloOpcode::kAsyncStart
                     ? instruction->async_wrapped_instruction()
                     : instruction)) {
          const HloOperandIndex& operand_index = operand_and_output_index.first;
          if (copied_operands.contains(operand_index.operand_number)) {
            continue;
          }

          bool can_share_buffer = false;
          if (can_share_buffer_ != nullptr) {
            auto maybe_can_share_buffer = can_share_buffer_(
                instruction, instruction->operand(operand_index.operand_number),
                operand_index.operand_index);
            if (maybe_can_share_buffer.has_value()) {
              can_share_buffer = maybe_can_share_buffer.value();
            }
          }

          // Skip copies for aliasing input/output pairs iff:
          // *) Operand can share buffer with 'instruction' output.
          // *) Instruction has frontend attribute which indicates that the
          //    write region of the input/output aliased buffer updated by
          //    'instruction' is disjoint from the read region of the shared
          //    buffer.
          // *) All uses of the operand are 'instruction'.
          if (can_share_buffer &&
              HasDisjointReadWriteRegionsAttr(instruction) &&
              absl::c_all_of(
                  instruction->operand(operand_index.operand_number)->users(),
                  [&instruction](const HloInstruction* user) {
                    return user == instruction;
                  })) {
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
  return absl::OkStatus();
}

absl::Status CopyInsertion::AddSpecialCaseCopies(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  return AddSpecialCaseCopies(*call_graph, execution_threads, module);
}

absl::Status CopyInsertion::AddSpecialCaseCopies(
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
  return absl::OkStatus();
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

absl::Status CopyInsertion::RemoveUnnecessaryCopies(
    HloModule* module, bool check_live_range_ordering,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    bool insert_post_scheduling_control_dependencies) {
  XLA_VLOG_LINES(
      4, module->ToString(HloPrintOptions().set_syntax_sugar_async_ops(false)));

  // Use SequentialHloOrdering if the module has a schedule. The schedule can
  // provide more information on the ordering, allowing for detecting more
  // redundant copies.
  std::unique_ptr<HloOrdering> ordering;
  if (module->has_schedule()) {
    ordering = std::make_unique<SequentialHloOrdering>(module->schedule());
  } else {
    ordering = std::make_unique<DependencyHloOrdering>(module);
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module, can_share_buffer_));
  CopyRemover copy_remover(*module, *alias_analysis, ordering.get(),
                           check_live_range_ordering, execution_threads);
  if (VLOG_IS_ON(3)) {
    LOG(INFO) << "Removing unnecessary copies in " << module->name();
    LOG(INFO) << "Buffer values, in dependency order: ";
    for (const HloBuffer& buffer : alias_analysis->buffers()) {
      LOG(INFO) << "    HloBuffer " << buffer.id();
    }
  }

  int64_t num_existing_copies = GetNumExistingCopies(module, execution_threads);
  bool changed = true;
  int64_t num_iterations = -1;
  VLOG(6) << "Copy Insertion analyzing module with instruction count = "
          << module->instruction_count();
  BoundNonLinearCompilerAnalysis allowance(module, name(), 10);
  while (changed) {
    CHECK_LE(++num_iterations, num_existing_copies);
    changed = false;
    VLOG(2) << "Running fixpoint iteration " << num_iterations
            << " of copy elision";
    for (HloComputation* computation :
         module->computations(execution_threads)) {
      VLOG(2) << "computation:" << computation->name();
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->opcode() != HloOpcode::kCopy) {
          continue;
        }

        // The region_analysis_cost_now is always set to
        // use_region_based_live_range_analysis_ if it is < 0, in which case the
        // analysis is always performed.
        int64_t region_analysis_cost_now =
            (use_region_based_live_range_analysis_ == 0)
                ? 0
                : std::min(allowance.analysis_allowance(),
                           use_region_based_live_range_analysis_);
        if (copy_remover.TryElideCopy(
                instruction, &region_analysis_cost_now,
                insert_post_scheduling_control_dependencies)) {
          changed = true;
          TF_RETURN_IF_ERROR(StripControlDependenciesFrom(instruction));
          TF_RETURN_IF_ERROR(
              instruction->ReplaceAllUsesWith(instruction->mutable_operand(0)));
          VLOG(6) << "succeeded in eliminating copy.";
        }
        if (allowance.ContinueAnalysis() && region_analysis_cost_now > 0) {
          VLOG(6) << "Copy Insertion analyzing module cost: "
                  << region_analysis_cost_now;
          VLOG(6) << "instruction:" << instruction->ToString();
          allowance.DeductCost(region_analysis_cost_now);
          VLOG(6) << "allowance:" << allowance.analysis_allowance();
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> CopyInsertion::Run(
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

  TF_RETURN_IF_ERROR(RemoveUnnecessaryCopies(module,
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
