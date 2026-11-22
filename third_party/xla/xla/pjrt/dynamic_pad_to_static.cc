/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/dynamic_pad_to_static.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/computation_layout.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {
namespace {

bool IsPadRealToStaticOp(mlir::Operation* op) {
  auto attr = op->getAttrOfType<mlir::StringAttr>("call_target_name");
  return attr != nullptr && attr.getValue() == "PadRealToStatic";
}

struct OperandSlot {
  const HloInstruction* instruction;
  int64_t operand_number;
};

bool ShapeIndexOverlaps(const ShapeIndex& a, const ShapeIndex& b) {
  const size_t min_size = std::min(a.size(), b.size());
  for (size_t i = 0; i < min_size; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

bool HasOutputToOperandAlias(const HloInstruction* instruction,
                             int64_t operand_number,
                             const ShapeIndex& operand_index) {
  return absl::c_any_of(
      instruction->output_operand_aliasing(), [&](const auto& alias) {
        return alias.second.first == operand_number &&
               ShapeIndexOverlaps(alias.second.second, operand_index);
      });
}

bool HasMatchingCallSignature(const HloInstruction* instruction,
                              const HloComputation* callee) {
  if (callee == nullptr || callee->root_instruction() == nullptr ||
      instruction->operand_count() != callee->num_parameters() ||
      !ShapeUtil::Compatible(instruction->shape(),
                             callee->root_instruction()->shape())) {
    return false;
  }
  for (int64_t i = 0; i < instruction->operand_count(); ++i) {
    if (!ShapeUtil::Compatible(instruction->operand(i)->shape(),
                               callee->parameter_instruction(i)->shape())) {
      return false;
    }
  }
  return true;
}

// Matches root embedded `kCustomCall` dispatchers of offloaded async wrappers
// that forward operands positionally (where `HloDataflowAnalysis` stops).
bool IsAsyncWrappedSubcomputationCall(const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kCustomCall ||
      instruction->IsCustomCall("PadRealToStatic") ||
      instruction->called_computations().size() != 1) {
    return false;
  }
  const HloComputation* wrapper = instruction->parent();
  if (wrapper == nullptr || wrapper->root_instruction() != instruction ||
      wrapper->execution_thread() == HloInstruction::kMainExecutionThread ||
      !HasMatchingCallSignature(instruction, instruction->to_apply())) {
    return false;
  }
  const absl::InlinedVector<HloInstruction*, 1> all_callers =
      wrapper->caller_instructions();
  const absl::InlinedVector<HloInstruction*, 1> start_callers =
      wrapper->caller_instructions(HloOpcode::kAsyncStart);
  return !start_callers.empty() &&
         absl::c_all_of(all_callers, [&](const HloInstruction* caller) {
           return caller->IsAsynchronous() &&
                  caller->async_wrapped_instruction() == instruction;
         });
}

// Maps each parameter of `async_inst`'s wrapped computation to its bound slot.
std::vector<OperandSlot> GetAsyncBoundOperandSlots(
    const HloInstruction* async_inst) {
  const HloInstruction* start_inst = async_inst->async_chain_start();
  const HloAsyncStartInstruction* start =
      start_inst != nullptr ? DynCast<HloAsyncStartInstruction>(start_inst)
                            : nullptr;
  const HloInstruction* done = async_inst->async_chain_done();
  const HloComputation* callee = async_inst->async_wrapped_computation();
  if (start == nullptr || done == nullptr || callee == nullptr) {
    return {};
  }
  std::vector<OperandSlot> slots;
  slots.reserve(callee->num_parameters());
  bool reached_done = false;
  for (const HloAsyncInstruction* inst : start->GetAsyncChain()) {
    const int64_t first_operand =
        inst->opcode() == HloOpcode::kAsyncStart ? 0 : 1;
    for (int64_t i = first_operand; i < inst->operand_count(); ++i) {
      const HloInstruction* operand = inst->operand(i);
      if (operand == nullptr ||
          slots.size() >= static_cast<size_t>(callee->num_parameters()) ||
          !ShapeUtil::Compatible(
              operand->shape(),
              callee->parameter_instruction(slots.size())->shape())) {
        return {};
      }
      slots.push_back({inst, i});
    }
    if (inst == done) {
      reached_done = true;
      break;
    }
  }
  if (!reached_done ||
      slots.size() != static_cast<size_t>(callee->num_parameters())) {
    return {};
  }
  return slots;
}

const HloInstruction* GetSubcomputationCallerOperand(
    const HloInstruction* caller, const HloComputation* callee,
    int64_t parameter_number) {
  if (parameter_number < 0 || parameter_number >= callee->num_parameters()) {
    return nullptr;
  }
  if (caller->opcode() == HloOpcode::kCall ||
      IsAsyncWrappedSubcomputationCall(caller)) {
    if (caller->to_apply() != callee ||
        !HasMatchingCallSignature(caller, callee)) {
      return nullptr;
    }
    return caller->operand(parameter_number);
  }
  if (caller->opcode() == HloOpcode::kAsyncStart &&
      caller->async_wrapped_computation() == callee) {
    const std::vector<OperandSlot> slots = GetAsyncBoundOperandSlots(caller);
    if (parameter_number < static_cast<int64_t>(slots.size())) {
      const OperandSlot& slot = slots[parameter_number];
      return slot.instruction->operand(slot.operand_number);
    }
  }
  return nullptr;
}

const HloInstruction* GetUnaliasedForwardedParameter(const HloUse& use) {
  const HloInstruction* user = use.instruction;
  if (user->opcode() == HloOpcode::kCall ||
      IsAsyncWrappedSubcomputationCall(user)) {
    const HloComputation* callee = user->to_apply();
    if (!HasMatchingCallSignature(user, callee) || use.operand_number < 0 ||
        use.operand_number >= callee->num_parameters() ||
        HasOutputToOperandAlias(user, use.operand_number, use.operand_index)) {
      return nullptr;
    }
    return callee->parameter_instruction(use.operand_number);
  }
  if (user->opcode() == HloOpcode::kAsyncStart ||
      (user->opcode() == HloOpcode::kAsyncUpdate && use.operand_number > 0)) {
    const std::vector<OperandSlot> slots = GetAsyncBoundOperandSlots(user);
    for (size_t param_no = 0; param_no < slots.size(); ++param_no) {
      if (slots[param_no].instruction == user &&
          slots[param_no].operand_number == use.operand_number) {
        if (HasOutputToOperandAlias(user, use.operand_number,
                                    use.operand_index) ||
            HasOutputToOperandAlias(user->async_chain_start(),
                                    static_cast<int64_t>(param_no),
                                    use.operand_index)) {
          return nullptr;
        }
        return user->async_wrapped_computation()->parameter_instruction(
            param_no);
      }
    }
  }
  return nullptr;
}

// Returns true if `use` is the pad itself, a bitcast of the buffer, or a
// positional call / async boundary whose callee parameter uses are verified.
bool IsAllowedPadRealToStaticUse(
    const HloDataflowAnalysis& dataflow,
    const absl::flat_hash_set<HloValue::Id>& verified_value_ids,
    const HloValue& value, const HloUse& use) {
  const HloInstruction* user = use.instruction;
  if (user->IsCustomCall("PadRealToStatic")) {
    return use.operand_number == 0 && use.operand_index.empty();
  }
  if (user->opcode() == HloOpcode::kBitcast) {
    return true;
  }
  if ((user->opcode() == HloOpcode::kAsyncUpdate ||
       user->opcode() == HloOpcode::kAsyncDone) &&
      use.operand_number == 0) {
    // Dataflow keeps bound async inputs in context slot {0} across
    // `async-update` until `async-done` consumes the context tuple.
    return user->async_chain_start() != nullptr && !use.operand_index.empty() &&
           use.operand_index.front() == 0 &&
           (user->opcode() != HloOpcode::kAsyncUpdate ||
            !HasOutputToOperandAlias(user, 0, use.operand_index));
  }
  const HloInstruction* param = GetUnaliasedForwardedParameter(use);
  if (param == nullptr ||
      !ShapeUtil::IndexIsValid(param->shape(), use.operand_index)) {
    return false;
  }
  const std::vector<const HloValue*>& param_values =
      dataflow.GetValueSet(param, use.operand_index).values();
  if (absl::c_linear_search(param_values, &value)) {
    return param->user_count() > 0 &&
           absl::c_any_of(value.GetUses(), [&](const HloUse& u) {
             return u.instruction->parent() == param->parent();
           });
  }
  return !param_values.empty() &&
         absl::c_all_of(param_values, [&](const HloValue* v) {
           return verified_value_ids.contains(v->id());
         });
}

absl::Status VerifyOnlyPadRealToStaticReadsValue(
    const HloDataflowAnalysis& dataflow,
    const absl::flat_hash_set<HloValue::Id>& verified_value_ids,
    const HloValue& value, const HloValue& entry_parameter_value) {
  const HloInstruction* parameter =
      entry_parameter_value.defining_instruction();
  const ShapeIndex& leaf_index = entry_parameter_value.defining_index();
  auto format_error = [&](absl::string_view detail,
                          absl::string_view via_name) {
    return InvalidArgument(
        "Entry parameter %d%s of module %s feeds PadRealToStatic and is also "
        "%s (through %s). A dynamic parameter consumed by PadRealToStatic is "
        "passed as a compact buffer without dynamic-shape metadata, which no "
        "other instruction can read, so it must have no other users.",
        parameter->parameter_number(),
        leaf_index.empty() ? "" : absl::StrCat(" ", leaf_index.ToString()),
        parameter->GetModule()->name(), detail, via_name);
  };

  for (const HloPosition& pos : value.positions()) {
    if (pos.instruction->IsRoot()) {
      return format_error(absl::StrCat("returned as a result of computation ",
                                       pos.instruction->parent()->name()),
                          pos.instruction->name());
    }
  }
  for (const HloUse& use : value.GetUses()) {
    if (!IsAllowedPadRealToStaticUse(dataflow, verified_value_ids, value,
                                     use)) {
      const HloInstruction* via = use.instruction->operand(use.operand_number);
      return format_error(
          absl::StrCat("used by ", use.instruction->ToString(),
                       " in computation ", use.instruction->parent()->name()),
          via->name());
    }
  }
  return absl::OkStatus();
}

struct VerifiedPadValue {
  const HloValue* value;
  const HloValue* entry_origin;
};

struct PadRealToStaticResolver {
  const HloModule& module;
  const HloDataflowAnalysis& dataflow;
  absl::flat_hash_map<HloValue::Id, std::vector<const HloValue*>> memo = {};
  absl::flat_hash_set<HloValue::Id> active_stack = {};
  absl::flat_hash_set<HloValue::Id> verified_value_ids = {};
  std::vector<VerifiedPadValue> verified_values = {};

  absl::Status ResolvePadOperand(const HloInstruction* pad) {
    const std::vector<const HloValue*>& values =
        dataflow.GetValueSet(pad->operand(0), {}).values();
    if (values.empty()) {
      return InvalidArgument(
          "Dynamic operand of PadRealToStatic (%s) in module %s does not "
          "resolve to an entry computation parameter.",
          pad->ToString(), module.name());
    }
    for (const HloValue* value : values) {
      active_stack.clear();
      ABSL_RETURN_IF_ERROR(ResolveEntryOrigins(pad, value).status());
    }
    return absl::OkStatus();
  }

  absl::StatusOr<std::vector<const HloValue*>> ResolveEntryOrigins(
      const HloInstruction* pad, const HloValue* value) {
    if (auto it = memo.find(value->id()); it != memo.end()) {
      return it->second;
    }

    const HloComputation* entry_comp = module.entry_computation();
    const HloInstruction* def = value->defining_instruction();
    auto non_entry_error = [&]() {
      return InvalidArgument(
          "Dynamic operand of PadRealToStatic (%s) in module %s must come from "
          "an entry computation parameter without being copied or recomputed, "
          "but its value is defined by %s in computation %s.",
          pad->ToString(), module.name(), def->ToString(),
          def->parent()->name());
    };

    if (def->opcode() != HloOpcode::kParameter) {
      return non_entry_error();
    }
    if (def->parent() == entry_comp) {
      const Shape& leaf_shape =
          ShapeUtil::GetSubshape(def->shape(), value->defining_index());
      if (leaf_shape.IsArray() && leaf_shape.is_dynamic() &&
          verified_value_ids.insert(value->id()).second) {
        verified_values.push_back({value, value});
      }
      std::vector<const HloValue*> result = {value};
      memo.emplace(value->id(), result);
      return result;
    }

    const HloComputation* sub_comp = def->parent();
    absl::InlinedVector<HloInstruction*, 1> callers =
        sub_comp->caller_instructions();
    if (callers.empty() || !active_stack.insert(value->id()).second) {
      return non_entry_error();
    }
    absl::c_sort(callers, [](const HloInstruction* a, const HloInstruction* b) {
      return a->unique_id() < b->unique_id();
    });

    std::vector<const HloValue*> resolved_entry_values;
    absl::flat_hash_set<HloValue::Id> seen_local_entry_values;
    for (const HloInstruction* caller : callers) {
      // `kAsyncStart` already covers the bound operands of its updates and
      // done.
      if ((caller->opcode() == HloOpcode::kAsyncUpdate ||
           caller->opcode() == HloOpcode::kAsyncDone) &&
          caller->async_chain_start() != nullptr &&
          caller->async_wrapped_computation() == sub_comp) {
        continue;
      }
      const HloInstruction* arg = GetSubcomputationCallerOperand(
          caller, sub_comp, def->parameter_number());
      if (arg == nullptr ||
          !ShapeUtil::IndexIsValid(arg->shape(), value->defining_index())) {
        active_stack.erase(value->id());
        return non_entry_error();
      }
      const std::vector<const HloValue*>& caller_values =
          dataflow.GetValueSet(arg, value->defining_index()).values();
      if (caller_values.empty()) {
        active_stack.erase(value->id());
        return non_entry_error();
      }
      for (const HloValue* caller_value : caller_values) {
        absl::StatusOr<std::vector<const HloValue*>> entry_vals =
            ResolveEntryOrigins(pad, caller_value);
        if (!entry_vals.ok()) {
          active_stack.erase(value->id());
          return entry_vals.status();
        }
        for (const HloValue* entry_val : *entry_vals) {
          if (seen_local_entry_values.insert(entry_val->id()).second) {
            resolved_entry_values.push_back(entry_val);
          }
        }
      }
    }
    active_stack.erase(value->id());

    if (resolved_entry_values.empty()) {
      return non_entry_error();
    }
    for (const HloValue* entry_val : resolved_entry_values) {
      const Shape& leaf_shape =
          ShapeUtil::GetSubshape(entry_val->defining_instruction()->shape(),
                                 entry_val->defining_index());
      if (leaf_shape.IsArray() && leaf_shape.is_dynamic()) {
        if (verified_value_ids.insert(value->id()).second) {
          verified_values.push_back({value, entry_val});
        }
        break;
      }
    }
    memo.emplace(value->id(), resolved_entry_values);
    return resolved_entry_values;
  }
};

}  // namespace

absl::Status ClearPadRealToStaticParameterMetadataPrefixes(
    const HloModule& hlo_module, absl::Span<Shape> parameter_shapes) {
  // Scan every computation (offloaded pads live outside the entry computation)
  // before running dataflow so modules without dynamic pads pay no cost.
  std::vector<const HloInstruction*> pad_calls;
  for (const HloComputation* comp : hlo_module.computations()) {
    for (const HloInstruction* instruction : comp->instructions()) {
      if (instruction->IsCustomCall("PadRealToStatic") &&
          instruction->operand_count() > 0 &&
          instruction->operand(0)->shape().IsArray() &&
          instruction->operand(0)->shape().is_dynamic()) {
        pad_calls.push_back(instruction);
      }
    }
  }
  if (pad_calls.empty()) {
    return absl::OkStatus();
  }

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloDataflowAnalysis> dataflow,
                   HloDataflowAnalysis::Run(hlo_module));

  const HloComputation* entry_comp = hlo_module.entry_computation();
  const ComputationLayout& layout = hlo_module.entry_computation_layout();
  const bool flattened_tuple_parameter =
      layout.parameter_count() == 1 && layout.parameter_shape(0).IsTuple();

  // Every pad operand must trace to a zero-prefix entry parameter (directly via
  // dataflow or across embedded offload dispatchers); other defs carry +1024B.
  PadRealToStaticResolver resolver{hlo_module, *dataflow};
  for (const HloInstruction* pad : pad_calls) {
    ABSL_RETURN_IF_ERROR(resolver.ResolvePadOperand(pad));
  }

  std::vector<const HloValue*> parameter_values;
  parameter_values.reserve(resolver.verified_values.size());
  for (const VerifiedPadValue& item : resolver.verified_values) {
    const HloInstruction* entry_def = item.entry_origin->defining_instruction();
    CHECK_EQ(entry_def->opcode(), HloOpcode::kParameter);
    CHECK_EQ(entry_def->parent(), entry_comp);
    ABSL_RETURN_IF_ERROR(VerifyOnlyPadRealToStaticReadsValue(
        *dataflow, resolver.verified_value_ids, *item.value,
        *item.entry_origin));
    if (item.value->defining_instruction()->parent() != entry_comp) {
      continue;
    }
    if (std::optional<ShapeIndex> aliased_output =
            hlo_module.input_output_alias_config().GetAliasedOutput(
                entry_def->parameter_number(), item.value->defining_index());
        aliased_output.has_value()) {
      return InvalidArgument(
          "Entry parameter %d%s of module %s feeds PadRealToStatic and is "
          "aliased to output %s. A dynamic parameter consumed by "
          "PadRealToStatic is passed as a compact runtime-sized buffer without "
          "dynamic-shape metadata, so it cannot back an aliased or donated "
          "output.",
          entry_def->parameter_number(),
          item.value->defining_index().empty()
              ? ""
              : absl::StrCat(" ", item.value->defining_index().ToString()),
          hlo_module.name(), aliased_output->ToString());
    }
    parameter_values.push_back(item.value);
  }

  // Mutate `parameter_shapes` only after all validations succeed so an error
  // leaves `parameter_shapes` untouched.
  for (const HloValue* value : parameter_values) {
    int64_t shape_no = value->defining_instruction()->parameter_number();
    ShapeIndex index = value->defining_index();
    if (flattened_tuple_parameter) {
      if (shape_no != 0 || index.empty()) {
        continue;
      }
      shape_no = index.front();
      index = ShapeIndex(index.begin() + 1, index.end());
    }
    if (shape_no < 0 || shape_no >= parameter_shapes.size() ||
        !ShapeUtil::IndexIsValid(parameter_shapes[shape_no], index)) {
      continue;
    }
    Shape* subshape =
        ShapeUtil::GetMutableSubshape(&parameter_shapes[shape_no], index);
    if (subshape->IsArray() && subshape->has_layout()) {
      subshape->mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(0);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> ShouldClearPadRealToStaticMetadataPrefix(
    mlir::func::FuncOp main, size_t arg_index) {
  if (arg_index >= main.getNumArguments()) {
    return InvalidArgument("Argument index %d out of range for %d arguments.",
                           arg_index, main.getNumArguments());
  }
  mlir::BlockArgument arg = main.getArgument(arg_index);
  // Operand 0 is the data tensor; the other operands are the sizes.
  const bool is_pad_real_to_static_input =
      llvm::any_of(arg.getUses(), [](mlir::OpOperand& use) {
        return use.getOperandNumber() == 0 &&
               IsPadRealToStaticOp(use.getOwner());
      });
  if (!is_pad_real_to_static_input) {
    return false;
  }
  if (!arg.hasOneUse()) {
    return InvalidArgument(
        "Entry argument %d feeds PadRealToStatic and has other users; "
        "PadRealToStatic dynamic inputs have no metadata prefix and "
        "must have no other users.",
        arg_index);
  }
  if (main.getArgAttr(arg_index, "tf.aliasing_output") != nullptr) {
    return InvalidArgument(
        "Entry argument %d feeds PadRealToStatic and is aliased via "
        "tf.aliasing_output; compact PadRealToStatic buffers cannot back "
        "aliased or donated outputs.",
        arg_index);
  }
  return true;
}

}  // namespace xla
