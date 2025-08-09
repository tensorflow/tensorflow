// Copyright 2025 The OpenXLA Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/hlo/tools/hlo_diff/graph/analysis/hlo_value_tracing.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"

namespace xla {
namespace {
// CalculatePostOrderSchedule traverses a module and assign a ordinal to each
// instruction based the postorder dependency.
int64_t CalculatePostOrderScheduleHelper(
    const HloComputation* comp, int64_t start_ordinal,
    absl::flat_hash_map<HloInstruction*, int64_t>* ordinal_map) {
  int64_t ordinal = start_ordinal;
  for (HloInstruction* instruction : comp->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kCall ||
        instruction->opcode() == HloOpcode::kAsyncStart ||
        instruction->opcode() == HloOpcode::kConditional) {
      for (const HloComputation* called_computation :
           instruction->called_computations()) {
        ordinal = CalculatePostOrderScheduleHelper(called_computation, ordinal,
                                                   ordinal_map);
      }
    }
    if (instruction->opcode() == HloOpcode::kWhile) {
      ordinal = CalculatePostOrderScheduleHelper(instruction->while_condition(),
                                                 ordinal, ordinal_map);
      ordinal = CalculatePostOrderScheduleHelper(instruction->while_body(),
                                                 ordinal, ordinal_map);
    }
    // It's possible that in some unit tests the computation graph is not
    // flatten (meaning we could have multiple callers for one computation). In
    // that case the oridinal_map will see the instruction multiple times. We
    // consider that case to be ok as it only shows up in unit tests.
    ordinal_map->insert({instruction, ordinal++});
  }
  return ordinal;
}

absl::flat_hash_map<HloInstruction*, int64_t> CalculatePostOrderSchedule(
    const HloModule& module) {
  absl::flat_hash_map<HloInstruction*, int64_t> map;
  CalculatePostOrderScheduleHelper(module.entry_computation(), 0, &map);
  return map;
}

}  // namespace

bool HloValueTracing::ValueIsDefinedAt(const HloInstruction* instruction,
                                       const ShapeIndex& index) const {
  const HloValueSet& value_set = GetValueSet(instruction, index);
  if (value_set.values().size() != 1) {
    return false;
  }
  return value_set.GetUniqueValue().defining_instruction() == instruction;
}

HloValueTracing::HloValueTracing(
    const HloModule& module,
    absl::flat_hash_set<absl::string_view> execution_threads)
    : module_(module),
      execution_threads_(std::move(execution_threads)),
      call_graph_(CallGraph::Build(&module)) {}

HloValue* HloValueTracing::NewHloValue(HloInstruction* instruction,
                                       const ShapeIndex& index, bool is_phi) {
  const int64_t value_id = next_value_id_++;
  auto result =
      values_.insert({value_id, std::make_unique<HloValue>(
                                    value_id, instruction, index, is_phi)});
  CHECK(result.second);


  return result.first->second.get();
}

void HloValueTracing::DeleteMarkedValues() {
  // Use a set to prevent deleting an id twice.
  absl::flat_hash_set<HloValue::Id> id_set(value_ids_to_delete_.begin(),
                                           value_ids_to_delete_.end());

  for (HloValue::Id value_id : id_set) {
    values_.erase(value_id);
  }
  value_ids_to_delete_.clear();
}

HloValue& HloValueTracing::GetValue(HloValue::Id value_id) {
  DCHECK(values_.contains(value_id)) << "Value not found: " << value_id;
  return *values_.find(value_id)->second;
}

HloValueSet HloValueTracing::GetFlattenedValueSet(
    const HloInstruction* instruction) const {
  HloValueSet value_set;

  const InstructionValueSet& value_set_tree =
      GetInstructionValueSet(instruction);

  std::vector<const HloValueSet*> all_sets;
  for (auto& pair : value_set_tree) {
    all_sets.push_back(&pair.second);
  }
  value_set.AssignUnionOf(all_sets);

  return value_set;
}

const HloValueSet& HloValueTracing::GetValueSet(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  return GetInstructionValueSet(instruction).element(index);
}

HloValueSet& HloValueTracing::GetValueSet(const HloInstruction* instruction,
                                          const ShapeIndex& index) {
  return *GetInstructionValueSet(instruction).mutable_element(index);
}

bool HloValueTracing::UpdateSendValueSet(HloInstruction* send) {
  CHECK_EQ(send->opcode(), HloOpcode::kSend);
  bool changed = false;
  // Send forwards the operand value to the output tuple at {0}.
  for (auto& pair : GetInstructionValueSet(send->operand(0))) {
    const ShapeIndex& operand_index = pair.first;
    const HloValueSet& operand_value_set = pair.second;

    ShapeIndex index = {0};
    for (int64_t i : operand_index) {
      index.push_back(i);
    }

    HloValueSet& value_set = GetValueSet(send, index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloValueTracing::UpdateAsyncStartValueSet(HloInstruction* async_start) {
  CHECK_EQ(async_start->opcode(), HloOpcode::kAsyncStart);
  bool changed = false;
  // AsyncStart forwards the operand values to element {0} of its output.
  for (int64_t i = 0; i < async_start->operand_count(); ++i) {
    const HloInstruction* operand = async_start->operand(i);
    ShapeUtil::ForEachSubshape(
        operand->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (!subshape.IsArray()) {
            return;
          }
          const HloValueSet& operand_value_set = GetValueSet(operand, index);

          ShapeIndex output_index = {0, i};
          output_index.insert(output_index.end(), index.begin(), index.end());

          HloValueSet& value_set = GetValueSet(async_start, output_index);
          if (value_set != operand_value_set) {
            value_set = operand_value_set;
            changed = true;
          }
        });
  }
  if (!HloInstruction::IsThreadIncluded(async_start->async_execution_thread(),
                                        execution_threads_)) {
    return changed;
  }
  // AsyncStart forwards the async wrapped computation root values to element
  // {1} of its output.
  HloInstruction* root =
      async_start->async_wrapped_computation()->root_instruction();
  ShapeUtil::ForEachSubshape(
      root->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return;
        }
        const HloValueSet& root_value_set = GetValueSet(root, index);

        ShapeIndex output_index = {1};
        output_index.insert(output_index.end(), index.begin(), index.end());

        HloValueSet& value_set = GetValueSet(async_start, output_index);
        if (value_set != root_value_set) {
          value_set = root_value_set;
          changed = true;
        }
      });
  return changed;
}

bool HloValueTracing::UpdateAsyncUpdateValueSet(HloInstruction* async_update) {
  CHECK_EQ(async_update->opcode(), HloOpcode::kAsyncUpdate);
  CHECK_EQ(async_update->shape(), async_update->operand(0)->shape());
  bool changed = false;
  HloInstruction* root =
      HloInstruction::IsThreadIncluded(async_update->async_execution_thread(),
                                       execution_threads_)
          ? async_update->async_wrapped_computation()->root_instruction()
          : nullptr;
  // AsyncUpdate forwards all of the operand values to corresponding elements of
  // its output.
  ShapeUtil::ForEachSubshape(
      async_update->operand(0)->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return;
        }
        const HloValueSet& operand_value_set =
            GetValueSet(async_update->operand(0), index);

        HloValueSet& value_set = GetValueSet(async_update, index);
        CHECK_GE(index.size(), 0);
        if (index[0] != 1) {
          if (value_set != operand_value_set) {
            value_set = operand_value_set;
            changed = true;
          }
        } else if (root != nullptr) {
          // If this subshape is an output (index {1}), we need to create the
          // union with the async wrapped computation root.
          ShapeIndex root_index(index.begin() + 1, index.end());
          const HloValueSet& root_value_set = GetValueSet(root, root_index);
          changed |=
              value_set.AssignUnionOf({&operand_value_set, &root_value_set});
        } else if (value_set != operand_value_set) {
          value_set = operand_value_set;
          changed = true;
        }
      });
  return changed;
}

bool HloValueTracing::UpdateAsyncDoneValueSet(HloInstruction* async_done) {
  CHECK_EQ(async_done->opcode(), HloOpcode::kAsyncDone);
  bool changed = false;
  HloInstruction* root =
      HloInstruction::IsThreadIncluded(async_done->async_execution_thread(),
                                       execution_threads_)
          ? async_done->async_wrapped_computation()->root_instruction()
          : nullptr;
  // AsyncDone creates a union of the operand values at {1} and the async
  // wrapped computation root to element {} of its output.
  ShapeUtil::ForEachSubshape(
      async_done->operand(0)->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray() || index.front() != 1) {
          return;
        }
        const HloValueSet& operand_value_set =
            GetValueSet(async_done->operand(0), index);

        ShapeIndex output_index(index.begin() + 1, index.end());
        HloValueSet& value_set = GetValueSet(async_done, output_index);
        if (root != nullptr) {
          const HloValueSet& root_value_set = GetValueSet(root, output_index);
          changed |=
              value_set.AssignUnionOf({&operand_value_set, &root_value_set});
        } else if (value_set != operand_value_set) {
          value_set = operand_value_set;
          changed = true;
        }
      });
  return changed;
}

bool HloValueTracing::UpdateCopyStartValueSet(HloInstruction* copy_start) {
  CHECK_EQ(copy_start->opcode(), HloOpcode::kCopyStart);
  bool changed = false;
  // CopyStart forwards the operand value to elements {0, 1} of its output.
  const HloValueSet& operand_value_set = GetValueSet(copy_start->operand(0));
  HloValueSet& first_value_set = GetValueSet(copy_start, {0});
  if (first_value_set != operand_value_set) {
    first_value_set = operand_value_set;
    changed = true;
  }

  HloValueSet& second_value_set = GetValueSet(copy_start, {1});
  if (second_value_set != operand_value_set) {
    second_value_set = operand_value_set;
    changed = true;
  }

  return changed;
}

bool HloValueTracing::UpdateCopyDoneValueSet(HloInstruction* copy_done) {
  CHECK_EQ(copy_done->opcode(), HloOpcode::kCopyDone);
  bool changed = false;
  // CopyDone forwards the operand value at {0} to element {} of its output.
  const HloValueSet& operand_value_set =
      GetValueSet(copy_done->operand(0), {0});
  HloValueSet& value_set = GetValueSet(copy_done);
  if (value_set != operand_value_set) {
    value_set = operand_value_set;
    changed = true;
  }
  return changed;
}

bool HloValueTracing::UpdateRecvDoneValueSet(HloInstruction* recv_done) {
  CHECK_EQ(recv_done->opcode(), HloOpcode::kRecvDone);
  bool changed = false;
  // RecvDone forwards the operand value at {0} to element {0} of its output.
  for (auto& pair : GetInstructionValueSet(recv_done)) {
    ShapeIndex& index = pair.first;
    HloValueSet& value_set = pair.second;

    if (index.empty() || index[0] != 0) {
      continue;
    }

    const HloValueSet& operand_value_set =
        GetValueSet(recv_done->operand(0), index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloValueTracing::UpdateCallValueSet(HloInstruction* call) {
  CHECK_EQ(call->opcode(), HloOpcode::kCall);
  InstructionValueSet& value_set = GetInstructionValueSet(call);
  InstructionValueSet& root_value_set =
      GetInstructionValueSet(call->to_apply()->root_instruction());
  if (value_set != root_value_set) {
    value_set = root_value_set;
    return true;
  }
  return false;
}

bool HloValueTracing::UpdateConditionalValueSet(HloInstruction* conditional) {
  CHECK_EQ(conditional->opcode(), HloOpcode::kConditional);
  std::vector<const InstructionValueSet*> inputs(conditional->branch_count());
  for (int j = 0; j < conditional->branch_count(); ++j) {
    inputs[j] = &GetInstructionValueSet(
        conditional->branch_computation(j)->root_instruction());
  }
  return GetInstructionValueSet(conditional).AssignUnionOf(inputs);
}

bool HloValueTracing::UpdateCopyValueSet(HloInstruction* copy) {
  CHECK_EQ(copy->opcode(), HloOpcode::kCopy);
  bool changed = false;
  for (auto& pair : GetInstructionValueSet(copy)) {
    const ShapeIndex& index = pair.first;

    HloValueSet& value_set = pair.second;
    HloValueSet& operand_value_set = GetValueSet(copy->operand(0), index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloValueTracing::UpdateOptimizationBarrierValueSet(
    HloInstruction* barrier) {
  // Optimization Barriers just forward their operand. Given that barriers can
  // have a tuple operand, we iterate through its indexes, like for copies.
  // Unlike copies though we also propagate the top-level value.
  CHECK_EQ(barrier->opcode(), HloOpcode::kOptimizationBarrier);
  bool changed = false;
  for (auto& pair : GetInstructionValueSet(barrier)) {
    const ShapeIndex& index = pair.first;
    HloValueSet& value_set = pair.second;
    HloValueSet& operand_value_set = GetValueSet(barrier->operand(0), index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloValueTracing::UpdateDomainValueSet(HloInstruction* domain) {
  // Domain instructions just forward their operand. Given that domains can have
  // a tuple operand, we iterate through its indexes, like for copies.
  // Unlike copies though we also propagate the top-level value.
  CHECK_EQ(domain->opcode(), HloOpcode::kDomain);
  bool changed = false;
  for (auto& pair : GetInstructionValueSet(domain)) {
    const ShapeIndex& index = pair.first;
    HloValueSet& value_set = pair.second;
    HloValueSet& operand_value_set = GetValueSet(domain->operand(0), index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloValueTracing::UpdateAddDependencyValueSet(
    HloInstruction* add_dependency) {
  // AddDependency just forwards the value of its zero-th operand.
  CHECK_EQ(add_dependency->opcode(), HloOpcode::kAddDependency);
  const InstructionValueSet& operand_set =
      GetInstructionValueSet(add_dependency->operand(0));
  InstructionValueSet& add_dependency_set =
      GetInstructionValueSet(add_dependency);
  if (operand_set != add_dependency_set) {
    add_dependency_set = operand_set;
    return true;
  }
  return false;
}

bool HloValueTracing::UpdateGetTupleElementValueSet(HloInstruction* gte) {
  CHECK_EQ(gte->opcode(), HloOpcode::kGetTupleElement);
  bool changed = false;
  // The GetTupleElement instruction forwards the values from the specified
  // tuple element.
  for (auto& pair : GetInstructionValueSet(gte)) {
    const ShapeIndex& index = pair.first;
    HloValueSet& value_set = pair.second;

    // The corresponding ShapeIndex of the operand is simply the GTE ShapeIndex
    // with the tuple element number prefixed.
    ShapeIndex operand_index = {gte->tuple_index()};
    for (int64_t i : index) {
      operand_index.push_back(i);
    }

    HloValueSet& operand_value_set =
        GetValueSet(gte->operand(0), operand_index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloValueTracing::UpdateParameterValueSet(HloInstruction* parameter) {
  CHECK_EQ(parameter->opcode(), HloOpcode::kParameter);
  const CallGraphNode& call_graph_node =
      call_graph_->GetNode(parameter->parent());

  // Subcomputations called in a parallel context (eg, map) do not have dataflow
  // from the caller operands.
  if (call_graph_node.caller_callsites().empty()) {
    return false;
  }

  std::vector<const InstructionValueSet*> inputs;
  for (const CallSite& callsite : call_graph_node.caller_callsites()) {
    const HloOpcode opcode = callsite.instruction()->opcode();
    if (opcode == HloOpcode::kCall || opcode == HloOpcode::kFusion) {
      // The operand values of a call instruction are forwarded to the
      // respective parameter instruction of the subcomputation.
      inputs.push_back(&GetInstructionValueSet(
          callsite.instruction()->operand(parameter->parameter_number())));
    } else if (opcode == HloOpcode::kWhile) {
      // In a while instruction, the while operand (ie, the init value) and the
      // backedge are dataflow inputs to the parameter instruction. This is the
      // case for parameters of both the body and condition computations.
      CHECK_EQ(parameter->parameter_number(), 0);
      inputs.push_back(
          &GetInstructionValueSet(callsite.instruction()->operand(0)));
      // If the parameter *is not* the root, parameter state would be
      // updated by the root, otherwise don't consider it's current state
      // (InstructionValueSet) as we are recomputing its current state.
      if (parameter !=
          callsite.instruction()->while_body()->root_instruction()) {
        inputs.push_back(&GetInstructionValueSet(
            callsite.instruction()->while_body()->root_instruction()));
      }
    } else if (opcode == HloOpcode::kConditional) {
      CHECK_EQ(parameter->parameter_number(), 0);
      auto* conditional = callsite.instruction();
      // Conditional has branch_count+1 operands. Operand 0 is the branch_index,
      // operands 1 and onward are the arguments to the branch computations.
      //
      // If the parameter belongs to conditional's branch 0 computation, then
      // operand 1 is forwarded to this parameter instruction. If the parameter
      // belongs to conditional's branch 5 computation, then operand 6 is
      // forwarded to this parameter instruction.
      bool found_parent = false;
      for (int j = 0; j < conditional->branch_count(); ++j) {
        if (parameter->parent() == conditional->branch_computation(j)) {
          inputs.push_back(
              &GetInstructionValueSet(conditional->operand(j + 1)));
          found_parent = true;
          break;
        }
      }
      CHECK(found_parent);
    } else if (opcode == HloOpcode::kAsyncStart) {
      inputs.push_back(&GetInstructionValueSet(
          callsite.instruction()->operand(parameter->parameter_number())));
    } else if (opcode == HloOpcode::kAsyncUpdate ||
               opcode == HloOpcode::kAsyncDone) {
      return GetInstructionValueSet(parameter).AssignUnionOf(
          GetInstructionValueSet(callsite.instruction()->operand(0)),
          {0, parameter->parameter_number()});
    } else {
      return false;
    }
  }

  return GetInstructionValueSet(parameter).AssignUnionOf(inputs);
}

bool HloValueTracing::UpdateTupleValueSet(HloInstruction* tuple) {
  CHECK_EQ(tuple->opcode(), HloOpcode::kTuple);
  bool changed = false;
  for (int64_t i = 0; i < tuple->operands().size(); ++i) {
    // Copy the value set(s) of each operand into the respective position in the
    // kTuple instruction's value sets.
    for (auto& pair : GetInstructionValueSet(tuple->operand(i))) {
      const ShapeIndex& operand_index = pair.first;
      HloValueSet& operand_value_set = pair.second;

      ShapeIndex index = {i};
      for (int64_t op_index : operand_index) {
        index.push_back(op_index);
      }
      HloValueSet& value_set = GetValueSet(tuple, index);

      if (value_set != operand_value_set) {
        value_set = operand_value_set;
        changed = true;
      }
    }
  }
  return changed;
}

bool HloValueTracing::UpdateWhileValueSet(HloInstruction* xla_while) {
  CHECK_EQ(xla_while->opcode(), HloOpcode::kWhile);
  const InstructionValueSet* const inputs[] = {
      &GetInstructionValueSet(xla_while->while_body()->root_instruction()),
      &GetInstructionValueSet(xla_while->operand(0))};
  return GetInstructionValueSet(xla_while).AssignUnionOf(inputs);
}

bool HloValueTracing::UpdateFusionValueSet(HloInstruction* fusion) {
  CHECK_EQ(fusion->opcode(), HloOpcode::kFusion);

  InstructionValueSet& value_set = GetInstructionValueSet(fusion);
  InstructionValueSet& root_value_set = GetInstructionValueSet(
      fusion->called_computations().front()->root_instruction());
  if (value_set != root_value_set) {
    value_set = root_value_set;
    return true;
  }

  return false;
}

bool HloValueTracing::UpdateAllGatherStartValueSet(
    HloInstruction* all_gather_start) {
  CHECK_EQ(all_gather_start->opcode(), HloOpcode::kAllGatherStart);
  bool changed = false;
  // AllGatherStart forwards the operand values to element {0} of its output.
  for (int64_t i = 0; i < all_gather_start->operand_count(); ++i) {
    const HloValueSet& operand_value_set =
        GetValueSet(all_gather_start->operand(i));

    ShapeIndex output_index = {0};
    if (all_gather_start->operand_count() > 1) {
      output_index.push_back(i);
    }

    HloValueSet& value_set = GetValueSet(all_gather_start, output_index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloValueTracing::UpdateAllGatherDoneValueSet(
    HloInstruction* all_gather_done) {
  CHECK_EQ(all_gather_done->opcode(), HloOpcode::kAllGatherDone);
  bool changed = false;
  // AllGatherDone forwards the operand value at {1} to its output. If the
  // output is a tuple, then that tuple is defined by all-gather-done, so
  // only update the value set for tuple leaf elements (arrays).
  for (auto& pair : GetInstructionValueSet(all_gather_done)) {
    const ShapeIndex& output_index = pair.first;
    HloValueSet& value_set = pair.second;

    if (!ShapeUtil::GetSubshape(all_gather_done->shape(), output_index)
             .IsArray()) {
      continue;
    }
    ShapeIndex operand_index = {1};
    for (int64_t i : output_index) {
      operand_index.push_back(i);
    }

    const HloValueSet& operand_value_set =
        GetValueSet(all_gather_done->operand(0), operand_index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloValueTracing::UpdateAllReduceDoneValueSet(
    HloInstruction* all_reduce_done) {
  CHECK_EQ(all_reduce_done->opcode(), HloOpcode::kAllReduceDone);
  bool changed = false;
  // AllReduceDone forwards its only operand.
  for (auto& pair : GetInstructionValueSet(all_reduce_done)) {
    const ShapeIndex& output_index = pair.first;
    HloValueSet& value_set = pair.second;

    ShapeIndex operand_index = {};
    for (int64_t i : output_index) {
      operand_index.push_back(i);
    }

    const HloValueSet& operand_value_set =
        GetValueSet(all_reduce_done->operand(0), operand_index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloValueTracing::UpdateCollectivePermuteStartValueSet(
    HloInstruction* collective_permute_start) {
  CHECK_EQ(collective_permute_start->opcode(),
           HloOpcode::kCollectivePermuteStart);
  bool changed = false;
  // CollectivePermuteStart forwards the operand value to element {0} of its
  // output.
  if (collective_permute_start->operand(0)->shape().IsTuple()) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(
                            collective_permute_start->operand(0)->shape());
         ++i) {
      const HloValueSet& operand_value_set =
          GetValueSet(collective_permute_start->operand(0), {i});
      HloValueSet& value_set = GetValueSet(collective_permute_start, {0, i});
      if (value_set != operand_value_set) {
        value_set = operand_value_set;
        changed = true;
      }
    }
  } else {
    const HloValueSet& operand_value_set =
        GetValueSet(collective_permute_start->operand(0));
    HloValueSet& value_set = GetValueSet(collective_permute_start, {0});
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloValueTracing::UpdateCollectivePermuteDoneValueSet(
    HloInstruction* collective_permute_done) {
  CHECK_EQ(collective_permute_done->opcode(),
           HloOpcode::kCollectivePermuteDone);
  bool changed = false;
  // CollectivePermuteDone forwards the operand value at {1} to its output.
  if (collective_permute_done->shape().IsTuple()) {
    for (int i = 0;
         i < ShapeUtil::TupleElementCount(collective_permute_done->shape());
         ++i) {
      const HloValueSet& operand_value_set =
          GetValueSet(collective_permute_done->operand(0), {1, i});
      HloValueSet& value_set = GetValueSet(collective_permute_done, {i});
      if (value_set != operand_value_set) {
        value_set = operand_value_set;
        changed = true;
      }
    }
  } else {
    const HloValueSet& operand_value_set =
        GetValueSet(collective_permute_done->operand(0), {1});
    HloValueSet& value_set = GetValueSet(collective_permute_done);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}

bool HloValueTracing::UpdateInstructionValueSet(HloInstruction* instruction) {
  // Recompute from operands.
  bool changed = false;
  switch (instruction->opcode()) {
    case HloOpcode::kAddDependency: {
      changed = UpdateAddDependencyValueSet(instruction);
      break;
    }
    case HloOpcode::kAllGatherStart: {
      changed = UpdateAllGatherStartValueSet(instruction);
      break;
    }
    case HloOpcode::kAllGatherDone: {
      changed = UpdateAllGatherDoneValueSet(instruction);
      break;
    }
    case HloOpcode::kAsyncStart: {
      changed = UpdateAsyncStartValueSet(instruction);
      break;
    }
    case HloOpcode::kAsyncUpdate: {
      changed = UpdateAsyncUpdateValueSet(instruction);
      break;
    }
    case HloOpcode::kAsyncDone: {
      changed = UpdateAsyncDoneValueSet(instruction);
      break;
    }
    case HloOpcode::kDomain: {
      changed = UpdateDomainValueSet(instruction);
      break;
    }
    case HloOpcode::kCopy: {
      changed = UpdateCopyValueSet(instruction);
      break;
    }
    case HloOpcode::kGetTupleElement: {
      changed = UpdateGetTupleElementValueSet(instruction);
      break;
    }
    case HloOpcode::kTuple: {
      changed = UpdateTupleValueSet(instruction);
      break;
    }
    case HloOpcode::kParameter: {
      changed = UpdateParameterValueSet(instruction);
      break;
    }
    case HloOpcode::kCall: {
      changed = UpdateCallValueSet(instruction);
      break;
    }
    case HloOpcode::kWhile: {
      changed = UpdateWhileValueSet(instruction);
      break;
    }
    case HloOpcode::kSend: {
      changed = UpdateSendValueSet(instruction);
      break;
    }
    case HloOpcode::kRecvDone: {
      changed = UpdateRecvDoneValueSet(instruction);
      break;
    }
    case HloOpcode::kCopyStart: {
      changed = UpdateCopyStartValueSet(instruction);
      break;
    }
    case HloOpcode::kCopyDone: {
      changed = UpdateCopyDoneValueSet(instruction);
      break;
    }
    case HloOpcode::kConditional: {
      changed = UpdateConditionalValueSet(instruction);
      break;
    }
    case HloOpcode::kAllReduceDone: {
      changed = UpdateAllReduceDoneValueSet(instruction);
      break;
    }
    case HloOpcode::kCollectivePermuteStart: {
      changed = UpdateCollectivePermuteStartValueSet(instruction);
      break;
    }
    case HloOpcode::kCollectivePermuteDone: {
      changed = UpdateCollectivePermuteDoneValueSet(instruction);
      break;
    }
    case HloOpcode::kOptimizationBarrier: {
      changed = UpdateOptimizationBarrierValueSet(instruction);
      break;
    }
    case HloOpcode::kFusion: {
      changed = UpdateFusionValueSet(instruction);
      break;
    }
    default:
      break;
  }

  return changed;
}

void HloValueTracing::Propagate() {
  using Work = std::pair<int64_t, HloInstruction*>;
  // Avoid duplicating work by preferring work items early in the post order
  // schedule. Intuitively, we start from entry parameters and propagate buffers
  // updates throughout the module only once.
  std::priority_queue<Work, std::vector<Work>, std::greater<Work>> worklist;
  absl::flat_hash_set<HloInstruction*> workset;
  auto priority_map = CalculatePostOrderSchedule(module_);
  auto add_to_worklist = [&priority_map, &worklist,
                          &workset](HloInstruction* instruction) {
    if (workset.insert(instruction).second) {
      worklist.emplace(priority_map[instruction], instruction);
    }
  };

  auto comps = module_.MakeComputationPostOrder();
  for (HloComputation* computation : comps) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads_)) {
      continue;
    }
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      add_to_worklist(instruction);
    }
  }

  while (!worklist.empty()) {
    HloInstruction* instruction = worklist.top().second;
    worklist.pop();

    workset.erase(workset.find(instruction));

    if (!UpdateInstructionValueSet(instruction)) {
      // No change to the instruction's value set.
      continue;
    }

    // Instruction value was updated. Add users to work list if we haven't
    // already.
    for (HloInstruction* user : instruction->users()) {
      add_to_worklist(user);

      // If user sequentially calls a computation, then the respective
      // parameter(s) of the computation need to be updated.
      if (user->opcode() == HloOpcode::kConditional) {
        // If operand 0 is the use of instruction, then no parameters need to be
        // updated, since that is the branch_index of the conditional.
        // If operand n+1 is the use of instruction, then the branch_computation
        // n's parameter need to be updated.
        //
        // Note that the same instruction can be used in multiple branches'
        // operands.
        for (int j = 0; j < user->branch_count(); ++j) {
          if (user->operand(j + 1) == instruction) {
            add_to_worklist(
                user->branch_computation(j)->parameter_instruction(0));
          }
        }
      } else if (user->opcode() == HloOpcode::kAsyncUpdate ||
                 user->opcode() == HloOpcode::kAsyncDone) {
        if (HloInstruction::IsThreadIncluded(user->async_execution_thread(),
                                             execution_threads_)) {
          // For async update and async done, we cannot distinguish which
          // parameter needs to be updated so add all to the worklist.
          for (int64_t parameter_number = 0;
               parameter_number <
               user->async_wrapped_computation()->num_parameters();
               ++parameter_number) {
            add_to_worklist(
                user->async_wrapped_computation()->parameter_instruction(
                    parameter_number));
          }
        }
      } else {
        for (HloComputation* called_computation : user->called_computations()) {
          if (!HloInstruction::IsThreadIncluded(
                  called_computation->execution_thread(), execution_threads_)) {
            continue;
          }
          const CallGraphNode& call_graph_node =
              call_graph_->GetNode(called_computation);
          if (call_graph_node.context() == CallContext::kControlFlow ||
              user->opcode() == HloOpcode::kFusion) {
            for (int64_t operand_number : user->OperandIndices(instruction)) {
              add_to_worklist(
                  called_computation->parameter_instruction(operand_number));
            }
          }
        }
      }
    }

    // If instruction is a root instruction, then propagate out to any calling
    // instruction and across any while backedge.
    if (instruction == instruction->parent()->root_instruction()) {
      const CallGraphNode& call_graph_node =
          call_graph_->GetNode(instruction->parent());
      for (const CallSite& callsite : call_graph_node.caller_callsites()) {
        if (callsite.instruction()->opcode() == HloOpcode::kWhile) {
          // Add the while itself, and the body and condition parameters.
          add_to_worklist(callsite.instruction());
          add_to_worklist(
              callsite.instruction()->while_body()->parameter_instruction(0));
          add_to_worklist(
              callsite.instruction()->while_condition()->parameter_instruction(
                  0));
        } else if (call_graph_node.context() == CallContext::kControlFlow ||
                   callsite.instruction()->opcode() ==
                       HloOpcode::kConditional ||
                   callsite.instruction()->opcode() == HloOpcode::kFusion) {
          add_to_worklist(callsite.instruction());
        }
      }
    }
  }
}

const InstructionValueSet& HloValueTracing::GetInstructionValueSet(
    const HloInstruction* instruction) const {
  DCHECK(value_sets_.contains(instruction))
      << "Instruction " << instruction->ToString() << " not found.";
  return *value_sets_.find(instruction)->second;
}

InstructionValueSet& HloValueTracing::GetInstructionValueSet(
    const HloInstruction* instruction) {
  DCHECK(value_sets_.contains(instruction))
      << "Instruction " << instruction->ToString() << " not found.";
  return *value_sets_.find(instruction)->second;
}

absl::Status HloValueTracing::InitializeInstructionValueSets() {
  for (const HloComputation* computation : module_.MakeComputationSorted()) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads_)) {
      continue;
    }
    const CallGraphNode& call_graph_node = call_graph_->GetNode(computation);
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      // Create an empty shape tree.
      value_sets_.insert({instruction, std::make_unique<InstructionValueSet>(
                                           &instruction->shape())});

      // For each sub-shape of the instruction shape, add a new HloValue to its
      // HloValueSet. should_define may be provided to define a subset of
      // values.
      auto define_all_values =
          [this, &instruction](
              absl::FunctionRef<bool(const ShapeIndex&)> should_define =
                  [](const ShapeIndex&) { return true; }) {
            for (auto& pair : GetInstructionValueSet(instruction)) {
              const ShapeIndex& index = pair.first;

              if (should_define(index)) {
                HloValue* value =
                    NewHloValue(instruction, index, /*is_phi=*/false);
                GetValueSet(instruction, index).AddValue(value);
              }
            }
          };

      // Add a new HloValue to the HloValueSet corresponding to the given index
      // of the instruction shape.
      auto define_value_at = [this, &instruction](const ShapeIndex& index) {
        HloValue* value = NewHloValue(instruction, index, /*is_phi=*/false);
        GetValueSet(instruction, index).AddValue(value);
      };

      switch (instruction->opcode()) {
        case HloOpcode::kAddDependency:
        case HloOpcode::kWhile:
        case HloOpcode::kCall:
        case HloOpcode::kConditional:
        case HloOpcode::kGetTupleElement:
        case HloOpcode::kDomain:
        case HloOpcode::kOptimizationBarrier:
        case HloOpcode::kCopy:
        case HloOpcode::kFusion:
          // These instructions define no values. The values in their output
          // flow from their operands or from cross computation dataflow.
          break;
        case HloOpcode::kParameter: {
          if (call_graph_node.context() == CallContext::kBoth) {
            // We do not support a subcomputation that is called from both a
            // parallel and sequential context. In this case, the parameter
            // would both define a value and propagate a value from its
            // caller. This limitation is not really a problem because the call
            // graph is typically flattened.
            return Unimplemented(
                "Computation %s is called in both a parallel (eg, kMap) and "
                "sequential (eg, kCall) context",
                computation->name());
          }
          if (call_graph_node.caller_callsites().empty()) {
            // Parameters of computations called in a parallel context (eg, map
            // and reduce) as well as parameters of dead computations define all
            // values in their output. Otherwise the values of the parameter
            // come from the caller (eg, operands to the kCall instruction).
            define_all_values();
          } else {
            HloOpcode caller_callsite_opcode =
                call_graph_node.caller_callsites()
                    .front()
                    .instruction()
                    ->opcode();
            if (caller_callsite_opcode != HloOpcode::kFusion) {
              define_all_values();
            }
          }
          break;
        }
        case HloOpcode::kTuple:
          // These instructions only define their top-level values. Any other
          // values flow from their operands.
          define_value_at(/*index=*/{});
          break;
        case HloOpcode::kAsyncStart: {
          // AsyncStart produces a tuple of {{aliased operands}, {destination},
          // contexts}. It defines all of the tuple-shaped values and the
          // contexts.
          // If the thread is excluded, then we don't track the contained
          // dataflow, and define the destination values too.
          bool thread_included = HloInstruction::IsThreadIncluded(
              instruction->async_execution_thread(), execution_threads_);
          define_all_values([&](const ShapeIndex& index) {
            return ShapeUtil::GetSubshape(instruction->shape(), index)
                       .IsTuple() ||
                   (!thread_included && index.front() == 1) ||
                   (index.front() > 1);
          });
          break;
        }
        case HloOpcode::kAsyncUpdate:
          // AsyncUpdate produces a tuple of {{aliased operands}, {destination},
          // contexts} where all of the array-typed values alias with the
          // operand. So, only tuple-shaped values are defined by AsyncUpdate.
          define_all_values([&](const ShapeIndex& index) {
            return ShapeUtil::GetSubshape(instruction->shape(), index)
                .IsTuple();
          });
          break;
        case HloOpcode::kAsyncDone:
          // AsyncDone's output aliases its output. It defines all remaining
          // tuple-shaped values.
          define_all_values([&](const ShapeIndex& index) {
            return ShapeUtil::GetSubshape(instruction->shape(), index)
                .IsTuple();
          });
          break;
        case HloOpcode::kCopyStart:
          // CopyStart produces a tuple of {destination buffer, aliased operand,
          // U32 context}.
          define_value_at(/*index=*/{});
          define_value_at(/*index=*/{2});
          break;
        case HloOpcode::kCopyDone:
          // CopyDone consumes a tuple produced by CopyStart and produces an
          // element. Its output aliases its input tuple element {0}.
          break;
        case HloOpcode::kAllGatherStart:
          // AllGatherStart produces a tuple of
          // {aliased operands, destination buffers}. If there is more than
          // one operand, then both aliased operands and destination buffers
          // will be tuples themselves. all-gather-start will define all tuples
          // and all tuple leaves (arrays) in tuple sub-index 1 (destination
          // buffers).
          define_all_values([&](const ShapeIndex& index) {
            return ShapeUtil::GetSubshape(instruction->shape(), index)
                       .IsTuple() ||
                   index.front() == 1;
          });
          break;
        case HloOpcode::kAllGatherDone:
          // AllGatherDone's output aliases its input tuple element {1}.
          if (instruction->shape().IsTuple()) {
            define_value_at(/*index=*/{});
          }
          break;
        case HloOpcode::kAllReduceDone:
          // AllReduceDone's output aliases its input.
          break;
        case HloOpcode::kCollectivePermuteStart:
          // CollectivePermuteStart produces a tuple of
          // {aliased operand, destination buffer, contexts}, where the context
          // data are optional.
          define_value_at(/*index=*/{});
          define_value_at(/*index=*/{1});
          for (int i = 2; i < instruction->shape().tuple_shapes().size(); ++i) {
            define_value_at(/*index=*/{i});
          }

          if (instruction->operand_count() > 1) {
            CHECK_EQ(instruction->operand_count(), 4);
            if (instruction->operand(1)->shape().IsTuple()) {
              for (int i = 0; i < ShapeUtil::TupleElementCount(
                                      instruction->operand(1)->shape());
                   ++i) {
                define_value_at(/*index=*/{1, i});
              }
            }
          }
          break;
        case HloOpcode::kCollectivePermuteDone:
          // CollectivePermuteDone's output aliases its input tuple element {1}.
          if (instruction->shape().IsTuple()) {
            define_value_at(/*index=*/{});
          }
          break;
        case HloOpcode::kRecvDone:
          // RecvDone produces a two-element tuple. Element zero aliases its
          // input tuple element {0}; element one is a token.
          define_value_at(/*index=*/{});
          define_value_at(/*index=*/{1});
          break;
        case HloOpcode::kSend:
          // Send produces a tuple of {aliased operand, U32 context, token},
          // therefore only defines the top-level tuple and the tuple elements
          // at {1} and {2}.
          define_value_at(/*index=*/{});
          define_value_at(/*index=*/{1});
          define_value_at(/*index=*/{2});
          break;
        default:
          define_all_values();
          break;
      }
    }
  }

  return absl::OkStatus();
}

/* static */
absl::StatusOr<std::unique_ptr<HloValueTracing>> HloValueTracing::Run(
    const HloModule& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto hlo_value_tracing =
      absl::WrapUnique(new HloValueTracing(module, execution_threads));

  TF_RETURN_IF_ERROR(hlo_value_tracing->InitializeInstructionValueSets());
  hlo_value_tracing->Propagate();

  // Delete all values marked for deletion.
  hlo_value_tracing->DeleteMarkedValues();

  // Gather and set all non-definition positions of all values. Value deletion
  // is rare, so just use a vector indexed by Value::Id rather than a map from
  // Value::Id to positions. There should be very few holes in the vector, and
  // lookup is faster.
  std::vector<std::vector<HloPosition>> value_positions(
      hlo_value_tracing->next_value_id_);
  for (const HloComputation* computation : module.computations()) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads)) {
      continue;
    }
    for (HloInstruction* instruction : computation->instructions()) {
      for (const auto& pair :
           hlo_value_tracing->GetInstructionValueSet(instruction)) {
        const ShapeIndex& index = pair.first;
        const HloValueSet& value_set = pair.second;
        for (const HloValue* value : value_set.values()) {
          if (value->defining_instruction() != instruction) {
            value_positions[value->id()].push_back(
                HloPosition{instruction, index});
          }
        }
      }
    }
  }
  for (auto& pair : hlo_value_tracing->values_) {
    HloValue::Id value_id = pair.first;
    HloValue& value = *pair.second;
    value.SetPositions(value_positions[value_id]);
  }

  // Construct vector of values.
  hlo_value_tracing->values_vector_.reserve(hlo_value_tracing->values_.size());
  for (const auto& pair : hlo_value_tracing->values_) {
    hlo_value_tracing->values_vector_.push_back(pair.second.get());
  }
  absl::c_sort(hlo_value_tracing->values_vector_, HloValue::IdLessThan);

  return hlo_value_tracing;
}
}  // namespace xla
