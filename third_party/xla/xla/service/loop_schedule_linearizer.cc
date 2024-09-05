/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/loop_schedule_linearizer.h"

#include <memory>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/graphcycles/graphcycles.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

// Calculate ordering for HLO, for fast online checking of whether adding
// additional dependencies would create cycles.
class ComputationInstructionOrdering {
 public:
  explicit ComputationInstructionOrdering(const HloComputation& computation) {
    for (const HloInstruction* instr : computation.instructions()) {
      for (const HloInstruction* control_pred : instr->control_predecessors()) {
        CHECK(this->InsertEdge(*control_pred, *instr))
            << "Graph already contained a cycle";
      }

      for (int op_id = 0; op_id < instr->operand_count(); op_id++) {
        const HloInstruction* op = instr->operand(op_id);
        CHECK(this->InsertEdge(*op, *instr))
            << "Graph already contained a cycle";
      }
    }
  }

  int32_t NodeIdForInstruction(const HloInstruction& instr) {
    int32_t instruction_id = instr.unique_id();
    auto it = node_id_to_graph_id_.find(instruction_id);

    if (it != node_id_to_graph_id_.end()) {
      return it->second;
    }
    int32_t node_id = graph_cycles_.NewNode();
    node_id_to_graph_id_[instruction_id] = node_id;
    return node_id;
  }

  // Returns `false` if adding an edge would have introduced a cycle. Does not
  // add an edge in that case. Returns `true` otherwise.
  bool InsertEdge(const HloInstruction& source, const HloInstruction& dest) {
    int32_t source_id = NodeIdForInstruction(source);
    int32_t dest_id = NodeIdForInstruction(dest);
    return graph_cycles_.InsertEdge(source_id, dest_id);
  }

 private:
  absl::flat_hash_map<int32_t, int32_t> node_id_to_graph_id_;
  tensorflow::GraphCycles graph_cycles_;
};

}  // namespace

static absl::StatusOr<bool> AddControlEdgesForLoopWrites(
    HloInstruction* xla_while, HloAliasAnalysis& alias_analysis) {
  HloDataflowAnalysis& dataflow = alias_analysis.dataflow_analysis();
  HloComputation* body = xla_while->while_body();
  HloInstruction* root = body->root_instruction();
  HloInstruction* input = body->parameter_instruction(0);

  bool changed = false;

  // Compute dependency ordering ourselves. The reason we don't reuse other
  // computations is because it is hard to extract the underlying graph from
  // those abstractions.
  ComputationInstructionOrdering ordering(*body);
  ShapeTree<bool> indices_to_copy(&xla_while->shape());

  for (auto& p : indices_to_copy) {
    const ShapeIndex& index = p.first;

    if (index.empty()) {
      continue;
    }

    if (dataflow.GetValueSet(root, index).values().size() > 1 ||
        dataflow.GetValueSet(input, index).values().size() > 1) {
      VLOG(2) << "Index " << index.ToString() << " is associated with multiple "
              << "values, not attempting to introduce stricter dependencies";
    } else {
      HloValue& value_at_root = dataflow.GetUniqueValueAt(root, index);
      HloValue& value_at_input = dataflow.GetUniqueValueAt(input, index);

      if (value_at_root.shape().IsTuple()) {
        // TODO(cheshire): For simplicity we currently do not handle nested
        // tuples, as we haven't seen them in the examples we care about.
        continue;
      }

      // TODO(cheshire): This is too conservative and does not take aliasing
      // into account.
      HloInstruction* write = value_at_root.defining_instruction();

      for (const HloUse& use : value_at_input.GetUses()) {
        HloInstruction* read = use.instruction;

        if (read != write &&
            value_at_root != value_at_input

            // TODO(cheshire): Parents sometimes differ in case of e.g. nested
            // loops, where the value is read/written into in the inner loop.
            // For now we skip this case for simplicity (as the inner loop
            // performance is more important in any case)
            && read->parent() == write->parent()) {
          VLOG(2) << "Inside " << body->name() << ", index "
                  << index.ToString();
          if (!ordering.InsertEdge(*read, *write)) {
            VLOG(2) << "Not adding a control dependency from "
                    << read->ToShortString() << " to " << write->ToShortString()
                    << " as it would introduce a cycle";
            continue;
          }

          // Add control dependency if it does not already exist.
          if (!absl::c_linear_search(read->control_successors(), write)) {
            // Unless we want a copy, read should happen before write.
            TF_RETURN_IF_ERROR(read->AddControlDependencyTo(write));
            VLOG(2) << "Adding dependency: " << read->ToShortString()
                    << " before " << write->ToShortString();
            changed = true;
          }
        }
      }
    }
  }
  return changed;
}

absl::StatusOr<bool> LoopScheduleLinearizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Constructing HloAliasAnalysis is expensive, so don't do it until we find at
  // least one kWhile op in the module.
  std::unique_ptr<HloAliasAnalysis> alias_analysis;

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kWhile) {
        continue;
      }

      // Skip loops that have async collectives, as the additional control deps
      // inserted by this pass can constrain scheduling and hamper compute
      // and communication overlap.
      const HloComputation* body = instruction->while_body();
      bool has_async_collectives =
          absl::c_any_of(body->instructions(), [](const HloInstruction* instr) {
            return hlo_query::IsAsyncCollectiveStartOp(
                       instr, /*include_send_recv=*/true) ||
                   hlo_query::IsAsyncCollectiveDoneOp(
                       instr, /*include_send_recv=*/true);
          });

      if (has_async_collectives) {
        VLOG(2) << "Skipping " << instruction->name()
                << " since body has async collectives";
        continue;
      }

      if (alias_analysis == nullptr) {
        TF_ASSIGN_OR_RETURN(alias_analysis,
                            HloAliasAnalysis::Run(module, can_share_buffer_));
      }
      TF_ASSIGN_OR_RETURN(bool updated_loop, AddControlEdgesForLoopWrites(
                                                 instruction, *alias_analysis));
      changed |= updated_loop;
    }
  }

  return changed;
}

}  // end namespace xla
