/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/transforms/bfloat16_propagation.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_operand_index.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/map_util.h"
#include "xla/service/float_support.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

// Returns true if `hlo` is an associative scan (one for which the
// ScanExpander chose not to lower into a while loop). Only associative
// scans survive past ScanExpander, and only those need the kWhile-style
// carry-aliasing handling implemented in this file. Non-associative scans
// are expanded into while loops elsewhere and reach this pass as kWhile.
bool IsAssociativeScan(const HloInstruction* hlo) {
  if (hlo->opcode() != HloOpcode::kScan) {
    return false;
  }
  return Cast<HloScanInstruction>(hlo)->is_associative() == TRI_STATE_TRUE;
}

}  // namespace

BFloat16Propagation::BFloat16Propagation(const FloatSupport* bfloat16_support,
                                         const AliasInfo* alias_info)
    : bfloat16_support_(bfloat16_support), alias_info_(alias_info) {
  DCHECK_EQ(bfloat16_support->LowPrecisionType(), BF16);
}

void BFloat16Propagation::DetermineFusionComputationPrecision(
    HloInstruction* fusion) {
  CHECK_EQ(fusion->opcode(), HloOpcode::kFusion);
  if (!bfloat16_support_->SupportsMixedPrecisions(*fusion)) {
    return;
  }

  // We are depending on the fusion node itself having already been analyzed
  // for whether it can output BF16 and this has been adjusted in the output
  // shape, and now we're looking to update the interior of the fusion node to
  // match the new output shape, as well as recursively process the whole fusion
  // node even if the output shape was not modified.
  auto root = fusion->fused_instructions_computation()->root_instruction();

  // Adjust root's element types according to the fusion's output shape.
  ShapeUtil::ForEachSubshape(
      root->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.element_type() != F32) {
          return;
        }
        if (OutputTypeAfterChange(fusion, index) == BF16) {
          AddToOrRemoveFromBF16ChangeSet(root, index, BF16);
          VLOG(2) << "Fused root " << root->ToString() << " at shape index "
                  << index << " changed to BF16 precision for fusion "
                  << fusion->ToString();
        }
      });

  // Propagate BF16 in the fusion computation.
  auto insts =
      fusion->fused_instructions_computation()->MakeInstructionPostOrder();
  for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
    DetermineInstructionPrecision(*inst_it, /*skip_parameters=*/false);
  }
  computations_visited_in_backward_pass_.insert(
      fusion->fused_instructions_computation());

  RevertIfFusionInternalBF16Changes(fusion);
}

void BFloat16Propagation::RevertIfFusionInternalBF16Changes(
    HloInstruction* fusion) {
  auto has_changes = [this](HloInstruction* inst) {
    auto it = changes_to_bf16_.find(inst);
    return it != changes_to_bf16_.end() && !it->second.empty();
  };

  auto root = fusion->fused_instructions_computation()->root_instruction();
  absl::flat_hash_set<const HloValue*> changed_root_buffers;

  auto root_changes_it = changes_to_bf16_.find(root);
  if (root_changes_it != changes_to_bf16_.end()) {
    for (const auto& entry : root_changes_it->second) {
      for (const HloValue* value :
           dataflow_->GetValueSet(root, entry.second).values()) {
        changed_root_buffers.insert(value);
      }
    }
  }

  auto aliases_changed_root_buffer = [this, &changed_root_buffers](
                                         const HloInstruction* inst) {
    bool aliasing = false;
    ShapeUtil::ForEachSubshape(inst->shape(), [&](const Shape& subshape,
                                                  const ShapeIndex& index) {
      if (aliasing) {
        // Skip if aliasing is already found.
        return;
      }
      // Only F32 buffers are considered for changing to BF16 in this
      // pass.
      if (subshape.element_type() != F32) {
        return;
      }

      aliasing = absl::c_any_of(dataflow_->GetValueSet(inst, index).values(),
                                IsValueIn(changed_root_buffers));
    });
    return aliasing;
  };

  for (auto inst :
       fusion->fused_instructions_computation()->MakeInstructionPostOrder()) {
    if (inst->opcode() == HloOpcode::kParameter) {
      continue;
    }
    if (aliases_changed_root_buffer(inst)) {
      continue;
    }
    if (inst->opcode() == HloOpcode::kFusion) {
      bool parameter_reverted = false;
      for (int64_t i = 0; i < inst->operand_count(); ++i) {
        if (has_changes(inst->mutable_operand(i))) {
          // Changes on the operand have not been reverted.
          continue;
        }
        auto* fused_parameter = inst->fused_parameter(i);
        if (has_changes(fused_parameter)) {
          changes_to_bf16_.erase(fused_parameter);
          parameter_reverted = true;
        }
      }
      if (parameter_reverted) {
        RevertIfFusionInternalBF16Changes(inst);
      }
    }
    if (!has_changes(inst)) {
      continue;
    }
    bool revert_changes = true;
    for (auto operand : inst->operands()) {
      if (has_changes(operand)) {
        revert_changes = false;
        break;
      }
    }
    if (revert_changes) {
      changes_to_bf16_.erase(inst);
    }
  }
}

void BFloat16Propagation::DetermineWhileComputationsPrecision(
    HloInstruction* while_hlo) {
  CHECK_EQ(while_hlo->opcode(), HloOpcode::kWhile);

  // We are depending on the while node itself having already been analyzed for
  // whether it can output BF16 and this has been adjusted in the output shape,
  // and now we're looking to update the body and condition computations to
  // match the new output shape, as well as recursively process the whole while
  // node even if the output shape was not modified.
  HloComputation* body = while_hlo->while_body();
  auto body_root = body->root_instruction();
  HloComputation* condition = while_hlo->while_condition();

  ShapeUtil::ForEachSubshape(
      body_root->shape(), [this, while_hlo, body_root](
                              const Shape& subshape, const ShapeIndex& index) {
        if (subshape.element_type() != F32) {
          return;
        }
        if (OutputTypeAfterChange(while_hlo, index) == BF16) {
          AddToOrRemoveFromBF16ChangeSet(body_root, index, BF16);
          VLOG(2) << "While body root " << body_root->ToString()
                  << " at shape index " << index
                  << " changed to BF16 precision for while "
                  << while_hlo->ToString();
        }
      });

  auto body_insts = body->MakeInstructionPostOrder();
  for (auto inst_it = body_insts.rbegin(); inst_it != body_insts.rend();
       ++inst_it) {
    DetermineInstructionPrecision(*inst_it, /*skip_parameters=*/false);
  }
  computations_visited_in_backward_pass_.insert(body);

  auto condition_insts = condition->MakeInstructionPostOrder();
  for (auto inst_it = condition_insts.rbegin();
       inst_it != condition_insts.rend(); ++inst_it) {
    DetermineInstructionPrecision(*inst_it, /*skip_parameters=*/false);
  }
  computations_visited_in_backward_pass_.insert(condition);
}

void BFloat16Propagation::DetermineScanComputationPrecision(
    HloInstruction* scan_hlo) {
  CHECK(IsAssociativeScan(scan_hlo));

  // We are depending on the scan node itself having already been analyzed
  // for whether it can output BF16 and this has been adjusted in the output
  // shape. Mirroring the kWhile path, we now push those output-slot precision
  // decisions down to the body root and propagate through the body.
  //
  // Scan tuple structure (verified by HloVerifier::HandleScan):
  //   * scan output slot i and body root slot i hold the same precision; the
  //     output is the body root with scan_dim added back (for i < num_outputs)
  //     or carried through verbatim (for i >= num_outputs).
  // So a per-shape-index walk over the body root mirrors the scan shape 1:1.
  HloComputation* body = scan_hlo->to_apply();
  HloInstruction* body_root = body->root_instruction();

  ShapeUtil::ForEachSubshape(body_root->shape(), [this, scan_hlo, body_root](
                                                     const Shape& subshape,
                                                     const ShapeIndex& index) {
    if (subshape.element_type() != F32) {
      return;
    }
    if (OutputTypeAfterChange(scan_hlo, index) == BF16) {
      AddToOrRemoveFromBF16ChangeSet(body_root, index, BF16);
      VLOG(2) << "Scan body root " << body_root->ToString()
              << " at shape index " << index
              << " changed to BF16 precision for scan " << scan_hlo->ToString();
    }
  });

  auto body_insts = body->MakeInstructionPostOrder();
  for (auto inst_it = body_insts.rbegin(); inst_it != body_insts.rend();
       ++inst_it) {
    DetermineInstructionPrecision(*inst_it, /*skip_parameters=*/false);
  }
  computations_visited_in_backward_pass_.insert(body);
}

void BFloat16Propagation::DetermineConditionalComputationsPrecision(
    HloInstruction* cond) {
  CHECK_EQ(cond->opcode(), HloOpcode::kConditional);
  for (int64_t i = 0; i < cond->branch_count(); ++i) {
    auto branch = cond->branch_computation(i);
    auto root = branch->root_instruction();
    ShapeUtil::ForEachSubshape(
        root->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.element_type() != F32) {
            return;
          }
          if (OutputTypeAfterChange(cond, index) == BF16) {
            AddToOrRemoveFromBF16ChangeSet(root, index, BF16);
            VLOG(2) << "Conditional branch " << i << " root "
                    << root->ToString() << " at shape index " << index
                    << " changed to BF16 precision for conditional "
                    << cond->ToString();
          }
        });
    auto insts = branch->MakeInstructionPostOrder();
    for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
      DetermineInstructionPrecision(*inst_it, /*skip_parameters=*/false);
    }
    computations_visited_in_backward_pass_.insert(branch);
  }
}

void BFloat16Propagation::DetermineAsyncComputationsPrecision(
    HloInstruction* async_start) {
  CHECK_EQ(async_start->opcode(), HloOpcode::kAsyncStart);

  HloComputation* wrapped_comp = async_start->async_wrapped_computation();
  HloInstruction* root = async_start->async_wrapped_instruction();
  HloInstruction* done = async_start->async_chain_done();
  if (wrapped_comp == nullptr || root == nullptr || done == nullptr) {
    return;
  }
  ShapeUtil::ForEachSubshape(root->shape(), [&](const Shape& subshape,
                                                const ShapeIndex& index) {
    if (subshape.element_type() != F32) {
      return;
    }
    if (OutputTypeAfterChange(done, index) == BF16) {
      AddToOrRemoveFromBF16ChangeSet(root, index, BF16);
      VLOG(2) << "Async wrapped computation root " << root->ToString()
              << " at shape index " << index
              << " changed to BF16 precision for async start "
              << async_start->ToString();
    }
  });
  auto insts = wrapped_comp->MakeInstructionPostOrder();
  for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
    DetermineInstructionPrecision(*inst_it, /*skip_parameters=*/false);
  }
  computations_visited_in_backward_pass_.insert(wrapped_comp);
}

void BFloat16Propagation::DetermineCalledComputationsPrecision(
    HloInstruction* call) {
  CHECK_EQ(call->opcode(), HloOpcode::kCall);

  auto root = call->to_apply()->root_instruction();
  ShapeUtil::ForEachSubshape(
      root->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.element_type() != F32) {
          return;
        }
        if (OutputTypeAfterChange(call, index) == BF16) {
          AddToOrRemoveFromBF16ChangeSet(root, index, BF16);
          VLOG(2) << "Called computation root " << root->ToString()
                  << " at shape index " << index
                  << " changed to BF16 precision for call " << call->ToString();
        }
      });
  auto insts = call->to_apply()->MakeInstructionPostOrder();
  for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
    DetermineInstructionPrecision(*inst_it, /*skip_parameters=*/false);
  }
  computations_visited_in_backward_pass_.insert(call->to_apply());
}

bool BFloat16Propagation::AllUsersConsumeBF16(const HloInstruction& hlo,
                                              const ShapeIndex& index) const {
  // If the subshape isn't floating point then none of the users will be BF16.
  const Shape& subshape = ShapeUtil::GetSubshape(hlo.shape(), index);
  if (subshape.element_type() != BF16 && subshape.element_type() != F32) {
    return false;
  }

  const HloValueSet& value_set = dataflow_->GetValueSet(&hlo, index);
  for (const HloValue* value : value_set.values()) {
    if (ContainsKey(values_that_must_be_kept_as_f32_, value)) {
      return false;
    }
    // Check the uses against the value's original type. Using the type
    // after pending changes would keep a tentative BF16 change even when
    // aliasing users require F32.
    if (value->shape().element_type() == BF16) {
      continue;
    }
    for (const HloUse& use : value->GetUses()) {
      if (!ContainsKey(instructions_visited_in_backward_pass_,
                       use.instruction)) {
        // We don't know yet whether use.instruction will consume BF16 since it
        // hasn't been visited. Although we visit instructions in reverse
        // topological order, this is still possible because there may be
        // unvisited instruction that alias the same buffer. In this case, we
        // aggressively skip this use, and if this causes inconsistency (e.g.,
        // one use is in BF16 but another use is in F32), it will be resolved at
        // the end of the BFloat16Propagation pass.
        continue;
      }
      if (use.instruction->HasSideEffectNoRecurse()) {
        // Keep side-effecting instruction's operands unchanged.
        return false;
      }
      // Any visited user that can accept BF16 has already been updated if
      // necessary, e.g., the output has been changed to BF16 if it propagates
      // precision, or a called computation's parameters have been changed to
      // BF16 for fusions or whiles.
      if (use.instruction->opcode() == HloOpcode::kFusion) {
        auto* fused_parameter =
            use.instruction->fused_parameter(use.operand_number);
        if (OutputTypeAfterChange(fused_parameter, use.operand_index) != BF16) {
          return false;
        }
        continue;
      } else if (use.instruction->opcode() == HloOpcode::kWhile) {
        auto* cond_parameter =
            use.instruction->while_condition()->parameter_instruction(
                use.operand_number);
        if (OutputTypeAfterChange(cond_parameter, use.operand_index) != BF16) {
          return false;
        }
        auto* body_parameter =
            use.instruction->while_body()->parameter_instruction(
                use.operand_number);
        if (OutputTypeAfterChange(body_parameter, use.operand_index) != BF16) {
          return false;
        }
        continue;
      } else if (use.instruction->opcode() == HloOpcode::kConditional) {
        auto* cond_parameter =
            use.instruction->branch_computation(use.operand_number - 1)
                ->parameter_instruction(0);
        if (OutputTypeAfterChange(cond_parameter, use.operand_index) != BF16) {
          return false;
        }
        continue;
      } else if (use.instruction->opcode() == HloOpcode::kAsyncStart &&
                 HloInstruction::IsThreadIncluded(
                     use.instruction->async_execution_thread(),
                     execution_threads_)) {
        HloComputation* wrapped_comp =
            use.instruction->async_wrapped_computation();
        if (wrapped_comp == nullptr) {
          return false;
        }
        auto* async_parameter =
            wrapped_comp->parameter_instruction(use.operand_number);
        if (OutputTypeAfterChange(async_parameter, use.operand_index) != BF16) {
          return false;
        }
        continue;
      } else if (use.instruction->opcode() == HloOpcode::kCall) {
        auto* call_parameter =
            use.instruction->to_apply()->parameter_instruction(
                use.operand_number);
        if (OutputTypeAfterChange(call_parameter, use.operand_index) != BF16) {
          return false;
        }
        continue;
      } else if (IsAssociativeScan(use.instruction)) {
        auto* body_parameter =
            use.instruction->to_apply()->parameter_instruction(
                use.operand_number);
        if (OutputTypeAfterChange(body_parameter, use.operand_index) != BF16) {
          return false;
        }
        continue;
      } else if (use.instruction->opcode() == HloOpcode::kAsyncDone) {
        // async-done consumes whatever async-start gives it.
        continue;
      }
      if (use.instruction->opcode() == HloOpcode::kBitcast &&
          use.instruction->operand(use.operand_number)
                  ->shape()
                  .element_type() == BF16) {
        continue;
      }
      if (bfloat16_support_->EffectiveOperandPrecisionIsLowPrecision(
              *use.instruction, use.operand_number)) {
        continue;
      }
      // If the op propagates precision and it outputs a BF16, then it's OK to
      // supply BF16 also as the input. In the backward pass, the users shapes
      // should have already been processed.
      if (bfloat16_support_->EffectiveOperandPrecisionIsOutputPrecision(
              *use.instruction, use.operand_number)) {
        if (use.instruction->opcode() == HloOpcode::kTuple ||
            (use.instruction->opcode() == HloOpcode::kAllReduce &&
             use.instruction->shape().IsTuple())) {
          ShapeIndex use_output_index{use.operand_number};
          for (int64_t i : use.operand_index) {
            use_output_index.push_back(i);
          }
          if (OutputTypeAfterChange(use.instruction, use_output_index) ==
              BF16) {
            continue;
          }
        } else if (use.instruction->opcode() == HloOpcode::kGetTupleElement) {
          ShapeIndex use_output_index;
          for (int64_t i = 1; i < use.operand_index.size(); ++i) {
            use_output_index.push_back(use.operand_index[i]);
          }
          if (OutputTypeAfterChange(use.instruction, use_output_index) ==
              BF16) {
            continue;
          }
        } else {
          if (OutputTypeAfterChange(use.instruction, use.operand_index) ==
              BF16) {
            continue;
          }
        }
      }
      return false;
    }
  }
  return true;
}

bool BFloat16Propagation::ShouldKeepPrecisionUnchanged(
    const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kFusion &&
      inst->fusion_kind() == HloInstruction::FusionKind::kCustom) {
    return ShouldKeepPrecisionUnchanged(
        inst->fused_instructions_computation()->root_instruction());
  }
  // Do not change precision for side-effecting instructions, control flow, and
  // bitcast-convert, because this pass might break the interfaces or
  // assumptions for them. It is safe to change precision for AllocateBuffer
  // since it is merely a buffer allocation and does not have any side effects.
  // Non-associative scans must have been expanded into while loops by
  // ScanExpander upstream of BFloat16Propagation; without expansion this pass
  // cannot consistently update both the scan output and the body and the
  // module would fail HloVerifier::HandleScan. Only associative scans are
  // expected here, and they are handled like other ops by the propagation.
  if (inst->opcode() == HloOpcode::kScan) {
    CHECK_EQ(Cast<HloScanInstruction>(inst)->is_associative(), TRI_STATE_TRUE)
        << "Non-associative kScan reached BFloat16Propagation; ScanExpander "
           "must run upstream. "
        << inst->ToString();
  }
  if (inst->opcode() == HloOpcode::kBitcast &&
      UnmutatedElementType(inst) != UnmutatedElementType(inst->operand(0))) {
    return true;
  }
  return (inst->opcode() == HloOpcode::kCustomCall &&
          !inst->IsCustomCall("AllocateBuffer")) ||
         inst->opcode() == HloOpcode::kBitcastConvert ||
         inst->HasSideEffectNoRecurse() ||
         (inst->IsAsynchronous() &&
          !HloInstruction::IsThreadIncluded(inst->async_execution_thread(),
                                            execution_threads_));
}

void BFloat16Propagation::DetermineInstructionPrecision(HloInstruction* hlo,
                                                        bool skip_parameters) {
  // We handle any fusion computation, while body/condition or conditional
  // branches after the instruction is handled, because we need to know the
  // output shape of a fusion or while before propagating inside its
  // computations.
  bool postpone_processing_called_computations = false;
  absl::Cleanup cleaner = [this, hlo,
                           &postpone_processing_called_computations] {
    if (!postpone_processing_called_computations) {
      if (hlo->opcode() == HloOpcode::kFusion) {
        DetermineFusionComputationPrecision(hlo);
      } else if (hlo->opcode() == HloOpcode::kWhile) {
        DetermineWhileComputationsPrecision(hlo);
      } else if (IsAssociativeScan(hlo)) {
        DetermineScanComputationPrecision(hlo);
      } else if (hlo->opcode() == HloOpcode::kConditional) {
        DetermineConditionalComputationsPrecision(hlo);
      } else if (hlo->opcode() == HloOpcode::kAsyncStart &&
                 HloInstruction::IsThreadIncluded(hlo->async_execution_thread(),
                                                  execution_threads_)) {
        DetermineAsyncComputationsPrecision(hlo);
      } else if (hlo->opcode() == HloOpcode::kCall) {
        DetermineCalledComputationsPrecision(hlo);
      }
    }
    instructions_visited_in_backward_pass_.insert(hlo);
  };

  if (hlo->opcode() == HloOpcode::kWhile &&
      (caller_counts_[hlo->while_condition()] > 1 ||
       caller_counts_[hlo->while_body()] > 1)) {
    postpone_processing_called_computations = true;
    return;
  }

  if (IsAssociativeScan(hlo)) {
    // Associative scan bodies are constructed alongside the scan op and not
    // shared with any other caller, so we never need to postpone processing
    // them (unlike kWhile/kCall whose called computations may have multiple
    // callers). Assert the single-caller invariant rather than relying on the
    // pre-existing caller_counts_ map (which is currently never populated).
    CHECK_EQ(hlo->to_apply()->caller_instructions().size(), 1u)
        << "Associative scan body has multiple callers: "
        << hlo->to_apply()->name();
  }

  if (hlo->opcode() == HloOpcode::kConditional &&
      absl::c_any_of(hlo->branch_computations(), [&](const HloComputation* c) {
        return caller_counts_[c] > 1;
      })) {
    postpone_processing_called_computations = true;
    return;
  }

  if (hlo->opcode() == HloOpcode::kAsyncStart &&
      HloInstruction::IsThreadIncluded(hlo->async_execution_thread(),
                                       execution_threads_) &&
      (hlo->async_wrapped_computation() == nullptr ||
       caller_counts_[hlo->async_wrapped_computation()] > 1)) {
    postpone_processing_called_computations = true;
    return;
  }

  if (hlo->opcode() == HloOpcode::kCall &&
      caller_counts_[hlo->to_apply()] > 1) {
    postpone_processing_called_computations = true;
    return;
  }

  // Prevent root instructions from having their output modified by recording
  // all F32 output values as needing to stay as F32.
  CHECK(hlo->parent() != nullptr);
  if (hlo == hlo->parent()->root_instruction()) {
    if (!hlo->parent()->IsFusionComputation()) {
      ShapeUtil::ForEachSubshape(hlo->shape(), [&](const Shape& /* subshape */,
                                                   const ShapeIndex& index) {
        if (OutputTypeAfterChange(hlo, index) != F32) {
          return;
        }
        for (const auto* value : dataflow_->GetValueSet(hlo, index).values()) {
          // Since we use HloValues from the dataflow analysis, this can also
          // affect HLO instructions beyond the root, e.g., if the root is a
          // Tuple HLO, then its operands are also affected.
          if (value->shape().element_type() == F32) {
            values_that_must_be_kept_as_f32_.insert(value);
          }
        }
      });
    }
    return;
  }

  if (ShouldKeepPrecisionUnchangedCached(hlo) ||
      (hlo->opcode() == HloOpcode::kParameter && skip_parameters)) {
    return;
  }

  if (!ContainsKey(consider_using_bfloat16_, hlo)) {
    return;
  }

  if (!bfloat16_support_->SupportsLowPrecisionOutput(*hlo)) {
    return;
  }

  ShapeUtil::ForEachSubshape(
      hlo->shape(),
      [hlo, this](const Shape& subshape, const ShapeIndex& index) {
        if (OutputTypeAfterChange(hlo, index) == F32 &&
            AllUsersConsumeBF16(*hlo, index) && subshape.has_layout() &&
            subshape.layout().memory_space() != Layout::kHostMemorySpace) {
          AddToOrRemoveFromBF16ChangeSet(hlo, index, BF16);
          VLOG(2) << "HloInstruction output at shape index " << index
                  << " changed to BF16 precision: " << hlo->ToString();
        }
      });
}

bool BFloat16Propagation::InstructionIsCandidateForBF16Output(
    HloInstruction* hlo) {
  if (!bfloat16_support_->SupportsMixedPrecisions(*hlo) &&
      hlo->opcode() != HloOpcode::kTuple &&
      hlo->opcode() != HloOpcode::kGetTupleElement &&
      hlo->opcode() != HloOpcode::kDomain &&
      hlo->shape().element_type() != BF16) {
    for (int64_t i = 0; i < hlo->operand_count(); ++i) {
      if (!bfloat16_support_->EffectiveOperandPrecisionIsOutputPrecision(*hlo,
                                                                         i) ||
          !ContainsKey(consider_using_bfloat16_, hlo->operand(i))) {
        return false;
      }
    }
  }
  if (hlo->opcode() == HloOpcode::kDynamicSlice ||
      hlo->opcode() == HloOpcode::kCopy) {
    // These two instructions are not candidates for BF16 output if their
    // source operand is in host memory space.
    if (hlo->operand(0)->shape().has_layout() &&
        hlo->operand(0)->shape().layout().memory_space() ==
            Layout::kHostMemorySpace) {
      return false;
    }
  }
  return true;
}

void BFloat16Propagation::BuildF32ConstraintGraph(HloModule* module) {
  const std::vector<HloComputation*> computations =
      module->MakeComputationPostOrder(execution_threads_);
  included_computations_ = absl::flat_hash_set<const HloComputation*>(
      computations.begin(), computations.end());

  // The seeds depend on the push index, so build it first.
  BuildCallBoundaryPushIndex(computations);
  for (HloComputation* computation : computations) {
    for (HloInstruction* hlo : computation->instructions()) {
      AddInPlaceEdges(hlo);
      AddKeepPrecisionSeeds(hlo);
      AddScanCarryEdges(hlo);
    }
  }
  for (const HloValue* value : dataflow_->values()) {
    AddValueSeedsAndUseEdges(value);
  }
}

void BFloat16Propagation::AddPushableParams(
    HloComputation* callee, absl::Span<HloInstruction* const> operands) {
  CHECK_EQ(operands.size(), callee->num_parameters());
  for (int64_t i = 0; i < operands.size(); ++i) {
    HloInstruction* parameter = callee->parameter_instruction(i);
    push_params_[operands[i]].push_back(parameter);
    ShapeUtil::ForEachSubshape(
        parameter->shape(),
        [&](const Shape& subshape, const ShapeIndex& index) {
          if (ShapeUtil::IsLeafIndex(parameter->shape(), index)) {
            bf16_pushable_positions_.insert({parameter, index});
          }
        });
  }
}

void BFloat16Propagation::AddPushableRoot(HloInstruction* call_site,
                                          HloComputation* callee) {
  HloInstruction* root = callee->root_instruction();
  // Every root takes F32 pushes; only BF16 pushes are gated on
  // ShouldKeepPrecisionUnchanged.
  push_roots_[call_site].push_back(root);
  if (ShouldKeepPrecisionUnchangedCached(root)) {
    return;
  }
  ShapeUtil::ForEachSubshape(
      root->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (ShapeUtil::IsLeafIndex(call_site->shape(), index)) {
          bf16_pushable_positions_.insert({root, index});
        }
      });
}

void BFloat16Propagation::BuildCallBoundaryPushIndex(
    absl::Span<HloComputation* const> computations) {
  for (HloComputation* computation : computations) {
    for (HloInstruction* hlo : computation->instructions()) {
      switch (hlo->opcode()) {
        case HloOpcode::kFusion:
          AddPushableParams(hlo->fused_instructions_computation(),
                            hlo->operands());
          AddPushableRoot(hlo, hlo->fused_instructions_computation());
          break;
        case HloOpcode::kWhile:
          AddPushableParams(hlo->while_condition(), hlo->operands());
          AddPushableParams(hlo->while_body(), hlo->operands());
          AddPushableRoot(hlo, hlo->while_body());
          break;
        case HloOpcode::kScan:
          if (IsAssociativeScan(hlo)) {
            AddPushableParams(hlo->to_apply(), hlo->operands());
            AddPushableRoot(hlo, hlo->to_apply());
          }
          break;
        case HloOpcode::kConditional:
          for (int64_t i = 0; i < hlo->branch_count(); ++i) {
            AddPushableParams(hlo->branch_computation(i),
                              {hlo->mutable_operand(i + 1)});
            AddPushableRoot(hlo, hlo->branch_computation(i));
          }
          break;
        case HloOpcode::kAsyncStart:
          if (HloInstruction::IsThreadIncluded(hlo->async_execution_thread(),
                                               execution_threads_) &&
              hlo->async_wrapped_computation() != nullptr) {
            AddPushableParams(hlo->async_wrapped_computation(),
                              hlo->operands());
            AddPushableRoot(hlo, hlo->async_wrapped_computation());
          }
          break;
        case HloOpcode::kCall:
          AddPushableParams(hlo->to_apply(), hlo->operands());
          AddPushableRoot(hlo, hlo->to_apply());
          break;
        default:
          break;
      }
    }
  }
}

void BFloat16Propagation::AddInPlaceEdges(HloInstruction* hlo) {
  if (ShouldKeepPrecisionUnchangedCached(hlo)) {
    return;  // The rules never adjusted kept instructions' outputs.
  }
  for (const auto& operand_and_output_index :
       GetInPlaceInputOutputPairsCached(hlo)) {
    const HloOperandIndex& operand_index = operand_and_output_index.first;
    for (const HloValue* value :
         dataflow_
             ->GetValueSet(hlo->operand(operand_index.operand_number),
                           operand_index.operand_index)
             .values()) {
      const PrimitiveType value_type = value->shape().element_type();
      if (value_type != F32 && value_type != BF16) {
        // A value that can never be BF16 (e.g. reached through a type
        // punning bitcast) forces the in place output F32 up front.
        static_f32_seed_positions_.push_back(
            {hlo, operand_and_output_index.second});
        continue;
      }
      if (value_type == BF16) {
        continue;  // A BF16 value imposes nothing on the output.
      }
      value_to_inplace_outputs_[value].push_back(
          HloPosition{hlo, operand_and_output_index.second});
    }
  }
}

void BFloat16Propagation::AddKeepPrecisionSeeds(HloInstruction* hlo) {
  if (!ShouldKeepPrecisionUnchangedCached(hlo)) {
    return;
  }
  // These positions always read as F32, unless a call boundary push can
  // still mark them BF16.
  ShapeUtil::ForEachSubshape(
      hlo->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.IsArray() && OutputTypeAfterChange(hlo, index) == F32 &&
            !bf16_pushable_positions_.contains({hlo, index})) {
          static_f32_seed_positions_.push_back({hlo, index});
        }
      });
}

void BFloat16Propagation::AddScanCarryEdges(HloInstruction* hlo) {
  if (!IsAssociativeScan(hlo)) {
    return;
  }
  auto* scan = Cast<HloScanInstruction>(hlo);
  const int64_t num_carries = scan->num_carries();
  const int64_t num_inputs = hlo->operand_count() - num_carries;
  const int64_t num_outputs =
      hlo->shape().IsTuple() ? hlo->shape().tuple_shapes().size() - num_carries
                             : 1 - num_carries;
  HloComputation* body = hlo->to_apply();
  HloInstruction* body_root = body->root_instruction();
  auto carry_index = [&](const Shape& shape, int64_t i) -> ShapeIndex {
    return shape.IsTuple() ? ShapeIndex{num_outputs + i} : ShapeIndex{};
  };
  auto is_leaf_array = [&](const HloInstruction* loc, const ShapeIndex& index) {
    auto subshape = ShapeUtil::TryGetSubshape(loc->shape(), index);
    return subshape.ok() && (*subshape)->IsArray();
  };
  for (int64_t i = 0; i < num_carries; ++i) {
    HloInstruction* carry_init = hlo->mutable_operand(num_inputs + i);
    HloInstruction* body_param = body->parameter_instruction(num_inputs + i);
    const ShapeIndex root_carry_index = carry_index(body_root->shape(), i);
    const ShapeIndex result_carry_index = carry_index(hlo->shape(), i);
    // Skip carries whose four locations are not all leaf arrays (e.g. tuple
    // carries); they have no single element type to align.
    if (!is_leaf_array(carry_init, {}) || !is_leaf_array(body_param, {}) ||
        !is_leaf_array(body_root, root_carry_index) ||
        !is_leaf_array(hlo, result_carry_index)) {
      continue;
    }
    scan_carry_edges_[HloPosition{carry_init, ShapeIndex{}}].push_back(
        {HloPosition{body_param, ShapeIndex{}},
         HloPosition{body_root, root_carry_index},
         HloPosition{hlo, result_carry_index}});
  }
}

void BFloat16Propagation::AddValueSeedsAndUseEdges(const HloValue* value) {
  const PrimitiveType value_type = value->shape().element_type();
  if (value_type != F32) {
    if (value_type != BF16) {
      // A value that is neither F32 nor BF16 forces every position it
      // appears at.
      for (const HloPosition& position : value->positions()) {
        static_f32_seed_positions_.push_back(position);
      }
    }
    return;
  }
  HloInstruction* defining_instruction = value->defining_instruction();
  const ShapeIndex& defining_index = value->defining_position().index;
  // An unmarked value stays F32, unless a call boundary push can still
  // mark it.
  if (values_that_must_be_kept_as_f32_.contains(value) ||
      (OutputTypeAfterChange(defining_instruction, defining_index) == F32 &&
       !bf16_pushable_positions_.contains(
           {defining_instruction, defining_index}))) {
    static_f32_seed_values_.push_back(value);
  }
  for (const HloUse& use : value->GetUses()) {
    AddEdgesForUse(value, use);
  }
}

void BFloat16Propagation::AddEdgesForUse(const HloValue* value,
                                         const HloUse& use) {
  if (!ContainsKey(instructions_visited_in_backward_pass_, use.instruction)) {
    return;  // AllUsersConsumeBF16 skips unvisited users.
  }
  if (use.instruction->HasSideEffectNoRecurse()) {
    static_f32_seed_values_.push_back(value);
    return;
  }
  auto add_reader = [&](HloInstruction* instruction, const ShapeIndex& index) {
    use_edges_[HloPosition{instruction, index}].push_back(value);
  };
  // Uses of called computations read the callee parameter position.
  bool handled = true;
  switch (use.instruction->opcode()) {
    case HloOpcode::kFusion:
      add_reader(use.instruction->fused_parameter(use.operand_number),
                 use.operand_index);
      break;
    case HloOpcode::kWhile:
      add_reader(use.instruction->while_condition()->parameter_instruction(
                     use.operand_number),
                 use.operand_index);
      add_reader(use.instruction->while_body()->parameter_instruction(
                     use.operand_number),
                 use.operand_index);
      break;
    case HloOpcode::kConditional:
      if (use.operand_number > 0) {
        add_reader(use.instruction->branch_computation(use.operand_number - 1)
                       ->parameter_instruction(0),
                   use.operand_index);
      }
      break;
    case HloOpcode::kAsyncStart:
      if (HloInstruction::IsThreadIncluded(
              use.instruction->async_execution_thread(), execution_threads_)) {
        HloComputation* wrapped_comp =
            use.instruction->async_wrapped_computation();
        if (wrapped_comp == nullptr) {
          // AllUsersConsumeBF16 statically fails such uses.
          static_f32_seed_values_.push_back(value);
          return;
        }
        add_reader(wrapped_comp->parameter_instruction(use.operand_number),
                   use.operand_index);
      } else {
        handled = false;
      }
      break;
    case HloOpcode::kCall:
      add_reader(use.instruction->to_apply()->parameter_instruction(
                     use.operand_number),
                 use.operand_index);
      break;
    case HloOpcode::kScan:
      if (IsAssociativeScan(use.instruction)) {
        add_reader(use.instruction->to_apply()->parameter_instruction(
                       use.operand_number),
                   use.operand_index);
      } else {
        handled = false;
      }
      break;
    case HloOpcode::kAsyncDone:
      break;  // async-done consumes whatever async-start gives it.
    default:
      handled = false;
      break;
  }
  if (handled) {
    return;
  }
  if (use.instruction->opcode() == HloOpcode::kBitcast &&
      use.instruction->operand(use.operand_number)->shape().element_type() ==
          BF16) {
    return;
  }
  if (bfloat16_support_->EffectiveOperandPrecisionIsLowPrecision(
          *use.instruction, use.operand_number)) {
    return;
  }
  if (!bfloat16_support_->EffectiveOperandPrecisionIsOutputPrecision(
          *use.instruction, use.operand_number)) {
    static_f32_seed_values_.push_back(value);
    return;
  }
  // Forwarding users read their own output position.
  ShapeIndex use_output_index;
  if (use.instruction->opcode() == HloOpcode::kTuple ||
      (use.instruction->opcode() == HloOpcode::kAllReduce &&
       use.instruction->shape().IsTuple())) {
    use_output_index.push_back(use.operand_number);
    for (int64_t i : use.operand_index) {
      use_output_index.push_back(i);
    }
  } else if (use.instruction->opcode() == HloOpcode::kGetTupleElement) {
    for (int64_t i = 1; i < use.operand_index.size(); ++i) {
      use_output_index.push_back(use.operand_index[i]);
    }
  } else {
    use_output_index = use.operand_index;
  }
  // What the user reads as depends on the read position's type. An original
  // F32 array can go either way, so it becomes an edge. A tuple (variadic
  // sort, tuple shaped collectives) or non BF16 array (widening convert) can
  // never read as BF16, so the use fails statically. An original BF16 array
  // always reads as BF16.
  auto read_subshape =
      ShapeUtil::TryGetSubshape(use.instruction->shape(), use_output_index);
  if (!read_subshape.ok() || !(*read_subshape)->IsArray() ||
      ((*read_subshape)->element_type() != F32 &&
       (*read_subshape)->element_type() != BF16)) {
    static_f32_seed_values_.push_back(value);
    return;
  }
  if ((*read_subshape)->element_type() == BF16) {
    return;
  }
  // A forwarder like a same type bitcast records a use that reads a position
  // of `value` itself, which only the value could reach: an edge would be a
  // dead self loop. If that position is unmarked it can never become BF16
  // (its own read fails), so the value must stay F32; if it is marked, the
  // read is consistent whichever way the value resolves.
  const HloValueSet& read_set =
      dataflow_->GetValueSet(use.instruction, use_output_index);
  for (const HloValue* read_value : read_set.values()) {
    if (read_value == value) {
      if (OutputTypeAfterChange(use.instruction, use_output_index) == F32) {
        static_f32_seed_values_.push_back(value);
      }
      return;
    }
  }
  add_reader(use.instruction, use_output_index);
}

void BFloat16Propagation::ConstrainValueToF32(const HloValue* value) {
  if (!f32_values_.insert(value).second) {
    return;
  }
  value_queue_.push_back(value);
}

void BFloat16Propagation::ConstrainPositionToF32(const HloPosition& position) {
  if (!included_computations_.contains(position.instruction->parent())) {
    return;
  }
  if (!f32_positions_.insert(position).second) {
    return;
  }
  position_queue_.push_back(position);
}

void BFloat16Propagation::PropagateFromValue(const HloValue* value) {
  for (const HloPosition& position : value->positions()) {
    ConstrainPositionToF32(position);
  }
  auto it = value_to_inplace_outputs_.find(value);
  if (it != value_to_inplace_outputs_.end()) {
    for (const HloPosition& position : it->second) {
      ConstrainPositionToF32(position);
    }
  }
}

void BFloat16Propagation::PropagateFromPosition(const HloPosition& position) {
  HloInstruction* instr = position.instruction;
  const ShapeIndex& index = position.index;
  const Shape& subshape = ShapeUtil::GetSubshape(instr->shape(), index);
  // The F32 values this position aliases.
  if (!ShouldKeepPrecisionUnchangedCached(instr) && subshape.IsArray() &&
      (subshape.element_type() == F32 || subshape.element_type() == BF16)) {
    for (const HloValue* value :
         dataflow_->GetValueSet(instr, index).values()) {
      if (value->shape().element_type() == F32) {
        ConstrainValueToF32(value);
      }
    }
  }
  if (!subshape.IsArray() || subshape.element_type() != F32) {
    return;  // Only F32 subshapes can read as F32.
  }
  // The values whose AllUsersConsumeBF16 check reads this position.
  auto readers_it = use_edges_.find(position);
  if (readers_it != use_edges_.end()) {
    for (const HloValue* value : readers_it->second) {
      ConstrainValueToF32(value);
    }
  }
  // Scan carry alignment.
  auto align_it = scan_carry_edges_.find(position);
  if (align_it != scan_carry_edges_.end()) {
    for (const auto& targets : align_it->second) {
      for (const HloPosition& target : targets) {
        ConstrainPositionToF32(target);
      }
    }
  }
  PushAcrossCallBoundaries(position);
}

void BFloat16Propagation::PushAcrossCallBoundaries(
    const HloPosition& position) {
  const ShapeIndex& index = position.index;
  auto roots_it = push_roots_.find(position.instruction);
  if (roots_it != push_roots_.end()) {
    for (HloInstruction* root : roots_it->second) {
      auto root_subshape = ShapeUtil::TryGetSubshape(root->shape(), index);
      if (!root_subshape.ok() || !(*root_subshape)->IsArray()) {
        continue;
      }
      // The root push also constrains the root's F32 values.
      ConstrainPositionToF32({root, index});
      for (const HloValue* value :
           dataflow_->GetValueSet(root, index).values()) {
        if (value->shape().element_type() == F32) {
          ConstrainValueToF32(value);
        }
      }
    }
  }
  auto params_it = push_params_.find(position.instruction);
  if (params_it != push_params_.end()) {
    for (HloInstruction* parameter : params_it->second) {
      auto param_subshape =
          ShapeUtil::TryGetSubshape(parameter->shape(), index);
      if (!param_subshape.ok() || !(*param_subshape)->IsArray()) {
        continue;
      }
      ConstrainPositionToF32({parameter, index});
    }
  }
}

void BFloat16Propagation::PropagateF32Constraints() {
  f32_values_.reserve(dataflow_->values().size());
  for (const HloValue* value : static_f32_seed_values_) {
    ConstrainValueToF32(value);
  }
  for (const HloPosition& position : static_f32_seed_positions_) {
    ConstrainPositionToF32(position);
  }
  while (!value_queue_.empty() || !position_queue_.empty()) {
    if (!value_queue_.empty()) {
      const HloValue* value = value_queue_.front();
      value_queue_.pop_front();
      PropagateFromValue(value);
      continue;
    }
    const HloPosition position = position_queue_.front();
    position_queue_.pop_front();
    PropagateFromPosition(position);
  }
}

void BFloat16Propagation::MaterializeResolvedPrecisions(HloModule* module) {
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads_)) {
    for (HloInstruction* hlo : computation->MakeInstructionPostOrder()) {
      const bool adjusted = !ShouldKeepPrecisionUnchangedCached(hlo);
      ShapeUtil::ForEachSubshape(hlo->shape(), [&](const Shape& subshape,
                                                   const ShapeIndex& index) {
        if (!subshape.IsArray() || subshape.element_type() != F32) {
          return;  // Only originally-F32 positions can change to BF16.
        }
        const bool must_be_f32 = f32_positions_.contains({hlo, index});
        if (!adjusted && !bf16_pushable_positions_.contains({hlo, index})) {
          // Nothing may mark this position BF16, but an F32 push still
          // removes a BF16 mark introduced by the backward pass.
          if (must_be_f32) {
            AddToOrRemoveFromBF16ChangeSet(hlo, index, F32);
          }
          return;
        }
        AddToOrRemoveFromBF16ChangeSet(hlo, index, must_be_f32 ? F32 : BF16);
      });
    }
  }
}

void BFloat16Propagation::ResolveInconsistencyOfAliasingBuffers(
    HloModule* module) {
  BuildF32ConstraintGraph(module);
  PropagateF32Constraints();
  MaterializeResolvedPrecisions(module);
}

absl::Status BFloat16Propagation::ResolveInconsistentFusions(
    HloModule* module) {
  // We could have changed a fusion computation's root shape to have a different
  // precision than the fusion node's output, if the fusion root does not
  // define a buffer (e.g., a tuple). Now we add conversions after such fusion
  // roots to make them match the fusion output. If the fusion output is a
  // (possibly nested) tuple, we first create get-tuple-elements, then convert
  // the unmatching leaf nodes, and finally create a new tuple as the fusion
  // computation's root. If tuples and get-tuple-elements are created, we will
  // run tuple simplifier and dead code elimination at the end (dead code is not
  // allowed in fusion computation). E.g.,
  //
  // (1)             (2)             (3)
  // a  b            a  b            a  b
  // |\ |            |\ |            |\ |
  // \ add   ->      |add    ->      | add
  //  \ |            \ |        convert |
  //  tuple         tuple             \ |
  //                 / \              tuple
  //               gte gte
  //                |   |
  //           convert  |
  //                 \  /
  //                 tuple
  // (1) a is F32 but tuple is BF16
  // (2) after adding conversion
  // (3) after tuple simplifier and DCE.
  for (auto computation :
       module->MakeComputationPostOrder(execution_threads_)) {
    auto insts = computation->MakeInstructionPostOrder();
    for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
      auto hlo = *inst_it;
      if (hlo->opcode() != HloOpcode::kFusion) {
        continue;
      }
      auto fusion_computation = hlo->fused_instructions_computation();
      auto fusion_root = fusion_computation->root_instruction();
      if (ShapeUtil::Compatible(fusion_root->shape(), hlo->shape())) {
        continue;
      }
      ShapeTree<HloInstruction*> converted_outputs(hlo->shape());
      // Deep copy the fusion root, and convert a leaf node only if its shape
      // does not match the fusion output.
      ABSL_ASSIGN_OR_RETURN(
          HloInstruction * copy,
          fusion_computation->DeepCopyInstructionWithCustomCopier(
              fusion_root,
              [hlo](HloInstruction* leaf, const ShapeIndex& leaf_index,
                    HloComputation* comp) {
                const Shape& hlo_subshape =
                    ShapeUtil::GetSubshape(hlo->shape(), leaf_index);
                if (ShapeUtil::Compatible(leaf->shape(), hlo_subshape)) {
                  return leaf;
                }
                return comp->AddInstruction(
                    HloInstruction::CreateConvert(hlo_subshape, leaf));
              }));
      fusion_computation->set_root_instruction(copy);
    }
  }
  return absl::OkStatus();
}

absl::Status BFloat16Propagation::ResolveConvertedConstants(HloModule* module) {
  // We may have converted some constants from F32 to BF16, so adjust the
  // constant literals in such cases. We do this here instead of when the
  // constant node's is changed because 1) the HloInstruction interface does not
  // allow resetting the literal so we have to create a new kConstant
  // instruction to replace the old one, which invalidates dataflow analysis,
  // and 2) it's possible that a kConstant's output gets changed to BF16 at the
  // beginning but later on adjusted back to F32, so converting literals here
  // can avoid repeated conversions.
  //
  // TODO(b/73833576): Consider resetting literal in HloInstruction.
  for (auto computation :
       module->MakeComputationPostOrder(execution_threads_)) {
    for (auto hlo : computation->MakeInstructionPostOrder()) {
      if (hlo->opcode() != HloOpcode::kConstant) {
        continue;
      }
      if (!Shape::Equal().MinorToMajorOnlyInLayout()(hlo->literal().shape(),
                                                     hlo->shape())) {
        ABSL_ASSIGN_OR_RETURN(auto converted_literal,
                         hlo->literal().ConvertToShape(hlo->shape()));
        auto new_constant = computation->AddInstruction(
            HloInstruction::CreateConstant(std::move(converted_literal)));
        UpdateLayout(new_constant->mutable_shape());
        ABSL_RETURN_IF_ERROR(hlo->ReplaceAllUsesWith(new_constant));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status BFloat16Propagation::SkipNoopConversions(HloModule* module) {
  for (auto computation : module->computations(execution_threads_)) {
    for (auto hlo : computation->MakeInstructionPostOrder()) {
      if (hlo->opcode() != HloOpcode::kConvert) {
        continue;
      }
      auto source = hlo->mutable_operand(0);
      if (!ShapeUtil::Equal(source->shape(), hlo->shape())) {
        continue;
      }
      const bool is_root = hlo == computation->root_instruction();
      ABSL_RETURN_IF_ERROR(hlo->ReplaceAllUsesWith(source));
      if (is_root) {
        computation->set_root_instruction(source);
      }
    }
  }
  return absl::OkStatus();
}

// The algorithm first does a forward pass (parameters to root) to determine a
// set of instructions to consider using bfloat16, then does a backward pass to
// determine the precisions of those instructions according to the need of
// their users. During the backward pass, the potential changes are stored in
// changes_to_bf16_. A final resolving pass finds, with one BFS over the F32
// constraint graph, everything that must stay F32 so that aliasing buffers
// agree on precision. The remaining changes are then applied to the HLOs.
absl::StatusOr<bool> BFloat16Propagation::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  consider_using_bfloat16_.clear();
  instructions_visited_in_backward_pass_.clear();
  computations_visited_in_backward_pass_.clear();
  values_that_must_be_kept_as_f32_.clear();
  caller_counts_.clear();
  changes_to_bf16_.clear();
  value_queue_.clear();
  position_queue_.clear();
  f32_values_.clear();
  f32_positions_.clear();
  included_computations_.clear();
  use_edges_.clear();
  value_to_inplace_outputs_.clear();
  static_f32_seed_values_.clear();
  static_f32_seed_positions_.clear();
  bf16_pushable_positions_.clear();
  push_roots_.clear();
  push_params_.clear();
  scan_carry_edges_.clear();
  keep_precision_unchanged_cache_.clear();
  inplace_input_output_pairs_cache_.clear();
  changed_ = false;
  execution_threads_ = execution_threads;

  auto computations_topological_order =
      module->MakeComputationPostOrder(execution_threads_);

  // Before running the propagation pass, we insert copies (kConvert to the same
  // type) of F32 inputs to while loops. This prevents other uses of the same
  // input from aliasing the while loop input/output, so that there's greater
  // chance to use BF16 inside the loop. If some of these added copies do not
  // help, they will remain F32 after BF16 propagation and will be removed since
  // they are no-ops.
  for (auto computation : computations_topological_order) {
    for (auto inst : computation->MakeInstructionPostOrder()) {
      if (inst->opcode() != HloOpcode::kWhile) {
        continue;
      }

      auto operand = inst->mutable_operand(0);
      ABSL_ASSIGN_OR_RETURN(
          HloInstruction * copy,
          computation->DeepCopyInstructionWithCustomCopier(
              operand, [](HloInstruction* leaf, const ShapeIndex& leaf_index,
                          HloComputation* comp) {
                if (leaf->shape().element_type() != F32) {
                  return leaf;
                }
                return comp->AddInstruction(
                    HloInstruction::CreateConvert(leaf->shape(), leaf));
              }));
      ABSL_RETURN_IF_ERROR(operand->ReplaceUseWith(inst, copy));
    }
  }

  ABSL_ASSIGN_OR_RETURN(dataflow_, HloDataflowAnalysis::Run(*module));

  // The first step is a forward pass (parameters to root), where we determine
  // the potential candidate instructions to use bfloat16 in the outputs that
  // are not likely to cause overhead from extra explicit conversions. This is
  // done forwardly because we determine whether an HLO is a candidate partially
  // based on whether its operands are candidates.
  for (auto computation : computations_topological_order) {
    for (auto inst : computation->MakeInstructionPostOrder()) {
      if (InstructionIsCandidateForBF16Output(inst)) {
        consider_using_bfloat16_.insert(inst);
      }
    }
  }

  // The second step is a backward pass (root to parameters), where we modify
  // the precisions of the instructions identified in the first step when
  // feasible. This is done backwardly because we determine the precision of an
  // HLO's output based on how it is later used.
  //
  // The precision of an instruction is determined by its users, so we do the
  // propagation in reverse topological order.
  for (auto comp_it = computations_topological_order.rbegin();
       comp_it != computations_topological_order.rend(); ++comp_it) {
    if (ContainsKey(computations_visited_in_backward_pass_, *comp_it)) {
      continue;
    }
    auto insts = (*comp_it)->MakeInstructionPostOrder();
    for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
      DetermineInstructionPrecision(*inst_it,
                                    /*skip_parameters=*/true);
    }
    computations_visited_in_backward_pass_.insert(*comp_it);
  }

  // It's possible that an instruction does not define a buffer, but the
  // defining instruction's shape has changed. So we need to adjust the output
  // shapes of instructions according to the HLO values they refer to.
  ResolveInconsistencyOfAliasingBuffers(module);

  // Apply the changes in changes_to_bf16_.
  for (auto& change : changes_to_bf16_) {
    auto inst = change.first;
    // It is possible that we marked inst to change precision even if it is an
    // unsupported change, when inst is the root of a fusion computation and it
    // has to match the fusion node's output precision. We do a convert instead
    // of in-place change for such cases.
    if (ShouldKeepPrecisionUnchanged(inst)) {
      auto users = inst->users();
      bool is_root = inst == inst->parent()->root_instruction();
      ABSL_ASSIGN_OR_RETURN(
          HloInstruction * copy,
          inst->parent()->DeepCopyInstructionWithCustomCopier(
              inst, [&](HloInstruction* leaf, const ShapeIndex& leaf_index,
                        HloComputation* comp) {
                if (!ContainsKey(change.second,
                                 ShapeUtil::GetMutableSubshape(
                                     inst->mutable_shape(), leaf_index))) {
                  return leaf;
                }
                auto converted_shape =
                    ShapeUtil::ChangeElementType(leaf->shape(), BF16);
                UpdateLayout(&converted_shape);
                return comp->AddInstruction(
                    HloInstruction::CreateConvert(converted_shape, leaf));
              }));
      for (auto user : users) {
        ABSL_RETURN_IF_ERROR(inst->ReplaceUseWithDifferentShape(user, copy));
      }
      if (is_root) {
        inst->parent()->set_root_instruction(copy,
                                             /*accept_different_shape=*/true);
      }
      continue;
    }
    for (const auto& entry : change.second) {
      auto subshape = entry.first;
      CHECK_EQ(subshape->element_type(), F32);
      subshape->set_element_type(BF16);
      UpdateLayout(subshape);
      changed_ = true;
    }
  }

  // Removes redundant HLOs added by this pass, either when inserting
  // de-aliasing copies to while loop inputs, or later when converting output
  // types.
  auto clean_up = [this, module]() -> absl::Status {
    ABSL_RETURN_IF_ERROR(SkipNoopConversions(module));
    TupleSimplifier tuple_simplifier;
    ABSL_RETURN_IF_ERROR(tuple_simplifier.Run(module, execution_threads_).status());
    HloDCE dce;
    ABSL_RETURN_IF_ERROR(dce.Run(module, execution_threads_).status());
    return absl::OkStatus();
  };

  if (!changed_) {
    ABSL_RETURN_IF_ERROR(clean_up());
    return false;
  }

  ABSL_RETURN_IF_ERROR(ResolveInconsistentFusions(module));
  ABSL_RETURN_IF_ERROR(ResolveInconsistentScans(module));
  ABSL_RETURN_IF_ERROR(ResolveConvertedConstants(module));

  ABSL_RETURN_IF_ERROR(clean_up());
  return true;
}

absl::Status BFloat16Propagation::ResolveInconsistentScans(HloModule* module) {
  // Each body root slot must match the corresponding scan output slot, as
  // enforced by the HloVerifier. The resolving pass can leave them
  // different, because no dataflow alias ties the two slots together:
  //   1. Per-step output slots: the scan output went BF16 while the body
  //      root value had to stay F32 for another use.
  //   2. Carry slots with a BF16 init and F32 body/scan slots.
  // Forcing the two sides to agree would change the precision seen by the
  // other users of the body root value, so insert a convert on the body
  // root slot instead, like ResolveInconsistentFusions.
  for (auto computation :
       module->MakeComputationPostOrder(execution_threads_)) {
    for (auto inst : computation->MakeInstructionPostOrder()) {
      if (!IsAssociativeScan(inst)) {
        continue;
      }
      HloComputation* body = inst->to_apply();
      HloInstruction* body_root = body->root_instruction();
      const Shape& scan_shape = inst->shape();
      if (ShapeUtil::Compatible(body_root->shape(), scan_shape)) {
        continue;
      }
      // Use DeepCopyInstructionWithCustomCopier to walk every leaf of the body
      // root and insert a precision-changing convert wherever the leaf type
      // disagrees with the corresponding scan output slot. This mirrors
      // ResolveInconsistentFusions and works regardless of body_root's opcode
      // (kTuple, kParameter, kCall, kFusion, ...): instead of mutating
      // body_root's operands directly we materialize a fresh GTE/Convert/Tuple
      // tree and swap it in as the new root. Body root tuple slot i maps to
      // scan output tuple slot i for both per-step outputs (i < num_outputs)
      // and carries (i >= num_outputs); carries simply pass through unchanged.
      ABSL_ASSIGN_OR_RETURN(
          HloInstruction * new_root,
          body->DeepCopyInstructionWithCustomCopier(
              body_root, [this, &scan_shape](HloInstruction* leaf,
                                             const ShapeIndex& leaf_index,
                                             HloComputation* comp) {
                const Shape& scan_subshape =
                    ShapeUtil::GetSubshape(scan_shape, leaf_index);
                if (ShapeUtil::Compatible(leaf->shape(), scan_subshape)) {
                  return leaf;
                }
                Shape converted_shape = leaf->shape();
                converted_shape.set_element_type(scan_subshape.element_type());
                UpdateLayout(&converted_shape);
                return comp->AddInstruction(
                    HloInstruction::CreateConvert(converted_shape, leaf));
              }));
      body->set_root_instruction(new_root,
                                 /*accept_different_shape=*/true);
    }
  }
  return absl::OkStatus();
}

PrimitiveType BFloat16Propagation::OutputTypeAfterChange(
    HloInstruction* hlo, const ShapeIndex& index) const {
  Shape* subshape = ShapeUtil::GetMutableSubshape(hlo->mutable_shape(), index);
  const PrimitiveType type_on_hlo = subshape->element_type();
  if (type_on_hlo != F32) {
    return type_on_hlo;
  }
  auto it = changes_to_bf16_.find(hlo);
  if (it == changes_to_bf16_.end()) {
    return type_on_hlo;
  }
  return ContainsKey(it->second, subshape) ? BF16 : F32;
}

void BFloat16Propagation::AddToOrRemoveFromBF16ChangeSet(
    HloInstruction* hlo, const ShapeIndex& index, PrimitiveType target_type) {
  if (target_type == BF16) {
    changes_to_bf16_[hlo].emplace(
        ShapeUtil::GetMutableSubshape(hlo->mutable_shape(), index), index);
    return;
  }
  CHECK_EQ(target_type, F32);
  auto it = changes_to_bf16_.find(hlo);
  if (it == changes_to_bf16_.end()) {
    return;
  }
  it->second.erase(ShapeUtil::GetMutableSubshape(hlo->mutable_shape(), index));
  if (it->second.empty()) {
    changes_to_bf16_.erase(it);
  }
}

bool BFloat16Propagation::ShouldKeepPrecisionUnchangedCached(
    const HloInstruction* inst) {
  auto it = keep_precision_unchanged_cache_.find(inst);
  if (it != keep_precision_unchanged_cache_.end()) {
    return it->second;
  }
  const bool result = ShouldKeepPrecisionUnchanged(inst);
  keep_precision_unchanged_cache_.emplace(inst, result);
  return result;
}

const std::vector<std::pair<HloOperandIndex, ShapeIndex>>&
BFloat16Propagation::GetInPlaceInputOutputPairsCached(
    const HloInstruction* hlo) {
  auto it = inplace_input_output_pairs_cache_.find(hlo);
  if (it == inplace_input_output_pairs_cache_.end()) {
    it = inplace_input_output_pairs_cache_
             .emplace(hlo, alias_info_->GetInPlaceInputOutputPairs(hlo))
             .first;
  }
  return it->second;
}

}  // namespace xla
