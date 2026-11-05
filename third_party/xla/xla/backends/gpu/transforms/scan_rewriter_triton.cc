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

#include "xla/backends/gpu/transforms/scan_rewriter_triton.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

namespace {

// A tiled scan computes one row per block and loops over the scan dimension.
// For scan dimensions of at least kMinLongScanDimSize, it only beats
// the hierarchical scan of AssociativeScanRewriter with at least
// kMinRowsForLongScan rows (crossover at 64 rows on H100 and 32 on B200).
// Shorter scans are faster with Triton at any batch size.
constexpr int64_t kMinLongScanDimSize = 32768;
constexpr int64_t kMinRowsForLongScan = 64;

// Unfused data-movement or indexing ops that PriorityFusion can merge with the
// AssociativeScanRewriter tree into a kLoop/kInput fusion, but cannot fuse with
// kScan into Triton (kTranspose is omitted because Triton fuses it directly).
// Native fusions do not exist yet because this pass runs before PriorityFusion.
bool IsLoopDataMovementOrIndexingOp(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kConcatenate:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGather:
    case HloOpcode::kPad:
    case HloOpcode::kReverse:
    case HloOpcode::kScatter:
    case HloOpcode::kSlice:
      return true;
    default:
      return false;
  }
}

// Performs a bounded BFS from `scan`'s input (through non-scalar transparent
// producers) and GTE(0) users (through transparent consumers) to check whether
// the scan's fusion neighborhood reaches any op in
// `IsLoopDataMovementOrIndexingOp`.
bool TouchesLoopDataMovementOrIndexingOp(const HloScanInstruction* scan) {
  constexpr size_t kMaxNeighborhoodVisitCount = 32;
  absl::flat_hash_set<const HloInstruction*> visited;
  std::vector<const HloInstruction*> queue;
  auto enqueue = [&](const HloInstruction* instr) {
    if (visited.insert(instr).second) {
      queue.push_back(instr);
    }
  };
  auto search = [&](bool walk_operands) {
    for (size_t i = 0; i < queue.size() && i < kMaxNeighborhoodVisitCount;
         ++i) {
      const HloInstruction* curr = queue[i];
      if (IsLoopDataMovementOrIndexingOp(curr)) {
        return true;
      }
      if (!curr->IsElementwise() && curr->opcode() != HloOpcode::kBroadcast &&
          curr->opcode() != HloOpcode::kBitcast &&
          curr->opcode() != HloOpcode::kReshape) {
        continue;
      }
      if (walk_operands) {
        for (const HloInstruction* operand : curr->operands()) {
          if (!ShapeUtil::IsEffectiveScalar(operand->shape())) {
            enqueue(operand);
          }
        }
      } else {
        for (const HloInstruction* user : curr->users()) {
          enqueue(user);
        }
      }
    }
    return false;
  };

  enqueue(scan->operand(0));
  if (search(/*walk_operands=*/true)) {
    return true;
  }
  queue.clear();
  for (const HloInstruction* gte : scan->users()) {
    for (const HloInstruction* user : gte->users()) {
      enqueue(user);
    }
  }
  return search(/*walk_operands=*/false);
}

// Complements `IsTritonSupportedInstruction` with graph-level checks (non-root,
// only GTE 0 users) and profitability heuristics.
bool IsEligibleScan(const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kScan) {
    return false;
  }
  if (instr->IsRoot()) {
    return false;
  }
  if (!instr->shape().IsTuple()) {
    return false;
  }
  if (instr->shape().tuple_shapes().size() != 2) {
    return false;
  }

  const auto* scan = Cast<HloScanInstruction>(instr);
  const Shape& input_shape = scan->operand(0)->shape();
  int64_t rank = input_shape.dimensions().size();
  int64_t scan_dim = scan->scan_dimension();
  int64_t scan_dim_size = input_shape.dimensions(scan_dim);
  if (scan_dim_size <= 1) {
    return false;
  }
  // Restrict to the most minor dimension for coalescing. Triton can handle
  // non-minor scans, but they are generally less efficient.
  if (input_shape.has_layout() &&
      LayoutUtil::Minor(input_shape.layout(), 0) != scan_dim) {
    return false;
  }

  int64_t num_rows = 1;
  for (int64_t i = 0; i < rank; ++i) {
    if (i != scan_dim) {
      num_rows *= input_shape.dimensions(i);
    }
  }
  if (scan_dim_size >= kMinLongScanDimSize && num_rows < kMinRowsForLongScan) {
    return false;
  }

  for (const HloInstruction* user : scan->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return false;
    }
    if (user->tuple_index() != 0) {
      return false;
    }
  }

  // For small scans (up to 3 radix-16 levels in AssociativeScanRewriter), a
  // native loop/input fusion that absorbs surrounding data-movement or
  // indexing ops beats splitting the fusion around a Triton scan kernel,
  // unless the input is already a custom (Triton) fusion.
  constexpr int64_t kMaxSmallScanDimSize = 1024;
  const HloInstruction* input = scan->operand(0);
  bool has_custom_fusion_producer =
      input->opcode() == HloOpcode::kFusion &&
      input->fusion_kind() == HloInstruction::FusionKind::kCustom;
  if (scan_dim_size <= kMaxSmallScanDimSize && !has_custom_fusion_producer &&
      TouchesLoopDataMovementOrIndexingOp(scan)) {
    return false;
  }

  return true;
}

// Wraps the scan instruction and its output get-tuple-element into a custom
// Triton fusion computation (kind=__triton) and replaces all original uses.
absl::Status RewriteScanToTritonFusion(HloInstruction* scan) {
  HloComputation* parent = scan->parent();
  HloModule* module = parent->parent();
  const Shape& output_shape = scan->shape().tuple_shapes(0);

  HloComputation::Builder builder("triton_scan_computation");
  std::vector<HloInstruction*> parameters;
  parameters.reserve(scan->operand_count());
  for (int64_t i = 0; i < scan->operand_count(); ++i) {
    parameters.push_back(builder.AddInstruction(HloInstruction::CreateParameter(
        i, scan->operand(i)->shape(), absl::StrCat("param_", i))));
  }

  HloInstruction* cloned_scan = builder.AddInstruction(
      scan->CloneWithNewOperands(scan->shape(), parameters));
  HloInstruction* gte_0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(output_shape, cloned_scan, 0));
  HloComputation* fusion_computation =
      module->AddComputationAndUnifyNamesAndIds(builder.Build(gte_0),
                                                /*is_entry=*/false);

  HloInstruction* fusion = parent->AddInstruction(HloInstruction::CreateFusion(
      output_shape, HloInstruction::FusionKind::kCustom, scan->operands(),
      fusion_computation));
  module->SetAndUniquifyInstrName(fusion, "triton_scan");

  // The block-level config is left to PriorityFusion or
  // FusionBlockLevelRewriter.
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   fusion->backend_config<GpuBackendConfig>());
  gpu_config.mutable_fusion_backend_config()->set_kind(kTritonFusionKind);
  ABSL_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

  std::vector<HloInstruction*> users = scan->users();
  for (HloInstruction* user : users) {
    if (user->IsRoot()) {
      parent->set_root_instruction(fusion);
    }
    ABSL_RETURN_IF_ERROR(fusion->CopyAllControlDepsFrom(user));
    ABSL_RETURN_IF_ERROR(user->DropAllControlDeps());
    ABSL_RETURN_IF_ERROR(user->ReplaceAllUsesWith(fusion));
    ABSL_RETURN_IF_ERROR(parent->RemoveInstruction(user));
  }

  ABSL_RETURN_IF_ERROR(fusion->CopyAllControlDepsFrom(scan));
  ABSL_RETURN_IF_ERROR(scan->DropAllControlDeps());
  ABSL_RETURN_IF_ERROR(parent->RemoveInstruction(scan));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> ScanRewriterTriton::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  ABSL_RETURN_IF_ERROR(EnsureTritonSupportsComputeCapability(
      device_info_.gpu_compute_capability()));

  std::vector<HloInstruction*> scans_to_rewrite;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : comp->instructions()) {
      if (IsEligibleScan(instr) &&
          IsTritonSupportedInstruction(*instr,
                                       device_info_.gpu_compute_capability())) {
        scans_to_rewrite.push_back(instr);
      }
    }
  }

  for (HloInstruction* scan : scans_to_rewrite) {
    ABSL_RETURN_IF_ERROR(RewriteScanToTritonFusion(scan));
  }
  return !scans_to_rewrite.empty();
}

}  // namespace gpu
}  // namespace xla
