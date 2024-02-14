/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/model/coalescing_analysis.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

// Returns true if all input reads are coalesced. If consumer is not nullptr,
// producer and consumer are considered as one fusion, otherwise it's only the
// producer.
bool IsReadCoalescedHeuristic(HloFusionAnalysis::EmitterFusionKind fusion_kind,
                              const HloInstruction* producer,
                              const HloInstruction* consumer) {
  // Transposing minor dimension breaks coalescing.
  if (fusion_kind != HloFusionAnalysis::EmitterFusionKind::kTranspose) {
    auto is_broadcast = [&](const HloInstruction* instr) {
      while (true) {
        if (instr->opcode() == HloOpcode::kBroadcast) return true;
        if (instr->operand_count() != 1) return false;
        if (instr->opcode() != HloOpcode::kBitcast && !instr->IsElementwise()) {
          return false;
        }
        instr = instr->operand(0);
      }
    };
    auto is_bad_transpose = [&](const HloInstruction* instr) {
      if (instr->opcode() == HloOpcode::kFusion) {
        for (auto* instr : instr->fused_instructions()) {
          // Hack: we allow transposes of broadcasts.
          if (TransposesMinorDimension(instr) &&
              !is_broadcast(instr->operand(0))) {
            return true;
          }
        }
        return false;
      }
      return TransposesMinorDimension(instr);
    };
    if (is_bad_transpose(producer)) return false;
    if (consumer && is_bad_transpose(consumer)) return false;
  }
  // Fusing two row reductions breaks coalescing.
  if (fusion_kind == HloFusionAnalysis::EmitterFusionKind::kReduction &&
      IsInputFusibleReduction(*producer) && consumer &&
      IsInputFusibleReduction(*consumer)) {
    return false;
  }
  return true;
}

namespace {

using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::MLIRContext;

bool IsCoalesced(const IndexingMap& thread_id_to_input_indexing_map) {
  MLIRContext* mlir_context = thread_id_to_input_indexing_map.GetMLIRContext();
  AffineExpr thread_x_dim = mlir::getAffineDimExpr(
      KernelFusionInterface::kIndexingMapThreadIdxDims[0], mlir_context);
  AffineExpr c0 = mlir::getAffineConstantExpr(0, mlir_context);
  IndexingMap thread_x_first_32_elements{
      AffineMap::get(1, 0, {thread_x_dim, c0, c0, c0, c0, c0}, mlir_context),
      {Range{0, 31}},
      {}};
  IndexingMap thread_x_to_linearized_input =
      thread_x_first_32_elements * thread_id_to_input_indexing_map;
  thread_x_to_linearized_input.Simplify();
  thread_x_to_linearized_input.RemoveUnusedSymbols();
  // That's quite a naive condition. It would be better to estimate the number
  // of memory transactions needed to cover the elements of the indexing map's
  // codomain.
  return thread_x_to_linearized_input.GetAffineMap().getResult(0) ==
         thread_x_dim;
}

}  // namespace

CoalescingAnalysis::CoalescingAnalysis(
    const HloInstruction* instr,
    HloFusionAnalysis::EmitterFusionKind fusion_kind,
    KernelFusionInterface* fusion_interface, mlir::MLIRContext* mlir_context,
    bool use_heuristic) {
  if (!use_heuristic && ComputeCoalescingForAllOperands(
                            instr, /*optional_producer=*/nullptr, fusion_kind,
                            fusion_interface, mlir_context)) {
    return;
  }
  // If ComputeCoalescingForAllOperands fails, fallback to using the heuristic.
  is_coalesced_computed_by_heuristic_ =
      IsReadCoalescedHeuristic(fusion_kind, instr);
}

CoalescingAnalysis::CoalescingAnalysis(
    const HloInstruction* producer, const HloInstruction* consumer,
    HloFusionAnalysis::EmitterFusionKind fusion_kind,
    KernelFusionInterface* fusion_interface, mlir::MLIRContext* mlir_context,
    bool use_heuristic) {
  if (!use_heuristic &&
      ComputeCoalescingForAllOperands(producer, consumer, fusion_kind,
                                      fusion_interface, mlir_context)) {
    return;
  }
  // If ComputeCoalescingForAllOperands fails, fallback to using the heuristic.
  is_coalesced_computed_by_heuristic_ =
      IsReadCoalescedHeuristic(fusion_kind, producer, consumer);
}

// Returns a linearized shape, i.e. tensor<num_elements(input) x element_type>.
Shape GetLinearizedShape(const Shape& shape) {
  std::vector<int64_t> dims{ShapeUtil::ElementsIn(shape)};
  auto result = Shape(shape.element_type(), dims,
                      absl::InlinedVector<bool, 4>(dims.size(), false), {});
  *result.mutable_layout() = xla::Layout({0});
  return result;
}

bool CoalescingAnalysis::ComputeCoalescingForAllOperands(
    const HloInstruction* instr, const HloInstruction* optional_producer,
    HloFusionAnalysis::EmitterFusionKind fusion_kind,
    KernelFusionInterface* fusion_interface, mlir::MLIRContext* mlir_context) {
  // Compute indexing from output to inputs for logical layout.
  auto instr_indexing = ComputeOutputToInputIndexing(instr, 0, mlir_context);
  auto instr_indexing_keyed_by_operands =
      GroupIndexingMapsByProducers(instr_indexing, instr);
  if (optional_producer) {
    DCHECK(FuseProducerConsumerOutputToInputIndexing(
        optional_producer, &instr_indexing_keyed_by_operands, mlir_context));
  }
  // Compute thread ID -> physical layout of output indexing map.
  std::optional<IndexingMap> thread_id_to_logical_output_map =
      fusion_interface->ComputeThreadIdToOutputIndexing(0, mlir_context);
  // If thread_id_to_physical_output_map is not defined, we return false. In
  // that case, we rely on heuristics to compute coalescing.
  if (!thread_id_to_logical_output_map.has_value()) {
    return false;
  }
  // For every operand compute thread ID -> physical layout of operand indexing
  // map.
  for (const auto& [operand, indexing_maps] :
       instr_indexing_keyed_by_operands) {
    const Shape& operand_shape = operand->shape();
    IndexingMap operand_logical_to_physical_map =
        GetIndexingMapFromLogicalToPhysicalLayout(operand_shape, mlir_context);
    IndexingMap operand_physical_to_linearized_shape = GetBitcastMap(
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
            operand_shape),
        GetLinearizedShape(operand_shape), mlir_context);
    IndexingMap operand_logical_to_linearized_physical_shape =
        operand_logical_to_physical_map * operand_physical_to_linearized_shape;
    operand_logical_to_linearized_physical_shape.Simplify();

    for (const IndexingMap& operand_indexing_map : indexing_maps) {
      // If one of the indexing maps for the operand is undefined, we remove all
      // indexing maps for it and store only the undefined one.
      if (operand_indexing_map.IsUndefined()) {
        coalescing_per_operand_[operand] = false;
        break;
      }
      IndexingMap physical_output_to_linearized_physical_input_map =
          operand_indexing_map * operand_logical_to_linearized_physical_shape;
      IndexingMap thread_id_to_linearized_physical_input_map =
          *thread_id_to_logical_output_map *
          physical_output_to_linearized_physical_input_map;
      thread_id_to_linearized_physical_input_map.Simplify();
      bool is_coalesced =
          IsCoalesced(thread_id_to_linearized_physical_input_map);
      auto [it, inserted] =
          coalescing_per_operand_.insert({operand, is_coalesced});
      if (!inserted) {
        it->second &= is_coalesced;
      }
      if (!is_coalesced) break;
    }
  }
  return true;
}

bool CoalescingAnalysis::IsReadCoalesced(const HloInstruction* operand) const {
  auto it = coalescing_per_operand_.find(operand);
  if (it == coalescing_per_operand_.end()) {
    return is_coalesced_computed_by_heuristic_;
  }
  return it->second;
}

}  // namespace gpu
}  // namespace xla
