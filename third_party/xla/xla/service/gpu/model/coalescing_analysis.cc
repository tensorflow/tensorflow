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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
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
        if (instr->opcode() == HloOpcode::kBroadcast ||
            instr->opcode() == HloOpcode::kIota) {
          return true;
        }
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
          // Hack: we allow transposes of broadcasts or iotas.
          if (TransposesMinorDimension(instr) &&
              !is_broadcast(instr->operand(0))) {
            return true;
          }
        }
        return false;
      }
      // Hack: we allow transposes of broadcasts or iotas.
      return TransposesMinorDimension(instr) &&
             !is_broadcast(instr->operand(0));
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
using mlir::getAffineConstantExpr;
using mlir::MLIRContext;

// Performs backtracking to find all feasible dimensions, symbols that satisfy
// the constraints and then evaluates the affine map at those.
// For example, for the following indexing map:
//   (d0)[s0] -> (d0 + s0)
//   domain:
//   d0 in [0, 3]
//   s0 in [0, 1, 2]
//   s0 mod 2 in [0, 0]
// The function will compute the following indices [0, 2, 1, 3, 2, 4, 3, 5].
void FindAllIndices(const IndexingMap& thread_id_to_physical_index,
                    MLIRContext* mlir_context, int dim_id, int symbol_id,
                    std::vector<AffineExpr>* dimensions,
                    std::vector<AffineExpr>* symbols,
                    std::vector<int64_t>* indices) {
  if (dim_id < thread_id_to_physical_index.GetDimensionCount()) {
    Interval dim_range = thread_id_to_physical_index.GetDimensionBound(dim_id);
    for (int64_t dim_value = dim_range.lower; dim_value <= dim_range.upper;
         ++dim_value) {
      dimensions->push_back(getAffineConstantExpr(dim_value, mlir_context));
      FindAllIndices(thread_id_to_physical_index, mlir_context, dim_id + 1,
                     symbol_id, dimensions, symbols, indices);
      dimensions->pop_back();
    }
    return;
  }
  if (symbol_id < thread_id_to_physical_index.GetRangeVarsCount()) {
    Interval symbol_range =
        thread_id_to_physical_index.GetSymbolBound(symbol_id);
    for (int64_t symbol_value = symbol_range.lower;
         symbol_value <= symbol_range.upper; ++symbol_value) {
      symbols->push_back(getAffineConstantExpr(symbol_value, mlir_context));
      FindAllIndices(thread_id_to_physical_index, mlir_context, dim_id,
                     symbol_id + 1, dimensions, symbols, indices);
      symbols->pop_back();
    }
    return;
  }
  if (!thread_id_to_physical_index.ConstraintsSatisfied(*dimensions,
                                                        *symbols)) {
    return;
  }
  indices->push_back(
      thread_id_to_physical_index.Evaluate(*dimensions, *symbols).front());
}

// Computes contiguous intervals of accessed elements.
// For example, for an indexing map
//   (thread_x) -> (thread_x * 4 + s0 + (thread_x floordiv 16) * 1984)
//   d0 in [0, 31]
//   s0 in [0, 3]
// The intervals are [0, 63] and [2047, 2111].
// TODO(b/325613460): Make it faster than O(number of elements in the domain).
std::vector<Interval> FindContiguousIntervals(
    const IndexingMap& thread_id_to_physical_index) {
  CHECK(thread_id_to_physical_index.GetAffineMap().getNumResults() == 1)
      << "Expects an affine map that maps to 1D.";
  MLIRContext* mlir_context = thread_id_to_physical_index.GetMLIRContext();

  // Find all linear indices, sort and deduplicate them.
  std::vector<AffineExpr> dimensions, symbols;
  std::vector<int64_t> linear_indices;
  FindAllIndices(thread_id_to_physical_index, mlir_context,
                 /*dim_id=*/0,
                 /*symbol_id=*/0, &dimensions, &symbols, &linear_indices);
  std::sort(linear_indices.begin(), linear_indices.end());
  linear_indices.erase(
      std::unique(linear_indices.begin(), linear_indices.end()),
      linear_indices.end());

  // Scan over the sorted unique indices and combine them in intervals.
  std::vector<Interval> intervals;
  for (int i = 0, start, end; i < linear_indices.size(); ++i) {
    start = linear_indices[i++];
    end = start;
    while (i < linear_indices.size() && linear_indices[i] == end + 1) {
      ++end;
      ++i;
    }
    intervals.push_back(Interval{start, end});
  }
  return intervals;
}

int64_t CeilDiv(int64_t a, int64_t b) { return a / b + (a % b != 0); }

// Approximately estimate the number of memory transactions needed to load all
// elements in every range and compare it with the "ideal" number of memory
// transactions, i.e. total number of elements in all ranges / WarpSize().
// Note, that later we would need to take the element type into account.
bool EstimateCoalescingViaMemoryTransactionsCount(
    absl::Span<const Interval> intervals, PrimitiveType element_type) {
  constexpr int64_t kBytesPerMemoryTransaction = 128;
  int64_t type_size = ShapeUtil::ByteSizeOfPrimitiveType(element_type);
  int memory_transactions = 0;
  int total_num_elements = 0;
  for (const auto& range : intervals) {
    int64_t num_elements = range.upper - range.lower + 1;
    memory_transactions +=
        CeilDiv(num_elements * type_size, kBytesPerMemoryTransaction);
    total_num_elements += num_elements;
  }
  if (memory_transactions == 0) {
    return true;
  }
  int memory_transactions_lower_bound =
      CeilDiv(total_num_elements * type_size, kBytesPerMemoryTransaction);
  // The magic value chosen by an uneducated guess.
  constexpr float kIsCoalescedThreshold = 0.9;
  return memory_transactions_lower_bound >
         memory_transactions * kIsCoalescedThreshold;
}

bool IsCoalesced(const IndexingMap& thread_id_to_input_indexing_map,
                 PrimitiveType element_type) {
  // Undefined indexing maps, i.e. those for which we don't know the indexing
  // are assumed to be uncoalesced.
  if (thread_id_to_input_indexing_map.IsUndefined()) {
    return false;
  }
  // 0d constants are coalesced.
  if (thread_id_to_input_indexing_map.GetAffineMap().getNumResults() == 0) {
    return true;
  }
  MLIRContext* mlir_context = thread_id_to_input_indexing_map.GetMLIRContext();
  AffineExpr thread_x_dim = mlir::getAffineDimExpr(
      KernelFusionInterface::kIndexingMapThreadIdxDims[0], mlir_context);
  AffineExpr c0 = mlir::getAffineConstantExpr(0, mlir_context);
  IndexingMap thread_x_first_32_elements{
      AffineMap::get(1, 0, {thread_x_dim, c0, c0, c0, c0, c0}, mlir_context),
      {DimVar{{0, 31}}},
      /*range_vars=*/{},
      /*rt_vars=*/{}};
  IndexingMap thread_x_to_linearized_input =
      thread_x_first_32_elements * thread_id_to_input_indexing_map;

  // If RTVars are present, replace them with constants.
  if (thread_x_to_linearized_input.GetRTVarsCount() > 0) {
    llvm::SmallVector<AffineExpr, 2> symbol_replacements;
    for (int64_t symbol_id = 0;
         symbol_id < thread_x_to_linearized_input.GetRangeVarsCount();
         ++symbol_id) {
      symbol_replacements.push_back(
          mlir::getAffineSymbolExpr(symbol_id, mlir_context));
    }
    for (const RTVar& rt_var : thread_x_to_linearized_input.GetRTVars()) {
      // Take midpoint of the feasible interval for the RT variable.
      symbol_replacements.push_back(getAffineConstantExpr(
          (rt_var.feasible_values.lower + rt_var.feasible_values.upper) / 2,
          mlir_context));
    }
    AffineMap thread_x_to_input_no_rt_symbols =
        thread_x_to_linearized_input.GetAffineMap().replaceDimsAndSymbols(
            {}, symbol_replacements,
            thread_x_to_linearized_input.GetDimVarsCount(),
            thread_x_to_linearized_input.GetRangeVarsCount());
    thread_x_to_linearized_input = IndexingMap{
        thread_x_to_input_no_rt_symbols,
        thread_x_to_linearized_input.GetDimVars(),
        thread_x_to_linearized_input.GetRangeVars(),
        thread_x_to_linearized_input.GetRTVars(),
    };
  }
  thread_x_to_linearized_input.Simplify(GetIndexingMapForInstruction);
  thread_x_to_linearized_input.RescaleSymbols();
  thread_x_to_linearized_input.RemoveUnusedSymbols();
  return EstimateCoalescingViaMemoryTransactionsCount(
      FindContiguousIntervals(thread_x_to_linearized_input), element_type);
}

// Returns a linearized shape, i.e. tensor<num_elements(input) x element_type>.
Shape GetLinearizedShape(const Shape& shape) {
  if (shape.rank() == 0) {
    return shape;
  }
  std::vector<int64_t> dims{ShapeUtil::ElementsIn(shape)};
  auto result = Shape(shape.element_type(), dims,
                      absl::InlinedVector<bool, 4>(dims.size(), false), {});
  *result.mutable_layout() = xla::Layout({0});
  return result;
}

// Returns thread ID to linearized physical layout indexing map for each operand
// of the fusion.
std::optional<GroupedByOpIndexingMap> GetThreadIdToInputMemoryLayoutsMaps(
    const HloFusionAdaptor& fusion_adaptor,
    absl::Span<const HloInstruction* const> operands,
    const HloFusionAnalysis& fusion_analysis,
    KernelFusionInterface* fusion_interface, mlir::MLIRContext* mlir_context) {
  GroupedByOpIndexingMap result;
  for (const auto& [root_index, hero] :
       llvm::enumerate(fusion_analysis.fusion_heroes())) {
    for (const auto& [hero_operand_index, hero_operand] :
         llvm::enumerate(hero->operands())) {
      if (hero_operand->shape().rank() == 0) {
        continue;
      }
      // Compute thread ID -> hero operand indexing map.
      std::optional<IndexingMap> thread_id_to_hero_operand_map =
          fusion_interface->ComputeThreadIdToInputIndexing(
              root_index, hero_operand_index, mlir_context);
      if (!thread_id_to_hero_operand_map.has_value()) {
        return std::nullopt;
      }
      // Compute indexing from output to inputs for logical layout.
      HloInstructionAdaptor hero_operand_adaptor(*hero_operand);
      GroupedByOpIndexingMap instr_indexing_keyed_by_operands =
          ComputeGroupedOutputToInputIndexing(
              fusion_adaptor, hero_operand_adaptor, mlir_context);
      // For every operand compute thread ID -> physical layout of operand
      // indexing map.
      for (const HloInstruction* operand : operands) {
        auto operand_indexing_maps_it =
            instr_indexing_keyed_by_operands.find(operand);
        if (operand_indexing_maps_it ==
            instr_indexing_keyed_by_operands.end()) {
          continue;
        }
        const Shape& operand_shape = operand->shape();

        IndexingMap operand_logical_to_physical_map =
            GetIndexingMapFromLogicalToPhysicalLayout(operand_shape,
                                                      mlir_context);
        IndexingMap operand_physical_to_linearized_shape = GetBitcastMap(
            ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
                operand_shape),
            GetLinearizedShape(operand_shape), mlir_context);
        IndexingMap operand_logical_to_linearized_physical_shape =
            operand_logical_to_physical_map *
            operand_physical_to_linearized_shape;
        operand_logical_to_linearized_physical_shape.Simplify(
            GetIndexingMapForInstruction);

        for (const IndexingMap& operand_indexing_map :
             operand_indexing_maps_it->second) {
          // If one of the indexing maps for the operand is undefined, we remove
          // all indexing maps for it and store only the undefined one.
          if (operand_indexing_map.IsUndefined()) {
            result[operand] = {operand_indexing_map};
            break;
          }
          IndexingMap logical_output_to_linearized_physical_input_map =
              operand_indexing_map *
              operand_logical_to_linearized_physical_shape;
          IndexingMap thread_id_to_linearized_physical_input_map =
              *thread_id_to_hero_operand_map *
              logical_output_to_linearized_physical_input_map;
          thread_id_to_linearized_physical_input_map.Simplify(
              GetIndexingMapForInstruction);
          result[operand].insert(thread_id_to_linearized_physical_input_map);
        }
      }
    }
  }
  return result;
}

}  // namespace

CoalescingAnalysis::CoalescingAnalysis(
    const HloInstruction* instr,
    absl::Span<const HloInstruction* const> operands,
    const HloFusionAnalysis& fusion_analysis,
    KernelFusionInterface* fusion_interface, mlir::MLIRContext* mlir_context,
    bool use_heuristic) {
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(instr);
  if (!use_heuristic && ComputeCoalescingForAllOperands(
                            *fusion_adaptor, operands, fusion_analysis,
                            fusion_interface, mlir_context)) {
    return;
  }
  // If ComputeCoalescingForAllOperands fails, fallback to using the heuristic.
  is_coalesced_computed_by_heuristic_ =
      IsReadCoalescedHeuristic(fusion_analysis.GetEmitterFusionKind(), instr);
}

CoalescingAnalysis::CoalescingAnalysis(
    const HloInstruction* producer, const HloInstruction* consumer,
    absl::Span<const HloInstruction* const> operands,
    const HloFusionAnalysis& fusion_analysis,
    KernelFusionInterface* fusion_interface, mlir::MLIRContext* mlir_context,
    bool use_heuristic) {
  ProducerConsumerFusion fusion_adaptor(producer, consumer);
  if (!use_heuristic &&
      ComputeCoalescingForAllOperands(fusion_adaptor, operands, fusion_analysis,
                                      fusion_interface, mlir_context)) {
    return;
  }
  // If ComputeCoalescingForAllOperands fails, fallback to using the heuristic.
  is_coalesced_computed_by_heuristic_ = IsReadCoalescedHeuristic(
      fusion_analysis.GetEmitterFusionKind(), producer, consumer);
}

bool CoalescingAnalysis::ComputeCoalescingForAllOperands(
    const HloFusionAdaptor& fusion_adaptor,
    absl::Span<const HloInstruction* const> operands,
    const HloFusionAnalysis& fusion_analysis,
    KernelFusionInterface* fusion_interface, mlir::MLIRContext* mlir_context) {
  std::optional<GroupedByOpIndexingMap> thread_id_to_input_memory_layouts =
      GetThreadIdToInputMemoryLayoutsMaps(fusion_adaptor, operands,
                                          fusion_analysis, fusion_interface,
                                          mlir_context);
  if (!thread_id_to_input_memory_layouts.has_value()) {
    return false;
  }
  for (const HloInstruction* operand : operands) {
    if (operand->shape().rank() == 0) {
      coalescing_per_operand_.insert({operand, true});
      continue;
    }
    auto operand_indexing_maps =
        thread_id_to_input_memory_layouts->find(operand);
    // If there is no indexing map for the operand, it means that it is not used
    // in the fusion cluster.
    if (operand_indexing_maps == thread_id_to_input_memory_layouts->end()) {
      coalescing_per_operand_.insert({operand, true});
      continue;
    }
    for (const IndexingMap& operand_indexing_map :
         operand_indexing_maps->second) {
      bool is_coalesced =
          IsCoalesced(operand_indexing_map, operand->shape().element_type());
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
