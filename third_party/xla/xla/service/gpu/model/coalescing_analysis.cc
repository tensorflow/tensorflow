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
#include <cstdlib>
#include <optional>
#include <stack>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/layout.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/affine_map_evaluator.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Returns true if all input reads are coalesced. If consumer is not nullptr,
// producer and consumer are considered as one fusion, otherwise it's only the
// producer.
bool IsReadCoalescedHeuristic(HloFusionAnalysis::EmitterFusionKind fusion_kind,
                              const se::DeviceDescription& device_info,
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
      IsInputFusibleReduction(*producer, device_info) && consumer &&
      IsInputFusibleReduction(*consumer, device_info)) {
    return false;
  }
  return true;
}

double BandwidthUtilizationRateHeuristicForTiledMemoryAccess(
    const TiledHloInstruction& hbm_access_instr,
    const se::DeviceDescription& device_info) {
  const HloInstruction* hlo = hbm_access_instr.hlo();
  const Shape& shape = hlo->shape();

  // Compute the number of elements in the contiguous part of the tile.
  int64_t contiguous_elements = 1;
  for (const auto dim_idx : shape.layout().minor_to_major()) {
    // This dimension is strided, so it's not contiguous.
    if (hbm_access_instr.tile_stride(dim_idx) != 1) {
      break;
    }

    int64_t tile_size = hbm_access_instr.tile_size(dim_idx);
    int64_t dim_size = shape.dimensions(dim_idx);

    // Make sure to ignore the mask if there is one.
    contiguous_elements *= std::min(tile_size, dim_size);

    // This dimension is only partially captured, so more major dimensions are
    // necessarily not captured contiguously.
    if (tile_size < dim_size) {
      break;
    }
  }

  // Compute the size of the contiguous part of the tile in bytes.
  int64_t contiguous_bytes_accessed =
      contiguous_elements *
      ShapeUtil::ByteSizeOfPrimitiveType(hlo->shape().element_type());

  // Memory accesses are fully coalesced if the memory access uses exactly a
  // multiple of the DRAM->L2 cache line size contiguously.
  int64_t transaction_size_bytes =
      device_info.dram_to_l2_transaction_size_bytes();
  int64_t effective_bytes_accessed =
      transaction_size_bytes *
      CeilOfRatio(contiguous_bytes_accessed, transaction_size_bytes);
  return 1.0 * contiguous_bytes_accessed / effective_bytes_accessed;
}

bool IsTiledReadCoalescedHeuristic(const TiledHloInstruction& operand,
                                   const se::DeviceDescription& device_info) {
  const Shape& shape = operand.hlo()->shape();

  // Compute the number of elements in the contiguous part of the tile.
  int64_t contiguous_read_elements = 1;
  for (const auto dim_idx : shape.layout().minor_to_major()) {
    // This dimension is strided, so it's not contiguous.
    if (operand.tile_stride(dim_idx) != 1) {
      break;
    }

    int64_t tile_size = operand.tile_size(dim_idx);
    int64_t dim_size = shape.dimensions(dim_idx);

    // Make sure to ignore the mask if there is one.
    contiguous_read_elements *= std::min(tile_size, dim_size);

    // This dimension is only partially captured, so more major dimensions are
    // necessarily not captured contiguously.
    if (tile_size < dim_size) {
      break;
    }
  }

  // Compute the size of the contiguous part of the tile in bytes.
  int64_t contiguous_bytes_accessed =
      contiguous_read_elements *
      ShapeUtil::ByteSizeOfPrimitiveType(operand.hlo()->shape().element_type());

  // We consider a read coalesced if the contiguous part of the read covers the
  // whole DRAM->L2 cache line.
  //
  // TODO(b/332714755): note that we don't check that we fully exploit all the
  // cache lines we read from if we happen to read through several of them.
  return contiguous_bytes_accessed >=
         device_info.dram_to_l2_transaction_size_bytes();
}

namespace {

using ::mlir::AffineBinaryOpExpr;
using ::mlir::AffineConstantExpr;
using ::mlir::AffineExpr;
using ::mlir::AffineExprKind;
using ::mlir::AffineMap;
using ::mlir::getAffineConstantExpr;
using ::mlir::MLIRContext;

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
    memory_transactions += llvm::divideCeilSigned(num_elements * type_size,
                                                  kBytesPerMemoryTransaction);
    total_num_elements += num_elements;
  }
  if (memory_transactions == 0) {
    return true;
  }
  int memory_transactions_lower_bound = llvm::divideCeilSigned(
      total_num_elements * type_size, kBytesPerMemoryTransaction);
  // The magic value chosen by an uneducated guess.
  constexpr float kIsCoalescedThreshold = 0.9;
  return memory_transactions_lower_bound >
         memory_transactions * kIsCoalescedThreshold;
}

// Returns a linearized shape, i.e. tensor<num_elements(input) x element_type>.
Shape GetLinearizedShape(const Shape& shape) {
  if (shape.dimensions().empty()) {
    return shape;
  }
  std::vector<int64_t> dims{ShapeUtil::ElementsIn(shape)};
  auto result = Shape(shape.element_type(), dims);
  *result.mutable_layout() = xla::Layout({0});
  return result;
}

// Returns thread ID to linearized physical layout indexing map for each operand
// of the fusion.
std::optional<GroupedByOpIndexingMap> GetThreadIdToInputMemoryLayoutsMaps(
    const HloFusionAdaptor& fusion_adaptor,
    absl::Span<const HloInstruction* const> operands,
    const HloFusionAnalysis& fusion_analysis,
    KernelFusionInterface* fusion_interface, MLIRContext* mlir_context) {
  GroupedByOpIndexingMap result;
  for (const auto& [root_index, hero] :
       llvm::enumerate(fusion_analysis.fusion_heroes())) {
    for (const auto& [hero_operand_index, hero_operand] :
         llvm::enumerate(hero.GetOperands())) {
      if (hero_operand.shape().dimensions().empty()) {
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
      GroupedByOpIndexingMap instr_indexing_keyed_by_operands =
          ComputeGroupedOutputToInputIndexing(fusion_adaptor, hero_operand,
                                              mlir_context);
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
        operand_logical_to_linearized_physical_shape.Simplify();

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
          thread_id_to_linearized_physical_input_map.Simplify();
          result[operand].insert(thread_id_to_linearized_physical_input_map);
        }
      }
    }
  }
  return result;
}

// Replaces RTVars with the midpoints of the feasible intervals.
void AssignValuesToRTVars(IndexingMap* indexing_map) {
  // If RTVars are present, replace them with constants.
  if (indexing_map->GetRTVarsCount() == 0) {
    return;
  }
  MLIRContext* mlir_context = indexing_map->GetMLIRContext();
  llvm::SmallVector<AffineExpr, 2> symbol_replacements;
  for (int64_t symbol_id = 0; symbol_id < indexing_map->GetRangeVarsCount();
       ++symbol_id) {
    symbol_replacements.push_back(
        mlir::getAffineSymbolExpr(symbol_id, mlir_context));
  }
  for (const IndexingMap::Variable& rt_var : indexing_map->GetRTVars()) {
    // Take midpoint of the feasible interval for the RT variable.
    symbol_replacements.push_back(getAffineConstantExpr(
        (rt_var.bounds.lower + rt_var.bounds.upper) / 2, mlir_context));
  }
  AffineMap thread_x_to_input_no_dim_symbols =
      indexing_map->GetAffineMap().replaceDimsAndSymbols(
          {}, symbol_replacements, indexing_map->GetDimVarsCount(),
          indexing_map->GetRangeVarsCount());
  *indexing_map = IndexingMap{thread_x_to_input_no_dim_symbols,
                              indexing_map->GetDimVars(),
                              indexing_map->GetRangeVars(),
                              {}};
  indexing_map->Simplify();
  indexing_map->RemoveUnusedSymbols();
}

// Replaces all but one RangeVars with the first elements in the range.
// At the moment, we assume that the last RangeVar symbol corresponds to the
// innermost loop induction variable.
void AssignValuesToOuterLoopIVs(IndexingMap* indexing_map) {
  if (indexing_map->GetRangeVarsCount() <= 1) {
    return;
  }
  MLIRContext* mlir_context = indexing_map->GetMLIRContext();
  llvm::SmallVector<AffineExpr, 2> symbol_replacements;
  for (int64_t symbol_id = 0; symbol_id < indexing_map->GetRangeVarsCount() - 1;
       ++symbol_id) {
    symbol_replacements.push_back(getAffineConstantExpr(
        indexing_map->GetRangeVar(symbol_id).bounds.lower, mlir_context));
  }
  symbol_replacements.push_back(mlir::getAffineSymbolExpr(0, mlir_context));

  AffineMap thread_x_to_input_no_dim_symbols =
      indexing_map->GetAffineMap().replaceDimsAndSymbols(
          {}, symbol_replacements, indexing_map->GetDimVarsCount(), 1);
  *indexing_map = IndexingMap{thread_x_to_input_no_dim_symbols,
                              indexing_map->GetDimVars(),
                              {indexing_map->GetRangeVars().back()},
                              {}};
  indexing_map->Simplify();
  indexing_map->RemoveUnusedSymbols();
}

// Result of partitioning of AffineExpr f(d0) + g(s0) into the summands.
struct PartitionedExpr {
  explicit PartitionedExpr(MLIRContext* mlir_context) {
    AffineExpr zero = getAffineConstantExpr(0, mlir_context);
    func_of_d0 = zero;
    func_of_s0 = zero;
  }
  AffineExpr func_of_d0;
  AffineExpr func_of_s0;
};

// Given an AffineExpr that depends on d0 and s0, attempts to split it into
// f(d0) + g(s0). If it is not possible, returns std::nullopt.
std::optional<PartitionedExpr> Partition(AffineExpr expr) {
  PartitionedExpr result(expr.getContext());

  std::vector<AffineExpr> summands;
  std::stack<AffineExpr> dfs;
  dfs.push(expr);
  while (!dfs.empty()) {
    auto top = dfs.top();
    dfs.pop();
    auto sum = mlir::dyn_cast<AffineBinaryOpExpr>(top);
    if (sum && sum.getKind() == AffineExprKind::Add) {
      dfs.push(sum.getLHS());
      dfs.push(sum.getRHS());
      continue;
    }
    bool depends_on_thread_x = top.isFunctionOfDim(0);
    bool depends_on_range = top.isFunctionOfSymbol(0);

    if (depends_on_thread_x && depends_on_range) {
      return std::nullopt;
    }
    if (depends_on_thread_x) {
      result.func_of_d0 = top + result.func_of_d0;
    }
    if (depends_on_range) {
      result.func_of_s0 = top + result.func_of_s0;
    }
  }
  return result;
}

// Performs backtracking to find all feasible dimensions, symbols that satisfy
// the constraints and then evaluates the affine map at those.
// For example, for the following indexing map:
//   (d0)[s0] -> (d0 + s0)
//   domain:
//   d0 in [0, 3]
//   s0 in [0, 1, 2]
//   s0 mod 2 in [0, 0]
// The function will compute the following indices [0, 2, 1, 3, 2, 4, 3, 5].
void FindAllIndices(AffineExpr expr, int dim_id, int symbol_id,
                    const std::vector<Interval>& dimension_ranges,
                    const std::vector<Interval>& symbol_ranges,
                    std::vector<int64_t>* dimensions,
                    std::vector<int64_t>* symbols,
                    std::vector<int64_t>* indices) {
  if (dim_id < dimension_ranges.size()) {
    Interval dim_range = dimension_ranges[dim_id];
    for (int64_t dim_value = dim_range.lower; dim_value <= dim_range.upper;
         ++dim_value) {
      dimensions->push_back(dim_value);
      FindAllIndices(expr, dim_id + 1, symbol_id, dimension_ranges,
                     symbol_ranges, dimensions, symbols, indices);
      dimensions->pop_back();
    }
    return;
  }
  if (symbol_id < symbol_ranges.size()) {
    Interval symbol_range = symbol_ranges[symbol_id];
    for (int64_t symbol_value = symbol_range.lower;
         symbol_value <= symbol_range.upper; ++symbol_value) {
      symbols->push_back(symbol_value);
      FindAllIndices(expr, dim_id, symbol_id + 1, dimension_ranges,
                     symbol_ranges, dimensions, symbols, indices);
      symbols->pop_back();
    }
    return;
  }
  indices->push_back(EvaluateAffineExpr(expr, *dimensions, *symbols));
}

// Computes contiguous intervals of accessed elements.
// For example, for an indexing map
//   (thread_x) -> (thread_x * 4 + s0 + (thread_x floordiv 16) * 1984)
//   d0 in [0, 31]
//   s0 in [0, 3]
// The intervals are [0, 63] and [2047, 2111].
std::vector<Interval> FindIntervals(
    AffineExpr expr, const std::vector<Interval>& dimension_ranges,
    const std::vector<Interval>& symbol_ranges = {}) {
  // Find all linear indices, sort and deduplicate them.
  std::vector<int64_t> dimensions, symbols;
  std::vector<int64_t> linear_indices;
  FindAllIndices(expr, 0, 0, dimension_ranges, symbol_ranges, &dimensions,
                 &symbols, &linear_indices);

  std::sort(linear_indices.begin(), linear_indices.end());
  linear_indices.erase(
      std::unique(linear_indices.begin(), linear_indices.end()),
      linear_indices.end());

  // Scan over the sorted unique indices and combine them in intervals.
  std::vector<Interval> intervals;
  for (int i = 0, start, end; i < linear_indices.size();) {
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

// Given a vector of interval [lb, ub] computes intervals [lb, ub + length] and
// then computes union of contiguous intervals.
std::vector<Interval> ExtendIntervals(const std::vector<Interval>& intervals,
                                      int64_t length) {
  // Compute union of overlapped intervals.
  std::vector<Interval> overlapped_intervals;
  for (int i = 0; i < intervals.size();) {
    int64_t lower = intervals[i].lower;
    int64_t upper = intervals[i].upper + length;
    ++i;
    while (i < intervals.size() && upper >= intervals[i].lower - 1) {
      upper = std::max(upper, intervals[i].upper + length);
      ++i;
    }
    overlapped_intervals.push_back(Interval{lower, upper});
  }
  return overlapped_intervals;
}

// Computes contiguous intervals, for the expression of type f(thread_x) + g(s).
std::vector<Interval> FindContiguousIntervals(
    const PartitionedExpr& partitioned_expr, const IndexingMap& indexing_map) {
  constexpr int64_t kNumThreadsPerWarp = 32;
  MLIRContext* mlir_context = indexing_map.GetMLIRContext();
  AffineExpr thread_x = mlir::getAffineDimExpr(0, mlir_context);
  AffineExpr range = mlir::getAffineSymbolExpr(0, mlir_context);

  // Case 1: f(thread_x) = thread_x * multiplier.
  // Case 1.1: multiplier == 1.
  if (partitioned_expr.func_of_d0 == thread_x) {
    return {Interval{0, kNumThreadsPerWarp - 1}};
  }
  if (auto mul =
          mlir::dyn_cast<AffineBinaryOpExpr>(partitioned_expr.func_of_d0);
      mul && mul.getKind() == AffineExprKind::Mul) {
    if (auto multiplier = mlir::dyn_cast<AffineConstantExpr>(mul.getRHS());
        multiplier) {
      // Case 1.2: multiplier == -1.
      if (multiplier.getValue() == -1) {
        return {Interval{0, kNumThreadsPerWarp - 1}};
      }
      // Case 1.3: |multiplier| != 1 and g(s) = s.
      if (partitioned_expr.func_of_s0 == range) {
        Interval range_interval = indexing_map.GetSymbolBound(0);
        int64_t num_elems = range_interval.GetLoopTripCount();
        // In this case we get a single interval, because the ranges that every
        // thread is reading overlap.
        if (num_elems >= std::abs(multiplier.getValue())) {
          return {Interval{0, multiplier.getValue() * (kNumThreadsPerWarp - 1) +
                                  num_elems - 1}};
        }
        std::vector<Interval> intervals;
        for (int i = 0, dm = 0; i < kNumThreadsPerWarp;
             ++i, dm += multiplier.getValue()) {
          intervals.push_back(
              {range_interval.lower + dm, range_interval.upper + dm});
        }
        return intervals;
      }
      // Case 1.4: |multiplier| != 1 and g(s) != s.
      std::vector<Interval> intervals;
      for (int i = 0, dm = 0; i < kNumThreadsPerWarp;
           ++i, dm += multiplier.getValue()) {
        intervals.push_back({dm, dm});
      }
      return intervals;
    }
  }
  // Case 2: f(thread_x) != thread_x * multiplier.
  auto intervals = FindIntervals(partitioned_expr.func_of_d0,
                                 {indexing_map.GetDimVar(0).bounds});
  // Case 2.1: g(s) != s.
  if (partitioned_expr.func_of_s0 != range) {
    return intervals;
  }
  // Case 2.2: g(s) = s.
  Interval range_interval = indexing_map.GetSymbolBound(0);
  return ExtendIntervals(intervals, range_interval.GetLoopTripCount() - 1);
}

bool IsIndexingCoalesced(IndexingMap& thread_x_to_linearized_input,
                         PrimitiveType element_type) {
  // Undefined indexing maps, i.e. those for which we don't know the indexing
  // are assumed to be uncoalesced.
  if (thread_x_to_linearized_input.IsUndefined()) {
    return false;
  }
  // 0d constants are coalesced.
  if (thread_x_to_linearized_input.GetAffineMap().getNumResults() == 0) {
    return true;
  }
  // Replace RTVars with the feasible values.
  AssignValuesToRTVars(&thread_x_to_linearized_input);

  // Compute the indexing map for the first [0, 31] threads. This should be
  // extended to sampling several warps.
  MLIRContext* mlir_context = thread_x_to_linearized_input.GetMLIRContext();
  AffineExpr thread_x_dim = mlir::getAffineDimExpr(
      KernelFusionInterface::kIndexingMapThreadIdxDims[0], mlir_context);
  AffineExpr c0 = getAffineConstantExpr(0, mlir_context);
  IndexingMap thread_x_first_32_elements{
      AffineMap::get(1, 0, {thread_x_dim, c0, c0, c0, c0, c0}, mlir_context),
      {IndexingMap::Variable{{0, 31}}},
      /*range_vars=*/{},
      /*rt_vars=*/{}};
  IndexingMap thread_x_to_input_sample =
      thread_x_first_32_elements * thread_x_to_linearized_input;
  thread_x_to_input_sample.Simplify();
  thread_x_to_input_sample.RescaleSymbols();
  thread_x_to_input_sample.RemoveUnusedSymbols();

  // If the indexing map is "empty", then the input is not used in this warp,
  // therefore, it's coalesced.
  if (thread_x_to_input_sample.IsKnownEmpty()) {
    return true;
  }
  AssignValuesToOuterLoopIVs(&thread_x_to_input_sample);
  auto partitioned_expr =
      Partition(thread_x_to_input_sample.GetAffineMap().getResult(0));
  if (!partitioned_expr.has_value()) {
    return false;
  }
  // Right now we support only thread_x maps what do not have any constraints or
  // have a single constraint that coincides with
  // thread_x_to_input_sample.getAffineMap().
  if (thread_x_to_input_sample.GetConstraintsCount() > 1 ||
      (thread_x_to_input_sample.GetConstraintsCount() == 1 &&
       thread_x_to_input_sample.GetConstraints().begin()->first !=
           partitioned_expr->func_of_d0 + partitioned_expr->func_of_s0)) {
    return false;
  }
  return EstimateCoalescingViaMemoryTransactionsCount(
      FindContiguousIntervals(*partitioned_expr, thread_x_to_input_sample),
      element_type);
}

}  // namespace

CoalescingAnalysis::CoalescingAnalysis(
    const HloInstruction* instr,
    absl::Span<const HloInstruction* const> operands,
    const HloFusionAnalysis& fusion_analysis,
    KernelFusionInterface* fusion_interface, MLIRContext* mlir_context,
    bool use_heuristic) {
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(instr);
  if (!use_heuristic && ComputeCoalescingForAllOperands(
                            *fusion_adaptor, operands, fusion_analysis,
                            fusion_interface, mlir_context)) {
    return;
  }
  // If ComputeCoalescingForAllOperands fails, fallback to using the heuristic.
  is_coalesced_computed_by_heuristic_ =
      IsReadCoalescedHeuristic(fusion_analysis.GetEmitterFusionKind(),
                               fusion_analysis.device_info(), instr);
}

CoalescingAnalysis::CoalescingAnalysis(
    const HloInstruction* producer, const HloInstruction* consumer,
    absl::Span<const HloInstruction* const> operands,
    const HloFusionAnalysis& fusion_analysis,
    KernelFusionInterface* fusion_interface, MLIRContext* mlir_context,
    bool use_heuristic) {
  auto fusion_adaptor =
      HloFusionAdaptor::ForProducerConsumer(producer, consumer);
  if (!use_heuristic && ComputeCoalescingForAllOperands(
                            *fusion_adaptor, operands, fusion_analysis,
                            fusion_interface, mlir_context)) {
    return;
  }
  // If ComputeCoalescingForAllOperands fails, fallback to using the heuristic.
  is_coalesced_computed_by_heuristic_ = IsReadCoalescedHeuristic(
      fusion_analysis.GetEmitterFusionKind(), fusion_analysis.device_info(),
      producer, consumer);
}

bool CoalescingAnalysis::ComputeCoalescingForAllOperands(
    const HloFusionAdaptor& fusion_adaptor,
    absl::Span<const HloInstruction* const> operands,
    const HloFusionAnalysis& fusion_analysis,
    KernelFusionInterface* fusion_interface, MLIRContext* mlir_context) {
  std::optional<GroupedByOpIndexingMap> thread_id_to_input_memory_layouts =
      GetThreadIdToInputMemoryLayoutsMaps(fusion_adaptor, operands,
                                          fusion_analysis, fusion_interface,
                                          mlir_context);
  if (!thread_id_to_input_memory_layouts.has_value()) {
    return false;
  }
  for (const HloInstruction* operand : operands) {
    if (operand->shape().dimensions().empty()) {
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
    for (IndexingMap operand_indexing_map : operand_indexing_maps->second) {
      bool is_coalesced = IsIndexingCoalesced(operand_indexing_map,
                                              operand->shape().element_type());
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
