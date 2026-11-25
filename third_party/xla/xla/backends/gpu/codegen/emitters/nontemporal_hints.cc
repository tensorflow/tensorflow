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

#include "xla/backends/gpu/codegen/emitters/nontemporal_hints.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"

namespace xla::gpu {

namespace {

// Who else reads an operand's buffer.
struct OtherReaders {
  // An instruction other than the fusion.
  bool instruction = false;
  // The fusion, through another operand.
  bool operand = false;
};

OtherReaders GetOtherReaders(const HloFusionInstruction& fusion,
                             int64_t operand_id,
                             const HloDataflowAnalysis* dataflow) {
  if (dataflow == nullptr) {
    return {/*instruction=*/true, /*operand=*/true};
  }
  const HloInstruction* entry_root =
      fusion.GetModule()->entry_computation()->root_instruction();
  OtherReaders readers;
  for (const HloValue* value :
       dataflow->GetValueSet(fusion.operand(operand_id)).values()) {
    for (const HloUse& use : value->GetUses()) {
      const HloInstruction* user = use.instruction;
      // Bitcasts and optimization barriers define no values, so their users
      // are uses of this value too. The entry root tuple only returns it.
      if (user->opcode() == HloOpcode::kBitcast ||
          user->opcode() == HloOpcode::kOptimizationBarrier ||
          (user == entry_root && user->opcode() == HloOpcode::kTuple)) {
        continue;
      }
      if (user != &fusion) {
        readers.instruction = true;
      } else if (use.operand_number != operand_id) {
        readers.operand = true;
      }
    }
  }
  return readers;
}

// A piece of a linearized index: a variable (divisor 1), or a variable
// floor-divided by or modulo a positive constant.
struct Term {
  int64_t var;
  SymbolicExprType type;
  int64_t divisor;

  template <typename H>
  friend H AbslHashValue(H h, const Term& t) {
    return H::combine(std::move(h), t.var, t.type, t.divisor);
  }
  bool operator==(const Term& other) const {
    return var == other.var && type == other.type && divisor == other.divisor;
  }
};

// Adds `scale` times `expr` to `coefficients`, one per piece. False if `expr`
// is not a sum of pieces times constants.
bool AddTermCoefficients(SymbolicExpr expr, int64_t scale, int64_t num_vars,
                         absl::flat_hash_map<Term, int64_t>& coefficients) {
  auto is_var = [&](SymbolicExpr e) {
    return e.GetType() == SymbolicExprType::kVariable && e.GetValue() >= 0 &&
           e.GetValue() < num_vars;
  };
  switch (expr.GetType()) {
    case SymbolicExprType::kConstant:
      return true;
    case SymbolicExprType::kVariable:
      if (!is_var(expr)) {
        return false;
      }
      coefficients[{expr.GetValue(), SymbolicExprType::kVariable, 1}] += scale;
      return true;
    case SymbolicExprType::kFloorDiv:
    case SymbolicExprType::kMod:
      if (!is_var(expr.GetLHS()) ||
          expr.GetRHS().GetType() != SymbolicExprType::kConstant ||
          expr.GetRHS().GetValue() <= 0) {
        return false;
      }
      coefficients[{expr.GetLHS().GetValue(), expr.GetType(),
                    expr.GetRHS().GetValue()}] += scale;
      return true;
    case SymbolicExprType::kAdd:
      return AddTermCoefficients(expr.GetLHS(), scale, num_vars,
                                 coefficients) &&
             AddTermCoefficients(expr.GetRHS(), scale, num_vars, coefficients);
    case SymbolicExprType::kMul:
      if (expr.GetRHS().GetType() == SymbolicExprType::kConstant) {
        return AddTermCoefficients(expr.GetLHS(),
                                   scale * expr.GetRHS().GetValue(), num_vars,
                                   coefficients);
      }
      if (expr.GetLHS().GetType() == SymbolicExprType::kConstant) {
        return AddTermCoefficients(expr.GetRHS(),
                                   scale * expr.GetLHS().GetValue(), num_vars,
                                   coefficients);
      }
      return false;
    default:
      return false;
  }
}

// Number of values `piece` takes over `bounds` of its variable, or an upper
// bound on the span of an interval holding them.
int64_t TermExtent(const Term& term, const Interval& bounds) {
  const int64_t c = term.divisor;
  switch (term.type) {
    case SymbolicExprType::kFloorDiv:
      return llvm::divideFloorSigned(bounds.upper, c) -
             llvm::divideFloorSigned(bounds.lower, c) + 1;
    case SymbolicExprType::kMod:
      return llvm::divideFloorSigned(bounds.lower, c) ==
                     llvm::divideFloorSigned(bounds.upper, c)
                 ? bounds.GetLoopTripCount()
                 : c;
    default:
      return bounds.GetLoopTripCount();
  }
}

// The place value of the digit of a mixed-radix index that `expr` extracts:
// D in e floorDiv D, (e floorDiv D) mod M and (e mod M) floorDiv D, else 1.
int64_t DigitDivisor(SymbolicExpr expr) {
  if (expr.GetType() == SymbolicExprType::kMod) {
    expr = expr.GetLHS();
  }
  if (expr.GetType() == SymbolicExprType::kFloorDiv &&
      expr.GetRHS().GetType() == SymbolicExprType::kConstant) {
    return expr.GetRHS().GetValue();
  }
  return 1;
}

// Whether `map` reads each element at most once, from the index into `dims`
// (row-major) that `results` give.
bool LinearIndexReadsOnce(const IndexingMap& map,
                          absl::Span<const int64_t> dims,
                          absl::Span<const SymbolicExpr> results) {
  mlir::MLIRContext* mlir_context = map.GetMLIRContext();
  const SymbolicMap& symbolic_map = map.GetSymbolicMap();
  const int64_t num_dims = symbolic_map.GetNumDims();
  IndexingMap linear(
      SymbolicMap::Get(mlir_context, num_dims, symbolic_map.GetNumSymbols(),
                       {LinearizeShape(dims, results, mlir_context)}),
      map.GetDimVars(), map.GetRangeVars(), map.GetRTVars(),
      map.GetSymbolicConstraints());
  linear.Simplify();
  if (linear.IsKnownEmpty()) {
    return true;
  }
  const int64_t num_vars = num_dims + symbolic_map.GetNumSymbols();
  absl::flat_hash_map<Term, int64_t> coefficients;
  if (!AddTermCoefficients(linear.GetSymbolicMap().GetResult(0), 1, num_vars,
                           coefficients)) {
    return false;
  }
  auto bounds_of = [&](int64_t var) -> const Interval& {
    return var < num_dims ? linear.GetDimensionBound(var)
                          : linear.GetSymbolBound(var - num_dims);
  };
  // Each variable that takes several values must be recoverable from its
  // pieces: it appears itself, or as v floorDiv c and v mod m with c | m.
  for (int64_t var = 0; var < num_vars; ++var) {
    if (bounds_of(var).GetLoopTripCount() <= 1) {
      continue;
    }
    bool plain = false;
    std::vector<int64_t> divisors, moduli;
    for (const auto& [term, coefficient] : coefficients) {
      if (term.var != var || coefficient == 0) {
        continue;
      }
      if (term.type == SymbolicExprType::kVariable) {
        plain = true;
      } else if (term.type == SymbolicExprType::kFloorDiv) {
        divisors.push_back(term.divisor);
      } else {
        moduli.push_back(term.divisor);
      }
    }
    const bool split = absl::c_any_of(divisors, [&](int64_t c) {
      return absl::c_any_of(moduli, [&](int64_t m) { return m % c == 0; });
    });
    if (!plain && !split) {
      return false;
    }
  }
  // Sorted by stride, each piece that takes several values must step past all
  // the elements the smaller strides reach.
  std::vector<std::pair<int64_t, int64_t>> strides_and_extents;
  for (const auto& [term, coefficient] : coefficients) {
    const int64_t extent = TermExtent(term, bounds_of(term.var));
    if (coefficient != 0 && extent > 1) {
      strides_and_extents.push_back({std::abs(coefficient), extent});
    }
  }
  absl::c_sort(strides_and_extents);
  int64_t reach = 0;
  for (const auto& [stride, extent] : strides_and_extents) {
    if (stride <= reach) {
      return false;
    }
    reach += stride * (extent - 1);
  }
  return true;
}

// Thread x, y, z and block x, y, z.
constexpr int64_t kNumThreadIdDims = 6;
constexpr int64_t kMaxVectorBytes = 16;
constexpr int64_t kMaxScalarLoads = 4;

// Evaluates a thread-id-to-index map at a thread of a block and an iteration of
// the inner loop, the innermost range variable. Other symbols stay at their
// lower bounds.
class ThreadIdToIndex {
 public:
  // `*map` must outlive `ThreadIdToIndex`.
  explicit ThreadIdToIndex(const IndexingMap* absl_nonnull map)
      : map_(map),
        result_(map->GetSymbolicMap().GetResult(0)),
        inner_(map->GetRangeVarsCount() - 1),
        inner_bounds_(inner_ >= 0 ? map->GetRangeVar(inner_).bounds
                                  : Interval{0, 0}),
        vars_(kNumThreadIdDims + map->GetSymbolCount(), 0) {
    for (int64_t i = 0; i < map->GetSymbolCount(); ++i) {
      vars_[kNumThreadIdDims + i] = map->GetSymbolBound(i).lower;
    }
  }

  int64_t block_threads() const { return Extent(0) * Extent(1) * Extent(2); }
  int64_t num_blocks() const { return Extent(3); }
  bool has_inner_loop() const { return inner_ >= 0; }
  // 1 without an inner loop.
  int64_t inner_count() const { return inner_bounds_.GetLoopTripCount(); }

  // Whether `thread` of `block` satisfies the constraints in iteration `k`.
  bool IsActive(int64_t block, int64_t thread, int64_t k) {
    Select(block, thread, k);
    for (const auto& [expr, interval] : map_->GetSymbolicConstraints()) {
      if (!interval.Contains(expr.Evaluate(vars_))) {
        return false;
      }
    }
    return true;
  }

  // The index `thread` of `block` reads in iteration `k`, if active.
  std::optional<int64_t> ActiveIndex(int64_t block, int64_t thread, int64_t k) {
    if (!IsActive(block, thread, k)) {
      return std::nullopt;
    }
    return result_.Evaluate(vars_);
  }

 private:
  int64_t Extent(int64_t dim) const {
    return map_->GetDimVar(dim).bounds.GetLoopTripCount();
  }

  void Select(int64_t block, int64_t thread, int64_t k) {
    vars_[0] = map_->GetDimVar(0).bounds.lower + thread % Extent(0);
    vars_[1] = map_->GetDimVar(1).bounds.lower + thread / Extent(0) % Extent(1);
    vars_[2] =
        map_->GetDimVar(2).bounds.lower + thread / (Extent(0) * Extent(1));
    vars_[3] = map_->GetDimVar(3).bounds.lower + block;
    vars_[4] = map_->GetDimVar(4).bounds.lower;
    vars_[5] = map_->GetDimVar(5).bounds.lower;
    if (has_inner_loop()) {
      vars_[kNumThreadIdDims + inner_] = inner_bounds_.lower + k;
    }
  }

  const IndexingMap* absl_nonnull map_;
  const SymbolicExpr result_;
  const int64_t inner_;
  const Interval inner_bounds_;
  std::vector<int64_t> vars_;
};

// Threads [first_thread, end_thread) of a block.
struct Wavefront {
  int64_t block;
  int64_t first_thread;
  int64_t end_thread;
};

int64_t SampledInnerCount(const ThreadIdToIndex& indexing,
                          int64_t element_bytes) {
  const int64_t inner_count = indexing.inner_count();
  if (indexing.has_inner_loop() &&
      inner_count <= kMaxVectorBytes / element_bytes) {
    return inner_count;
  }
  return std::min(inner_count, kMaxScalarLoads);
}

bool HasActiveThread(ThreadIdToIndex& indexing, int64_t block,
                     int64_t num_threads, int64_t num_inner) {
  for (int64_t thread = 0; thread < num_threads; ++thread) {
    for (int64_t k = 0; k < num_inner; ++k) {
      if (indexing.IsActive(block, thread, k)) {
        return true;
      }
    }
  }
  return false;
}

// The first two wavefronts of the first two blocks with an active thread, or
// nullopt if the scan reaches block 1024 first. A leading pad can leave the
// first blocks without active threads.
std::optional<std::vector<Wavefront>> SampleActiveWavefronts(
    ThreadIdToIndex& indexing, int64_t wavefront_size, int64_t element_bytes) {
  constexpr int64_t kMaxScannedBlocks = 1024;
  const int64_t block_threads = indexing.block_threads();
  const int64_t sampled_threads =
      std::min<int64_t>(block_threads, 2 * wavefront_size);
  const int64_t sampled_inner = SampledInnerCount(indexing, element_bytes);
  std::vector<Wavefront> wavefronts;
  int64_t sampled_blocks = 0;
  for (int64_t block = 0; block < indexing.num_blocks() && sampled_blocks < 2;
       ++block) {
    if (block == kMaxScannedBlocks) {
      return std::nullopt;
    }
    if (!HasActiveThread(indexing, block, sampled_threads, sampled_inner)) {
      continue;
    }
    ++sampled_blocks;
    for (int64_t first = 0; first < sampled_threads; first += wavefront_size) {
      wavefronts.push_back(
          {block, first, std::min(first + wavefront_size, block_threads)});
    }
  }
  return wavefronts;
}

// Whether each thread reads its inner loop with one vector load: at most 16
// bytes at consecutive indices. A thread with some iterations masked reads
// value by value.
bool LoadsInnerLoopAsVector(ThreadIdToIndex& indexing,
                            absl::Span<const Wavefront> wavefronts,
                            int64_t element_bytes) {
  const int64_t inner_count = indexing.inner_count();
  if (!indexing.has_inner_loop() || inner_count <= 1 ||
      inner_count > kMaxVectorBytes / element_bytes) {
    return false;
  }
  for (const Wavefront& wavefront : wavefronts) {
    for (int64_t thread = wavefront.first_thread; thread < wavefront.end_thread;
         ++thread) {
      int64_t previous = 0;
      int64_t num_active = 0;
      for (int64_t k = 0; k < inner_count; ++k) {
        std::optional<int64_t> index =
            indexing.ActiveIndex(wavefront.block, thread, k);
        if (!index.has_value()) {
          continue;
        }
        if (num_active > 0 && *index != previous + 1) {
          return false;
        }
        previous = *index;
        ++num_active;
      }
      if (num_active != 0 && num_active != inner_count) {
        return false;
      }
    }
  }
  return true;
}

// The elements load `load` of `wavefront` reads: the whole inner loop for a
// vector load, iteration `load` for a scalar one.
std::vector<int64_t> LoadedElements(ThreadIdToIndex& indexing,
                                    const Wavefront& wavefront, int64_t load,
                                    bool vector_inner) {
  const int64_t first_k = vector_inner ? 0 : load;
  const int64_t end_k = vector_inner ? indexing.inner_count() : load + 1;
  std::vector<int64_t> elements;
  for (int64_t thread = wavefront.first_thread; thread < wavefront.end_thread;
       ++thread) {
    for (int64_t k = first_k; k < end_k; ++k) {
      if (std::optional<int64_t> index =
              indexing.ActiveIndex(wavefront.block, thread, k)) {
        elements.push_back(*index);
      }
    }
  }
  return elements;
}

// Whether `elements` cover part of a 128-byte line. The last line ends at the
// end of the buffer.
bool CoversPartOfLine(std::vector<int64_t> elements, int64_t element_bytes,
                      int64_t buffer_bytes) {
  constexpr int64_t kLineBytes = 128;
  absl::c_sort(elements);
  elements.erase(std::unique(elements.begin(), elements.end()), elements.end());
  absl::flat_hash_map<int64_t, int64_t> line_bytes;
  for (int64_t element : elements) {
    line_bytes[llvm::divideFloorSigned(element * element_bytes, kLineBytes)] +=
        element_bytes;
  }
  for (const auto& [line, bytes] : line_bytes) {
    const int64_t line_end = std::min((line + 1) * kLineBytes, buffer_bytes);
    if (bytes < line_end - line * kLineBytes) {
      return true;
    }
  }
  return false;
}

}  // namespace

std::vector<bool> ComputeReadElsewhere(const HloFusionInstruction& fusion,
                                       const HloDataflowAnalysis* dataflow) {
  std::vector<bool> read_elsewhere(fusion.operand_count());
  for (int64_t operand_id = 0; operand_id < fusion.operand_count();
       ++operand_id) {
    const OtherReaders readers = GetOtherReaders(fusion, operand_id, dataflow);
    read_elsewhere[operand_id] = readers.instruction || readers.operand;
  }
  return read_elsewhere;
}

bool ReadsEachElementAtMostOnce(const IndexingMap& map, const Shape& shape) {
  if (map.IsUndefined() || !shape.IsArray() ||
      static_cast<int64_t>(map.GetSymbolicMap().GetNumResults()) !=
          shape.dimensions().size()) {
    return false;
  }
  if (map.IsKnownEmpty()) {
    return true;
  }
  const SymbolicMap& symbolic_map = map.GetSymbolicMap();
  const int64_t rank = shape.dimensions().size();
  // Any order of the dimensions gives a one-to-one linear index. The floorDiv
  // and mod that bitcasts add recombine in the order of the digits they
  // extract: the physical order, or that of a transpose before the bitcast.
  std::vector<std::vector<int64_t>> orders;
  if (shape.has_layout()) {
    const auto& minor_to_major = shape.layout().minor_to_major();
    orders.emplace_back(minor_to_major.rbegin(), minor_to_major.rend());
  }
  std::vector<int64_t> by_divisor(rank);
  absl::c_iota(by_divisor, 0);
  absl::c_stable_sort(by_divisor, [&](int64_t a, int64_t b) {
    return DigitDivisor(symbolic_map.GetResult(a)) >
           DigitDivisor(symbolic_map.GetResult(b));
  });
  orders.push_back(std::move(by_divisor));
  std::vector<int64_t> logical(rank);
  absl::c_iota(logical, 0);
  orders.push_back(std::move(logical));

  for (int64_t i = 0; i < static_cast<int64_t>(orders.size()); ++i) {
    if (absl::c_contains(absl::MakeConstSpan(orders).first(i), orders[i])) {
      continue;
    }
    std::vector<int64_t> dims;
    std::vector<SymbolicExpr> results;
    for (int64_t dim : orders[i]) {
      dims.push_back(shape.dimensions(dim));
      results.push_back(symbolic_map.GetResult(dim));
    }
    if (LinearIndexReadsOnce(map, dims, results)) {
      return true;
    }
  }
  return false;
}

std::vector<OperandReads> ComputeOperandReads(
    const HloFusionInstruction& fusion, const HloDataflowAnalysis* dataflow,
    mlir::MLIRContext* mlir_context) {
  // Scatter operands have no indexing map; a runtime-var producer fails
  // verification.
  if (absl::c_any_of(fusion.fused_instructions(),
                     HloPredicateIsOp<HloOpcode::kScatter>)) {
    return {};
  }

  // SymbolicExpr needs its storage registered on this context.
  RegisterSymbolicExprStorage(mlir_context);

  const int64_t num_operands = fusion.operand_count();
  // Count roots, not outputs: a variadic reduce is one root.
  const int64_t num_roots =
      HloFusionAdaptor::ForInstruction(&fusion)->GetRoots().size();

  // Union the per-operand indexing sets across all roots.
  std::vector<OperandIndexingSet> combined(num_operands);
  for (int64_t root_id = 0; root_id < num_roots; ++root_id) {
    HloInstructionIndexing indexing =
        ComputeOutputToInputIndexing(&fusion, root_id, mlir_context);
    for (int64_t operand_id = 0;
         operand_id < static_cast<int64_t>(indexing.indexing_maps.size()) &&
         operand_id < num_operands;
         ++operand_id) {
      const OperandIndexingSet& operand_set =
          indexing.indexing_maps[operand_id];
      combined[operand_id].insert(operand_set.begin(), operand_set.end());
    }
  }

  std::vector<OperandReads> reads(num_operands);
  for (int64_t operand_id = 0; operand_id < num_operands; ++operand_id) {
    const OtherReaders readers = GetOtherReaders(fusion, operand_id, dataflow);
    reads[operand_id].read_elsewhere = readers.instruction || readers.operand;
    const OperandIndexingSet& operand_set = combined[operand_id];
    if (operand_set.empty()) {
      continue;
    }
    const Shape& shape = fusion.operand(operand_id)->shape();
    bool not_broadcast = true;
    bool has_range_vars = false;
    for (const auto& operand_indexing : operand_set) {
      const IndexingMap& map = operand_indexing.map();
      if (map.IsUndefined() || map.GetRTVarsCount() != 0 ||
          !ReadsEachElementAtMostOnce(map, shape)) {
        not_broadcast = false;
        break;
      }
      has_range_vars |= map.GetRangeVarsCount() != 0;
    }
    reads[operand_id].not_broadcast = not_broadcast;
    reads[operand_id].streamed_once = not_broadcast &&
                                      operand_set.size() == 1 &&
                                      !has_range_vars && !readers.operand;
  }
  return reads;
}

bool ReadsLinesInPieces(const IndexingMap& thread_id_to_index,
                        int64_t wavefront_size, int64_t element_bytes,
                        int64_t buffer_bytes) {
  const IndexingMap& map = thread_id_to_index;
  if (map.IsUndefined() || map.IsKnownEmpty() ||
      map.GetDimVarsCount() != kNumThreadIdDims ||
      map.GetSymbolicMap().GetNumResults() != 1 || element_bytes <= 0 ||
      wavefront_size <= 0) {
    return false;
  }
  ThreadIdToIndex indexing(&map);
  std::optional<std::vector<Wavefront>> wavefronts =
      SampleActiveWavefronts(indexing, wavefront_size, element_bytes);
  if (!wavefronts.has_value()) {
    return true;
  }
  const bool vector_inner =
      LoadsInnerLoopAsVector(indexing, *wavefronts, element_bytes);
  // One vector load per thread, or one scalar load per inner iteration.
  const int64_t num_loads =
      vector_inner ? 1 : std::min(indexing.inner_count(), kMaxScalarLoads);
  for (const Wavefront& wavefront : *wavefronts) {
    for (int64_t load = 0; load < num_loads; ++load) {
      if (CoversPartOfLine(
              LoadedElements(indexing, wavefront, load, vector_inner),
              element_bytes, buffer_bytes)) {
        return true;
      }
    }
  }
  return false;
}

void AnnotateNontemporalHintFacts(mlir::ModuleOp module,
                                  absl::string_view entry_function_name,
                                  const HloInstruction& hlo_instruction,
                                  const HloDataflowAnalysis* dataflow,
                                  OutputContiguity output_contiguity) {
  auto entry = module.lookupSymbol<mlir::func::FuncOp>(
      llvm::StringRef(entry_function_name.data(), entry_function_name.size()));
  if (!entry) {
    return;
  }

  mlir::Builder builder(module.getContext());
  if (output_contiguity == OutputContiguity::kContiguous) {
    entry->setAttr(emitters::kXlaContiguousOutputAttr, builder.getUnitAttr());
  }

  const auto* fusion = DynCast<HloFusionInstruction>(&hlo_instruction);
  if (fusion == nullptr) {
    return;
  }

  const std::vector<OperandReads> reads =
      ComputeOperandReads(*fusion, dataflow, module.getContext());
  const int64_t num_operands = fusion->operand_count();
  for (int64_t index = 0;
       index < num_operands && index < static_cast<int64_t>(reads.size());
       ++index) {
    if (reads[index].streamed_once) {
      entry.setArgAttr(index, emitters::kXlaStreamedOnceAttr,
                       builder.getUnitAttr());
    }
    if (reads[index].not_broadcast) {
      entry.setArgAttr(index, emitters::kXlaNotBroadcastAttr,
                       builder.getUnitAttr());
    }
    if (reads[index].read_elsewhere) {
      entry.setArgAttr(index, emitters::kXlaReadElsewhereAttr,
                       builder.getUnitAttr());
    }
  }
}

}  // namespace xla::gpu
