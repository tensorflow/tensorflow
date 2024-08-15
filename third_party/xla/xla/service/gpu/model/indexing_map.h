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

#ifndef XLA_SERVICE_GPU_MODEL_INDEXING_MAP_H_
#define XLA_SERVICE_GPU_MODEL_INDEXING_MAP_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/affine_map_printer.h"

namespace xla {
namespace gpu {

// Interval represents a closed interval [lower_bound, upper_bound].
struct Interval {
  std::string ToString() const;
  void Print(std::ostream& out) const;

  bool IsPoint() const { return lower == upper; }
  bool IsFeasible() const { return lower <= upper; }

  // Returns the number of elements in the interval. Asserts that the number of
  // elements fits in an int64_t. For this reason, this should only be used for
  // intervals corresponding to symbols, not for general intervals. Use
  // `IsFeasible` to check if the interval is non-empty.
  int64_t GetLoopTripCount() const;

  bool Contains(int64_t value) const {
    return value >= lower && value <= upper;
  }

  // Returns true if this interval contains the entire other interval.
  bool Contains(Interval other) const { return Intersect(other) == other; }

  // The result of a range comparison. We wrap std::optional in a struct to
  // avoid accidental implicit conversion to bool:
  // if (range < 42) {
  //   Executed if the result of the comparison is known to be false!
  // }
  struct ComparisonResult {
    // true or false if the result is known, nullopt otherwise.
    std::optional<bool> result;

    ComparisonResult operator!() const {
      if (result) return {!*result};
      return {result};
    }
    bool operator==(const ComparisonResult& other) const {
      return result == other.result;
    }
    bool operator==(bool other) const { return result && *result == other; }
    bool operator==(std::nullopt_t) const { return !result; }
    bool operator!=(std::nullopt_t) const { return result.has_value(); }
    bool operator*() const { return *result; }
  };

  // All comparison operators here return true or false if the result is known,
  // or nullopt if it may be either true or false.
  // We don't use operators here, because the "==" used for hashing is not the
  // same as "Eq".
  ComparisonResult Gt(const Interval& b) const;
  ComparisonResult Lt(const Interval& b) const { return b.Gt(*this); }
  ComparisonResult Ge(const Interval& b) const { return !b.Gt(*this); }
  ComparisonResult Le(const Interval& b) const { return !this->Gt(b); }
  // This is not the same as "==".  See the implementations.
  ComparisonResult Eq(const Interval& b) const;
  // This is not the same as "!=".  See the implementations.
  ComparisonResult Ne(const Interval& b) const { return !this->Eq(b); }

  Interval Intersect(const Interval& rhs) const {
    Interval result{std::max(lower, rhs.lower), std::min(upper, rhs.upper)};
    if (result.upper < result.lower) {
      // Normalize empty results such that NumElements returns 0.
      result.upper = result.lower - 1;
    }
    return result;
  }

  Interval Union(const Interval& rhs) const {
    return {std::min(lower, rhs.lower), std::max(upper, rhs.upper)};
  }

  // Computes the range of the sum of the two intervals. Implements saturating
  // semantics (i.e. overflow and underflow get clamped to the maximum and
  // minimum int64). Additionally, bounds of the minimum/maximum value are
  // considered to be possibly saturated, i.e. `{-2 ** 63, 0} + {42, 42}`
  // returns `{-2 ** 63, 42}`, not `{-2 ** 63 + 42, 42}`.
  Interval operator+(const Interval& rhs) const;
  // Computes the range of the product of the two intervals. Implements
  // saturating semantics.
  Interval operator*(const Interval& rhs) const;
  // Computes the range of the difference of the two intervals. Implements
  // saturating semantics.
  Interval operator-(const Interval& rhs) const { return *this + (-rhs); }
  Interval operator-() const;
  Interval FloorDiv(int64_t rhs) const;

  Interval min(const Interval& rhs) const {
    return {std::min(lower, rhs.lower), std::min(upper, rhs.upper)};
  }

  Interval max(const Interval& rhs) const {
    return {std::max(lower, rhs.lower), std::max(upper, rhs.upper)};
  }

  // This is not the same as "Eq".  See the implementations.
  bool operator==(const Interval& rhs) const {
    return lower == rhs.lower && upper == rhs.upper;
  }
  // This is not the same as "Ne".  See the implementations.
  bool operator!=(const Interval& rhs) const { return !(*this == rhs); }

  int64_t lower = 0;
  int64_t upper = 0;
};

std::ostream& operator<<(std::ostream& out, const Interval& range);
inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                     const Interval& interval) {
  os << absl::StrFormat("[%d, %d]", interval.lower, interval.upper);
  return os;
}

template <typename H>
H AbslHashValue(H h, const Interval& range) {
  return H::combine(std::move(h), range.lower, range.upper);
}

// For use in llvm::hash_combine.
inline size_t hash_value(const Interval& range) {
  return llvm::hash_combine(range.lower, range.upper);
}

class IndexingMap;

// Evaluates lower and upper bounds for expressions given the domain.
// Not thread safe. Lifetime is tied to the owning IndexingMap's lifetime.
class RangeEvaluator {
 public:
  RangeEvaluator(const IndexingMap& indexing_map,
                 mlir::MLIRContext* mlir_context, bool use_constraints = true);

  // Checks whether an `AffineExpr` always describes a non-negative value.
  bool IsAlwaysPositiveOrZero(mlir::AffineExpr expr);

  // Checks whether an `AffineExpr` always describes a non-positive value.
  bool IsAlwaysNegativeOrZero(mlir::AffineExpr expr);

  // Computes the range of expression using its subexpression ranges.
  Interval ComputeExpressionRange(mlir::AffineExpr expr);

  // Return MLIR context.
  mlir::MLIRContext* GetMLIRContext() const { return mlir_context_; }

 private:
  mlir::MLIRContext* mlir_context_;
  const IndexingMap& indexing_map_;
  bool use_constraints_;
};

// Dimension variable represents a dimension of a tensor or a GPU grid.
// Dimensions correspond to the dimension parameter of `affine_map_`.
struct DimVar {
  Interval bounds;
};
bool operator==(const DimVar& lhs, const DimVar& rhs);
inline bool operator!=(const DimVar& lhs, const DimVar& rhs) {
  return !(lhs == rhs);
}

template <typename H>
H AbslHashValue(H h, const DimVar& dimension) {
  return H::combine(std::move(h), dimension.bounds);
}

inline size_t hash_value(const DimVar& dim_var) {
  return llvm::hash_combine(dim_var.bounds);
}

// RangeSymbol variable represents a range of values, e.g. to compute a single
// element of the reduction's result we need a range of values from the input
// tensor. RangeSymbol variables correspond to the front portion of the
// symbols in `affine_map_`.
struct RangeVar {
  Interval range;
};
bool operator==(const RangeVar& lhs, const RangeVar& rhs);
inline bool operator!=(const RangeVar& lhs, const RangeVar& rhs) {
  return !(lhs == rhs);
}

template <typename H>
H AbslHashValue(H h, const RangeVar& range_var) {
  return H::combine(std::move(h), range_var.range);
}

inline size_t hash_value(const RangeVar& range_var) {
  return llvm::hash_combine(range_var.range);
}

// RTSymbol variable represents a runtime symbol, e.g. a dynamic offset in
// HLO dynamic-update-slice op. RTSymbol variables correspond to the back
// portion of the symbols in `affine_map_`.
struct RTVar {
  Interval feasible_values;
  const HloInstruction* hlo;
  // This is a map from the iteration space of the corresponding indexing map to
  // the iteration space of `hlo`. It shows what element of `hlo` we need to
  // extract to get the runtime value for the RTVar.
  mlir::AffineMap map;
};
bool operator==(const RTVar& lhs, const RTVar& rhs);
inline bool operator!=(const RTVar& lhs, const RTVar& rhs) {
  return !(lhs == rhs);
}

template <typename H>
H AbslHashValue(H h, const RTVar& rt_var) {
  llvm::hash_code map_hash = llvm::hash_combine(rt_var.map);
  return H::combine(std::move(h), rt_var.feasible_values, rt_var.hlo,
                    static_cast<size_t>(map_hash));
}

std::vector<DimVar> DimVarsFromTensorSizes(
    absl::Span<const int64_t> tensor_sizes);

std::vector<RangeVar> RangeVarsFromTensorSizes(
    absl::Span<const int64_t> tensor_sizes);

// Contains an affine map with N dimension expressions and M symbols:
//   (d0, ..., d_{N - 1})[s_0, ..., s_{M - 1}] -> f(d_i, s_j)
// Dimensions d_i correspond to the iteration space of the output tensor. Some
// or all of the dimensions of the input operands can be expressed as a function
// of dimensions of output. For example, for broadcasts and cwise ops all
// dimensions of the inputs are covered by the output dimensions.
// Domain specifies for what ranges of values the indexing map is specified.
//
// Example:
//
// 1. Indexing map for the input of the following reduction
// ```
//   p0 = f32[150, 20, 10, 50] parameter(0)
//   reduce = f32[150, 10] reduce(p0, p0_init), dimensions={3, 1}
// ```
// can be written as `(d0, d1)[s0, s1] -> (d0, s0, d1, s1)`  with
// d0 in [0, 149], d1 in [0, 9], s0 in [0, 19] and s1 in [0, 49].
//
// 2. Indexing map for the input of the reverse op
// ```
//  %p0 = f32[1, 17, 9, 9] parameter(0)
//  reverse = f32[1, 17, 9, 9] reverse(%p0), dimensions={1, 2}
// ```
// can be written as `(d0, d1, d2, d3) -> (d0, -d1 + 16, -d2 + 8, d3)` with
// d0 in [0, 1), d1 in [0, 16], d2 in [0, 8] and d3 in [0, 8].
class IndexingMap {
 public:
  IndexingMap(
      mlir::AffineMap affine_map, std::vector<DimVar> dimensions,
      std::vector<RangeVar> range_vars, std::vector<RTVar> rt_vars,
      absl::Span<std::pair<mlir::AffineExpr, Interval> const> constraints = {});

  IndexingMap(mlir::AffineMap affine_map, std::vector<DimVar> dimensions,
              std::vector<RangeVar> range_vars, std::vector<RTVar> rt_vars,
              const llvm::DenseMap<mlir::AffineExpr, Interval>& constraints);

  // Returns an undefined indexing map.
  static IndexingMap GetUndefined() { return IndexingMap(); }

  static IndexingMap FromTensorSizes(
      mlir::AffineMap affine_map, absl::Span<const int64_t> dim_upper_bounds,
      absl::Span<const int64_t> symbol_upper_bounds);

  std::string ToString(
      const AffineMapPrinter& printer = AffineMapPrinter()) const;

  void Print(std::ostream& out, const AffineMapPrinter& printer) const;

  // Returns true if the map was simplified.
  bool Simplify();

  // Return MLIRContext.
  mlir::MLIRContext* GetMLIRContext() const;

  // Returns the affine map.
  mlir::AffineMap GetAffineMap() const { return affine_map_; }
  mlir::AffineMap& GetMutableAffineMap() { return affine_map_; }

  // Returns the range evaluator for the indexing map's domain.
  RangeEvaluator GetRangeEvaluator() const;

  // Getters for dimension vars.
  const DimVar& GetDimVars(int64_t id) const { return dim_vars_[id]; }
  const std::vector<DimVar>& GetDimVars() const { return dim_vars_; }
  int64_t GetDimVarsCount() const { return dim_vars_.size(); }

  // Getters for range vars.
  const RangeVar& GetRangeVar(int64_t id) const { return range_vars_[id]; }
  const std::vector<RangeVar>& GetRangeVars() const { return range_vars_; }
  int64_t GetRangeVarsCount() const { return range_vars_.size(); }

  // Getters for runtime vars.
  const RTVar& GetRTVar(int64_t id) const { return rt_vars_[id]; }
  const std::vector<RTVar>& GetRTVars() const { return rt_vars_; }
  int64_t GetRTVarsCount() const { return rt_vars_.size(); }

  // Gets bounds of `affine_map_` dimensions.
  const Interval& GetDimensionBound(int64_t dim_id) const;
  Interval& GetMutableDimensionBound(int64_t dim_id);
  std::vector<Interval> GetDimensionBounds() const;
  int64_t GetDimensionCount() const { return affine_map_.getNumDims(); }

  // Gets bounds of `affine_map_` symbols.
  const Interval& GetSymbolBound(int64_t symbol_id) const;
  Interval& GetMutableSymbolBound(int64_t symbol_id);
  std::vector<Interval> GetSymbolBounds() const;
  int64_t GetSymbolCount() const { return affine_map_.getNumSymbols(); }

  // Getters for affine expression constraints.
  const llvm::DenseMap<mlir::AffineExpr, Interval>& GetConstraints() const {
    return constraints_;
  }
  int64_t GetConstraintsCount() const { return constraints_.size(); }

  // Allows to add bounds for the affine expression `expr`. If there are
  // bounds for the `expr`, then computes intersection of the current and new
  // ranges.
  void AddConstraint(mlir::AffineExpr expr, Interval range);
  void ClearConstraints() { constraints_.clear(); }
  void EraseConstraint(mlir::AffineExpr expr);

  // Evaluates the constraints at a given point and returns `true` if all
  // constraints are satisfied.
  bool ConstraintsSatisfied(
      llvm::ArrayRef<mlir::AffineExpr> dim_const_exprs,
      llvm::ArrayRef<mlir::AffineExpr> symbol_const_exprs) const;

  // Evaluates indexing map results at a given point.
  llvm::SmallVector<int64_t, 4> Evaluate(
      llvm::ArrayRef<mlir::AffineExpr> dim_const_exprs,
      llvm::ArrayRef<mlir::AffineExpr> symbol_const_exprs) const;

  // Returns true if the domain is empty. If it returns false, that does not
  // mean that the domain is not effectively empty.
  // For example, if there are two constraints 0 <= d0 mod 7 <= 0 and
  // 0 <= d0 mod 11 <= 0 for a dimension 0<= d0 <= 50 then there is no d0 that
  // satisfies both constraints.
  bool IsKnownEmpty() const { return is_known_empty_; }

  bool IsUndefined() const { return affine_map_ == mlir::AffineMap(); }

  // Removes unused symbols from the `affine_map_` and constraints.
  // Returns a bit vector of symbols that were removed. If none of the symbols
  // were removed, returns {}.
  llvm::SmallBitVector RemoveUnusedSymbols();

  // Removes unused dimensions and symbols from the `affine_map_` and
  // constraints. Returns a bit vector of all variables [dimensions, symbols]
  // that were removed. If none of the symbols were removed, returns {}.
  llvm::SmallBitVector RemoveUnusedVars();

  // Rescales all symbols that are sufficiently constrained through `s? mod x =
  // [N, N]` constraints. Returns true if a rescale took place, otherwise false.
  bool RescaleSymbols();

  // Does `symbol` correspond to a range var?
  bool IsRangeVarSymbol(mlir::AffineSymbolExpr symbol) const;

  // Does `symbol` correspond to an RTVar?
  bool IsRTVarSymbol(mlir::AffineSymbolExpr symbol) const;

  IndexingMap GetSubMap(unsigned int result_index) const {
    return {affine_map_.getSubMap({result_index}), dim_vars_, range_vars_,
            rt_vars_, constraints_};
  }

 private:
  IndexingMap() = default;

  // Merges "mod" constraints for the same AffineExpr.
  // Returns true if simplification was performed.
  bool MergeModConstraints();

  // Replace RTVars that yield constants by indexing expressions.
  // Returns true if simplification was performed.
  bool ReplaceConstantRTVars();

  // Removes DimVars, RangeVars, RTVars that correspond to the unused dimensions
  // and symbols. If unused_dims is empty, then dims won't be removed. The same
  // applies to unused_symbols. Returns true, if anything was removed.
  bool CompressVars(const llvm::SmallBitVector& unused_dims,
                    const llvm::SmallBitVector& unused_symbols);

  // Resets the indexing map to the canonical "known" empty indexing map, i.e.
  // (d0...)[s0...] -> (0...) affine map. Does not change the number of symbols,
  // dimensions or results.
  void ResetToKnownEmpty();

  // Verify if all intervals for DimVars, RangeVars and RTVars are feasible.
  bool VerifyVariableIntervals();

  // Verify if all intervals for constraints.
  bool VerifyConstraintIntervals();

  mlir::AffineMap affine_map_;
  std::vector<DimVar> dim_vars_;
  std::vector<RangeVar> range_vars_;
  std::vector<RTVar> rt_vars_;
  // Inequality constraints for affine expressions. They restrict the feasible
  // set for the domain of the indexing map. It contains affine expressions
  // other than AffineDimExpr and AffineSymbolExpr.
  llvm::DenseMap<mlir::AffineExpr, Interval> constraints_;
  // Flag to indicate that the domain is empty.
  bool is_known_empty_ = false;
};
std::ostream& operator<<(std::ostream& out, const IndexingMap& indexing_map);
bool operator==(const IndexingMap& lhs, const IndexingMap& rhs);
inline bool operator!=(const IndexingMap& lhs, const IndexingMap& rhs) {
  return !(lhs == rhs);
}
IndexingMap operator*(const IndexingMap& lhs, const IndexingMap& rhs);

// Composes affine maps, i.e. second ∘ first.
IndexingMap ComposeIndexingMaps(const IndexingMap& first,
                                const IndexingMap& second);

// Prints the RTVars.
//
// This is exposed to allow SymbolicTile to reuse it.
//
// `first_rt_var_symbol_index`: The index of the symbol associated with the
// first RTVar. The RTVars will be printed with consequent symbol indices
// starting with `first_rt_var_symbol_index`. For example, if `rt_vars.size()
// == 3` and `first_rt_var_symbol_index == 4`, then the symbol names "s4",
// "s5" and "s6" will be used.
//
// TODO(b/334043862): Unexpose this function if possible.
void PrintRTVars(const std::vector<RTVar>& rt_vars,
                 int first_rt_var_symbol_index, std::ostream& out,
                 const AffineMapPrinter& printer);

template <typename H>
H AbslHashValue(H h, const IndexingMap& indexing_map) {
  llvm::hash_code affine_map_hash =
      llvm::hash_combine(indexing_map.GetAffineMap());
  llvm::SmallVector<size_t> constraint_hashes;
  constraint_hashes.reserve(indexing_map.GetConstraintsCount());
  for (const auto& [expr, interval] : indexing_map.GetConstraints()) {
    constraint_hashes.push_back(llvm::hash_combine(expr, interval));
  }
  h = H::combine(std::move(h), static_cast<size_t>(affine_map_hash),
                 indexing_map.GetDimVars(), indexing_map.GetRangeVars(),
                 indexing_map.GetRTVars());
  h = H::combine_unordered(std::move(h), constraint_hashes.begin(),
                           constraint_hashes.end());
  return h;
}

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_INDEXING_MAP_H_
