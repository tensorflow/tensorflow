/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_HLO_ANALYSIS_SYMBOLIC_MAP_H_
#define XLA_HLO_ANALYSIS_SYMBOLIC_MAP_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"

namespace xla {

// SymbolicMap abstracts away the fact that dimensions and symbols are both
// implemented as SymbolicExpr variables. These free functions provide a way to
// work with them without a SymbolicMap instance.
inline SymbolicExpr CreateDimExpr(unsigned dim_id, mlir::MLIRContext* context) {
  return CreateSymbolicVariable(dim_id, context);
}

inline SymbolicExpr CreateSymbolExpr(unsigned symbol_id, int64_t num_dims,
                                     mlir::MLIRContext* context) {
  return CreateSymbolicVariable(symbol_id + num_dims, context);
}

inline bool IsDimension(SymbolicExpr expr, int64_t num_dims) {
  return expr.GetType() == SymbolicExprType::kVariable &&
         expr.GetValue() < num_dims;
}

inline bool IsSymbol(SymbolicExpr expr, int64_t num_dims) {
  return expr.GetType() == SymbolicExprType::kVariable &&
         expr.GetValue() >= num_dims;
}

inline int64_t GetDimensionIndex(SymbolicExpr expr, int64_t num_dims) {
  CHECK(IsDimension(expr, num_dims));
  return expr.GetValue();
}

inline int64_t GetSymbolIndex(SymbolicExpr expr, int64_t num_dims) {
  CHECK(IsSymbol(expr, num_dims));
  return expr.GetValue() - num_dims;
}

// Maps a set of input variables to a set of output SymbolicExpr trees.
class SymbolicMap {
 public:
  SymbolicMap() = default;
  static SymbolicMap Get(mlir::MLIRContext* ctx, int64_t num_dimensions,
                         int64_t num_symbols,
                         llvm::SmallVector<SymbolicExpr> exprs);

  explicit operator bool() const { return ctx_ != nullptr; }
  bool operator!() const { return ctx_ == nullptr; }

  mlir::MLIRContext* GetContext() const { return ctx_; }
  int64_t GetNumDims() const { return num_dimensions_; }
  int64_t GetNumSymbols() const { return num_symbols_; }
  SymbolicExpr GetDimExpression(unsigned idx) const {
    return CreateDimExpr(idx, ctx_);
  }
  SymbolicExpr GetSymbolExpression(unsigned idx) const {
    return CreateSymbolExpr(idx, num_dimensions_, ctx_);
  }
  int64_t GetNumResults() const { return exprs_.size(); }
  llvm::ArrayRef<SymbolicExpr> GetResults() const { return exprs_; }
  SymbolicExpr GetResult(unsigned idx) const { return exprs_[idx]; }
  std::string ToString() const;

  bool IsEmpty() const { return exprs_.empty(); }

  // Returns true if each result expression is a direct mapping of the dimension
  // at the same index. Symbols are not considered in this check.
  bool IsIdentity() const;

  // Returns true if all result expressions are constant.
  bool IsConstant() const;

  // Returns true if any result expression depends on the given dimension.
  bool IsFunctionOfDim(int64_t dim_id) const;

  // Returns true if any result expression depends on the given symbol.
  bool IsFunctionOfSymbol(int64_t symbol_id) const;

  // Returns a vector containing the values of all the results. CHECK-fails if
  // any result expression is not a constant.
  llvm::SmallVector<int64_t> GetConstantResults() const;

  // Replaces the dimensions and symbols in the map with the given expressions.
  // The number of dimension and symbol replacements must match the number of
  // dimensions and symbols in the map. The new map will have the given number
  // of dimensions and symbols.
  SymbolicMap ReplaceDimsAndSymbols(
      absl::Span<const SymbolicExpr> dim_replacements,
      absl::Span<const SymbolicExpr> sym_replacements, int64_t num_result_dims,
      int64_t num_result_symbols) const;

  // Composes this map with another map. The number of dimensions of this map
  // must match the number of results of the other map. The resulting map will
  // have the same number of dimensions as the other map, and the number of
  // symbols will be the sum of the number of symbols in both maps.
  //
  // The variables in the composed map are ordered as follows:
  // * dimensions of the other map
  // * symbols of this map
  // * symbols of the other map
  //
  // Example:
  // this: (d0, d1, s0) -> (d0 + s0, d1)
  // other: (d0, s0, s1) -> (d0 * 2 + 3 * s0, d0 + s1)
  // this.compose(other): (d0, s0, s1, s2) -> (d0 * 2 + 3 * s1 + s0, d0 + s2)
  SymbolicMap Compose(const SymbolicMap& other) const;

  // Creates a new SymbolicMap with a subset of the results of this map.
  SymbolicMap GetSubMap(absl::Span<const size_t> result_indices) const;

  SymbolicMap Replace(SymbolicExpr expr, SymbolicExpr replacement) const;

  /// Replaces multiple sub-expressions at once by applying
  /// `SymbolicExpr::Replace(map)` to each expression. Returns a new SymbolicMap
  /// with the new results and with the specified number of dims and symbols.
  SymbolicMap Replace(const llvm::DenseMap<SymbolicExpr, SymbolicExpr>& map,
                      int64_t numResultDims, int64_t numResultSyms) const;

  bool operator==(const SymbolicMap& other) const;
  bool operator!=(const SymbolicMap& other) const { return !(*this == other); }

  template <typename H>
  friend H AbslHashValue(H h, const SymbolicMap& map) {
    return H::combine(std::move(h), map.num_dimensions_, map.num_symbols_,
                      map.exprs_);
  }

  friend ::llvm::hash_code hash_value(const SymbolicMap& map) {
    return ::llvm::hash_combine(
        map.num_dimensions_, map.num_symbols_,
        ::llvm::hash_combine_range(map.exprs_.begin(), map.exprs_.end()));
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const SymbolicMap& map) {
    sink.Append(map.ToString());
  }

 private:
  SymbolicMap(mlir::MLIRContext* ctx, int64_t num_dimensions,
              int64_t num_symbols, llvm::SmallVector<SymbolicExpr> exprs);

  mlir::MLIRContext* ctx_;
  int64_t num_dimensions_;
  int64_t num_symbols_;
  llvm::SmallVector<SymbolicExpr> exprs_;
};

// Returns a bitvector marking dimensions that are not used in any expression in
// the map.
llvm::SmallBitVector GetUnusedDimensionsBitVector(const SymbolicMap& map);

// Returns a bitvector marking symbols that are not used in any expression in
// the map.
llvm::SmallBitVector GetUnusedSymbolsBitVector(const SymbolicMap& map);

// Creates a new SymbolicMap with unused dimensions removed.
// Expressions are updated to use the new dimension indices.
SymbolicMap CompressDims(const SymbolicMap& map,
                         const llvm::SmallBitVector& unused_dims);

// Creates a new SymbolicMap with unused symbols removed.
// Expressions are updated to use the new symbol indices.
SymbolicMap CompressSymbols(const SymbolicMap& map,
                            const llvm::SmallBitVector& unused_symbols);

template <typename H>
H AbslHashValue(H h, const llvm::SmallVector<SymbolicExpr>& vec) {
  return H::combine(std::move(h), absl::MakeSpan(vec));
}

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_SYMBOLIC_MAP_H_
