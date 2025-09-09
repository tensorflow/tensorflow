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

#ifndef XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_MAP_H_
#define XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_MAP_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/types/span.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/service/gpu/model/experimental/symbolic_expr.h"

namespace xla {
namespace gpu {

class SymbolicExprContext;

// Maps a set of input variables to a set of output SymbolicExpr trees.
class SymbolicMap {
 public:
  SymbolicMap() = default;
  static SymbolicMap Get(SymbolicExprContext* ctx, int64_t num_dimensions,
                         int64_t num_symbols,
                         llvm::SmallVector<SymbolicExpr> exprs);

  SymbolicExprContext* GetContext() const { return ctx_; }
  int64_t GetNumDims() const { return num_dimensions_; }
  int64_t GetNumSymbols() const { return num_symbols_; }
  SymbolicExpr GetDimExpression(unsigned idx) const {
    return ctx_->CreateVariable(idx);
  }
  SymbolicExpr GetSymbolExpression(unsigned idx) const {
    return ctx_->CreateVariable(num_dimensions_ + idx);
  }
  int64_t GetNumResults() const { return exprs_.size(); }
  const llvm::SmallVector<SymbolicExpr>& GetResults() const { return exprs_; }
  SymbolicExpr GetResult(unsigned idx) const { return exprs_[idx]; }
  std::string ToString() const;

  bool IsEmpty() const { return exprs_.empty(); }

  // Returns true if each result expression is a direct mapping of the dimension
  // at the same index. Symbols are not considered in this check.
  bool IsIdentity() const;

  // Returns true if all result expressions are constant.
  bool IsConstant() const;

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

  bool operator==(const SymbolicMap& other) const;
  bool operator!=(const SymbolicMap& other) const { return !(*this == other); }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const SymbolicMap& map) {
    sink.Append(map.ToString());
  }

 private:
  SymbolicMap(SymbolicExprContext* ctx, int64_t num_dimensions,
              int64_t num_symbols, llvm::SmallVector<SymbolicExpr> exprs);

  SymbolicExprContext* ctx_;
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

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_MAP_H_
