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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "xla/service/gpu/model/affine_map_printer.h"

namespace xla {
namespace gpu {

// Range represents a closed interval [lower_bound, upper_bound].
struct Range {
  std::string ToString() const;
  void Print(std::ostream& out) const;

  bool IsPoint() const { return lower_bound == upper_bound; }

  int64_t lower_bound = 0;
  int64_t upper_bound = 0;
};
std::ostream& operator<<(std::ostream& out, const Range& range);
bool operator==(const Range& lhs, const Range& rhs);

template <typename H>
H AbslHashValue(H h, const Range& range) {
  return H::combine(std::move(h), range.lower_bound, range.upper_bound);
}

// Domain contains ranges for symbols and dimensions of an affine map.
struct Domain {
  std::string ToString(
      const AffineMapPrinter& printer = AffineMapPrinter()) const;

  void Print(std::ostream& out, const AffineMapPrinter& printer) const;

  static Domain FromUpperBounds(
      absl::Span<const int64_t> dimension_upper_bounds,
      absl::Span<const int64_t> symbol_upper_bounds);

  std::vector<Range> dimension_ranges;
  std::vector<Range> symbol_ranges;
};
std::ostream& operator<<(std::ostream& out, const Domain& domain);
bool operator==(const Domain& lhs, const Domain& rhs);

template <typename H>
H AbslHashValue(H h, const Domain& domain) {
  return H::combine(std::move(h), domain.dimension_ranges,
                    domain.symbol_ranges);
}

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
struct IndexingMap {
  std::string ToString(
      const AffineMapPrinter& printer = AffineMapPrinter()) const;

  void Print(std::ostream& out, const AffineMapPrinter& printer) const;

  // Returns true if the map was simplified.
  bool Simplify();

  mlir::AffineMap affine_map;
  Domain domain;
};
std::ostream& operator<<(std::ostream& out, const IndexingMap& indexing_map);
bool operator==(const IndexingMap& lhs, const IndexingMap& rhs);

// Composes affine maps, i.e. consumer_map ∘ producer_map.
// Right now the ranges of the composed indexing map are correct only when there
// is no composition with concat.
// TODO(b/319410501): Generalize domain modelling.
std::optional<IndexingMap> ComposeIndexingMaps(
    const std::optional<IndexingMap>& producer_map,
    const std::optional<IndexingMap>& consumer_map);

template <typename H>
H AbslHashValue(H h, const IndexingMap& indexing_map) {
  llvm::hash_code affine_map_hash = llvm::hash_combine(indexing_map.affine_map);
  return H::combine(std::move(h), static_cast<size_t>(affine_map_hash),
                    indexing_map.domain);
}

class IndexingMapSimplifier {
 public:
  explicit IndexingMapSimplifier(mlir::MLIRContext* mlir_context)
      : mlir_context_(mlir_context) {}

  // Derives an indexing map simplifier for the parameter indexing map.
  static IndexingMapSimplifier FromIndexingMap(const IndexingMap& indexing_map);

  // Sets the [lower, upper] range for the given expression. It can be used to
  // set bounds for dimensions and symbols.
  void SetRange(mlir::AffineExpr expr, int64_t lower, int64_t upper);

  // Simplifies the map as much as possible.
  mlir::AffineMap Simplify(mlir::AffineMap affine_map);

  // Simplifies the expression as much as possible.
  mlir::AffineExpr Simplify(mlir::AffineExpr expr);

  // Checks whether an `AffineExpr` always describes a non-negative value.
  bool IsAlwaysPositiveOrZero(mlir::AffineExpr expr);

  // Checks whether an `AffineExpr` always describes a non-positive value.
  bool IsAlwaysNegativeOrZero(mlir::AffineExpr expr);

 private:
  Range GetRange(mlir::AffineExpr expr);

  std::optional<int64_t> GetConstantRhsMultiplier(mlir::AffineExpr expr);

  // Simplifier for mod.
  // - Rewrites (a * 100 + ...) % 100 to (...) % 100
  // - Rewrites a % b to a if a is known to be less than b.
  mlir::AffineExpr RewriteMod(mlir::AffineBinaryOpExpr mod);

  // Simplifier for floordiv.
  // - Rewrites (a * 100 + ...) / 100 to a + (...) / 100
  // - Rewrites a / 100 to 0 when a is known to be less than 100.
  mlir::AffineExpr RewriteFloorDiv(mlir::AffineBinaryOpExpr div);

  mlir::AffineExpr RewriteSumIf(
      mlir::AffineExpr expr, const std::function<bool(mlir::AffineExpr)>& pred);

  // Attempts to simplify the expression, but doesn't attempt to simplify the
  // result further.
  mlir::AffineExpr SimplifyOnce(mlir::AffineExpr expr);

  mlir::MLIRContext* mlir_context_;
  llvm::DenseMap<mlir::AffineExpr, Range> ranges_{};
};


}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_INDEXING_MAP_H_
