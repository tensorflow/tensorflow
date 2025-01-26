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

#ifndef XLA_SERVICE_GPU_MODEL_SYMBOLIC_TILE_H_
#define XLA_SERVICE_GPU_MODEL_SYMBOLIC_TILE_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "xla/hlo/analysis/indexing_map.h"

namespace xla {
namespace gpu {

// `ConstraintExpression` represents a "flat" constraint expression of the form
//   ((expr0 in interval0) && (expr1 in interval1)...) ||
//   ((expr{n} in interval{n}) &&...)...
//
// The underlying constraints are stored in a vector of vectors, such that each
// innermost vector represents the conjunction of some constraints, and the
// outermost vector represents the disjunction of all its elements
// (conjunctions). This representation is effective because `&&` (`And`) is
// distributive over `||` (`Or`), ensuring that we can always flatten any given
// `ConstraintExpression` in this way, and that we have reasonable combinators
// for `&&` and `||`.
//
// We store a boolean `is_satisfiable_` to indicate whether we expect that the
// constraints can be satisfied. When set to `false`, we expect the
// `ConstraintExpression` to be empty (bottom).
class ConstraintExpression {
 public:
  struct Constraint {
    mlir::AffineExpr expr;
    Interval interval;

    bool operator==(const Constraint& other) const {
      CHECK_EQ(expr.getContext(), other.expr.getContext())
          << "AffineExpr should be from the same MLIRContext.";
      return expr == other.expr && interval == other.interval;
    }
  };

 private:
  using ConjointConstraints = llvm::SmallVector<Constraint, 2>;
  explicit ConstraintExpression(bool is_satisfiable)
      : is_satisfiable_(is_satisfiable) {}

 public:
  // Constructs a `ConstraintExpression` from a single `Constraint`.
  explicit ConstraintExpression(const Constraint& constraint)
      : disjoint_conjoint_constraints_({{constraint}}) {}

  // Constructs a `ConstraintExpression` that is always satisfied.
  static ConstraintExpression GetAlwaysSatisfied() {
    return ConstraintExpression(true);
  }

  // Constructs a `ConstraintExpression` that is unsatisfiable.
  static ConstraintExpression GetUnsatisfiable() {
    return ConstraintExpression(false);
  }

  // Takes the conjunction of the constraints of `first` and `second`.
  friend ConstraintExpression operator&&(const ConstraintExpression& first,
                                         const ConstraintExpression& second);

  // Takes the disjunction of the constraints of `first` and `second`.
  friend ConstraintExpression operator||(const ConstraintExpression& first,
                                         const ConstraintExpression& second);

  // Whether the constraints can be satisfied.
  bool is_satisfiable() const { return is_satisfiable_; }

  // Returns `true` if the constraint expression is marked satisfiable and does
  // not contain any constraint.
  bool IsAlwaysSatisfied() const {
    return is_satisfiable_ && disjoint_conjoint_constraints_.empty();
  }

  // Returns `true` if the constraint expression is satisfied by the provided
  // dim_values, and `false` otherwise.  The caller is responsible for ensuring
  // that the number of provided dim_values is sufficient to verify the
  // constraints.
  bool IsSatisfiedBy(absl::Span<const int64_t> dim_values) const;

  std::string ToString() const;

  void Print(std::ostream& out) const;

  // Simplifies the constraint expression.
  //
  // We remove conjunctions that are always satisfied, and we remove
  // disjunctions that are unsatisfiable. If we can deduce that the whole
  // expression is unsatisfiable or always satisfied, than we change the whole
  // expression to the canonical form.
  //
  // E.g., if we find that one of the conjunctions is always satisfied, we don't
  // just throw away that part---we throw away everything and make the
  // ConstraintExpression canonically always satisfied.
  void Simplify();

 private:
  // This allows GUnit to print the expression.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ConstraintExpression& expr) {
    sink.Append(expr.ToString());
  }

  template <typename H>
  friend H AbslHashValue(H h, const Constraint& constraint) {
    llvm::hash_code expr_hash = mlir::hash_value(constraint.expr);
    return H::combine(std::move(h), static_cast<size_t>(expr_hash),
                      constraint.interval);
  }

  template <typename H>
  friend H AbslHashValue(H h, const ConjointConstraints& conjoint_constraints) {
    for (const auto& constraint : conjoint_constraints) {
      h = H::combine(std::move(h), constraint);
    }
    return h;
  }

  // When this is set to `false`, disjoint_conjoint_constraints_ must be empty.
  bool is_satisfiable_ = true;
  llvm::SmallVector<ConjointConstraints, 2> disjoint_conjoint_constraints_;
};

// Logical operators between `ConstraintExpression` and `Constraint`.
inline ConstraintExpression operator&&(
    const ConstraintExpression::Constraint& first,
    const ConstraintExpression& second) {
  return ConstraintExpression(first) && second;
}

inline ConstraintExpression operator&&(
    const ConstraintExpression& first,
    const ConstraintExpression::Constraint& second) {
  return first && ConstraintExpression(second);
}

inline ConstraintExpression operator&&(
    const ConstraintExpression::Constraint& first,
    const ConstraintExpression::Constraint& second) {
  return ConstraintExpression(first) && ConstraintExpression(second);
}

inline ConstraintExpression operator||(
    const ConstraintExpression::Constraint& first,
    const ConstraintExpression& second) {
  return ConstraintExpression(first) || second;
}

inline ConstraintExpression operator||(
    const ConstraintExpression& first,
    const ConstraintExpression::Constraint& second) {
  return first || ConstraintExpression(second);
}

inline ConstraintExpression operator||(
    const ConstraintExpression::Constraint& first,
    const ConstraintExpression::Constraint& second) {
  return ConstraintExpression(first) || ConstraintExpression(second);
}

// Tiling in the simpler case, when we don't have dynamic offsets (see the
// general case later):
//
// An N-dimensional *tile* describes a structured subset of
// indices inside an N-dimensional array, where the set of indices captured
// along each dimension can be expressed as a strided expression
//     offset + stride * iota(size)
//
// where offset and stride are non-negative integers, size is a strictly
// positive integer and `iota` is the usual range function.
//
// An *N-dimensional symbolic tile* is a function from an M-dimensional
// tile to an N-dimensional tile. The input tile is assumed to have all offsets
// equal to 0 and all strides equal to 1.
//
// It is represented with "tile_map()", which is an IndexingMap of this form:
// (size0, ..., size{n-1}) ->  (offset0, ..., offset{n-1},
//                              size'0, ..., size'{n-1},
//                              stride0, ..., stride{n-1})
//
// We can get three AffineMap projections of tile_map(), which are just
// convenience methods to get the components that we need:
//     offset_map(): (size0, ..., size{M-1}) -> (offset0, ..., offset{N-1})
//     size_map():   (size0, ..., size{M-1}) -> (size'0, ..., size'{N-1})
//     stride_map(): (size0, ..., size{M-1}) -> (stride0, ..., stride{N-1})
//
// The maps respectively encode the offset, size, and stride component of each
// strided expression in the result tile.
//
// A symbolic tile with M symbols and N results is constructed using an
// `IndexingMap` with M input dimensions and N results. The construction of the
// symbolic tile may fail if any one of the resulting expressions is not a
// strided expression as described above.
//
// Tiling in the general case:
//
// In general, the offsets of the tile can depend on runtime variables. Runtime
// variables are evaluated to an element of a tensor at runtime for each
// multi-index of the output tensor. This allows support for dynamic offsets,
// for example in dynamic-slice. Until runtime, they are kept in a symbolic
// form. In the following reasoning, we assume that the runtime variables would
// evaluate to the same value for all elements of any given input tile. (This is
// trivially true for dynamic-slice, but we have to choose tiles wisely for
// gather for example.) In the following math, with the help of the previous
// assumption, we represent runtime variables as integer parameters. Note that
// the earlier concepts are here redefined in a more general form.
//
// Def. An n-dimensional tile is a function:
// t: Z^k -> P(N^n) =
//    rt_vars -> CartesianProduct_{i=1, ..., n-1}({
//           offsets(rt_vars)[i] + strides[i] * 0,
//           ...,
//           offsets(rt_vars)[i] + strides[i] * (sizes[i]-1)
//         })
// where
//    Z is the set of integers,
//    P is the power set operator (set of all subsets),
//    N is the set of non-negative integers
//    N+ is the set of positive integers
//    N^n meaning the set of n-tuples of non-negative integers
//
//    rt_vars: Z^k (so called "runtime variables")
//    offsets: Z^k -> N^n
//    strides: N^n
//    sizes: (N+)^n
//
// Notation. We can represent n-dimensional tiles as:
//   (offsets, strides, sizes): (Z^k -> N^n) x N^n x (N+)^n
// where A x B means a Cartesian product.
//
// Def. Let Tiles(n) denote the set of n-dimensional tiles.
//
// Def. An n-dimensional "symbolic tile" is a function:
// s: U_{m: N} (Tiles(m) -> Tiles(n))
// where U represents a union of sets.
//
// Notation. We can represent n-dimensional symbolic tiles of the form
// (offsets, strides, sizes) : Tiles(m)
//   -> (offsets', strides', sizes') : Tiles(n)
// as a vector of functions:
//   (offset_map, stride_map, size_map) where:
//     offset_map: ((Z^j -> N^m) x N^m x (N+)^m) -> (Z^k -> N^n)
//     stride_map: ((Z^j -> N^m) x N^m x (N+)^m) -> N^n
//     size_map: ((Z^j -> N^m) x N^m x (N+)^m) -> (N+)^n
// where each "map" returns one component of the result Tile.
//
// If we assume that offsets=({} -> {0, ..., 0}) and strides={1, ..., 1}, then
// we can simplify the definition:
//     offset_map: (N+)^m -> (Z^k -> N^n)
//     stride_map: (N+)^m -> N^n
//     size_map: (N+)^m -> (N+)^n
//
// As a notation, we can further simplify the structure of offset_map:
//   offset_map: (N+)^m x Z^k -> N^n
// As a notation, we can call the last k parameters of offset_map "rt_vars".
//
// In the code we represent a symbolic tile with "tile_map()", which is an
// IndexingMap of this form:
// (size0, ..., size{n-1})
// [rt_var0, ..., rt_var{k-1}] -> (offset0, ..., offset{n-1},
//                                 size'0, ..., size'{n-1},
//                                 stride0, ..., stride{n-1})
//
// We can get three AffineMap projections of tile_map(), which are just
// convenience methods to get the components that we need:
// offset_map(): (sizes...)[rt_vars...] -> offsets'
// size_map():   (sizes...) -> sizes'
// stride_map(): (sizes...) -> strides'
//
// The size parameters of the projections may be arbitrarily constrained, in
// order to ensure that applying the symbolic tile on an input tile yields a
// valid tile. Such constraints are exposed through the constraints() method.
// It may happen that constraints are unsatisfiable; in that case, the boolean
// is_satisfiable() is set to false. This boolean should always be checked
// before using the content of constraints().
//
// To correctly evaluate the RTVars for a given tile, we have to feed an
// index from the original tile (a tile of the output tensor) to the RTVar's
// affine map. (The RTVars in the symbolic tile are not adjusted to take indices
// from the result tile.)
//
// Note: Currently runtime offsets are relative to the whole tensor, while other
// offsets are local to the position of the input tile. This will be probably
// simplified later.
class SymbolicTile {
 public:
  static std::optional<SymbolicTile> FromIndexingMap(IndexingMap indexing_map);

  // For printing in tests.
  std::string ToString() const;

  void Print(std::ostream& out) const;

  mlir::AffineMap offset_map() const;
  mlir::AffineMap size_map() const;
  mlir::AffineMap stride_map() const;

  // Constraints on the `sizes` of the input tile. Content is irrelevant when
  // `is_satisfiable()` is false.
  const ConstraintExpression& constraints() const {
    CHECK(constraints_.is_satisfiable());
    return constraints_;
  }

  // Whether the `SymbolicTile` constraints can be satisfied. When this is set
  // to `false`, the domain of the `SymbolicTile` must be considered empty.
  bool is_satisfiable() const { return constraints_.is_satisfiable(); }

  // A map from one tile's sizes and RTVars to another tile's offsets, sizes,
  // and strides.
  //
  // (size0, ..., size{n-1})
  // [rt_var0, ..., rt_var{k-1}] -> (offset0, ..., offset{n-1},
  //                                 size'0, ..., size'{n-1},
  //                                 stride0, ..., stride{n-1})
  const IndexingMap& tile_map() const { return tile_map_; }

  // This allows GUnit to print the tile.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const SymbolicTile& tile) {
    sink.Append(tile.ToString());
  }

 private:
  // See the comment of tile_map().
  IndexingMap tile_map_;

  // See the comment of constraints().
  ConstraintExpression constraints_;

  explicit SymbolicTile(IndexingMap tile_map, ConstraintExpression constraints)
      : tile_map_(std::move(tile_map)), constraints_(std::move(constraints)) {}
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_SYMBOLIC_TILE_H_
