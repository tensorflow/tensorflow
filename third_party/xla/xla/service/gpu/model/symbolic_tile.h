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

#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "xla/service/gpu/model/affine_map_printer.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {

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
// offset_map(): ()[size0, ..., size{M-1}] -> (offset0, ..., offset{N-1})
// size_map():   ()[size0, ..., size{M-1}] -> (size'0, ..., size'{N-1})
// stride_map(): ()[size0, ..., size{M-1}] -> (stride0, ..., stride{N-1})
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
// (offsets, strides, sizes): (Z^k -> N^n) x N^n x (N+)^n
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
// offset_map(): ()[sizes..., rt_vars...] -> offsets'
// size_map():   ()[sizes...] -> sizes'
// stride_map(): ()[sizes...] -> strides'
//
// Other than this, the SymbolicTile object also contains a vector of RTVars
// (rt_vars()) which describe how to evaluate the runtime value of rt_vars.
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
  static std::optional<SymbolicTile> FromIndexingMap(
      const IndexingMap& indexing_map);

  // For printing in tests.
  std::string RtVarsToString(
      const AffineMapPrinter& printer = AffineMapPrinter()) const;
  std::string ToString(
      const AffineMapPrinter& printer = AffineMapPrinter()) const;

  void Print(std::ostream& out, const AffineMapPrinter& printer) const;

  mlir::AffineMap offset_map() const;
  mlir::AffineMap size_map() const;
  mlir::AffineMap stride_map() const;

  // A map from one tile's sizes and RTVars to another tile's offsets, sizes,
  // and strides.
  //
  // (size0, ..., size{n-1})
  // [rt_var0, ..., rt_var{k-1}] -> (offset0, ..., offset{n-1},
  //                                 size'0, ..., size'{n-1},
  //                                 stride0, ..., stride{n-1})
  //
  //
  // Its type is IndexingMap, but it's not a map of indices.
  // This indexing map wraps the relevant domain constraints.
  //
  // Warning: The dimensions and symbols in tile_map do not match the dimensions
  // and symbols in offset_map, size_map, and stride_map.
  const IndexingMap& tile_map() const { return tile_map_; }

  // This allows GUnit to print the tile.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const SymbolicTile& tile) {
    sink.Append(tile.ToString());
  }

 private:
  // See the comment of tile_map().
  IndexingMap tile_map_;

  explicit SymbolicTile(IndexingMap tile_map)
      : tile_map_(std::move(tile_map)) {}
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_SYMBOLIC_TILE_H_
