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

#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "xla/service/gpu/model/affine_map_printer.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {

// A tile describes a structured subset of indices inside an N-dimensional
// array, where the set of indices captured along each dimension can be
// expressed as a strided expression
//     offset + stride * iota(size)
// with offset, stride, and size three integers, and iota the usual range
// function. These values may never be negative.
//
// A N-dimensional symbolic tile is a function from offsets, strides, and sizes
// to a N-dimensional tile. It can be represented as three affine maps with
// domain
//     ()[size0, ..., size{M-1}]
// and respective co-domains
//     (offset0, ..., offset{N-1})     (offset_map())
//     (size0', ..., size'{N-1})       (size_map())
//     (stride0, ..., stride{N-1})     (stride_map())
// where maps respectively encode the offset, size, and stride component of
// each strided expression in the tile. The parameters to the maps above are all
// assumed to be strictly positive. The input offsets are assumed to be all 0s,
// and the input strides are assumed to be all 1s.
//
// A symbolic tile with M symbols and N results is constructed using an
// `IndexingMap` with M input dimensions and N results. The construction of the
// symbolic tile may fail if any one of the resulting expressions is not a
// strided expression as described above.
class SymbolicTile {
 public:
  static std::optional<SymbolicTile> FromIndexingMap(
      const IndexingMap& indexing_map);

  std::string ToString(
      const AffineMapPrinter& printer = AffineMapPrinter()) const;

  void Print(std::ostream& out, const AffineMapPrinter& printer) const;

  mlir::AffineMap offset_map() const { return offset_map_; }
  mlir::AffineMap size_map() const { return size_map_; }
  mlir::AffineMap stride_map() const { return stride_map_; }

 private:
  mlir::AffineMap offset_map_;
  mlir::AffineMap size_map_;
  mlir::AffineMap stride_map_;

  SymbolicTile(mlir::AffineMap offset_map, mlir::AffineMap size_map,
               mlir::AffineMap stride_map)
      : offset_map_(offset_map), size_map_(size_map), stride_map_(stride_map) {}
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_SYMBOLIC_TILE_H_
