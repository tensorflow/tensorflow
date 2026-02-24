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

#include "xla/codegen/tiling/experimental/symbolic_tile.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"

namespace xla::gpu::experimental {
namespace {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::mlir::AffineExpr;
using ::mlir::getAffineConstantExpr;
using ::mlir::getAffineDimExpr;
using ::mlir::getAffineSymbolExpr;
using ::mlir::MLIRContext;

SmallVector<std::string> GetVarNames(int64_t num_vars, llvm::StringRef prefix) {
  SmallVector<std::string> var_names;
  var_names.reserve(num_vars);
  for (int64_t i = 0; i < num_vars; ++i) {
    var_names.push_back(absl::StrFormat("%s%d", prefix, i));
  }
  return var_names;
}

}  // namespace

DimTile GetFullDimTile(int64_t dim_size, MLIRContext* ctx) {
  return DimTile{getAffineConstantExpr(0, ctx),
                 getAffineConstantExpr(llvm::PowerOf2Ceil(dim_size), ctx),
                 getAffineConstantExpr(1, ctx),
                 getAffineConstantExpr(dim_size, ctx)};
}

DimTile GetDefaultDimTile(int64_t id, int64_t dim_size, MLIRContext* ctx) {
  auto tile_id = getAffineDimExpr(id, ctx);
  auto tile_size = getAffineSymbolExpr(id, ctx);
  return DimTile{tile_id * tile_size, tile_size, getAffineConstantExpr(1, ctx),
                 getAffineConstantExpr(dim_size, ctx)};
}

bool DimTile::operator==(const DimTile& other) const {
  return offset == other.offset && size == other.size &&
         stride == other.stride && upper_bound == other.upper_bound;
}

SymbolicTile::SymbolicTile(const TilingSpace& tiling_space,
                           ArrayRef<AffineExpr> offsets,
                           ArrayRef<AffineExpr> sizes,
                           ArrayRef<AffineExpr> strides,
                           ArrayRef<AffineExpr> upper_bounds)
    : tiling_space_(&tiling_space) {
  dim_tiles_.reserve(offsets.size());
  for (auto [offset, size, stride, upper_bound] :
       llvm::zip(offsets, sizes, strides, upper_bounds)) {
    dim_tiles_.push_back(DimTile{offset, size, stride, upper_bound});
  }
}

SymbolicTile::SymbolicTile(const TilingSpace& tiling_space,
                           llvm::SmallVector<DimTile> dim_tiles)
    : tiling_space_(&tiling_space), dim_tiles_(std::move(dim_tiles)) {}

MLIRContext* SymbolicTile::mlir_context() const {
  return tiling_space_->mlir_context();
}

std::string SymbolicTile::ToString(bool print_variables) const {
  int64_t num_dimensions = tiling_space_->num_dimensions();
  auto tid_names = GetVarNames(num_dimensions, "tid_");
  auto ts_names = GetVarNames(num_dimensions, "ts_");
  auto rt_names = GetVarNames(tiling_space_->num_rt_vars(), "rt_");

  std::string s;
  llvm::raw_string_ostream ss(s);

  if (print_variables) {
    // Tile IDs.
    ss << '(' << absl::StrJoin(tid_names, ", ") << ')';
    // Tile size.
    ss << '[' << absl::StrJoin(ts_names, ", ") << ']';
    // Runtime identifiers.
    if (!rt_names.empty()) {
      ss << '{' << absl::StrJoin(rt_names, ", ") << '}';
    }
    ss << " ->";
  }
  SmallVector<std::string, 3> symbol_names;
  symbol_names.reserve(ts_names.size() + rt_names.size());
  symbol_names.append(ts_names.begin(), ts_names.end());
  symbol_names.append(rt_names.begin(), rt_names.end());
  auto print_expr = [&](AffineExpr expr) {
    ss << ::xla::ToString(expr, tid_names, symbol_names);
  };
  // Print offsets.
  ss << " offsets [";
  llvm::interleaveComma(offsets(), ss, print_expr);
  ss << "] sizes [";
  llvm::interleaveComma(sizes(), ss, print_expr);
  ss << "] strides [";
  llvm::interleaveComma(strides(), ss, print_expr);
  ss << "] upper bounds [";
  llvm::interleaveComma(upper_bounds(), ss, print_expr);
  ss << ']';
  return s;
}

SmallVector<AffineExpr> SymbolicTile::offsets() const {
  SmallVector<AffineExpr> offsets;
  offsets.reserve(offsets.size());
  for (const DimTile& dim_tile : dim_tiles_) {
    offsets.push_back(dim_tile.offset);
  }
  return offsets;
}

SmallVector<AffineExpr> SymbolicTile::sizes() const {
  SmallVector<AffineExpr> sizes;
  sizes.reserve(sizes.size());
  for (const DimTile& dim_tile : dim_tiles_) {
    sizes.push_back(dim_tile.size);
  }
  return sizes;
}

SmallVector<AffineExpr> SymbolicTile::strides() const {
  SmallVector<AffineExpr> strides;
  strides.reserve(strides.size());
  for (const DimTile& dim_tile : dim_tiles_) {
    strides.push_back(dim_tile.stride);
  }
  return strides;
}

SmallVector<AffineExpr> SymbolicTile::upper_bounds() const {
  SmallVector<AffineExpr> upper_bounds;
  upper_bounds.reserve(upper_bounds.size());
  for (const DimTile& dim_tile : dim_tiles_) {
    upper_bounds.push_back(dim_tile.upper_bound);
  }
  return upper_bounds;
}

bool SymbolicTile::operator==(const SymbolicTile& other) const {
  return tiling_space_ == other.tiling_space_ && dim_tiles_ == other.dim_tiles_;
}

}  // namespace xla::gpu::experimental
