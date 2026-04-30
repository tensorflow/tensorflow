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

#include "xla/codegen/tiling/experimental/tile.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"

namespace xla::gpu::experimental {
namespace {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::mlir::MLIRContext;

SmallVector<std::string> GetVarNames(int64_t num_vars, llvm::StringRef prefix) {
  SmallVector<std::string> var_names;
  var_names.reserve(num_vars);
  for (int64_t i = 0; i < num_vars; ++i) {
    var_names.push_back(absl::StrFormat("%s%d", prefix, i));
  }
  return var_names;
}

absl::StatusOr<SmallVector<int64_t>> ConvertSymbolicExprsToInts(
    ArrayRef<SymbolicExpr> symbolic_exprs) {
  SmallVector<int64_t> result;
  result.reserve(symbolic_exprs.size());
  for (const auto& symbolic_expr : symbolic_exprs) {
    SymbolicExpr canonical_expr = symbolic_expr.Canonicalize();
    if (canonical_expr.GetType() != SymbolicExprType::kConstant) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Symbolic expression is not a constant: ", canonical_expr));
    }
    result.push_back(canonical_expr.GetValue());
  }
  return result;
}

}  // namespace

DimTile GetFullDimTile(int64_t dim_size, MLIRContext* ctx) {
  return DimTile{CreateSymbolicConstant(0, ctx),
                 CreateSymbolicConstant(llvm::PowerOf2Ceil(dim_size), ctx),
                 CreateSymbolicConstant(1, ctx),
                 CreateSymbolicConstant(dim_size, ctx)};
}

DimTile GetDefaultDimTile(int64_t id, SymbolicExpr tile_size,
                          int64_t dim_size) {
  MLIRContext* ctx = tile_size.GetContext();
  auto tile_id = CreateDimExpr(id, ctx);
  return DimTile{tile_id * tile_size, tile_size, CreateSymbolicConstant(1, ctx),
                 CreateSymbolicConstant(dim_size, ctx)};
}

bool DimTile::operator==(const DimTile& other) const {
  return offset == other.offset && size == other.size &&
         stride == other.stride && upper_bound == other.upper_bound;
}

Tile::Tile(const TilingSpace& tiling_space, ArrayRef<SymbolicExpr> offsets,
           ArrayRef<SymbolicExpr> sizes, ArrayRef<SymbolicExpr> strides,
           ArrayRef<SymbolicExpr> upper_bounds)
    : tiling_space_(&tiling_space) {
  dim_tiles_.reserve(offsets.size());
  for (auto [offset, size, stride, upper_bound] :
       llvm::zip(offsets, sizes, strides, upper_bounds)) {
    dim_tiles_.push_back(DimTile{offset, size, stride, upper_bound});
  }
}

Tile::Tile(const TilingSpace& tiling_space,
           llvm::SmallVector<DimTile> dim_tiles)
    : tiling_space_(&tiling_space), dim_tiles_(std::move(dim_tiles)) {}

MLIRContext* Tile::mlir_context() const {
  return tiling_space_->mlir_context();
}

std::string Tile::ToString(bool print_variables) const {
  int64_t num_dimensions = tiling_space_->num_dimensions();
  auto tid_names = GetVarNames(num_dimensions, "tid_");
  auto ts_names = GetVarNames(num_dimensions, "ts_");
  auto rt_names = GetVarNames(tiling_space_->num_rt_vars(), "rt_");

  std::string s;
  llvm::raw_string_ostream ss(s);

  if (print_variables) {
    // Tile IDs.
    ss << '(' << absl::StrJoin(tid_names, ", ") << ')';
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
  auto print_expr = [&](SymbolicExpr expr) {
    ss << expr.ToString(tid_names, symbol_names);
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
  return ss.str();
}

SmallVector<SymbolicExpr> Tile::offsets() const {
  SmallVector<SymbolicExpr> offsets;
  offsets.reserve(dim_tiles_.size());
  for (const DimTile& dim_tile : dim_tiles_) {
    offsets.push_back(dim_tile.offset);
  }
  return offsets;
}

SmallVector<SymbolicExpr> Tile::sizes() const {
  SmallVector<SymbolicExpr> sizes;
  sizes.reserve(dim_tiles_.size());
  for (const DimTile& dim_tile : dim_tiles_) {
    sizes.push_back(dim_tile.size);
  }
  return sizes;
}

SmallVector<SymbolicExpr> Tile::strides() const {
  SmallVector<SymbolicExpr> strides;
  strides.reserve(dim_tiles_.size());
  for (const DimTile& dim_tile : dim_tiles_) {
    strides.push_back(dim_tile.stride);
  }
  return strides;
}

SmallVector<SymbolicExpr> Tile::upper_bounds() const {
  SmallVector<SymbolicExpr> upper_bounds;
  upper_bounds.reserve(dim_tiles_.size());
  for (const DimTile& dim_tile : dim_tiles_) {
    upper_bounds.push_back(dim_tile.upper_bound);
  }
  return upper_bounds;
}

absl::StatusOr<llvm::SmallVector<int64_t>> Tile::GetStaticTileSizes() const {
  return ConvertSymbolicExprsToInts(sizes());
}

absl::StatusOr<llvm::SmallVector<int64_t>> Tile::GetStaticTileStrides() const {
  return ConvertSymbolicExprsToInts(strides());
}

void Tile::Replace(const llvm::DenseMap<SymbolicExpr, SymbolicExpr>& map) {
  for (DimTile& dim_tile : dim_tiles_) {
    dim_tile.offset = dim_tile.offset.Replace(map);
    dim_tile.size = dim_tile.size.Replace(map);
    dim_tile.stride = dim_tile.stride.Replace(map);
    dim_tile.upper_bound = dim_tile.upper_bound.Replace(map);
  }
}

bool Tile::operator==(const Tile& other) const {
  return tiling_space_ == other.tiling_space_ && dim_tiles_ == other.dim_tiles_;
}

}  // namespace xla::gpu::experimental
