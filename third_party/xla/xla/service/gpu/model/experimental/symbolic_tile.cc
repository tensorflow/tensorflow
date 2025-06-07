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

#include "xla/service/gpu/model/experimental/symbolic_tile.h"

#include <cstdint>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla::gpu {
namespace {

using ::llvm::SmallVector;
using ::mlir::AffineExpr;
using ::mlir::AffineMap;

SmallVector<std::string> GetVarNames(int64_t num_vars, llvm::StringRef prefix) {
  SmallVector<std::string> var_names;
  var_names.reserve(num_vars);
  for (int64_t i = 0; i < num_vars; ++i) {
    var_names.push_back(absl::StrFormat("%s%d", prefix, i));
  }
  return var_names;
}

std::string ToStringImpl(AffineMap affine_map,
                         absl::Span<const HloInstruction* const> rt_vars,
                         absl::Span<const std::string> tid_names,
                         absl::Span<const std::string> ts_names,
                         absl::Span<const std::string> rt_names) {
  CHECK_EQ(tid_names.size(), affine_map.getNumDims());
  CHECK_EQ(ts_names.size() + rt_names.size(), affine_map.getNumSymbols());

  std::string s;
  llvm::raw_string_ostream ss(s);

  // Tile IDs.
  ss << '(' << absl::StrJoin(tid_names, ", ") << ')';
  // Tile size.
  ss << '[' << absl::StrJoin(ts_names, ", ") << ']';
  // Runtime identifiers.
  if (!rt_names.empty()) {
    ss << '{' << absl::StrJoin(rt_names, ", ") << '}';
  }
  SmallVector<std::string, 3> symbol_names;
  symbol_names.reserve(ts_names.size() + rt_names.size());
  symbol_names.append(ts_names.begin(), ts_names.end());
  symbol_names.append(rt_names.begin(), rt_names.end());
  int64_t num_tiled_results = affine_map.getNumResults() / 3;
  auto map_results = affine_map.getResults();
  auto print_expr = [&](AffineExpr expr) {
    ss << ::xla::ToString(expr, tid_names, symbol_names);
  };
  // Print offsets.
  ss << " -> [";
  llvm::interleaveComma(map_results.take_front(num_tiled_results), ss,
                        print_expr);
  ss << "] [";
  llvm::interleaveComma(
      map_results.drop_front(num_tiled_results).take_front(num_tiled_results),
      ss, print_expr);
  ss << "] [";
  llvm::interleaveComma(map_results.take_back(num_tiled_results), ss,
                        print_expr);
  ss << ']';
  for (const auto& [hlo, rt_var_name] : llvm::zip(rt_vars, rt_names)) {
    ss << " " << rt_var_name << ": " << (hlo ? hlo->ToString() : "nullptr")
       << "\n";
  }
  return s;
}

}  // namespace

std::string ExperimentalSymbolicTile::ToString() const {
  return ToStringImpl(tile_map, rt_vars, GetVarNames(num_tids(), "tid_"),
                      GetVarNames(num_tids(), "ts_"),
                      GetVarNames(num_rt_vars(), "rt_"));
}

AffineMap ExperimentalSymbolicTile::offset_map() const {
  int64_t num_dims = tile_map.getNumResults();
  return tile_map.getSliceMap(0, num_dims);
}

AffineMap ExperimentalSymbolicTile::size_map() const {
  int64_t num_dims = tile_map.getNumResults();
  return tile_map.getSliceMap(num_dims, num_dims);
}

AffineMap ExperimentalSymbolicTile::stride_map() const {
  int64_t num_dims = tile_map.getNumResults();
  return tile_map.getSliceMap(2 * num_dims, num_dims);
}

}  // namespace xla::gpu
