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

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::mlir::AffineExpr;

SmallVector<std::string> GetVarNames(int64_t num_vars, llvm::StringRef prefix) {
  SmallVector<std::string> var_names;
  var_names.reserve(num_vars);
  for (int64_t i = 0; i < num_vars; ++i) {
    var_names.push_back(absl::StrFormat("%s%d", prefix, i));
  }
  return var_names;
}

}  // namespace

ExperimentalSymbolicTile::ExperimentalSymbolicTile(
    mlir::MLIRContext* mlir_context, int64_t num_tile_ids,
    ArrayRef<AffineExpr> offsets, ArrayRef<AffineExpr> sizes,
    ArrayRef<AffineExpr> strides, ArrayRef<AffineExpr> upper_bounds,
    ArrayRef<const HloInstruction*> rt_vars)
    : mlir_context_(mlir_context),
      num_tile_ids_(num_tile_ids),
      offsets_(offsets.begin(), offsets.end()),
      sizes_(sizes.begin(), sizes.end()),
      strides_(strides.begin(), strides.end()),
      upper_bounds_(upper_bounds.begin(), upper_bounds.end()),
      rt_vars_(rt_vars.begin(), rt_vars.end()) {}

std::string ExperimentalSymbolicTile::ToString() const {
  auto tid_names = GetVarNames(num_tile_ids(), "tid_");
  auto ts_names = GetVarNames(num_tile_ids(), "ts_");
  auto rt_names = GetVarNames(num_rt_vars(), "rt_");

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
  auto print_expr = [&](AffineExpr expr) {
    ss << ::xla::ToString(expr, tid_names, symbol_names);
  };
  // Print offsets.
  ss << " -> offsets [";
  llvm::interleaveComma(offsets_, ss, print_expr);
  ss << "] sizes [";
  llvm::interleaveComma(sizes_, ss, print_expr);
  ss << "] strides [";
  llvm::interleaveComma(strides_, ss, print_expr);
  ss << "] upper bounds [";
  llvm::interleaveComma(upper_bounds_, ss, print_expr);
  ss << ']';
  for (const auto& [hlo, rt_var_name] : llvm::zip(rt_vars_, rt_names)) {
    ss << " " << rt_var_name << ": " << (hlo ? hlo->ToString() : "nullptr")
       << "\n";
  }
  return s;
}

}  // namespace xla::gpu
