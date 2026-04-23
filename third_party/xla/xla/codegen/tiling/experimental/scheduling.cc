/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/codegen/tiling/experimental/scheduling.h"

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/util.h"

namespace xla::gpu::experimental {

std::string Schedule::ToString() const {
  std::set<int64_t> dim_ids;
  for (const auto& [dim_id, expr] : dim_id_to_pid_expr) {
    dim_ids.insert(dim_id);
  }
  std::vector<std::string> expr_strs;
  expr_strs.reserve(dim_ids.size());
  for (int64_t dim_id : dim_ids) {
    expr_strs.push_back(
        absl::StrFormat("d%d -> %s", dim_id,
                        dim_id_to_pid_expr.at(dim_id).ToString({"pid"}, {})));
  }
  return absl::StrCat(absl::StrJoin(expr_strs, ", "),
                      ", pid_bounds=", pid_bounds.ToString());
}

absl::StatusOr<Schedule> GetSchedule(
    const TiledHloComputation& tiled_computation) {
  // Compute the block counts for each parallel dimension.
  llvm::SmallVector<int64_t, 4> parallel_dim_block_counts;
  llvm::SmallVector<int64_t, 4> parallel_dim_ids;
  for (const auto& [dim_id, dimension] :
       llvm::enumerate(tiled_computation.tiling_space().dimensions())) {
    if (dimension.type != TilingSpace::DimensionSemantics::kParallel) {
      continue;
    }
    parallel_dim_block_counts.push_back(
        CeilOfRatio(dimension.dimension_size, dimension.tile_size));
    parallel_dim_ids.push_back(dim_id);
  }
  mlir::MLIRContext* ctx = tiled_computation.GetMLIRContext();
  SymbolicExpr pid = CreateDimExpr(0, ctx);
  llvm::SmallVector<SymbolicExpr, 4> delinearized_pid =
      DelinearizeIndex(parallel_dim_block_counts, pid, ctx);
  Schedule schedule;
  for (const auto& [parallel_dim_id, expr] :
       llvm::zip(parallel_dim_ids, delinearized_pid)) {
    schedule.dim_id_to_pid_expr[parallel_dim_id] = expr;
  }
  schedule.pid_bounds = Interval{0, Product(parallel_dim_block_counts) - 1};
  return schedule;
}

}  // namespace xla::gpu::experimental
