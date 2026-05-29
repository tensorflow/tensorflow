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

#ifndef XLA_CODEGEN_TILING_EXPERIMENTAL_SCHEDULING_H_
#define XLA_CODEGEN_TILING_EXPERIMENTAL_SCHEDULING_H_

#include <cstdint>
#include <string>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/MapVector.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"

namespace xla::gpu::experimental {

struct Schedule {
  // Maps from the parallel dimension ID to the symbolic expression that depends
  // on the program ID.
  llvm::MapVector<int64_t, SymbolicExpr> dim_id_to_pid_expr;

  // The bounds of the program ID.
  Interval pid_bounds;

  // This allows GUnit to print the tile.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Schedule& tiled_hlo) {
    sink.Append(tiled_hlo.ToString());
  }
  SymbolicExpr GetPidExpr(int64_t dim_id) const {
    auto it = dim_id_to_pid_expr.find(dim_id);
    CHECK(it != dim_id_to_pid_expr.end())
        << "Dimension ID " << dim_id << " not found in the schedule.";
    return it->second;
  }

  std::string ToString() const;
};

// Returns a map from pid to the tile indices.
absl::StatusOr<Schedule> GetSchedule(
    const TiledHloComputation& tiled_computation);

}  // namespace xla::gpu::experimental

#endif  // XLA_CODEGEN_TILING_EXPERIMENTAL_SCHEDULING_H_
