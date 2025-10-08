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

#ifndef XLA_CODEGEN_TILING_TILED_HLO_SCHEDULE_H_
#define XLA_CODEGEN_TILING_TILED_HLO_SCHEDULE_H_

#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// A `TiledHloSchedule` exposes methods for scheduling a `TiledHloComputation`,
// i.e. it specifies an iteration order over tiles.
class TiledHloSchedule {
 public:
  virtual ~TiledHloSchedule() = default;

  // Returns a schedule for the given root instruction as an indexing map.
  //
  // The resulting indexing map must satisfy the following properties:
  // (1) the map must have exactly as many parameters as there are tiling
  //     parameters in `parameter_mapping`;
  // (2) the parameters in the resulting map must appear in the same order as
  //     they appear in `parameter_mapping`;
  // (3) the map must have as many results as there are output dimensions in
  //     the instruction---although the results are allowed to be outside the
  //     range of the instruction's output space;
  // (4) iterating over the entire input space of the map must yield the
  //     entire output space of the instruction.
  virtual absl::StatusOr<IndexingMap> RootSchedule(
      const HloInstruction* root,
      const TilingSpecification::ParameterMapping& parameter_mapping,
      mlir::MLIRContext* ctx) const = 0;
};

// The indexing map returned by this schedule uses parameters
// in major-to-minor order (i.e. in the order in which they are specified in
// the relevant parameter mapping).
class MajorToMinorTiledHloSchedule : public TiledHloSchedule {
 public:
  absl::StatusOr<IndexingMap> RootSchedule(
      const HloInstruction* root,
      const TilingSpecification::ParameterMapping& parameter_mapping,
      mlir::MLIRContext* ctx) const override;
};

// TODO(b/417977182): implement the `PlanarSnakeTiledHloSchedule` schedule.

}  // namespace xla

#endif  // XLA_CODEGEN_TILING_TILED_HLO_SCHEDULE_H_
