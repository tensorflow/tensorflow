/* Copyright 2024 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineMap.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/model/symbolic_tile.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"

#ifndef XLA_SERVICE_GPU_MODEL_TRITON_EMITTER_CONSTRAINTS_H_
#define XLA_SERVICE_GPU_MODEL_TRITON_EMITTER_CONSTRAINTS_H_

namespace xla {
namespace gpu {

// Triton-specific constraints on tile sizes.
class TritonEmitterConstraints : public EmitterSpecificConstraints {
 public:
  static EmitterSpecificConstraintsBuilder GetBuilder(
      const se::DeviceDescription& device_description);

  absl::StatusOr<bool> ParametersSatisfyConstraints(
      absl::Span<const int64_t> tile_parameters) const override;

  bool HasCustomConstraints() const { return !custom_constraints_.empty(); }

 private:
  // Holds a constraint expression over derived parameters (s'0, ..., s'm) where
  //   (s'0, ..., s'm) = tile_parameters_transform(tile_parameters).
  struct CustomConstraints {
    mlir::AffineMap tile_parameters_transform;
    ConstraintExpression constraints;
  };

  // Holds the info needed to validate whether the tiling parameters satisfy the
  // constraint that they are either powers of 2, or equal to the dimension
  // size.
  struct RootTileInfo {
    mlir::AffineMap size_map;
    std::vector<int64_t> dim_sizes;
  };

  explicit TritonEmitterConstraints(
      llvm::SmallVector<mlir::AffineMap, 4> tile_size_maps,
      llvm::SmallVector<RootTileInfo, 2> roots,
      std::vector<CustomConstraints> custom_constraints,
      const Shape& root_shape, const se::DeviceDescription& device_info)
      : tile_size_maps_(std::move(tile_size_maps)),
        roots_(std::move(roots)),
        custom_constraints_(std::move(custom_constraints)),
        root_shape_(root_shape),
        device_info_(device_info) {}

  // Derives a vector of `CustomConstraints` to be checked within
  // `ParametersSatisfyConstraints` from a vector of
  // `SymbolicTiledHloInstruction`s representing a symbolically tiled HLO
  // computation. The fusion adaptor is used to figure out which instructions
  // within the computation are operands of the fusion.
  //
  // Currently, this is used to work around an issue with reshapes/bitcasts when
  // instructions are tiled with non-power-of-2 shapes. The resulting custom
  // constraints contain
  //   * the reshape/bitcast's tile size map; this to allow deriving the
  //     output tile sizes for the reshape/bitcast instruction;
  //   * the constraint expression corresponding to the SymbolicTile derived
  //     from the reshape/bitcast instruction's output-to-input indexing map
  //     "in a vacuum" (i.e., without composing with any other indexing map).
  //
  // TODO(b/365727080): move tile derivation to have powers of 2 tiles
  // everywhere, and deprecate this.
  static std::vector<CustomConstraints> DeriveCustomConstraints(
      const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
          instructions,
      const HloFusionAdaptor& fusion_adaptor);

  // A collection of unique size maps from all the SymbolicTiledHloInstructions.
  //
  // Different TiledHloInstructions often have the same size map, so we keep a
  // collection of unique maps to improve compilation time.
  llvm::SmallVector<mlir::AffineMap, 4> tile_size_maps_;

  // Holds the info for all fusion roots necessary to check whether the tile
  // sizes evaluate to powers of 2 or have the same size as the dimension.
  llvm::SmallVector<RootTileInfo, 2> roots_;

  // Custom emitter-specific constraints to check in
  // `ParametersSatisfyConstraints`.
  std::vector<CustomConstraints> custom_constraints_;

  // Shape of the root instruction.
  Shape root_shape_;

  se::DeviceDescription device_info_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_TRITON_EMITTER_CONSTRAINTS_H_
