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
#include <utility>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineMap.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"

#ifndef XLA_SERVICE_GPU_MODEL_TRITON_EMITTER_CONSTRAINTS_H_
#define XLA_SERVICE_GPU_MODEL_TRITON_EMITTER_CONSTRAINTS_H_

namespace xla {
namespace gpu {

// Triton-specific constraints on tile sizes.
class TritonEmitterConstraints : public EmitterSpecificConstraints {
 public:
  static EmitterSpecificConstraintsBuilder GetBuilder();

  explicit TritonEmitterConstraints(
      llvm::SmallVector<mlir::AffineMap, 4> tile_size_maps)
      : tile_size_maps_(std::move(tile_size_maps)) {}

  absl::StatusOr<bool> ParametersSatisfyConstraints(
      absl::Span<const int64_t> tile_parameters) const override;

 private:
  // A collection of unique size maps from all the SymbolicTiledHloInstructions.
  //
  // Different TiledHloInstructions often have the same size map, so we keep a
  // collection of unique maps to improve compilation time.
  llvm::SmallVector<mlir::AffineMap, 4> tile_size_maps_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_TRITON_EMITTER_CONSTRAINTS_H_
