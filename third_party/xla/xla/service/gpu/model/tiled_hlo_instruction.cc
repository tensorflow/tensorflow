/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/model/tiled_hlo_instruction.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace gpu {

namespace {

// Checks the preconditions that must be fulfilled when creating a
// `TiledHloInstruction` or derived class.
absl::Status VerifyTiledHloInstructionConstructorPreconditions(
    const HloInstruction* hlo, llvm::SmallVector<int64_t> tile_sizes,
    llvm::SmallVector<int64_t> tile_strides,
    std::optional<IndexingMap> tile_offsets_indexing) {
  const int rank =
      hlo->shape().IsArray() ? hlo->shape().dimensions().size() : 0;

  if (tile_sizes.size() != rank) {
    return absl::InvalidArgumentError(
        absl::StrCat("Number of tile sizes must be equal to the rank of the "
                     "hlo shape. tile_sizes = ",
                     tile_sizes.size(), ", hlo = ", hlo->ToString()));
  }

  if (tile_strides.size() != rank) {
    return absl::InvalidArgumentError(
        absl::StrCat("Number of tile strides must be equal to the rank of the "
                     "hlo shape. tile_sizes = ",
                     tile_strides.size(), ", hlo = ", hlo->ToString()));
  }

  if (tile_offsets_indexing.has_value() &&
      tile_offsets_indexing->GetDimensionCount() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("tile_offsets_indexing must have 1 dim. "
                     "tile_offsets_indexing = ",
                     ToString(*tile_offsets_indexing)));
  }

  if (tile_offsets_indexing.has_value() &&
      tile_offsets_indexing->GetAffineMap().getNumResults() != rank) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "tile_offsets_indexing must have the same number of results as the "
        "rank of the hlo shape. tile_offsets_indexing = %s, hlo = %s",
        ToString(*tile_offsets_indexing), hlo->ToString()));
  }

  return absl::OkStatus();
}

}  // namespace

/*static*/
absl::StatusOr<std::unique_ptr<TiledHloInstruction>>
TiledHloInstruction::Create(
    const HloInstruction* hlo,
    llvm::SmallVector<const TiledHloInstruction*> operands,
    llvm::SmallVector<int64_t> tile_sizes,
    llvm::SmallVector<int64_t> tile_strides,
    std::optional<IndexingMap> tile_offsets_indexing) {
  TF_RETURN_IF_ERROR(VerifyTiledHloInstructionConstructorPreconditions(
      hlo, tile_sizes, tile_strides, tile_offsets_indexing));

  return absl::WrapUnique(new TiledHloInstruction(
      hlo, std::move(operands), std::move(tile_sizes), std::move(tile_strides),
      std::move(tile_offsets_indexing)));
}

std::string TiledHloInstruction::ToString() const {
  std::stringstream ss;
  ss << "\thlo: " << hlo_->ToString() << "\n";
  ss << "\ttile_sizes: (" << absl::StrJoin(tile_sizes_, ", ") << ")\n";
  ss << "\ttile_strides: (" << absl::StrJoin(tile_strides_, ", ") << ")\n";
  ss << "\ttile_offsets_indexing: "
     << (tile_offsets_indexing_.has_value()
             ? ::xla::ToString(*tile_offsets_indexing_)
             : "nullopt");
  return ss.str();
}

/*static*/
absl::StatusOr<std::unique_ptr<TiledHloFusionInstruction>>
TiledHloFusionInstruction::Create(
    const HloInstruction* hlo,
    llvm::SmallVector<const TiledHloInstruction*> operands,
    std::unique_ptr<TiledHloComputation> called_computation,
    llvm::SmallVector<int64_t> tile_sizes,
    llvm::SmallVector<int64_t> tile_strides,
    std::optional<IndexingMap> tile_offsets_indexing) {
  TF_RETURN_IF_ERROR(VerifyTiledHloInstructionConstructorPreconditions(
      hlo, tile_sizes, tile_strides, tile_offsets_indexing));

  return absl::WrapUnique(new TiledHloFusionInstruction(
      hlo, std::move(operands), std::move(called_computation),
      std::move(tile_sizes), std::move(tile_strides),
      std::move(tile_offsets_indexing)));
}

TiledHloFusionInstruction::TiledHloFusionInstruction(
    const HloInstruction* hlo,
    llvm::SmallVector<const TiledHloInstruction*> operands,
    std::unique_ptr<TiledHloComputation> called_computation,
    llvm::SmallVector<int64_t> tile_sizes,
    llvm::SmallVector<int64_t> tile_strides,
    std::optional<IndexingMap> tile_offsets_indexing)
    : TiledHloInstruction(hlo, std::move(operands), std::move(tile_sizes),
                          std::move(tile_strides),
                          std::move(tile_offsets_indexing)),
      called_computation_(std::move(called_computation)) {}

}  // namespace gpu
}  // namespace xla
