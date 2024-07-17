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

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

/*static*/
absl::StatusOr<std::unique_ptr<TiledHloInstruction>>
TiledHloInstruction::Create(const HloInstruction* hlo,
                            llvm::SmallVector<int64_t> tile_sizes,
                            llvm::SmallVector<int64_t> tile_strides,
                            IndexingMap tile_offsets_indexing) {
  int rank = hlo->shape().rank();

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

  if (tile_offsets_indexing.GetAffineMap().getNumResults() != rank) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "tile_offsets_indexing must have the same number of results as the "
        "rank of the hlo shape. tile_offsets_indexing = %s, hlo = %s",
        tile_offsets_indexing.ToString(), hlo->ToString()));
  }

  return absl::WrapUnique(new TiledHloInstruction(
      hlo, std::move(tile_sizes), std::move(tile_strides),
      std::move(tile_offsets_indexing)));
}

std::string TiledHloInstruction::ToString() const {
  std::stringstream ss;
  ss << "\thlo: " << hlo_->ToString() << "\n";
  ss << "\ttile_sizes: (" << absl::StrJoin(tile_sizes_, ", ") << ")\n";
  ss << "\ttile_strides: (" << absl::StrJoin(tile_strides_, ", ") << ")\n";
  ss << "\ttile_offsets_indexing: " << tile_offsets_indexing_;
  return ss.str();
}

}  // namespace gpu
}  // namespace xla
