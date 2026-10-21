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

#ifndef XLA_CODEGEN_XTILE_TILING_FROM_BLOCK_PARAMETERS_H_
#define XLA_CODEGEN_XTILE_TILING_FROM_BLOCK_PARAMETERS_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/codegen/xtile/block_level_parameters.h"
#include "xla/codegen/xtile/xtile_config.pb.h"

namespace xla::xtile {

// Returns concrete tiles sizes tiling space dimensions - in the order of
// dimensions in the tiling space.
// Sizes are returned "as is", i.e. they are not rounded up to the next power
// of 2.
absl::StatusOr<llvm::SmallVector<int64_t>> GetTilingSpaceConcreteSizes(
    const ::xla::gpu::experimental::TilingSpace& tiling_space,
    const xla::xtile::BlockLevelParameters& block_level_parameters,
    bool enable_same_shape_multi_output_fusion);

}  // namespace xla::xtile

#endif  // XLA_CODEGEN_XTILE_TILING_FROM_BLOCK_PARAMETERS_H_
