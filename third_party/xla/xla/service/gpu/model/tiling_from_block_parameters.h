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

#ifndef XLA_SERVICE_GPU_MODEL_TILING_FROM_BLOCK_PARAMETERS_H_
#define XLA_SERVICE_GPU_MODEL_TILING_FROM_BLOCK_PARAMETERS_H_

#include "absl/status/statusor.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/service/gpu/model/block_level_parameters.h"

namespace xla::gpu {

// Given a tiling specification for an annotated fusion, derives a tiling for
// this fusion.
//
// Note that the tiling extracted here is voluntarily not checked against the
// specification, which means that it could be invalid. This should only be the
// case, though, if this logic gets stale, or if the fusion does not contain
// the required annotations. Checking constraints is not cheap, so we left it up
// to the caller to decide when to check the constraints.
absl::StatusOr<Tiling> TilingFromAnnotatedFusion(
    const SymbolicTileAnalysis& symbolic_tile_analysis,
    const BlockLevelParameters& block_level_parameters);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_TILING_FROM_BLOCK_PARAMETERS_H_
