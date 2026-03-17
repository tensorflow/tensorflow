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

#include "absl/status/statusor.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/hlo/analysis/indexing_map.h"

namespace xla::gpu::experimental {

// Returns a map from pid to the tile indices.
absl::StatusOr<IndexingMap> Schedule(
    const TiledHloComputation& tiled_computation);

}  // namespace xla::gpu::experimental

#endif  // XLA_CODEGEN_TILING_EXPERIMENTAL_SCHEDULING_H_
