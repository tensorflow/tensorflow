/* Copyright 2023 The JAX Authors.

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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANSFORMS_INFER_MEMREF_LAYOUT_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANSFORMS_INFER_MEMREF_LAYOUT_H_

#include <array>
#include <cstdint>
#include <string_view>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

FailureOr<MemRefType> inferMemref(MemRefType memref, int hardware_generation,
                                  std::array<int64_t, 2> target_shape,
                                  const TpuTilingFlags& tpu_tiling_flags,
                                  bool is_kernel_argument,
                                  int64_t leading_tile_rows = 0);

const std::string_view kLeadingTileRows = "leading_tile_rows";

}  // namespace mlir::tpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANSFORMS_INFER_MEMREF_LAYOUT_H_
