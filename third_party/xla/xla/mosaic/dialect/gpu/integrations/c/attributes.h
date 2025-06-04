/* Copyright 2025 The JAX Authors.

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

#ifndef JAXLIB_MOSAIC_DIALECT_GPU_INTEGRATIONS_C_ATTRIBUTES_H_
#define JAXLIB_MOSAIC_DIALECT_GPU_INTEGRATIONS_C_ATTRIBUTES_H_

#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// TileTransformAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirMosaicGpuIsATileTransformAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirMosaicGpuTileTransformAttrGet(
    MlirContext ctx, int32_t* tiling, int32_t tiling_size);

MLIR_CAPI_EXPORTED int32_t
mlirMosaicGpuTileTransformAttrGetTilingSize(MlirAttribute attr);

MLIR_CAPI_EXPORTED int32_t
mlirMosaicGpuTileTransformAttrGetTiling(MlirAttribute attr, int32_t index);

//===----------------------------------------------------------------------===//
// TransposeTransformAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirMosaicGpuIsATransposeTransformAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirMosaicGpuTransposeTransformAttrGet(
    MlirContext ctx, int32_t* permutation, int32_t permutation_size);

MLIR_CAPI_EXPORTED int32_t
mlirMosaicGpuTransposeTransformAttrGetPermutationSize(MlirAttribute attr);

MLIR_CAPI_EXPORTED int32_t mlirMosaicGpuTransposeTransformAttrGetPermutation(
    MlirAttribute attr, int32_t index);

//===----------------------------------------------------------------------===//
// SwizzleTransformAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirMosaicGpuIsASwizzleTransformAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirMosaicGpuSwizzleTransformAttrGet(MlirContext ctx, int32_t swizzle);

MLIR_CAPI_EXPORTED int32_t
mlirMosaicGpuSwizzleTransformAttrGetSwizzle(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif  // JAXLIB_MOSAIC_DIALECT_GPU_INTEGRATIONS_C_ATTRIBUTES_H_