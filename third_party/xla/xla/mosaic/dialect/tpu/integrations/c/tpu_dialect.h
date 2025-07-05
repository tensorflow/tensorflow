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

// Refer to the corresponding C++ declarations in layout.h and
// apply_vector_layout.h for documentation on the functions in this file

#ifndef JAXLIB_MOSAIC_DIALECT_TPU_INTEGRATIONS_C_TPU_DIALECT_H_
#define JAXLIB_MOSAIC_DIALECT_TPU_INTEGRATIONS_C_TPU_DIALECT_H_

#include "xla/mosaic/dialect/tpu/integrations/c/tpu_dialect.h"
#ifndef __cplusplus
#include <stdbool.h>
#endif
#include <stddef.h>
#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "xla/mosaic/dialect/tpu/integrations/c/tpu_passes.capi.h.inc"

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TPU, tpu);

MLIR_CAPI_EXPORTED bool mlirTPUAttributeIsATiledLayoutAttr(MlirAttribute attr);

/// Encodes the tiles as an ArrayAttr of DenseI64ArrayAttrs.
MLIR_CAPI_EXPORTED MlirAttribute
mlirTPUTiledLayoutAttrGetTiles(MlirAttribute attr);

MLIR_CAPI_EXPORTED void mlirTPUAnalyzePotentialCommunication(
    MlirOperation op, bool* has_communication, bool* has_custom_barrier);

typedef enum MlirTpuImplicitDim {
  MlirTpuImplicitDimNone = 0,
  MlirTpuImplicitDimMinor = 1,
  MlirTpuImplicitDimSecondMinor = 2,
} MlirTpuImplicitDim;

typedef enum MlirTpuDirection {
  MlirTpuDirectionSublanes,
  MlirTpuDirectionLanes,
  MlirTpuDirectionSubelements
} MlirTpuDirection;

// Opaque reference to an owned layout
typedef struct MlirTpuVectorLayout {
  void* ptr;
} MlirTpuVectorLayout;

// Opaque reference to owned data bounds
typedef struct MlirTpuVregDataBounds {
  void* ptr;
} MlirTpuVregDataBounds;

// mlir::ArrayRef<int64_t> equivalent
// Unlike mlir::ArrayRef, the data may or may not be owned (this should be
// defined by the producer of the struct).
typedef struct MlirTpuI64ArrayRef {
  int64_t* ptr;
  size_t size;
} MlirTpuI64ArrayRef;

// Shaped array of values
typedef struct MlirTpuValueArray {
  MlirTpuI64ArrayRef shape;  // May or may not be owned
  MlirValue* vals;           // Size given by the shape
} MlirTpuValueArray;

typedef struct MlirTpuLayoutOffsets {
  // Use -1 for replicated
  int64_t sublane;
  int64_t lane;
} MlirTpuLayoutOffsets;

typedef struct MlirTpuI64TargetTuple {
  int64_t sublane;
  int64_t lane;
} MlirTpuI64TargetTuple;

typedef struct MlirTpuMxuShape {
  int64_t contracting_size;
  int64_t non_contracting_size;
} MlirTpuMxuShape;

typedef struct MlirTpuBoolTargetTuple {
  bool sublane;
  bool lane;
} MlirTpuBoolTargetTuple;

// An insertion point within a block.
// The MLIR C API does not already have a similar struct, unfortunately.
typedef struct MlirTpuInsertionPoint {
  MlirBlock block;  // Only used when ref_operation is unspecified (null)
  MlirOperation ref_operation;
} MlirTpuInsertionPoint;

typedef struct MlirTpuApplyVectorLayoutContext {
  int hardware_generation = -1;
  MlirTpuI64TargetTuple target_shape = {8, 128};
  MlirTpuMxuShape mxu_shape = {128, 128};
  int64_t max_sublanes_in_scratch = 0;
} MlirTpuApplyVectorLayoutContext;

// Caller owns the returned object and is responsible for calling
// mlirTpuVectorLayoutDestroy
MLIR_CAPI_EXPORTED MlirTpuVectorLayout mlirTpuVectorLayoutCreate(
    int bitwidth, MlirTpuLayoutOffsets offsets, MlirTpuI64TargetTuple tiling,
    MlirTpuImplicitDim implicit_dim);

MLIR_CAPI_EXPORTED void mlirTpuVectorLayoutDestroy(MlirTpuVectorLayout);

MLIR_CAPI_EXPORTED int mlirTpuVectorLayoutGetBitwidth(
    MlirTpuVectorLayout layout);

MLIR_CAPI_EXPORTED MlirTpuLayoutOffsets
mlirTpuVectorLayoutGetOffsets(MlirTpuVectorLayout layout);

MLIR_CAPI_EXPORTED MlirTpuI64TargetTuple
mlirTpuVectorLayoutGetTiling(MlirTpuVectorLayout layout);

MLIR_CAPI_EXPORTED MlirTpuImplicitDim
mlirTpuVectorLayoutGetImplicitDim(MlirTpuVectorLayout layout);

MLIR_CAPI_EXPORTED int mlirTpuVectorLayoutGetPacking(
    MlirTpuVectorLayout layout);

MLIR_CAPI_EXPORTED int mlirTpuVectorLayoutGetLayoutRank(
    MlirTpuVectorLayout layout);

MLIR_CAPI_EXPORTED bool mlirTpuVectorLayoutEquals(MlirTpuVectorLayout lhs,
                                                  MlirTpuVectorLayout rhs);

MLIR_CAPI_EXPORTED int64_t mlirTpuVectorLayoutTilesPerVreg(
    MlirTpuVectorLayout layout, MlirTpuI64TargetTuple target_shape);

MLIR_CAPI_EXPORTED int64_t mlirTpuVectorLayoutSublanesPerTile(
    MlirTpuVectorLayout layout, MlirTpuI64TargetTuple target_shape);

MLIR_CAPI_EXPORTED MlirTpuI64TargetTuple mlirTpuVectorLayoutVregSlice(
    MlirTpuVectorLayout layout, MlirTpuI64TargetTuple target_shape);

// Caller is responsible for calling free on the returned pointer
MLIR_CAPI_EXPORTED MlirTpuI64ArrayRef mlirTpuVectorLayoutImplicitShape(
    MlirTpuVectorLayout layout, MlirTpuI64ArrayRef shape);

// Caller is responsible for calling free on the returned pointer.
MLIR_CAPI_EXPORTED MlirTpuI64ArrayRef mlirTpuVectorLayoutTileArrayShape(
    MlirTpuVectorLayout layout, MlirTpuI64ArrayRef shape,
    MlirTpuI64TargetTuple target_shape);

// Caller owns the returned object and is responsible for calling
// mlirTpuVectorLayoutVregDataBoundsDestroy
MLIR_CAPI_EXPORTED MlirTpuVregDataBounds mlirTpuVectorLayoutTileDataBounds(
    MlirTpuVectorLayout layout, MlirContext ctx, int64_t* full_shape,
    int64_t* idxs, size_t size, MlirTpuI64TargetTuple target_shape,
    MlirTpuBoolTargetTuple allow_replicated);

MLIR_CAPI_EXPORTED bool mlirTpuVectorLayoutHasNaturalTopology(
    MlirTpuVectorLayout layout, MlirTpuI64TargetTuple target_shape);

MLIR_CAPI_EXPORTED bool mlirTpuVectorLayoutHasNativeTiling(
    MlirTpuVectorLayout layout, MlirTpuI64TargetTuple target_shape);

// `shape` is optional, pass a shape with a null `ptr` to return true iff the
// "generalizes" relationship applies to all shapes.
MLIR_CAPI_EXPORTED bool mlirTpuVectorLayoutGeneralizes(
    MlirTpuVectorLayout layout, MlirTpuVectorLayout other,
    MlirTpuI64ArrayRef shape, MlirTpuI64TargetTuple target_shape);

// `shape` is optional, pass a shape with a null `ptr` to return true iff the
// "equivalent to" relationship applies to all shapes.
MLIR_CAPI_EXPORTED bool mlirTpuVectorLayoutEquivalentTo(
    MlirTpuVectorLayout layout, MlirTpuVectorLayout other,
    MlirTpuI64ArrayRef shape, MlirTpuI64TargetTuple target_shape);

MLIR_CAPI_EXPORTED void mlirTpuVectorLayoutPrint(MlirTpuVectorLayout layout,
                                                 MlirStringCallback callback,
                                                 void* user_data);

MLIR_CAPI_EXPORTED bool mlirTpuVectorLayoutIsValid(
    MlirTpuVectorLayout layout, MlirTpuI64TargetTuple target_shape);

MLIR_CAPI_EXPORTED void mlirTpuVregDataBoundsDestroy(
    MlirTpuVregDataBounds data_bounds);

MLIR_CAPI_EXPORTED bool mlirTpuVregDataBoundsMaskVariesAlong(
    MlirTpuVregDataBounds data_bounds, MlirTpuDirection direction,
    MlirTpuI64TargetTuple target_shape);

MLIR_CAPI_EXPORTED bool mlirTpuVregDataBoundsIsComplete(
    MlirTpuVregDataBounds data_bounds, MlirTpuI64TargetTuple target_shape);
// Returns null on failure
MLIR_CAPI_EXPORTED MlirValue mlirTpuVregDataBoundsGetVectorMask(
    MlirTpuVregDataBounds data_bounds, MlirTpuInsertionPoint insertion_point,
    MlirLocation location, int generation, MlirTpuI64TargetTuple target_shape);

MLIR_CAPI_EXPORTED MlirAttribute mlirTpuVregDataBoundsGetSublaneMask(
    MlirTpuVregDataBounds data_bounds, MlirContext ctx,
    MlirTpuI64TargetTuple target_shape);

// vals are copied, ownership is not stolen.
MLIR_CAPI_EXPORTED MlirOperation
mlirTpuAssemble(MlirTpuInsertionPoint insertion_point, MlirType vector_type,
                MlirTpuVectorLayout layout, MlirTpuValueArray vals,
                MlirTpuI64TargetTuple target_shape);

// Returns null on failure
// Caller owns the returned object and is responsible for calling free on shape
// and vals
MLIR_CAPI_EXPORTED MlirTpuValueArray mlirTpuDisassemble(
    MlirTpuInsertionPoint insertion_point, MlirTpuVectorLayout layout,
    MlirValue val, MlirTpuI64TargetTuple target_shape);

MLIR_CAPI_EXPORTED MlirLogicalResult
mlirTpuApplyLayoutOp(MlirTpuApplyVectorLayoutContext ctx, MlirOperation op);

// Returns null on failure
MLIR_CAPI_EXPORTED MlirValue
mlirTpuRelayout(MlirTpuInsertionPoint insertion_point, MlirValue val,
                MlirTpuVectorLayout src, MlirTpuVectorLayout dst,
                MlirTpuApplyVectorLayoutContext ctx);

MLIR_CAPI_EXPORTED void mlirTpuRegisterMosaicSerdePass();

#ifdef __cplusplus
}
#endif

#endif  // JAXLIB_MOSAIC_DIALECT_TPU_INTEGRATIONS_C_TPU_DIALECT_H_
