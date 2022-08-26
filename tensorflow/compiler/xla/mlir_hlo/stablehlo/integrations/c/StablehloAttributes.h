/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.
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
#ifndef STABLEHLO_INTEGRATIONS_C_STABLEHLO_ATTRIBUTES_H
#define STABLEHLO_INTEGRATIONS_C_STABLEHLO_ATTRIBUTES_H

#include <sys/types.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// ScatterDimensionNumbers
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloScatterDimensionNumbersGet(
    MlirContext ctx,                                                  //
    intptr_t nUpdateWindowDims, const int64_t *updateWindowDims,      //
    intptr_t nInsertedWindowDims, const int64_t *insertedWindowDims,  //
    intptr_t nScatteredDimsToOperandDims,                             //
    const int64_t *scatteredDimsToOperandDims,                        //
    int64_t indexVectorDim);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsAScatterDimensionNumbers(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t
stablehloScatterDimensionNumbersGetUpdateWindowDimsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloScatterDimensionNumbersGetUpdateWindowDimsElem(MlirAttribute attr,
                                                        intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
stablehloScatterDimensionNumbersGetInsertedWindowDimsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloScatterDimensionNumbersGetInsertedWindowDimsElem(MlirAttribute attr,
                                                          intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize(
    MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem(
    MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED int64_t
stablehloDimensionNumbersGetIndexVectorDim(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// GatherDimensionNumbers
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloGatherDimensionNumbersGet(
    MlirContext ctx, intptr_t nOffsetDims, const int64_t *offsetDims,
    intptr_t nCollapsedSliceDims, const int64_t *collapsedSliceDims,
    intptr_t nStartIndexMap, const int64_t *startIndexMap,
    int64_t indexVectorDim);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsAGatherDimensionNumbers(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t
stablehloGatherDimensionNumbersGetOffsetDimsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t stablehloGatherDimensionNumbersGetOffsetDimsElem(
    MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem(MlirAttribute attr,
                                                         intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
stablehloGatherDimensionNumbersGetStartIndexMapSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t stablehloGatherDimensionNumbersGetStartIndexMapElem(
    MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED int64_t
stablehloGatherDimensionNumbersGetIndexVectorDim(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// DotDimensionNumbers
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloDotDimensionNumbersGet(
    MlirContext ctx,                                                        //
    intptr_t nLhsBatchingDimensions, const int64_t *lhsBatchingDimensions,  //
    intptr_t nRhsBatchingDimensions, const int64_t *rhsBatchingDimensions,  //
    intptr_t nLhsContractingDimensions,                                     //
    const int64_t *lhsContractingDimensions,                                //
    intptr_t nRhsContractingDimensions,                                     //
    const int64_t *rhsContractingDimensions);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsADotDimensionNumbers(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t
stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem(MlirAttribute attr,
                                                         intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem(MlirAttribute attr,
                                                         intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
stablehloDotDimensionNumbersGetLhsContractingDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloDotDimensionNumbersGetLhsContractingDimensionsElem(MlirAttribute attr,
                                                            intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
stablehloDotDimensionNumbersGetRhsContractingDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloDotDimensionNumbersGetRhsContractingDimensionsElem(MlirAttribute attr,
                                                            intptr_t pos);

//===----------------------------------------------------------------------===//
// ConvDimensionNumbers
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloConvDimensionNumbersGet(
    MlirContext ctx, int64_t inputBatchDimension, int64_t inputFeatureDimension,
    intptr_t nInputSpatialDimensions, const int64_t *inputSpatialDimensions,
    int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
    intptr_t nKernelSpatialDimensions, const int64_t *kernelSpatialDimensions,
    int64_t outputBatchDimension, int64_t outputFeatureDimension,
    intptr_t nOutputSpatialDimensions, const int64_t *outputSpatialDimensions);

// Returns true of the given attribute is a ConvDimensionNumbers attribute.
MLIR_CAPI_EXPORTED bool stablehloAttributeIsAConvDimensionNumbers(
    MlirAttribute attr);

// Returns the properties of ConvDimensionNumbers attributes.
MLIR_CAPI_EXPORTED int64_t
stablehloConvDimensionNumbersGetInputBatchDimension(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloConvDimensionNumbersGetInputFeatureDimension(MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t
stablehloConvDimensionNumbersGetInputSpatialDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloConvDimensionNumbersGetInputSpatialDimensionsElem(MlirAttribute attr,
                                                           intptr_t pos);
MLIR_CAPI_EXPORTED int64_t
stablehloConvDimensionNumbersGetKernelInputFeatureDimension(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloConvDimensionNumbersGetKernelOutputFeatureDimension(
    MlirAttribute attr);
MLIR_CAPI_EXPORTED intptr_t
stablehloConvDimensionNumbersGetKernelSpatialDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloConvDimensionNumbersGetKernelSpatialDimensionsElem(MlirAttribute attr,
                                                            intptr_t pos);
MLIR_CAPI_EXPORTED int64_t
stablehloConvDimensionNumbersGetOutputBatchDimension(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloConvDimensionNumbersGetOutputFeatureDimension(MlirAttribute attr);
MLIR_CAPI_EXPORTED intptr_t
stablehloConvDimensionNumbersGetOutputSpatialDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloConvDimensionNumbersGetOutputSpatialDimensionsElem(MlirAttribute attr,
                                                            intptr_t pos);

//===----------------------------------------------------------------------===//
// ComparisonDirectionAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
stablehloComparisonDirectionAttrGet(MlirContext ctx, MlirStringRef direction);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsAComparisonDirectionAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
stablehloComparisonDirectionAttrGetDirection(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// ComparisonTypeAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
stablehloComparisonTypeAttrGet(MlirContext ctx, MlirStringRef type);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsAComparisonTypeAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
stablehloComparisonTypeAttrGetType(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// PrecisionAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloPrecisionAttrGet(MlirContext ctx,
                                                           MlirStringRef type);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsAPrecisionAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
stablehloPrecisionAttrGetPrecision(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// FftTypeAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloFftTypeAttrGet(MlirContext ctx,
                                                         MlirStringRef type);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsAFftTypeAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
stablehloFftTypeAttrGetFftType(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// TransposeAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloTransposeAttrGet(MlirContext ctx,
                                                           MlirStringRef type);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsATransposeAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
stablehloTransposeAttrGetTranspose(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// RngDistributionAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
stablehloRngDistributionAttrGet(MlirContext ctx, MlirStringRef distribution);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsARngDistributionAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
stablehloRngDistributionAttrGetRngDistribution(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// RngAlgorithmAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
stablehloRngAlgorithmAttrGet(MlirContext ctx, MlirStringRef algorithm);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsARngAlgorithmAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
stablehloRngAlgorithmAttrGetRngAlgorithm(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// ChannelHandle
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloChannelHandleGet(MlirContext ctx,
                                                           int64_t handle,
                                                           int64_t type);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsChannelHandle(MlirAttribute attr);

MLIR_CAPI_EXPORTED int64_t stablehloChannelHandleGetHandle(MlirAttribute attr);

MLIR_CAPI_EXPORTED int64_t stablehloChannelHandleGetType(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// TypeExtensions
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloTypeExtensionsGet(
    MlirContext ctx, intptr_t nBounds, const int64_t *bounds);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsTypeExtensions(MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t
stablehloTypeExtensionsGetBoundsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloTypeExtensionsGetBoundsElem(MlirAttribute attr, intptr_t pos);

#ifdef __cplusplus
}
#endif

#endif  // STABLEHLO_INTEGRATIONS_C_STABLEHLO_ATTRIBUTES_H
