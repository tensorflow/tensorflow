/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
#ifndef MLIR_HLO_C_ATTRIBUTES_H
#define MLIR_HLO_C_ATTRIBUTES_H

#include <sys/types.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates a new ScatterDimensionNumbers attribute with the given parameters.
// The first three pairs of arguments are interpreted as arrays.
MLIR_CAPI_EXPORTED MlirAttribute mlirMhloScatterDimensionNumbersGet(
    MlirContext ctx,                                                  //
    intptr_t nUpdateWindowDims, const int64_t *updateWindowDims,      //
    intptr_t nInsertedWindowDims, const int64_t *insertedWindowDims,  //
    intptr_t nScatteredDimsToOperandDims,                             //
    const int64_t *scatteredDimsToOperandDims,                        //
    int64_t indexVectorDim);

// Returns true if the given attribute is a ScatterDimenionNumbers attribute.
MLIR_CAPI_EXPORTED bool mlirMhloAttributeIsAScatterDimensionNumbers(
    MlirAttribute attr);

// Returns the properties of ScatterDimensionNumbers attributes.
MLIR_CAPI_EXPORTED intptr_t
mlirMhloScatterDimensionNumbersGetUpdateWindowDimsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
mlirMhloScatterDimensionNumbersGetUpdateWindowDimsElem(MlirAttribute attr,
                                                       intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
mlirMhloScatterDimensionNumbersGetInsertedWindowDimsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
mlirMhloScatterDimensionNumbersGetInsertedWindowDimsElem(MlirAttribute attr,
                                                         intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
mlirMhloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize(
    MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
mlirMhloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem(
    MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED int64_t
mlirMhloDimensionNumbersGetIndexVectorDim(MlirAttribute attr);

// Creates a new GatherDimensionNumbers attribute with the given parameters. The
// first three pairs of arguments are interpreted as arrays.
MLIR_CAPI_EXPORTED MlirAttribute mlirMhloGatherDimensionNumbersGet(
    MlirContext ctx, intptr_t nOffsetDims, const int64_t *offsetDims,
    intptr_t nCollapsedSliceDims, const int64_t *collapsedSliceDims,
    intptr_t nStartIndexMap, const int64_t *startIndexMap,
    int64_t indexVectorDim);

// Returns true if the given attribute is a GatherDimensionNumbers attribute.
MLIR_CAPI_EXPORTED bool mlirMhloAttributeIsAGatherDimensionNumbers(
    MlirAttribute attr);

// Returns the properties of GatherDimensionNumbers attributes.
MLIR_CAPI_EXPORTED intptr_t
mlirMhloGatherDimensionNumbersGetOffsetDimsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t mlirMhloGatherDimensionNumbersGetOffsetDimsElem(
    MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
mlirMhloGatherDimensionNumbersGetCollapsedSliceDimsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
mlirMhloGatherDimensionNumbersGetCollapsedSliceDimsElem(MlirAttribute attr,
                                                        intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
mlirMhloGatherDimensionNumbersGetStartIndexMapSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t mlirMhloGatherDimensionNumbersGetStartIndexMapElem(
    MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED int64_t
mlirMhloGatherDimensionNumbersGetIndexVectorDim(MlirAttribute attr);

// Creates a new DotDimensionNumbers attribute with the given parameters. The
// argument pairs are interpreted as arrays with the leading argument being the
// number of elements and the trailing argument being the pointer to the first
// element of the array.
MLIR_CAPI_EXPORTED MlirAttribute mlirMhloDotDimensionNumbersGet(
    MlirContext ctx,                                                        //
    intptr_t nLhsBatchingDimensions, const int64_t *lhsBatchingDimensions,  //
    intptr_t nRhsBatchingDimensions, const int64_t *rhsBatchingDimensions,  //
    intptr_t nLhsContractingDimensions,                                     //
    const int64_t *lhsContractingDimensions,                                //
    intptr_t nRhsContractingDimensions,                                     //
    const int64_t *rhsContractingDimensions);

// Returns true of the given attribute is a DotDimensionNumbers attribute.
MLIR_CAPI_EXPORTED bool mlirMhloAttributeIsADotDimensionNumbers(
    MlirAttribute attr);

// Returns the properties of DotDimensionNumbers attributes.
MLIR_CAPI_EXPORTED intptr_t
mlirMhloDotDimensionNumbersGetLhsBatchingDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
mlirMhloDotDimensionNumbersGetLhsBatchingDimensionsElem(MlirAttribute attr,
                                                        intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
mlirMhloDotDimensionNumbersGetRhsBatchingDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
mlirMhloDotDimensionNumbersGetRhsBatchingDimensionsElem(MlirAttribute attr,
                                                        intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
mlirMhloDotDimensionNumbersGetLhsContractingDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
mlirMhloDotDimensionNumbersGetLhsContractingDimensionsElem(MlirAttribute attr,
                                                           intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
mlirMhloDotDimensionNumbersGetRhsContractingDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
mlirMhloDotDimensionNumbersGetRhsContractingDimensionsElem(MlirAttribute attr,
                                                           intptr_t pos);

// Creates a new ConvDimensionNumbers attribute with the given parameters. The
// pairs of consecutive intptr_t / int64_t* arguments are interpeted as sized
// arrays.
MLIR_CAPI_EXPORTED MlirAttribute mlirMhloConvDimensionNumbersGet(
    MlirContext ctx, int64_t inputBatchDimension, int64_t inputFeatureDimension,
    intptr_t nInputSpatialDimensions, const int64_t *inputSpatialDimensions,
    int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
    intptr_t nKernelSpatialDimensions, const int64_t *kernelSpatialDimensions,
    int64_t outputBatchDimension, int64_t outputFeatureDimension,
    intptr_t nOutputSpatialDimensions, const int64_t *outputSpatialDimensions);

// Returns true of the given attribute is a ConvDimensionNumbers attribute.
MLIR_CAPI_EXPORTED bool mlirMhloAttributeIsAConvDimensionNumbers(
    MlirAttribute attr);

// Returns the properties of ConvDimensionNumbers attributes.
MLIR_CAPI_EXPORTED int64_t
mlirMhloConvDimensionNumbersGetInputBatchDimension(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
mlirMhloConvDimensionNumbersGetInputFeatureDimension(MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t
mlirMhloConvDimensionNumbersGetInputSpatialDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
mlirMhloConvDimensionNumbersGetInputSpatialDimensionsElem(MlirAttribute attr,
                                                          intptr_t pos);
MLIR_CAPI_EXPORTED int64_t
mlirMhloConvDimensionNumbersGetKernelInputFeatureDimension(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
mlirMhloConvDimensionNumbersGetKernelOutputFeatureDimension(MlirAttribute attr);
MLIR_CAPI_EXPORTED intptr_t
mlirMhloConvDimensionNumbersGetKernelSpatialDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
mlirMhloConvDimensionNumbersGetKernelSpatialDimensionsElem(MlirAttribute attr,
                                                           intptr_t pos);
MLIR_CAPI_EXPORTED int64_t
mlirMhloConvDimensionNumbersGetOutputBatchDimension(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
mlirMhloConvDimensionNumbersGetOutputFeatureDimension(MlirAttribute attr);
MLIR_CAPI_EXPORTED intptr_t
mlirMhloConvDimensionNumbersGetOutputSpatialDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
mlirMhloConvDimensionNumbersGetOutputSpatialDimensionsElem(MlirAttribute attr,
                                                           intptr_t pos);

//
// ComparisonDirectionAttr.
//
// Creates a new ComparisonDirection attribute with the given
// 'direction' string parameter.
MLIR_CAPI_EXPORTED MlirAttribute
mlirMhloComparisonDirectionAttrGet(MlirContext ctx, MlirStringRef direction);

// Returns true if the given attribute is a ComparisonDirection attribute.
MLIR_CAPI_EXPORTED bool mlirMhloAttributeIsAComparisonDirectionAttr(
    MlirAttribute attr);

// Returns the direction string associated with ComparisonDirection attribute.
MLIR_CAPI_EXPORTED MlirStringRef
mlirMhloComparisonDirectionAttrGetDirection(MlirAttribute attr);

//
// ComparisonTypeAttr.
//
// Creates a new ComparisonType attribute with the given 'type' string
// parameter.
MLIR_CAPI_EXPORTED MlirAttribute
mlirMhloComparisonTypeAttrGet(MlirContext ctx, MlirStringRef type);

// Returns true if the given attribute is a ComparisonType attribute.
MLIR_CAPI_EXPORTED bool mlirMhloAttributeIsAComparisonTypeAttr(
    MlirAttribute attr);

// Returns the type string associated with ComparisonType attribute.
MLIR_CAPI_EXPORTED MlirStringRef
mlirMhloComparisonTypeAttrGetType(MlirAttribute attr);

//
// DomainKindAttr.
//
// Creates a new DomainKind attribute with the given 'kind' string
// parameter.
MLIR_CAPI_EXPORTED MlirAttribute mlirMhloDomainKindAttrGet(MlirContext ctx,
                                                           MlirStringRef kind);

// Returns true if the given attribute is a DomainKind attribute.
MLIR_CAPI_EXPORTED bool mlirMhloAttributeIsADomainKindAttr(MlirAttribute attr);

// Returns the type string associated with DomainKind attribute.
MLIR_CAPI_EXPORTED MlirStringRef
mlirMhloDomainKindAttrGetType(MlirAttribute attr);

//
// PrecisionAttr.
//
// Creates a new Precision attribute with the given 'type' string
// parameter.
MLIR_CAPI_EXPORTED MlirAttribute mlirMhloPrecisionAttrGet(MlirContext ctx,
                                                          MlirStringRef type);

// Returns true if the given attribute is a Precision attribute.
MLIR_CAPI_EXPORTED bool mlirMhloAttributeIsAPrecisionAttr(MlirAttribute attr);

// Returns the type string associated with Precision attribute.
MLIR_CAPI_EXPORTED MlirStringRef
mlirMhloPrecisionAttrGetPrecision(MlirAttribute attr);

//
// FftTypeAttr.
//
// Creates a new FftType attribute with the given 'type' string parameter.
MLIR_CAPI_EXPORTED MlirAttribute mlirMhloFftTypeAttrGet(MlirContext ctx,
                                                        MlirStringRef type);

// Returns true if the given attribute is a FftType attribute.
MLIR_CAPI_EXPORTED bool mlirMhloAttributeIsAFftTypeAttr(MlirAttribute attr);

// Returns the type string associated with FftType attribute.
MLIR_CAPI_EXPORTED MlirStringRef
mlirMhloFftTypeAttrGetFftType(MlirAttribute attr);

//
// DequantizeModeAttr.
//
// Creates a new DequantizeMode attribute with the given 'mode' string
// parameter.
MLIR_CAPI_EXPORTED MlirAttribute
mlirMhloDequantizeModeAttrGet(MlirContext ctx, MlirStringRef mode);

// Returns true if the given attribute is a DequantizeMode attribute.
MLIR_CAPI_EXPORTED bool mlirMhloAttributeIsADequantizeModeAttr(
    MlirAttribute attr);

// Returns the mode string associated with DequantizeMode attribute.
MLIR_CAPI_EXPORTED MlirStringRef
mlirMhloDequantizeModeAttrGetDequantizeMode(MlirAttribute attr);

//
// TransposeAttr.
//
// Creates a new Transpose attribute with the given 'type' string parameter.
MLIR_CAPI_EXPORTED MlirAttribute mlirMhloTransposeAttrGet(MlirContext ctx,
                                                          MlirStringRef type);

// Returns true if the given attribute is a Transpose attribute.
MLIR_CAPI_EXPORTED bool mlirMhloAttributeIsATransposeAttr(MlirAttribute attr);

// Returns the type string associated with Transpose attribute.
MLIR_CAPI_EXPORTED MlirStringRef
mlirMhloTransposeAttrGetTranspose(MlirAttribute attr);

//
// FusionKindAttr.
//
// Creates a new FusionKind attribute with the given 'kind' string parameter.
MLIR_CAPI_EXPORTED MlirAttribute mlirMhloFusionKindAttrGet(MlirContext ctx,
                                                           MlirStringRef kind);

// Returns true if the given attribute is a FusionKind attribute.
MLIR_CAPI_EXPORTED bool mlirMhloAttributeIsAFusionKindAttr(MlirAttribute attr);

// Returns the fusion-kind string associated with FusionKind attribute.
MLIR_CAPI_EXPORTED MlirStringRef
mlirMhloFusionKindAttrGetFusionKind(MlirAttribute attr);

//
// RngAlgorithmAttr.
//
// Creates a new RngAlgorithm attribute with the given 'algorithm' string
// parameter.
MLIR_CAPI_EXPORTED MlirAttribute
mlirMhloRngAlgorithmAttrGet(MlirContext ctx, MlirStringRef algorithm);

// Returns true if the given attribute is a RngAlgorithm attribute.
MLIR_CAPI_EXPORTED bool mlirMhloAttributeIsARngAlgorithmAttr(
    MlirAttribute attr);

// Returns the rng-algorithm string associated with RngAlgorithm attribute.
MLIR_CAPI_EXPORTED MlirStringRef
mlirMhloRngAlgorithmAttrGetRngAlgorithm(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif  // MLIR_HLO_C_ATTRIBUTES_H
