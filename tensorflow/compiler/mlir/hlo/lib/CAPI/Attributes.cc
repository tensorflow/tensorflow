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

#include "mlir-hlo-c/Attributes.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

//
// ScatterDimensionNumbersAttr.
//

MlirAttribute mlirMhloScatterDimensionNumbersGet(
    MlirContext ctx, intptr_t nUpdateWindowDims,
    const int64_t *updateWindowDims, intptr_t nInsertedWindowDims,
    const int64_t *insertedWindowDims, intptr_t nScatteredDimsToOperandDims,
    const int64_t *scatteredDimsToOperandDims, int64_t indexVectorDim) {
  return wrap(mlir::mhlo::ScatterDimensionNumbersAttr::get(
      unwrap(ctx), llvm::makeArrayRef(updateWindowDims, nUpdateWindowDims),
      llvm::makeArrayRef(insertedWindowDims, nInsertedWindowDims),
      llvm::makeArrayRef(scatteredDimsToOperandDims,
                         nScatteredDimsToOperandDims),
      indexVectorDim));
}

bool mlirMhloAttributeIsAScatterDimensionNumbers(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::mhlo::ScatterDimensionNumbersAttr>();
}

intptr_t mlirMhloScatterDimensionNumbersGetUpdateWindowDimsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::ScatterDimensionNumbersAttr>()
      .getUpdateWindowDims()
      .size();
}

int64_t mlirMhloScatterDimensionNumbersGetUpdateWindowDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::mhlo::ScatterDimensionNumbersAttr>()
      .getUpdateWindowDims()[pos];
}

intptr_t mlirMhloScatterDimensionNumbersGetInsertedWindowDimsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::ScatterDimensionNumbersAttr>()
      .getInsertedWindowDims()
      .size();
}

int64_t mlirMhloScatterDimensionNumbersGetInsertedWindowDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::mhlo::ScatterDimensionNumbersAttr>()
      .getInsertedWindowDims()[pos];
}

intptr_t mlirMhloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::ScatterDimensionNumbersAttr>()
      .getScatterDimsToOperandDims()
      .size();
}

int64_t mlirMhloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::mhlo::ScatterDimensionNumbersAttr>()
      .getScatterDimsToOperandDims()[pos];
}

int64_t mlirMhloDimensionNumbersGetIndexVectorDim(MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::ScatterDimensionNumbersAttr>()
      .getIndexVectorDim();
}

//
// GatherDimensionNumbersAttr.
//

MlirAttribute mlirMhloGatherDimensionNumbersGet(
    MlirContext ctx, intptr_t nOffsetDims, const int64_t *offsetDims,
    intptr_t nCollapsedSliceDims, const int64_t *collapsedSliceDims,
    intptr_t nStartIndexMap, const int64_t *startIndexMap,
    int64_t indexVectorDim) {
  return wrap(mlir::mhlo::GatherDimensionNumbersAttr::get(
      unwrap(ctx), llvm::makeArrayRef(offsetDims, nOffsetDims),
      llvm::makeArrayRef(collapsedSliceDims, nCollapsedSliceDims),
      llvm::makeArrayRef(startIndexMap, nStartIndexMap), indexVectorDim));
}

bool mlirMhloAttributeIsAGatherDimensionNumbers(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::mhlo::GatherDimensionNumbersAttr>();
}

intptr_t mlirMhloGatherDimensionNumbersGetOffsetDimsSize(MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::GatherDimensionNumbersAttr>()
      .getOffsetDims()
      .size();
}

int64_t mlirMhloGatherDimensionNumbersGetOffsetDimsElem(MlirAttribute attr,
                                                        intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::mhlo::GatherDimensionNumbersAttr>()
      .getOffsetDims()[pos];
}

intptr_t mlirMhloGatherDimensionNumbersGetCollapsedSliceDimsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::GatherDimensionNumbersAttr>()
      .getCollapsedSliceDims()
      .size();
}

int64_t mlirMhloGatherDimensionNumbersGetCollapsedSliceDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::mhlo::GatherDimensionNumbersAttr>()
      .getCollapsedSliceDims()[pos];
}

intptr_t mlirMhloGatherDimensionNumbersGetStartIndexMapSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::GatherDimensionNumbersAttr>()
      .getStartIndexMap()
      .size();
}

int64_t mlirMhloGatherDimensionNumbersGetStartIndexMapElem(MlirAttribute attr,
                                                           intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::mhlo::GatherDimensionNumbersAttr>()
      .getStartIndexMap()[pos];
}

int64_t mlirMhloGatherDimensionNumbersGetIndexVectorDim(MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::GatherDimensionNumbersAttr>()
      .getIndexVectorDim();
}

//
// DotDimensionNumbersAttr.
//

MlirAttribute mlirMhloDotDimensionNumbersGet(
    MlirContext ctx, intptr_t nLhsBatchingDimensions,
    const int64_t *lhsBatchingDimensions, intptr_t nRhsBatchingDimensions,
    const int64_t *rhsBatchingDimensions, intptr_t nLhsContractingDimensions,
    const int64_t *lhsContractingDimensions, intptr_t nRhsContractingDimensions,
    const int64_t *rhsContractingDimensions) {
  return wrap(mlir::mhlo::DotDimensionNumbersAttr::get(
      unwrap(ctx),
      llvm::makeArrayRef(lhsBatchingDimensions, nLhsBatchingDimensions),
      llvm::makeArrayRef(rhsBatchingDimensions, nRhsBatchingDimensions),
      llvm::makeArrayRef(lhsContractingDimensions, nLhsContractingDimensions),
      llvm::makeArrayRef(rhsContractingDimensions, nRhsContractingDimensions)));
}

bool mlirMhloAttributeIsADotDimensionNumbers(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::mhlo::DotDimensionNumbersAttr>();
}

intptr_t mlirMhloDotDimensionNumbersGetLhsBatchingDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::DotDimensionNumbersAttr>()
      .getLhsBatchingDimensions()
      .size();
}

int64_t mlirMhloDotDimensionNumbersGetLhsBatchingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::mhlo::DotDimensionNumbersAttr>()
      .getLhsBatchingDimensions()[pos];
}

intptr_t mlirMhloDotDimensionNumbersGetRhsBatchingDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::DotDimensionNumbersAttr>()
      .getRhsBatchingDimensions()
      .size();
}

int64_t mlirMhloDotDimensionNumbersGetRhsBatchingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::mhlo::DotDimensionNumbersAttr>()
      .getRhsBatchingDimensions()[pos];
}

intptr_t mlirMhloDotDimensionNumbersGetLhsContractingDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::DotDimensionNumbersAttr>()
      .getLhsContractingDimensions()
      .size();
}

int64_t mlirMhloDotDimensionNumbersGetLhsContractingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::mhlo::DotDimensionNumbersAttr>()
      .getLhsContractingDimensions()[pos];
}

intptr_t mlirMhloDotDimensionNumbersGetRhsContractingDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::DotDimensionNumbersAttr>()
      .getRhsContractingDimensions()
      .size();
}

int64_t mlirMhloDotDimensionNumbersGetRhsContractingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::mhlo::DotDimensionNumbersAttr>()
      .getRhsContractingDimensions()[pos];
}

//
// ConvDimensionNumbersAttr.
//

MlirAttribute mlirMhloConvDimensionNumbersGet(
    MlirContext ctx, int64_t inputBatchDimension, int64_t inputFeatureDimension,
    intptr_t nInputSpatialDimensions, const int64_t *inputSpatialDimensions,
    int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
    intptr_t nKernelSpatialDimensions, const int64_t *kernelSpatialDimensions,
    int64_t outputBatchDimension, int64_t outputFeatureDimension,
    intptr_t nOutputSpatialDimensions, const int64_t *outputSpatialDimensions) {
  return wrap(mlir::mhlo::ConvDimensionNumbersAttr::get(
      unwrap(ctx), inputBatchDimension, inputFeatureDimension,
      llvm::makeArrayRef(inputSpatialDimensions, nInputSpatialDimensions),
      kernelInputFeatureDimension, kernelOutputFeatureDimension,
      llvm::makeArrayRef(kernelSpatialDimensions, nKernelSpatialDimensions),
      outputBatchDimension, outputFeatureDimension,
      llvm::makeArrayRef(outputSpatialDimensions, nOutputSpatialDimensions)));
}

bool mlirMhloAttributeIsAConvDimensionNumbers(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::mhlo::ConvDimensionNumbersAttr>();
}

int64_t mlirMhloConvDimensionNumbersGetInputBatchDimension(MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getInputBatchDimension();
}

int64_t mlirMhloConvDimensionNumbersGetInputFeatureDimension(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getInputFeatureDimension();
}

intptr_t mlirMhloConvDimensionNumbersGetInputSpatialDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getInputSpatialDimensions()
      .size();
}

int64_t mlirMhloConvDimensionNumbersGetInputSpatialDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getInputSpatialDimensions()[pos];
}

int64_t mlirMhloConvDimensionNumbersGetKernelInputFeatureDimension(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getKernelInputFeatureDimension();
}

int64_t mlirMhloConvDimensionNumbersGetKernelOutputFeatureDimension(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getKernelOutputFeatureDimension();
}

intptr_t mlirMhloConvDimensionNumbersGetKernelSpatialDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getKernelSpatialDimensions()
      .size();
}

int64_t mlirMhloConvDimensionNumbersGetKernelSpatialDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getKernelSpatialDimensions()[pos];
}

int64_t mlirMhloConvDimensionNumbersGetOutputBatchDimension(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getOutputBatchDimension();
}

int64_t mlirMhloConvDimensionNumbersGetOutputFeatureDimension(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getOutputFeatureDimension();
}

intptr_t mlirMhloConvDimensionNumbersGetOutputSpatialDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getOutputSpatialDimensions()
      .size();
}

int64_t mlirMhloConvDimensionNumbersGetOutputSpatialDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getOutputSpatialDimensions()[pos];
}

//
// ComparisonDirectionAttr.
//
MlirAttribute mlirMhloComparisonDirectionAttrGet(MlirContext ctx,
                                                 MlirStringRef direction) {
  llvm::Optional<mlir::mhlo::ComparisonDirection> compareDirection =
      mlir::mhlo::symbolizeComparisonDirection(unwrap(direction));
  if (!compareDirection)
    llvm_unreachable("Invalid comparison-direction specified.");
  return wrap(mlir::mhlo::ComparisonDirectionAttr::get(
      unwrap(ctx), compareDirection.getValue()));
}

bool mlirMhloAttributeIsAComparisonDirectionAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::mhlo::ComparisonDirectionAttr>();
}

MlirStringRef mlirMhloComparisonDirectionAttrGetDirection(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyComparisonDirection(
      unwrap(attr).cast<mlir::mhlo::ComparisonDirectionAttr>().getValue()));
}

//
// ComparisonTypeAttr.
//

MlirAttribute mlirMhloComparisonTypeAttrGet(MlirContext ctx,
                                            MlirStringRef type) {
  llvm::Optional<mlir::mhlo::ComparisonType> compareType =
      mlir::mhlo::symbolizeComparisonType(unwrap(type));
  if (!compareType) llvm_unreachable("Invalid comparison-type specified.");
  return wrap(
      mlir::mhlo::ComparisonTypeAttr::get(unwrap(ctx), compareType.getValue()));
}

bool mlirMhloAttributeIsAComparisonTypeAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::mhlo::ComparisonTypeAttr>();
}

MlirStringRef mlirMhloComparisonTypeAttrGetType(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyComparisonType(
      unwrap(attr).cast<mlir::mhlo::ComparisonTypeAttr>().getValue()));
}

//
// DomainKindAttr.
//

MlirAttribute mlirMhloDomainKindAttrGet(MlirContext ctx, MlirStringRef kind) {
  llvm::Optional<mlir::mhlo::DomainKind> domainKind =
      mlir::mhlo::symbolizeDomainKind(unwrap(kind));
  if (!domainKind) llvm_unreachable("Invalid domain kind specified.");
  return wrap(
      mlir::mhlo::DomainKindAttr::get(unwrap(ctx), domainKind.getValue()));
}

bool mlirMhloAttributeIsADomainKindAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::mhlo::DomainKindAttr>();
}

MlirStringRef mlirMhloDomainKindAttrGetType(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyDomainKind(
      unwrap(attr).cast<mlir::mhlo::DomainKindAttr>().getValue()));
}

//
// PrecisionAttr.
//

MlirAttribute mlirMhloPrecisionAttrGet(MlirContext ctx, MlirStringRef type) {
  llvm::Optional<mlir::mhlo::Precision> precisionType =
      mlir::mhlo::symbolizePrecision(unwrap(type));
  if (!precisionType) llvm_unreachable("Invalid precision-type specified.");
  return wrap(
      mlir::mhlo::PrecisionAttr::get(unwrap(ctx), precisionType.getValue()));
}

bool mlirMhloAttributeIsAPrecisionAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::mhlo::PrecisionAttr>();
}

MlirStringRef mlirMhloPrecisionAttrGetPrecision(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyPrecision(
      unwrap(attr).cast<mlir::mhlo::PrecisionAttr>().getValue()));
}

//
// FftTypeAttr.
//

MlirAttribute mlirMhloFftTypeAttrGet(MlirContext ctx, MlirStringRef type) {
  llvm::Optional<mlir::mhlo::FftType> fftType =
      mlir::mhlo::symbolizeFftType(unwrap(type));
  if (!fftType) llvm_unreachable("Invalid fft-type specified.");
  return wrap(mlir::mhlo::FftTypeAttr::get(unwrap(ctx), fftType.getValue()));
}

bool mlirMhloAttributeIsAFftTypeAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::mhlo::FftTypeAttr>();
}

MlirStringRef mlirMhloFftTypeAttrGetFftType(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyFftType(
      unwrap(attr).cast<mlir::mhlo::FftTypeAttr>().getValue()));
}

//
// DequantizeModeAttr.
//

MlirAttribute mlirMhloDequantizeModeAttrGet(MlirContext ctx,
                                            MlirStringRef mode) {
  llvm::Optional<mlir::mhlo::DequantizeMode> dequantizeMode =
      mlir::mhlo::symbolizeDequantizeMode(unwrap(mode));
  if (!dequantizeMode) llvm_unreachable("Invalid dequantize-mode specified.");
  return wrap(mlir::mhlo::DequantizeModeAttr::get(unwrap(ctx),
                                                  dequantizeMode.getValue()));
}

bool mlirMhloAttributeIsADequantizeModeAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::mhlo::DequantizeModeAttr>();
}

MlirStringRef mlirMhloDequantizeModeAttrGetDequantizeMode(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyDequantizeMode(
      unwrap(attr).cast<mlir::mhlo::DequantizeModeAttr>().getValue()));
}

//
// TransposeAttr.
//

MlirAttribute mlirMhloTransposeAttrGet(MlirContext ctx, MlirStringRef type) {
  llvm::Optional<mlir::mhlo::Transpose> transposeType =
      mlir::mhlo::symbolizeTranspose(unwrap(type));
  if (!transposeType) llvm_unreachable("Invalid transpose-type specified.");
  return wrap(
      mlir::mhlo::TransposeAttr::get(unwrap(ctx), transposeType.getValue()));
}

bool mlirMhloAttributeIsATransposeAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::mhlo::TransposeAttr>();
}

MlirStringRef mlirMhloTransposeAttrGetTranspose(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyTranspose(
      unwrap(attr).cast<mlir::mhlo::TransposeAttr>().getValue()));
}

//
// FusionKindAttr.
//

MlirAttribute mlirMhloFusionKindAttrGet(MlirContext ctx, MlirStringRef kind) {
  llvm::Optional<mlir::mhlo::FusionKind> fusionKind =
      mlir::mhlo::symbolizeFusionKind(unwrap(kind));
  if (!fusionKind) llvm_unreachable("Invalid fusion-kind specified.");
  return wrap(
      mlir::mhlo::FusionKindAttr::get(unwrap(ctx), fusionKind.getValue()));
}

bool mlirMhloAttributeIsAFusionKindAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::mhlo::FusionKindAttr>();
}

MlirStringRef mlirMhloFusionKindAttrGetFusionKind(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyFusionKind(
      unwrap(attr).cast<mlir::mhlo::FusionKindAttr>().getValue()));
}

//
// RngAlgorithmAttr.
//

MlirAttribute mlirMhloRngAlgorithmAttrGet(MlirContext ctx,
                                          MlirStringRef algorithm) {
  llvm::Optional<mlir::mhlo::RngAlgorithm> rngAlgorithm =
      mlir::mhlo::symbolizeRngAlgorithm(unwrap(algorithm));
  if (!rngAlgorithm) llvm_unreachable("Invalid rng-algorithm specified.");
  return wrap(
      mlir::mhlo::RngAlgorithmAttr::get(unwrap(ctx), rngAlgorithm.getValue()));
}

bool mlirMhloAttributeIsARngAlgorithmAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::mhlo::RngAlgorithmAttr>();
}

MlirStringRef mlirMhloRngAlgorithmAttrGetRngAlgorithm(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyRngAlgorithm(
      unwrap(attr).cast<mlir::mhlo::RngAlgorithmAttr>().getValue()));
}
