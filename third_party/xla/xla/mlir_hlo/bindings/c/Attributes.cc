/* Copyright 2021 The OpenXLA Authors.
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

#include "bindings/c/Attributes.h"

#include <optional>

#include "mhlo/IR/hlo_ops.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Support/LLVM.h"

//
// ScatterDimensionNumbersAttr.
//

MlirAttribute mlirMhloScatterDimensionNumbersGet(
    MlirContext ctx, intptr_t nUpdateWindowDims,
    const int64_t *updateWindowDims, intptr_t nInsertedWindowDims,
    const int64_t *insertedWindowDims, intptr_t nScatteredDimsToOperandDims,
    const int64_t *scatteredDimsToOperandDims, int64_t indexVectorDim) {
  return wrap(mlir::mhlo::ScatterDimensionNumbersAttr::get(
      unwrap(ctx), llvm::ArrayRef(updateWindowDims, nUpdateWindowDims),
      llvm::ArrayRef(insertedWindowDims, nInsertedWindowDims),
      llvm::ArrayRef(scatteredDimsToOperandDims, nScatteredDimsToOperandDims),
      indexVectorDim));
}

bool mlirMhloAttributeIsAScatterDimensionNumbers(MlirAttribute attr) {
  return mlir::isa<mlir::mhlo::ScatterDimensionNumbersAttr>(unwrap(attr));
}

intptr_t mlirMhloScatterDimensionNumbersGetUpdateWindowDimsSize(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getUpdateWindowDims()
      .size();
}

int64_t mlirMhloScatterDimensionNumbersGetUpdateWindowDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return mlir::cast<mlir::mhlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getUpdateWindowDims()[pos];
}

intptr_t mlirMhloScatterDimensionNumbersGetInsertedWindowDimsSize(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getInsertedWindowDims()
      .size();
}

int64_t mlirMhloScatterDimensionNumbersGetInsertedWindowDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return mlir::cast<mlir::mhlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getInsertedWindowDims()[pos];
}

intptr_t mlirMhloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getScatterDimsToOperandDims()
      .size();
}

int64_t mlirMhloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return mlir::cast<mlir::mhlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getScatterDimsToOperandDims()[pos];
}

int64_t mlirMhloDimensionNumbersGetIndexVectorDim(MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::ScatterDimensionNumbersAttr>(unwrap(attr))
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
      unwrap(ctx), llvm::ArrayRef(offsetDims, nOffsetDims),
      llvm::ArrayRef(collapsedSliceDims, nCollapsedSliceDims),
      llvm::ArrayRef(startIndexMap, nStartIndexMap), indexVectorDim));
}

bool mlirMhloAttributeIsAGatherDimensionNumbers(MlirAttribute attr) {
  return mlir::isa<mlir::mhlo::GatherDimensionNumbersAttr>(unwrap(attr));
}

intptr_t mlirMhloGatherDimensionNumbersGetOffsetDimsSize(MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getOffsetDims()
      .size();
}

int64_t mlirMhloGatherDimensionNumbersGetOffsetDimsElem(MlirAttribute attr,
                                                        intptr_t pos) {
  return mlir::cast<mlir::mhlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getOffsetDims()[pos];
}

intptr_t mlirMhloGatherDimensionNumbersGetCollapsedSliceDimsSize(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getCollapsedSliceDims()
      .size();
}

int64_t mlirMhloGatherDimensionNumbersGetCollapsedSliceDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return mlir::cast<mlir::mhlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getCollapsedSliceDims()[pos];
}

intptr_t mlirMhloGatherDimensionNumbersGetStartIndexMapSize(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getStartIndexMap()
      .size();
}

int64_t mlirMhloGatherDimensionNumbersGetStartIndexMapElem(MlirAttribute attr,
                                                           intptr_t pos) {
  return mlir::cast<mlir::mhlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getStartIndexMap()[pos];
}

int64_t mlirMhloGatherDimensionNumbersGetIndexVectorDim(MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::GatherDimensionNumbersAttr>(unwrap(attr))
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
      llvm::ArrayRef(lhsBatchingDimensions, nLhsBatchingDimensions),
      llvm::ArrayRef(rhsBatchingDimensions, nRhsBatchingDimensions),
      llvm::ArrayRef(lhsContractingDimensions, nLhsContractingDimensions),
      llvm::ArrayRef(rhsContractingDimensions, nRhsContractingDimensions)));
}

bool mlirMhloAttributeIsADotDimensionNumbers(MlirAttribute attr) {
  return mlir::isa<mlir::mhlo::DotDimensionNumbersAttr>(unwrap(attr));
}

intptr_t mlirMhloDotDimensionNumbersGetLhsBatchingDimensionsSize(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getLhsBatchingDimensions()
      .size();
}

int64_t mlirMhloDotDimensionNumbersGetLhsBatchingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return mlir::cast<mlir::mhlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getLhsBatchingDimensions()[pos];
}

intptr_t mlirMhloDotDimensionNumbersGetRhsBatchingDimensionsSize(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getRhsBatchingDimensions()
      .size();
}

int64_t mlirMhloDotDimensionNumbersGetRhsBatchingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return mlir::cast<mlir::mhlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getRhsBatchingDimensions()[pos];
}

intptr_t mlirMhloDotDimensionNumbersGetLhsContractingDimensionsSize(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getLhsContractingDimensions()
      .size();
}

int64_t mlirMhloDotDimensionNumbersGetLhsContractingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return mlir::cast<mlir::mhlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getLhsContractingDimensions()[pos];
}

intptr_t mlirMhloDotDimensionNumbersGetRhsContractingDimensionsSize(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getRhsContractingDimensions()
      .size();
}

int64_t mlirMhloDotDimensionNumbersGetRhsContractingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return mlir::cast<mlir::mhlo::DotDimensionNumbersAttr>(unwrap(attr))
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
      llvm::ArrayRef(inputSpatialDimensions, nInputSpatialDimensions),
      kernelInputFeatureDimension, kernelOutputFeatureDimension,
      llvm::ArrayRef(kernelSpatialDimensions, nKernelSpatialDimensions),
      outputBatchDimension, outputFeatureDimension,
      llvm::ArrayRef(outputSpatialDimensions, nOutputSpatialDimensions)));
}

bool mlirMhloAttributeIsAConvDimensionNumbers(MlirAttribute attr) {
  return mlir::isa<mlir::mhlo::ConvDimensionNumbersAttr>(unwrap(attr));
}

int64_t mlirMhloConvDimensionNumbersGetInputBatchDimension(MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getInputBatchDimension();
}

int64_t mlirMhloConvDimensionNumbersGetInputFeatureDimension(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getInputFeatureDimension();
}

intptr_t mlirMhloConvDimensionNumbersGetInputSpatialDimensionsSize(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getInputSpatialDimensions()
      .size();
}

int64_t mlirMhloConvDimensionNumbersGetInputSpatialDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return mlir::cast<mlir::mhlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getInputSpatialDimensions()[pos];
}

int64_t mlirMhloConvDimensionNumbersGetKernelInputFeatureDimension(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getKernelInputFeatureDimension();
}

int64_t mlirMhloConvDimensionNumbersGetKernelOutputFeatureDimension(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getKernelOutputFeatureDimension();
}

intptr_t mlirMhloConvDimensionNumbersGetKernelSpatialDimensionsSize(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getKernelSpatialDimensions()
      .size();
}

int64_t mlirMhloConvDimensionNumbersGetKernelSpatialDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return mlir::cast<mlir::mhlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getKernelSpatialDimensions()[pos];
}

int64_t mlirMhloConvDimensionNumbersGetOutputBatchDimension(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getOutputBatchDimension();
}

int64_t mlirMhloConvDimensionNumbersGetOutputFeatureDimension(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getOutputFeatureDimension();
}

intptr_t mlirMhloConvDimensionNumbersGetOutputSpatialDimensionsSize(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getOutputSpatialDimensions()
      .size();
}

int64_t mlirMhloConvDimensionNumbersGetOutputSpatialDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return mlir::cast<mlir::mhlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getOutputSpatialDimensions()[pos];
}

//
// OutputOperandAliasAttr.
//

MLIR_CAPI_EXPORTED MlirAttribute mlirMhloOutputOperandAliasGet(
    MlirContext ctx, intptr_t nOutputTupleIndices,
    const int64_t *outputTupleIndices, int64_t operandIndex,
    intptr_t nOperandTupleIndices, const int64_t *operandTupleIndices) {
  return wrap(mlir::mhlo::OutputOperandAliasAttr::get(
      unwrap(ctx), llvm::ArrayRef(outputTupleIndices, nOutputTupleIndices),
      operandIndex, llvm::ArrayRef(operandTupleIndices, nOperandTupleIndices)));
}

bool mlirMhloAttributeIsAOutputOperandAlias(MlirAttribute attr) {
  return mlir::isa<mlir::mhlo::OutputOperandAliasAttr>(unwrap(attr));
}

intptr_t mlirMhloOutputOperandAliasGetOutputTupleIndicesSize(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::OutputOperandAliasAttr>(unwrap(attr))
      .getOutputTupleIndices()
      .size();
}

int64_t mlirMhloOutputOperandAliasGetOutputTupleIndicesElem(MlirAttribute attr,
                                                            intptr_t pos) {
  return mlir::cast<mlir::mhlo::OutputOperandAliasAttr>(unwrap(attr))
      .getOutputTupleIndices()[pos];
}

int64_t mlirMhloOutputOperandAliasGetOperandIndex(MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::OutputOperandAliasAttr>(unwrap(attr))
      .getOperandIndex();
}

intptr_t mlirMhloOutputOperandAliasGetOperandTupleIndicesSize(
    MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::OutputOperandAliasAttr>(unwrap(attr))
      .getOperandTupleIndices()
      .size();
}

int64_t mlirMhloOutputOperandAliasGetOperandTupleIndicesElem(MlirAttribute attr,
                                                             intptr_t pos) {
  return mlir::cast<mlir::mhlo::OutputOperandAliasAttr>(unwrap(attr))
      .getOperandTupleIndices()[pos];
}

//
// ComparisonDirectionAttr.
//
MlirAttribute mlirMhloComparisonDirectionAttrGet(MlirContext ctx,
                                                 MlirStringRef value) {
  std::optional<mlir::mhlo::ComparisonDirection> comparisonDirection =
      mlir::mhlo::symbolizeComparisonDirection(unwrap(value));
  if (!comparisonDirection) llvm_unreachable("Invalid value.");
  return wrap(mlir::mhlo::ComparisonDirectionAttr::get(
      unwrap(ctx), comparisonDirection.value()));
}

bool mlirMhloAttributeIsAComparisonDirectionAttr(MlirAttribute attr) {
  return mlir::isa<mlir::mhlo::ComparisonDirectionAttr>(unwrap(attr));
}

MlirStringRef mlirMhloComparisonDirectionAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyComparisonDirection(
      mlir::cast<mlir::mhlo::ComparisonDirectionAttr>(unwrap(attr))
          .getValue()));
}

//
// ComparisonTypeAttr.
//

MlirAttribute mlirMhloComparisonTypeAttrGet(MlirContext ctx,
                                            MlirStringRef value) {
  std::optional<mlir::mhlo::ComparisonType> comparisonType =
      mlir::mhlo::symbolizeComparisonType(unwrap(value));
  if (!comparisonType) llvm_unreachable("Invalid value.");
  return wrap(
      mlir::mhlo::ComparisonTypeAttr::get(unwrap(ctx), comparisonType.value()));
}

bool mlirMhloAttributeIsAComparisonTypeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::mhlo::ComparisonTypeAttr>(unwrap(attr));
}

MlirStringRef mlirMhloComparisonTypeAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyComparisonType(
      mlir::cast<mlir::mhlo::ComparisonTypeAttr>(unwrap(attr)).getValue()));
}

//
// DomainKindAttr.
//

MlirAttribute mlirMhloDomainKindAttrGet(MlirContext ctx, MlirStringRef value) {
  std::optional<mlir::mhlo::DomainKind> domainKind =
      mlir::mhlo::symbolizeDomainKind(unwrap(value));
  if (!domainKind) llvm_unreachable("Invalid value.");
  return wrap(mlir::mhlo::DomainKindAttr::get(unwrap(ctx), domainKind.value()));
}

bool mlirMhloAttributeIsADomainKindAttr(MlirAttribute attr) {
  return mlir::isa<mlir::mhlo::DomainKindAttr>(unwrap(attr));
}

MlirStringRef mlirMhloDomainKindAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyDomainKind(
      mlir::cast<mlir::mhlo::DomainKindAttr>(unwrap(attr)).getValue()));
}

//
// PrecisionAttr.
//

MlirAttribute mlirMhloPrecisionAttrGet(MlirContext ctx, MlirStringRef value) {
  std::optional<mlir::mhlo::Precision> precision =
      mlir::mhlo::symbolizePrecision(unwrap(value));
  if (!precision) llvm_unreachable("Invalid value specified.");
  return wrap(mlir::mhlo::PrecisionAttr::get(unwrap(ctx), precision.value()));
}

bool mlirMhloAttributeIsAPrecisionAttr(MlirAttribute attr) {
  return mlir::isa<mlir::mhlo::PrecisionAttr>(unwrap(attr));
}

MlirStringRef mlirMhloPrecisionAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyPrecision(
      mlir::cast<mlir::mhlo::PrecisionAttr>(unwrap(attr)).getValue()));
}

//
// FftTypeAttr.
//

MlirAttribute mlirMhloFftTypeAttrGet(MlirContext ctx, MlirStringRef value) {
  std::optional<mlir::mhlo::FftType> fftType =
      mlir::mhlo::symbolizeFftType(unwrap(value));
  if (!fftType) llvm_unreachable("Invalid value.");
  return wrap(mlir::mhlo::FftTypeAttr::get(unwrap(ctx), fftType.value()));
}

bool mlirMhloAttributeIsAFftTypeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::mhlo::FftTypeAttr>(unwrap(attr));
}

MlirStringRef mlirMhloFftTypeAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyFftType(
      mlir::cast<mlir::mhlo::FftTypeAttr>(unwrap(attr)).getValue()));
}

//
// DequantizeModeAttr.
//

MlirAttribute mlirMhloDequantizeModeAttrGet(MlirContext ctx,
                                            MlirStringRef value) {
  std::optional<mlir::mhlo::DequantizeMode> dequantizeMode =
      mlir::mhlo::symbolizeDequantizeMode(unwrap(value));
  if (!dequantizeMode) llvm_unreachable("Invalid value.");
  return wrap(
      mlir::mhlo::DequantizeModeAttr::get(unwrap(ctx), dequantizeMode.value()));
}

bool mlirMhloAttributeIsADequantizeModeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::mhlo::DequantizeModeAttr>(unwrap(attr));
}

MlirStringRef mlirMhloDequantizeModeAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyDequantizeMode(
      mlir::cast<mlir::mhlo::DequantizeModeAttr>(unwrap(attr)).getValue()));
}

//
// TransposeAttr.
//

MlirAttribute mlirMhloTransposeAttrGet(MlirContext ctx, MlirStringRef value) {
  std::optional<mlir::mhlo::Transpose> transpose =
      mlir::mhlo::symbolizeTranspose(unwrap(value));
  if (!transpose) llvm_unreachable("Invalid value.");
  return wrap(mlir::mhlo::TransposeAttr::get(unwrap(ctx), transpose.value()));
}

bool mlirMhloAttributeIsATransposeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::mhlo::TransposeAttr>(unwrap(attr));
}

MlirStringRef mlirMhloTransposeAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyTranspose(
      mlir::cast<mlir::mhlo::TransposeAttr>(unwrap(attr)).getValue()));
}

//
// FusionKindAttr.
//

MlirAttribute mlirMhloFusionKindAttrGet(MlirContext ctx, MlirStringRef value) {
  std::optional<mlir::mhlo::FusionKind> fusionKind =
      mlir::mhlo::symbolizeFusionKind(unwrap(value));
  if (!fusionKind) llvm_unreachable("Invalid value.");
  return wrap(mlir::mhlo::FusionKindAttr::get(unwrap(ctx), fusionKind.value()));
}

bool mlirMhloAttributeIsAFusionKindAttr(MlirAttribute attr) {
  return mlir::isa<mlir::mhlo::FusionKindAttr>(unwrap(attr));
}

MlirStringRef mlirMhloFusionKindAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyFusionKind(
      mlir::cast<mlir::mhlo::FusionKindAttr>(unwrap(attr)).getValue()));
}

//
// RngDistributionAttr.
//

MlirAttribute mlirMhloRngDistributionAttrGet(MlirContext ctx,
                                             MlirStringRef value) {
  std::optional<mlir::mhlo::RngDistribution> rngDistribution =
      mlir::mhlo::symbolizeRngDistribution(unwrap(value));
  if (!rngDistribution) llvm_unreachable("Invalid value.");
  return wrap(mlir::mhlo::RngDistributionAttr::get(unwrap(ctx),
                                                   rngDistribution.value()));
}

bool mlirMhloAttributeIsARngDistributionAttr(MlirAttribute attr) {
  return mlir::isa<mlir::mhlo::RngDistributionAttr>(unwrap(attr));
}

MlirStringRef mlirMhloRngDistributionAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyRngDistribution(
      mlir::cast<mlir::mhlo::RngDistributionAttr>(unwrap(attr)).getValue()));
}

//
// RngAlgorithmAttr.
//

MlirAttribute mlirMhloRngAlgorithmAttrGet(MlirContext ctx,
                                          MlirStringRef value) {
  std::optional<mlir::mhlo::RngAlgorithm> rngAlgorithm =
      mlir::mhlo::symbolizeRngAlgorithm(unwrap(value));
  if (!rngAlgorithm) llvm_unreachable("Invalid value.");
  return wrap(
      mlir::mhlo::RngAlgorithmAttr::get(unwrap(ctx), rngAlgorithm.value()));
}

bool mlirMhloAttributeIsARngAlgorithmAttr(MlirAttribute attr) {
  return mlir::isa<mlir::mhlo::RngAlgorithmAttr>(unwrap(attr));
}

MlirStringRef mlirMhloRngAlgorithmAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::mhlo::stringifyRngAlgorithm(
      mlir::cast<mlir::mhlo::RngAlgorithmAttr>(unwrap(attr)).getValue()));
}

//
// ChannelHandle
//

MlirAttribute mlirMhloChannelHandleGet(MlirContext ctx, int64_t handle,
                                       int64_t type) {
  return wrap(mlir::mhlo::ChannelHandleAttr::get(unwrap(ctx), handle, type));
}

bool mlirMhloAttributeIsChannelHandle(MlirAttribute attr) {
  return mlir::isa<mlir::mhlo::ChannelHandleAttr>(unwrap(attr));
}

int64_t mlirMhloChannelHandleGetHandle(MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::ChannelHandleAttr>(unwrap(attr)).getHandle();
}

int64_t mlirMhloChannelHandleGetType(MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::ChannelHandleAttr>(unwrap(attr)).getType();
}

//
// TypeExtensions
//

MlirAttribute mlirMhloTypeExtensionsGet(MlirContext ctx, intptr_t nBounds,
                                        const int64_t *bounds) {
  return wrap(mlir::mhlo::TypeExtensionsAttr::get(
      unwrap(ctx), llvm::ArrayRef(bounds, nBounds)));
}

bool mlirMhloAttributeIsTypeExtensions(MlirAttribute attr) {
  return mlir::isa<mlir::mhlo::TypeExtensionsAttr>(unwrap(attr));
}

intptr_t mlirMhloTypeExtensionsGetBoundsSize(MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::TypeExtensionsAttr>(unwrap(attr))
      .getBounds()
      .size();
}

int64_t mlirMhloTypeExtensionsGetBoundsElem(MlirAttribute attr, intptr_t pos) {
  return mlir::cast<mlir::mhlo::TypeExtensionsAttr>(unwrap(attr))
      .getBounds()[pos];
}

//
// SparsityDescriptor
//

MlirAttribute mlirMhloSparsityDescriptorGet(MlirContext ctx, int64_t dimension,
                                            int64_t n, int64_t m) {
  return wrap(
      mlir::mhlo::SparsityDescriptorAttr::get(unwrap(ctx), dimension, n, m));
}

bool mlirMhloAttributeIsASparsityDescriptor(MlirAttribute attr) {
  return mlir::isa<mlir::mhlo::SparsityDescriptorAttr>(unwrap(attr));
}

int64_t mlirMhloSparsityDescriptorGetDimension(MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::SparsityDescriptorAttr>(unwrap(attr))
      .getDimension();
}

int64_t mlirMhloSparsityDescriptorGetN(MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::SparsityDescriptorAttr>(unwrap(attr)).getN();
}

int64_t mlirMhloSparsityDescriptorGetM(MlirAttribute attr) {
  return mlir::cast<mlir::mhlo::SparsityDescriptorAttr>(unwrap(attr)).getM();
}
