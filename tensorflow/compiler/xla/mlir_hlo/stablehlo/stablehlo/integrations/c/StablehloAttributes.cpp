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

#include "stablehlo/integrations/c/StablehloAttributes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "stablehlo/dialect/StablehloOps.h"

//===----------------------------------------------------------------------===//
// ScatterDimensionNumbers
//===----------------------------------------------------------------------===//

MlirAttribute stablehloScatterDimensionNumbersGet(
    MlirContext ctx, intptr_t nUpdateWindowDims,
    const int64_t *updateWindowDims, intptr_t nInsertedWindowDims,
    const int64_t *insertedWindowDims, intptr_t nScatteredDimsToOperandDims,
    const int64_t *scatteredDimsToOperandDims, int64_t indexVectorDim) {
  return wrap(mlir::stablehlo::ScatterDimensionNumbersAttr::get(
      unwrap(ctx), llvm::makeArrayRef(updateWindowDims, nUpdateWindowDims),
      llvm::makeArrayRef(insertedWindowDims, nInsertedWindowDims),
      llvm::makeArrayRef(scatteredDimsToOperandDims,
                         nScatteredDimsToOperandDims),
      indexVectorDim));
}

bool stablehloAttributeIsAScatterDimensionNumbers(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::ScatterDimensionNumbersAttr>();
}

intptr_t stablehloScatterDimensionNumbersGetUpdateWindowDimsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ScatterDimensionNumbersAttr>()
      .getUpdateWindowDims()
      .size();
}

int64_t stablehloScatterDimensionNumbersGetUpdateWindowDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ScatterDimensionNumbersAttr>()
      .getUpdateWindowDims()[pos];
}

intptr_t stablehloScatterDimensionNumbersGetInsertedWindowDimsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ScatterDimensionNumbersAttr>()
      .getInsertedWindowDims()
      .size();
}

int64_t stablehloScatterDimensionNumbersGetInsertedWindowDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ScatterDimensionNumbersAttr>()
      .getInsertedWindowDims()[pos];
}

intptr_t stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ScatterDimensionNumbersAttr>()
      .getScatterDimsToOperandDims()
      .size();
}

int64_t stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ScatterDimensionNumbersAttr>()
      .getScatterDimsToOperandDims()[pos];
}

int64_t stablehloDimensionNumbersGetIndexVectorDim(MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ScatterDimensionNumbersAttr>()
      .getIndexVectorDim();
}

//===----------------------------------------------------------------------===//
// GatherDimensionNumbers
//===----------------------------------------------------------------------===//

MlirAttribute stablehloGatherDimensionNumbersGet(
    MlirContext ctx, intptr_t nOffsetDims, const int64_t *offsetDims,
    intptr_t nCollapsedSliceDims, const int64_t *collapsedSliceDims,
    intptr_t nStartIndexMap, const int64_t *startIndexMap,
    int64_t indexVectorDim) {
  return wrap(mlir::stablehlo::GatherDimensionNumbersAttr::get(
      unwrap(ctx), llvm::makeArrayRef(offsetDims, nOffsetDims),
      llvm::makeArrayRef(collapsedSliceDims, nCollapsedSliceDims),
      llvm::makeArrayRef(startIndexMap, nStartIndexMap), indexVectorDim));
}

bool stablehloAttributeIsAGatherDimensionNumbers(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::GatherDimensionNumbersAttr>();
}

intptr_t stablehloGatherDimensionNumbersGetOffsetDimsSize(MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::GatherDimensionNumbersAttr>()
      .getOffsetDims()
      .size();
}

int64_t stablehloGatherDimensionNumbersGetOffsetDimsElem(MlirAttribute attr,
                                                         intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::GatherDimensionNumbersAttr>()
      .getOffsetDims()[pos];
}

intptr_t stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::GatherDimensionNumbersAttr>()
      .getCollapsedSliceDims()
      .size();
}

int64_t stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::GatherDimensionNumbersAttr>()
      .getCollapsedSliceDims()[pos];
}

intptr_t stablehloGatherDimensionNumbersGetStartIndexMapSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::GatherDimensionNumbersAttr>()
      .getStartIndexMap()
      .size();
}

int64_t stablehloGatherDimensionNumbersGetStartIndexMapElem(MlirAttribute attr,
                                                            intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::GatherDimensionNumbersAttr>()
      .getStartIndexMap()[pos];
}

int64_t stablehloGatherDimensionNumbersGetIndexVectorDim(MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::GatherDimensionNumbersAttr>()
      .getIndexVectorDim();
}

//===----------------------------------------------------------------------===//
// DotDimensionNumbers
//===----------------------------------------------------------------------===//

MlirAttribute stablehloDotDimensionNumbersGet(
    MlirContext ctx, intptr_t nLhsBatchingDimensions,
    const int64_t *lhsBatchingDimensions, intptr_t nRhsBatchingDimensions,
    const int64_t *rhsBatchingDimensions, intptr_t nLhsContractingDimensions,
    const int64_t *lhsContractingDimensions, intptr_t nRhsContractingDimensions,
    const int64_t *rhsContractingDimensions) {
  return wrap(mlir::stablehlo::DotDimensionNumbersAttr::get(
      unwrap(ctx),
      llvm::makeArrayRef(lhsBatchingDimensions, nLhsBatchingDimensions),
      llvm::makeArrayRef(rhsBatchingDimensions, nRhsBatchingDimensions),
      llvm::makeArrayRef(lhsContractingDimensions, nLhsContractingDimensions),
      llvm::makeArrayRef(rhsContractingDimensions, nRhsContractingDimensions)));
}

bool stablehloAttributeIsADotDimensionNumbers(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::DotDimensionNumbersAttr>();
}

intptr_t stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::DotDimensionNumbersAttr>()
      .getLhsBatchingDimensions()
      .size();
}

int64_t stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::DotDimensionNumbersAttr>()
      .getLhsBatchingDimensions()[pos];
}

intptr_t stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::DotDimensionNumbersAttr>()
      .getRhsBatchingDimensions()
      .size();
}

int64_t stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::DotDimensionNumbersAttr>()
      .getRhsBatchingDimensions()[pos];
}

intptr_t stablehloDotDimensionNumbersGetLhsContractingDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::DotDimensionNumbersAttr>()
      .getLhsContractingDimensions()
      .size();
}

int64_t stablehloDotDimensionNumbersGetLhsContractingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::DotDimensionNumbersAttr>()
      .getLhsContractingDimensions()[pos];
}

intptr_t stablehloDotDimensionNumbersGetRhsContractingDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::DotDimensionNumbersAttr>()
      .getRhsContractingDimensions()
      .size();
}

int64_t stablehloDotDimensionNumbersGetRhsContractingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::DotDimensionNumbersAttr>()
      .getRhsContractingDimensions()[pos];
}

//===----------------------------------------------------------------------===//
// ConvDimensionNumbers
//===----------------------------------------------------------------------===//

MlirAttribute stablehloConvDimensionNumbersGet(
    MlirContext ctx, int64_t inputBatchDimension, int64_t inputFeatureDimension,
    intptr_t nInputSpatialDimensions, const int64_t *inputSpatialDimensions,
    int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
    intptr_t nKernelSpatialDimensions, const int64_t *kernelSpatialDimensions,
    int64_t outputBatchDimension, int64_t outputFeatureDimension,
    intptr_t nOutputSpatialDimensions, const int64_t *outputSpatialDimensions) {
  return wrap(mlir::stablehlo::ConvDimensionNumbersAttr::get(
      unwrap(ctx), inputBatchDimension, inputFeatureDimension,
      llvm::makeArrayRef(inputSpatialDimensions, nInputSpatialDimensions),
      kernelInputFeatureDimension, kernelOutputFeatureDimension,
      llvm::makeArrayRef(kernelSpatialDimensions, nKernelSpatialDimensions),
      outputBatchDimension, outputFeatureDimension,
      llvm::makeArrayRef(outputSpatialDimensions, nOutputSpatialDimensions)));
}

bool stablehloAttributeIsAConvDimensionNumbers(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::ConvDimensionNumbersAttr>();
}

int64_t stablehloConvDimensionNumbersGetInputBatchDimension(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getInputBatchDimension();
}

int64_t stablehloConvDimensionNumbersGetInputFeatureDimension(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getInputFeatureDimension();
}

intptr_t stablehloConvDimensionNumbersGetInputSpatialDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getInputSpatialDimensions()
      .size();
}

int64_t stablehloConvDimensionNumbersGetInputSpatialDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getInputSpatialDimensions()[pos];
}

int64_t stablehloConvDimensionNumbersGetKernelInputFeatureDimension(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getKernelInputFeatureDimension();
}

int64_t stablehloConvDimensionNumbersGetKernelOutputFeatureDimension(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getKernelOutputFeatureDimension();
}

intptr_t stablehloConvDimensionNumbersGetKernelSpatialDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getKernelSpatialDimensions()
      .size();
}

int64_t stablehloConvDimensionNumbersGetKernelSpatialDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getKernelSpatialDimensions()[pos];
}

int64_t stablehloConvDimensionNumbersGetOutputBatchDimension(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getOutputBatchDimension();
}

int64_t stablehloConvDimensionNumbersGetOutputFeatureDimension(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getOutputFeatureDimension();
}

intptr_t stablehloConvDimensionNumbersGetOutputSpatialDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getOutputSpatialDimensions()
      .size();
}

int64_t stablehloConvDimensionNumbersGetOutputSpatialDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getOutputSpatialDimensions()[pos];
}

//===----------------------------------------------------------------------===//
// ComparisonDirectionAttr
//===----------------------------------------------------------------------===//

MlirAttribute stablehloComparisonDirectionAttrGet(MlirContext ctx,
                                                  MlirStringRef direction) {
  llvm::Optional<mlir::stablehlo::ComparisonDirection> compareDirection =
      mlir::stablehlo::symbolizeComparisonDirection(unwrap(direction));
  if (!compareDirection)
    llvm_unreachable("Invalid comparison-direction specified.");
  return wrap(mlir::stablehlo::ComparisonDirectionAttr::get(
      unwrap(ctx), compareDirection.value()));
}

bool stablehloAttributeIsAComparisonDirectionAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::ComparisonDirectionAttr>();
}

MlirStringRef stablehloComparisonDirectionAttrGetDirection(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyComparisonDirection(
      unwrap(attr)
          .cast<mlir::stablehlo::ComparisonDirectionAttr>()
          .getValue()));
}

//===----------------------------------------------------------------------===//
// ComparisonTypeAttr
//===----------------------------------------------------------------------===//

MlirAttribute stablehloComparisonTypeAttrGet(MlirContext ctx,
                                             MlirStringRef type) {
  llvm::Optional<mlir::stablehlo::ComparisonType> compareType =
      mlir::stablehlo::symbolizeComparisonType(unwrap(type));
  if (!compareType) llvm_unreachable("Invalid comparison-type specified.");
  return wrap(mlir::stablehlo::ComparisonTypeAttr::get(unwrap(ctx),
                                                       compareType.value()));
}

bool stablehloAttributeIsAComparisonTypeAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::ComparisonTypeAttr>();
}

MlirStringRef stablehloComparisonTypeAttrGetType(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyComparisonType(
      unwrap(attr).cast<mlir::stablehlo::ComparisonTypeAttr>().getValue()));
}

//===----------------------------------------------------------------------===//
// PrecisionAttr
//===----------------------------------------------------------------------===//

MlirAttribute stablehloPrecisionAttrGet(MlirContext ctx, MlirStringRef type) {
  llvm::Optional<mlir::stablehlo::Precision> precisionType =
      mlir::stablehlo::symbolizePrecision(unwrap(type));
  if (!precisionType) llvm_unreachable("Invalid precision-type specified.");
  return wrap(
      mlir::stablehlo::PrecisionAttr::get(unwrap(ctx), precisionType.value()));
}

bool stablehloAttributeIsAPrecisionAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::PrecisionAttr>();
}

MlirStringRef stablehloPrecisionAttrGetPrecision(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyPrecision(
      unwrap(attr).cast<mlir::stablehlo::PrecisionAttr>().getValue()));
}

//===----------------------------------------------------------------------===//
// FftTypeAttr
//===----------------------------------------------------------------------===//

MlirAttribute stablehloFftTypeAttrGet(MlirContext ctx, MlirStringRef type) {
  llvm::Optional<mlir::stablehlo::FftType> fftType =
      mlir::stablehlo::symbolizeFftType(unwrap(type));
  if (!fftType) llvm_unreachable("Invalid fft-type specified.");
  return wrap(mlir::stablehlo::FftTypeAttr::get(unwrap(ctx), fftType.value()));
}

bool stablehloAttributeIsAFftTypeAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::FftTypeAttr>();
}

MlirStringRef stablehloFftTypeAttrGetFftType(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyFftType(
      unwrap(attr).cast<mlir::stablehlo::FftTypeAttr>().getValue()));
}

//===----------------------------------------------------------------------===//
// TransposeAttr
//===----------------------------------------------------------------------===//

MlirAttribute stablehloTransposeAttrGet(MlirContext ctx, MlirStringRef type) {
  llvm::Optional<mlir::stablehlo::Transpose> transposeType =
      mlir::stablehlo::symbolizeTranspose(unwrap(type));
  if (!transposeType) llvm_unreachable("Invalid transpose-type specified.");
  return wrap(
      mlir::stablehlo::TransposeAttr::get(unwrap(ctx), transposeType.value()));
}

bool stablehloAttributeIsATransposeAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::TransposeAttr>();
}

MlirStringRef stablehloTransposeAttrGetTranspose(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyTranspose(
      unwrap(attr).cast<mlir::stablehlo::TransposeAttr>().getValue()));
}

//===----------------------------------------------------------------------===//
// RngDistributionAttr
//===----------------------------------------------------------------------===//

MlirAttribute stablehloRngDistributionAttrGet(MlirContext ctx,
                                              MlirStringRef distribution) {
  llvm::Optional<mlir::stablehlo::RngDistribution> rngDistribution =
      mlir::stablehlo::symbolizeRngDistribution(unwrap(distribution));
  if (!rngDistribution) llvm_unreachable("Invalid rng-distribution specified.");
  return wrap(mlir::stablehlo::RngDistributionAttr::get(
      unwrap(ctx), rngDistribution.value()));
}

bool stablehloAttributeIsARngDistributionAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::RngDistributionAttr>();
}

MlirStringRef stablehloRngDistributionAttrGetRngDistribution(
    MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyRngDistribution(
      unwrap(attr).cast<mlir::stablehlo::RngDistributionAttr>().getValue()));
}

//===----------------------------------------------------------------------===//
// RngAlgorithmAttr
//===----------------------------------------------------------------------===//

MlirAttribute stablehloRngAlgorithmAttrGet(MlirContext ctx,
                                           MlirStringRef algorithm) {
  llvm::Optional<mlir::stablehlo::RngAlgorithm> rngAlgorithm =
      mlir::stablehlo::symbolizeRngAlgorithm(unwrap(algorithm));
  if (!rngAlgorithm) llvm_unreachable("Invalid rng-algorithm specified.");
  return wrap(mlir::stablehlo::RngAlgorithmAttr::get(unwrap(ctx),
                                                     rngAlgorithm.value()));
}

bool stablehloAttributeIsARngAlgorithmAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::RngAlgorithmAttr>();
}

MlirStringRef stablehloRngAlgorithmAttrGetRngAlgorithm(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyRngAlgorithm(
      unwrap(attr).cast<mlir::stablehlo::RngAlgorithmAttr>().getValue()));
}

//===----------------------------------------------------------------------===//
// ChannelHandle
//===----------------------------------------------------------------------===//

MlirAttribute stablehloChannelHandleGet(MlirContext ctx, int64_t handle,
                                        int64_t type) {
  return wrap(
      mlir::stablehlo::ChannelHandleAttr::get(unwrap(ctx), handle, type));
}

bool stablehloAttributeIsChannelHandle(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::ChannelHandleAttr>();
}

int64_t stablehloChannelHandleGetHandle(MlirAttribute attr) {
  return unwrap(attr).cast<mlir::stablehlo::ChannelHandleAttr>().getHandle();
}

int64_t stablehloChannelHandleGetType(MlirAttribute attr) {
  return unwrap(attr).cast<mlir::stablehlo::ChannelHandleAttr>().getType();
}

//===----------------------------------------------------------------------===//
// TypeExtensions
//===----------------------------------------------------------------------===//

MlirAttribute stablehloTypeExtensionsGet(MlirContext ctx, intptr_t nBounds,
                                         const int64_t *bounds) {
  return wrap(mlir::stablehlo::TypeExtensionsAttr::get(
      unwrap(ctx), llvm::makeArrayRef(bounds, nBounds)));
}

bool stablehloAttributeIsTypeExtensions(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::TypeExtensionsAttr>();
}

intptr_t stablehloTypeExtensionsGetBoundsSize(MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::TypeExtensionsAttr>()
      .getBounds()
      .size();
}

int64_t stablehloTypeExtensionsGetBoundsElem(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::TypeExtensionsAttr>()
      .getBounds()[pos];
}
