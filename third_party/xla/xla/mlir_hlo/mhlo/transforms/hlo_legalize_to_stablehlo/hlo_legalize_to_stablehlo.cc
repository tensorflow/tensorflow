/* Copyright 2022 The OpenXLA Authors.

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

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/map_stablehlo_to_hlo_op.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo {
namespace {

// PRIVATE MHLO features are internal to XLA and not used by any ML frontends.
// These should never be converted to StableHLO, as they are not a good fit for
// StableHLO.
template <typename HloOpTy>
bool hasPrivateFeaturesNotInStablehlo(HloOpTy hloOp) {
  // To the best of our knowledge, none of the ML frontends are using these ops
  // directly or indirectly, so we categorized them as private to XLA.
  // Please let us know if we missed something, and we'll recategorize them.
  if constexpr (llvm::is_one_of<
                    HloOpTy, mhlo::AddDependencyOp, mhlo::AsyncDoneOp,
                    mhlo::AsyncStartOp, mhlo::AsyncUpdateOp, mhlo::BitcastOp,
                    mhlo::CopyOp, mhlo::FusionOp, mhlo::StochasticConvertOp,
                    mhlo::XlaRngGetAndUpdateStateOp>::value) {
    return true;
  }
  if constexpr (std::is_same<HloOpTy, mhlo::ConvolutionOp>::value) {
    // StableHLO convolution doesn't support "unknown" dimensions.
    // This is an esoteric feature of MHLO convolutions, and it's different
    // from the notion of dynamic dimensions. For more context, here's the
    // commit which introduced it:
    // https://github.com/tensorflow/mlir-hlo/commit/4d6dc3163c1c9289d86455d9f4de5711465c50fb
    // This feature isn't supported in HLO and doesn't have documentation, so
    // we may end up removing it from MHLO as well.
    auto dimensionNumbers = debugString(hloOp.getDimensionNumbers());
    if (dimensionNumbers.find('?') != std::string::npos) return true;
  }
  if constexpr (std::is_same<HloOpTy, mhlo::CustomCallOp>::value) {
    // To the best of our knowledge, none of the ML frontends are using this
    // enum, so we categorized it as private to XLA.
    // Please let us know if we missed something, and we'll recategorize it.
    if (hloOp.getCustomCallSchedule() != mhlo::CustomCallSchedule::NONE)
      return true;
  }
  return false;
}

// EXPERIMENTAL MHLO features are being explored by ML frontends but do not have
// any agreed upon compatibility guarantees. By default, these features cannot
// be converted to StableHLO, although the allow-experimental-features flag can
// be used to manually enable the conversion. Such features might be a good fit
// for StableHLO, and they are usually accompanied by a StableHLO GitHub ticket.
template <typename HloOpTy>
bool hasExperimentalFeaturesNotInStablehlo(HloOpTy hloOp) {
  if constexpr (std::is_same<HloOpTy, mhlo::AllToAllOp>::value) {
    // StableHLO AllToAll doesn't support the tuple form yet.
    // Proposal: https://github.com/openxla/stablehlo/issues/574.
    if (hloOp.getNumOperands() != 1) return true;
  }
  return false;
}

// PUBLIC MHLO features are not yet in StableHLO but are agreed upon internally
// to have limited compatibility guarantees. These features are used by ML
// frontends but are not yet part of StableHLO. Such features might be a good
// fit for StableHLO, and are usually accompanied by a StableHLO GitHub ticket.
template <typename HloOpTy>
std::optional<int64_t> getPublicFeaturesNotInStablehlo(HloOpTy hloOp) {
  // StableHLO doesn't support TopK yet.
  // Proposal: https://github.com/openxla/stablehlo/pull/1593
  if constexpr (std::is_same<HloOpTy, mhlo::TopKOp>::value) {
    // Version 1: Initial version for TopK.
    return 1;
  }
  // StableHLO doesn't support TopK yet.
  if constexpr (std::is_same<HloOpTy, mhlo::ErfOp>::value) {
    // Version 1: Initial version for ErfOp.
    return 1;
  }
  // StableHLO doesn't support Acosh yet.
  if constexpr (std::is_same<HloOpTy, mhlo::AcoshOp>::value) {
    // Version 1: Initial version for AcoshOp.
    return 1;
  }
  // StableHLO doesn't support Acos yet.
  if constexpr (std::is_same<HloOpTy, mhlo::AcosOp>::value) {
    // Version 1: Initial version for AcosOp.
    return 1;
  }
  // StableHLO doesn't support Atanh yet.
  if constexpr (std::is_same<HloOpTy, mhlo::AtanhOp>::value) {
    // Version 1: Initial version for AtanhOp.
    return 1;
  }
  // StableHLO doesn't support Cosh yet.
  if constexpr (std::is_same<HloOpTy, mhlo::CoshOp>::value) {
    // Version 1: Initial version for CoshOp.
    return 1;
  }
  // StableHLO doesn't support Sinh yet.
  if constexpr (std::is_same<HloOpTy, mhlo::SinhOp>::value) {
    // Version 1: Initial version for CoshOp.
    return 1;
  }
  // StableHLO doesn't support Asin yet.
  if constexpr (std::is_same<HloOpTy, mhlo::AsinOp>::value) {
    // Version 1: Initial version for AsinOp.
    return 1;
  }
  // StableHLO doesn't support Asinh yet.
  if constexpr (std::is_same<HloOpTy, mhlo::AsinhOp>::value) {
    // Version 1: Initial version for AsinhOp.
    return 1;
  }
  // StableHLO doesn't support Mulhi yet.
  if constexpr (std::is_same<HloOpTy, mhlo::MulhiOp>::value) {
    // Version 1: Initial version for MulhiOp.
    return 1;
  }
  // StableHLO doesn't support Scan yet.
  if constexpr (std::is_same_v<HloOpTy, mhlo::ScanOp>) {
    // Version 1: Initial version for ScanOp.
    return 1;
  }
  return std::nullopt;
}

bool isDenseI64Array(Operation* hloOp, llvm::StringRef hloName) {
  if (isa<mhlo::BroadcastOp>(hloOp)) return hloName == "broadcast_sizes";
  if (isa<mhlo::BroadcastInDimOp>(hloOp)) {
    return hloName == "broadcast_dimensions";
  }
  if (isa<mhlo::ConvolutionOp, mhlo::DynamicConvOp>(hloOp)) {
    return hloName == "window_strides" || hloName == "lhs_dilation" ||
           hloName == "rhs_dilation";
  }
  if (isa<mhlo::DynamicBroadcastInDimOp>(hloOp)) {
    return hloName == "broadcast_dimensions" ||
           hloName == "known_expanding_dimensions" ||
           hloName == "known_nonexpanding_dimensions";
  }
  if (isa<mhlo::DynamicSliceOp, mhlo::GatherOp>(hloOp)) {
    return hloName == "slice_sizes";
  }
  if (isa<mhlo::FftOp>(hloOp)) return hloName == "fft_length";
  if (isa<mhlo::MapOp, mhlo::ReduceOp, mhlo::ReverseOp>(hloOp)) {
    return hloName == "dimensions";
  }
  if (isa<mhlo::PadOp>(hloOp)) {
    return hloName == "edge_padding_low" || hloName == "edge_padding_high" ||
           hloName == "interior_padding";
  }
  if (isa<mhlo::ReduceWindowOp>(hloOp)) {
    return hloName == "window_dimensions" || hloName == "window_strides" ||
           hloName == "base_dilations" || hloName == "window_dilations";
  }
  if (isa<mhlo::SelectAndScatterOp>(hloOp)) {
    return hloName == "window_dimensions" || hloName == "window_strides";
  }
  if (isa<mhlo::SliceOp>(hloOp)) {
    return hloName == "start_indices" || hloName == "limit_indices" ||
           hloName == "strides";
  }
  if (isa<mhlo::TransposeOp>(hloOp)) return hloName == "permutation";
  return false;
}

Attribute convertDenseArray(Operation* hloOp, NamedAttribute hloAttr) {
  auto denseInts = mlir::dyn_cast<DenseIntElementsAttr>(hloAttr.getValue());
  if (!denseInts) return {};

  llvm::StringRef hloName = hloAttr.getName().getValue();
  if (isa<mhlo::ConvolutionOp, mhlo::DynamicConvOp>(hloOp) &&
      hloName == "window_reversal") {
    return DenseBoolArrayAttr::get(
        denseInts.getContext(), llvm::to_vector(denseInts.getValues<bool>()));
  }

  // Handle DenseIntElementsAttr --> DenseI64ArrayAttr for StableHLO ops that
  // use dense arrays. This is temporary while MHLO integrates this change.
  if (isDenseI64Array(hloOp, hloName)) {
    return DenseI64ArrayAttr::get(
        denseInts.getContext(),
        llvm::to_vector(denseInts.getValues<int64_t>()));
  }

  return {};
}

#define RETURN_CONVERTED_ENUM_ATTR(Name)                      \
  auto hloValue = mhlo::stringify##Name(attr.getValue());     \
  auto stablehloValue = stablehlo::symbolize##Name(hloValue); \
  if (!stablehloValue.has_value()) return {};                 \
  return stablehlo::Name##Attr::get(attr.getContext(), stablehloValue.value())

stablehlo::ResultAccuracyMode convertResultAccuracyMode(
    mhlo::ResultAccuracyMode mode) {
  switch (mode) {
    case mhlo::ResultAccuracyMode::DEFAULT:
      return stablehlo::ResultAccuracyMode::DEFAULT;
    case mhlo::ResultAccuracyMode::HIGHEST:
      return stablehlo::ResultAccuracyMode::HIGHEST;
    case mhlo::ResultAccuracyMode::TOLERANCE:
      return stablehlo::ResultAccuracyMode::TOLERANCE;
    default:
      return {};
  }
}

Attribute convertAttr(Attribute hloAttr) {
  // Handle MHLO attributes.
  // The logic that handles attributes from other dialects (e.g. builtin
  // attributes) lives below.
  if (auto rgv3 = mlir::dyn_cast<mhlo::ReplicaGroupMeshAxesAttr>(hloAttr)) {
    SmallVector<Attribute> stablehloAxes;
    for (Attribute axisAttr : rgv3.getAxes().getValue()) {
      stablehloAxes.push_back(convertAttr(axisAttr));
    }
    return stablehlo::ReplicaGroupMeshAxesAttr::get(
        rgv3.getContext(), rgv3.getMesh(),
        ArrayAttr::get(rgv3.getContext(), stablehloAxes));
  }
  if (auto attr = mlir::dyn_cast<mhlo::AxisRefAttr>(hloAttr)) {
    stablehlo::SubAxisInfoAttr stablehloSubAxis;
    if (auto sub = attr.getSubAxisInfo()) {
      stablehloSubAxis =
          llvm::cast<stablehlo::SubAxisInfoAttr>(convertAttr(sub));
    }
    return stablehlo::AxisRefAttr::get(attr.getContext(), attr.getName(),
                                       stablehloSubAxis);
  }
  if (auto attr = mlir::dyn_cast<mhlo::SubAxisInfoAttr>(hloAttr)) {
    return stablehlo::SubAxisInfoAttr::get(attr.getContext(), attr.getPreSize(),
                                           attr.getSize());
  }

  if (auto attr = mlir::dyn_cast<mhlo::ChannelHandleAttr>(hloAttr)) {
    return stablehlo::ChannelHandleAttr::get(attr.getContext(),
                                             attr.getHandle(), attr.getType());
  }
  if (auto attr = mlir::dyn_cast<mhlo::ComparisonDirectionAttr>(hloAttr)) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonDirection);
  }
  if (auto attr = mlir::dyn_cast<mhlo::ComparisonTypeAttr>(hloAttr)) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonType);
  }
  if (auto attr = mlir::dyn_cast<mhlo::ConvDimensionNumbersAttr>(hloAttr)) {
    return stablehlo::ConvDimensionNumbersAttr::get(
        attr.getContext(), attr.getInputBatchDimension(),
        attr.getInputFeatureDimension(), attr.getInputSpatialDimensions(),
        attr.getKernelInputFeatureDimension(),
        attr.getKernelOutputFeatureDimension(),
        attr.getKernelSpatialDimensions(), attr.getOutputBatchDimension(),
        attr.getOutputFeatureDimension(), attr.getOutputSpatialDimensions());
  }
  // NOTE: We cannot process CustomCallApiVersionAttr here because
  // `dyn_cast<mhlo::CustomCallApiVersionAttr>()` succeeds for IntegerAttr too.
  if (auto attr = mlir::dyn_cast<mhlo::DotAlgorithmAttr>(hloAttr)) {
    return stablehlo::DotAlgorithmAttr::get(
        attr.getContext(), attr.getLhsPrecisionType(),
        attr.getRhsPrecisionType(), attr.getAccumulationType(),
        attr.getLhsComponentCount(), attr.getRhsComponentCount(),
        attr.getNumPrimitiveOperations(), attr.getAllowImpreciseAccumulation());
  }
  if (auto attr = mlir::dyn_cast<mhlo::DotDimensionNumbersAttr>(hloAttr)) {
    return stablehlo::DotDimensionNumbersAttr::get(
        attr.getContext(), attr.getLhsBatchingDimensions(),
        attr.getRhsBatchingDimensions(), attr.getLhsContractingDimensions(),
        attr.getRhsContractingDimensions());
  }
  if (auto attr = mlir::dyn_cast<mhlo::FftTypeAttr>(hloAttr)) {
    RETURN_CONVERTED_ENUM_ATTR(FftType);
  }
  if (auto attr = mlir::dyn_cast<mhlo::GatherDimensionNumbersAttr>(hloAttr)) {
    return stablehlo::GatherDimensionNumbersAttr::get(
        attr.getContext(), attr.getOffsetDims(), attr.getCollapsedSliceDims(),
        attr.getOperandBatchingDims(), attr.getStartIndicesBatchingDims(),
        attr.getStartIndexMap(), attr.getIndexVectorDim());
  }
  if (auto attr = mlir::dyn_cast<mhlo::OutputOperandAliasAttr>(hloAttr)) {
    return stablehlo::OutputOperandAliasAttr::get(
        attr.getContext(), attr.getOutputTupleIndices(), attr.getOperandIndex(),
        attr.getOperandTupleIndices());
  }
  if (auto attr = mlir::dyn_cast<mhlo::PrecisionAttr>(hloAttr)) {
    RETURN_CONVERTED_ENUM_ATTR(Precision);
  }
  if (auto attr = mlir::dyn_cast<mhlo::RngAlgorithmAttr>(hloAttr)) {
    RETURN_CONVERTED_ENUM_ATTR(RngAlgorithm);
  }
  if (auto attr = mlir::dyn_cast<mhlo::RngDistributionAttr>(hloAttr)) {
    RETURN_CONVERTED_ENUM_ATTR(RngDistribution);
  }
  if (auto attr = mlir::dyn_cast<mhlo::ScatterDimensionNumbersAttr>(hloAttr)) {
    return stablehlo::ScatterDimensionNumbersAttr::get(
        attr.getContext(), attr.getUpdateWindowDims(),
        attr.getInsertedWindowDims(), attr.getInputBatchingDims(),
        attr.getScatterIndicesBatchingDims(),
        attr.getScatterDimsToOperandDims(), attr.getIndexVectorDim());
  }
  if (auto attr = mlir::dyn_cast<mhlo::TransposeAttr>(hloAttr)) {
    RETURN_CONVERTED_ENUM_ATTR(Transpose);
  }
  if (auto attr = mlir::dyn_cast<mhlo::ResultAccuracyModeAttr>(hloAttr)) {
    RETURN_CONVERTED_ENUM_ATTR(ResultAccuracyMode);
  }
  if (auto attr = mlir::dyn_cast<mhlo::ResultAccuracyAttr>(hloAttr)) {
    stablehlo::ResultAccuracyModeAttr modeAttr;
    modeAttr = stablehlo::ResultAccuracyModeAttr::get(
        attr.getContext(),
        convertResultAccuracyMode(attr.getMode().getValue()));

    return stablehlo::ResultAccuracyAttr::get(attr.getContext(), attr.getAtol(),
                                              attr.getRtol(), attr.getUlps(),
                                              modeAttr);
  }
  if (auto attr = mlir::dyn_cast<mhlo::SubAxisInfoAttr>(hloAttr)) {
    return stablehlo::SubAxisInfoAttr::get(attr.getContext(), attr.getPreSize(),
                                           attr.getSize());
  }
  if (auto attr = mlir::dyn_cast<mhlo::AxisRefAttr>(hloAttr)) {
    stablehlo::SubAxisInfoAttr subAxisInfo;
    if (auto hloSubAxisInfo = attr.getSubAxisInfo()) {
      subAxisInfo =
          llvm::cast<stablehlo::SubAxisInfoAttr>(convertAttr(hloSubAxisInfo));
    }
    return stablehlo::AxisRefAttr::get(attr.getContext(), attr.getName(),
                                       subAxisInfo);
  }
  if (auto attr = mlir::dyn_cast<mhlo::OriginalValueAttr>(hloAttr)) {
    return attr;
  }
  if (hloAttr.getDialect().getNamespace() ==
      mhlo::MhloDialect::getDialectNamespace()) {
    // Our guiding principle is to support all StableHLO functionality in MHLO.
    // The inverse is not necessarily true - some MHLO attributes are missing
    // from StableHLO (either deliberately or haven't yet been proposed).
    // As a result, these MHLO attributes will fail here.
    return {};
  }

  // Handle non-MHLO attributes.
  // If an attribute is not defined in MHLO, then it is unchanged,
  // with the exception of ArrayAttr which is converted recursively.
  if (auto hloAttrs = mlir::dyn_cast<ArrayAttr>(hloAttr)) {
    SmallVector<Attribute> stablehloAttrs;
    for (auto hloAttr : hloAttrs) {
      auto stablehloAttr = convertAttr(hloAttr);
      if (!stablehloAttr) return {};
      stablehloAttrs.push_back(stablehloAttr);
    }
    return ArrayAttr::get(hloAttrs.getContext(), stablehloAttrs);
  }
  if (auto hloAttrs = mlir::dyn_cast<DictionaryAttr>(hloAttr)) {
    SmallVector<NamedAttribute> stablehloAttrs;
    for (auto hloAttr : hloAttrs) {
      auto stablehloAttr = convertAttr(hloAttr.getValue());
      if (!stablehloAttr) return {};
      stablehloAttrs.push_back({hloAttr.getName(), stablehloAttr});
    }
    return DictionaryAttr::get(hloAttrs.getContext(), stablehloAttrs);
  }
  return hloAttr;
}

#undef RETURN_CONVERTED_ENUM_ATTR

// Convert array of enum attrs to an array of enum strings
//   [#mhlo<precision HiGHEST>] -> ["HIGHEST"]
//
// This is stable as long as enum names are not changed. This is needed to avoid
// a dependency on upstream printing / parsing. If an attribute name is changed,
// we can fork and  modify the code of `stringifyPrecision` as needed for
// compatibility.
Attribute encodePrecisionConfig(ArrayAttr precisionConfigAttr) {
  SmallVector<Attribute> stablehloAttrs;
  for (auto hloAttr : precisionConfigAttr) {
    auto precisionAttr = mlir::dyn_cast<mhlo::PrecisionAttr>(hloAttr);
    if (!precisionAttr) return {};
    StringRef precisionStr = mhlo::stringifyPrecision(precisionAttr.getValue());
    if (precisionStr.empty()) return {};
    stablehloAttrs.push_back(
        StringAttr::get(hloAttr.getContext(), precisionStr));
  }
  return ArrayAttr::get(precisionConfigAttr.getContext(), stablehloAttrs);
}

template <typename FailedToConvertTy>
LogicalResult notifyConversionFailure(ConversionPatternRewriter& rewriter,
                                      Operation* op,
                                      const std::string& errorMessage,
                                      FailedToConvertTy ty) {
  return rewriter.notifyMatchFailure(
      op, [=](Diagnostic& diag) { diag << errorMessage << ": " << ty; });
}

// Converts region to function.
// Returns failure if region has more than one block.
// Example:
//  %0:2 = "mhlo.all_reduce"(%arg0, %arg1) ({
//  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
//    %2 = mhlo.add %arg2, %arg3 : tensor<f32>
//    mhlo.return %2 : tensor<f32>
//  }) {...} : (tensor<8xf32>, tensor<f32>) -> (tensor<8xf32>, tensor<f32>)
// ==>
//  func.func @all_reduce0(%arg0: tensor<f32>, %arg1: tensor<f32>)
//       -> tensor<f32> {
//    %0 = mhlo.add %arg0, %arg1 : tensor<f32>
//    mhlo.return %0 : tensor<f32>
//  }
FailureOr<func::FuncOp> rewriteMhloRegionAsFunc(
    Operation* op, ConversionPatternRewriter& rewriter,
    const TypeConverter* typeConverter) {
  auto& region = op->getRegion(0);
  if (!region.hasOneBlock()) return failure();

  // Must be isolated from above
  SetVector<Value> values;
  getUsedValuesDefinedAbove(region, values);
  if (!values.empty()) {
    return notifyConversionFailure(
        rewriter, op,
        "MHLO feature serialization in StableHLO only supports regions that "
        "do not capture SSA values from above",
        op);
  }

  // Insert into the parent module
  OpBuilder::InsertionGuard g(rewriter);
  auto module = op->getParentOfType<ModuleOp>();
  SymbolTable symTable(module);

  // Convert so that function signature is correct
  if (failed(rewriter.convertRegionTypes(&region, *typeConverter,
                                         /*entryConversion=*/nullptr))) {
    return notifyConversionFailure(rewriter, op,
                                   "failed to convert region types", op);
  }

  // Create function with args that match block inputs / return types
  rewriter.setInsertionPointToEnd(&module.getBodyRegion().front());
  auto& block = region.getBlocks().front();
  auto type = rewriter.getFunctionType(
      block.getArgumentTypes(), block.getTerminator()->getOperandTypes());
  auto funcOp = func::FuncOp::create(rewriter, region.getLoc(),
                                     op->getName().stripDialect(), type);
  // The function only backs the custom_call's called_computations reference;
  // keep it private so symbol DCE can clean it up after recomposition.
  funcOp.setPrivate();
  symTable.insert(funcOp);

  // Move region into new function
  rewriter.inlineRegionBefore(region, funcOp.getFunctionBody(), funcOp.end());

  return funcOp;
}

// Convert HLO attributes to StableHLO, using `convertAttr`.
//
// Has additional logic for skipping experimental features or default values
// of features that aren't in StableHLO like custom_call_schedule.
//
LogicalResult convertAttributes(ConversionPatternRewriter& rewriter,
                                Operation* hloOp,
                                SmallVector<NamedAttribute>& stablehloAttrs) {
  auto customCallOp = dyn_cast<mhlo::CustomCallOp>(hloOp);
  for (NamedAttribute hloAttr : hloOp->getAttrs()) {
    // Skip custom_call_schedule if using the default attribute.
    // custom_call_schedule is private to XLA, but we still want to allow
    // #mhlo<custom_call_schedule NONE> (by ignoring it).
    if (customCallOp &&
        hloAttr.getName().getValue() == "custom_call_schedule" &&
        customCallOp.getCustomCallSchedule() ==
            mhlo::CustomCallSchedule::NONE) {
      continue;
    }

    // Handle DenseElements --> DenseArray for certain StableHLO ops.
    Attribute stablehloAttr = convertDenseArray(hloOp, hloAttr);

    // Generic handler for all other attributes.
    if (!stablehloAttr) stablehloAttr = convertAttr(hloAttr.getValue());

    if (!stablehloAttr) {
      return notifyConversionFailure(rewriter, hloOp, "failed to convert attr ",
                                     hloAttr.getValue());
    }
    hloAttr.setValue(stablehloAttr);
    stablehloAttrs.push_back(hloAttr);
  }
  return success();
}

// Experimental and public ops in MHLO that do not exist yet in StableHLO can be
// encoded as a StableHLO CustomCallOp to allow round-tripping between dialects.
//
// Example:
//   %0 = "mhlo.dot"(%arg0, %arg1) {
//     precision_config = [#mhlo<precision HIGHEST>] } ...
//  ==>
//  %0 = stablehlo.custom_call @mhlo.dot {
//    mhlo.attributes = {precision_config = ["HIGHEST"]}}
LogicalResult rewriteMhloOpAsCustomCall(Operation* hloOp,
                                        ConversionPatternRewriter& rewriter,
                                        const TypeConverter* typeConverter,
                                        ValueRange stablehloOperands,
                                        std::optional<int64_t> featureVersion) {
  if (hloOp->getNumRegions() > 1) {
    // Extensibility protocol for regions is only supported for single-region
    // ops. Support for multiple regions is not yet implemented.
    // In principle, it should be straightforward to implement by
    // converting regions into functions and calling them out in
    // "called_computations" in the order the regions appear in the op.
    // https://github.com/openxla/stablehlo/issues/593.
    return failure();
  }

  // Convert MHLO attributes to StableHLO equivalents.
  SmallVector<Type> stablehloTypes;
  if (failed(typeConverter->convertTypes(hloOp->getResultTypes(),
                                         stablehloTypes))) {
    return notifyConversionFailure(rewriter, hloOp,
                                   "failed to convert op types", hloOp);
  }

  // Convert MHLO attributes to StableHLO equivalents.
  SmallVector<NamedAttribute> stablehloConvertedAttrs;
  if (failed(convertAttributes(rewriter, hloOp, stablehloConvertedAttrs))) {
    return failure();
  }

  // Create functions from regions
  std::optional<func::FuncOp> stablehloConvertedRegion;
  if (hloOp->getNumRegions() == 1) {
    auto funcOp = rewriteMhloRegionAsFunc(hloOp, rewriter, typeConverter);
    if (failed(funcOp)) return failure();
    stablehloConvertedRegion = funcOp.value();
  }

  auto stablehloCallTargetName = hloOp->getName().getStringRef();
  SmallVector<NamedAttribute> stablehloAttrs;
  stablehloAttrs.push_back(rewriter.getNamedAttr(
      "call_target_name", rewriter.getStringAttr(stablehloCallTargetName)));
  stablehloAttrs.push_back(rewriter.getNamedAttr(
      "mhlo.attributes", rewriter.getDictionaryAttr(stablehloConvertedAttrs)));
  if (stablehloConvertedRegion) {
    stablehloAttrs.push_back(rewriter.getNamedAttr(
        "called_computations",
        rewriter.getArrayAttr(FlatSymbolRefAttr::get(
            rewriter.getContext(), stablehloConvertedRegion->getSymName()))));
  }
  if (featureVersion) {
    stablehloAttrs.push_back(rewriter.getNamedAttr(
        "mhlo.version", rewriter.getI64IntegerAttr(featureVersion.value())));
  }
  rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
      hloOp, stablehloTypes, stablehloOperands, stablehloAttrs);
  return success();
}

LogicalResult convertTypesAndAttributes(
    ConversionPatternRewriter& rewriter, const TypeConverter* typeConverter,
    Operation* hloOp, SmallVector<Type>& stablehloTypes,
    SmallVector<NamedAttribute>& stablehloAttrs) {
  if (failed(typeConverter->convertTypes(hloOp->getResultTypes(),
                                         stablehloTypes))) {
    return failure();
  }
  return convertAttributes(rewriter, hloOp, stablehloAttrs);
}

LogicalResult convertRegions(ConversionPatternRewriter& rewriter,
                             const TypeConverter* typeConverter,
                             Operation* hloOp, Operation* stablehloOp) {
  for (auto [hloRegion, stablehloRegion] :
       llvm::zip_equal(hloOp->getRegions(), stablehloOp->getRegions())) {
    rewriter.inlineRegionBefore(hloRegion, stablehloRegion,
                                stablehloRegion.end());
    if (failed(rewriter.convertRegionTypes(&stablehloRegion, *typeConverter,
                                           /*entryConversion=*/nullptr))) {
      return failure();
    }
  }
  return success();
}

// This converter is only used for MHLO ops that are not in StableHLO but may
// need to be encoded in StableHLO CustomCall.
template <typename HloOpTy>
class HloToStablehloCustomCallOpConverter
    : public OpConversionPattern<HloOpTy> {
 public:
  HloToStablehloCustomCallOpConverter(TypeConverter& converter,
                                      MLIRContext* context,
                                      bool allowExperimentalFeatures)
      : OpConversionPattern<HloOpTy>::OpConversionPattern(converter, context),
        allowExperimentalFeatures(allowExperimentalFeatures) {}

  LogicalResult matchAndRewrite(
      HloOpTy hloOp, typename HloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (hasPrivateFeaturesNotInStablehlo(hloOp)) {
      return notifyConversionFailure(
          rewriter, hloOp, "op has private features not in StableHLO", hloOp);
    }
    bool hasExperimentalFeatures = hasExperimentalFeaturesNotInStablehlo(hloOp);
    if (!allowExperimentalFeatures && hasExperimentalFeatures) {
      return notifyConversionFailure(
          rewriter, hloOp,
          "op has experimental features, but conversion not enabled", hloOp);
    }
    auto publicFeatureVersion = getPublicFeaturesNotInStablehlo(hloOp);
    if (publicFeatureVersion.has_value() || hasExperimentalFeatures) {
      return rewriteMhloOpAsCustomCall(
          hloOp, rewriter, this->getTypeConverter(), adaptor.getOperands(),
          publicFeatureVersion);
    }
    return notifyConversionFailure(
        rewriter, hloOp, "op not candidate for custom_call conversion", hloOp);
  }

  bool allowExperimentalFeatures;
};

template <typename StablehloOpTy>
class HloToStablehloOpConverter
    : public OpConversionPattern<StablehloToHloOp<StablehloOpTy>> {
  using HloOpTy = StablehloToHloOp<StablehloOpTy>;

 public:
  HloToStablehloOpConverter(TypeConverter& converter, MLIRContext* context,
                            bool allowExperimentalFeatures,
                            bool allowXlaFeatures)
      : OpConversionPattern<HloOpTy>::OpConversionPattern(converter, context),
        allowExperimentalFeatures(allowExperimentalFeatures),
        allowXlaFeatures(allowXlaFeatures) {}

  LogicalResult matchAndRewrite(
      HloOpTy hloOp, typename HloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // Most MHLO ops which end up here are fully supported by StableHLO.
    // However, some of these ops are supported only partially because they
    // have features that are not supported in StableHLO.
    // These MHLO features fall into two distinct categories:
    //   1) Features that are private to the XLA compiler, so they are not
    //      a good fit for StableHLO. Conversion of such features should fail.
    //   2) Features that might be a good fit for StableHLO but haven't yet
    //      been proposed or approved in StableHLO. Conversion of such features
    //      should succeed using custom_call extensibility protocol (see below).
    if (!allowXlaFeatures && hasPrivateFeaturesNotInStablehlo(hloOp))
      return failure();

    // These operands have already been converted to StableHLO by
    // the dialect conversion infrastructure.
    ValueRange stablehloOperands = adaptor.getOperands();

    // Extensibility protocol for MHLO ops with public MHLO features that
    // are not yet supported in StableHLO.
    //   1) The op is represented by stablehlo::CustomCallOp.
    //   2) The full name, e.g. "mhlo.all_to_all" is stored in the
    //      `call_target_name` attribute of the CustomCallOp.
    //   3) The operands become operands of the CustomCallOp.
    //   4) The attributes are wrapped in a DictionaryAttr, which is
    //      prettyprinted and then stored in the `backend_config` attribute
    //      of the CustomCallOp.
    //   5) The result types become result types of the CustomCallOp.
    //
    // This StableHLO representation does not come with any compatibility
    // guarantees. For example, when it is roundtripped back to MHLO, it may
    // turn out that the original MHLO op no longer exists or has different
    // attributes in the current version.
    bool hasExperimentalFeatures = hasExperimentalFeaturesNotInStablehlo(hloOp);
    if (!allowXlaFeatures && !allowExperimentalFeatures &&
        hasExperimentalFeatures) {
      return failure();
    }
    auto publicFeatureVersion = getPublicFeaturesNotInStablehlo(hloOp);
    if (!allowXlaFeatures &&
        (publicFeatureVersion.has_value() || hasExperimentalFeatures)) {
      return rewriteMhloOpAsCustomCall(hloOp, rewriter,
                                       this->getTypeConverter(),
                                       stablehloOperands, publicFeatureVersion);
    }

    // Convert MHLO types and attributes to StableHLO equivalents.
    // If a type or attribute is not defined in MHLO, then it is unchanged,
    // with the exception of RankedTensorType, TupleType, and ArrayAttr which
    // are converted recursively.
    // See `HloToStablehloTypeConverter` for more information on when this
    // conversion will succeed or fail.
    SmallVector<Type> stablehloTypes;
    SmallVector<NamedAttribute> stablehloAttrs;
    if (failed(convertTypesAndAttributes(rewriter, this->getTypeConverter(),
                                         hloOp, stablehloTypes,
                                         stablehloAttrs))) {
      return failure();
    }

    // Convert the MHLO operation to a StableHLO equivalent. This can almost be
    // done in a generic fashion, except for ops with a variadic number of
    // regions which means an additional argument for the generic builder.
    HloToStablehloOp<HloOpTy> stablehloOp;
    if constexpr (HloOpTy::template hasTrait<OpTrait::VariadicRegions>()) {
      stablehloOp = HloToStablehloOp<HloOpTy>::create(
          rewriter, hloOp.getLoc(), stablehloTypes, stablehloOperands,
          stablehloAttrs, hloOp.getNumRegions());
    } else {
      stablehloOp = HloToStablehloOp<HloOpTy>::create(
          rewriter, hloOp.getLoc(), stablehloTypes, stablehloOperands,
          stablehloAttrs);
    }

    // Finally, populate the regions while converting argument types
    // and nested operations.
    if constexpr (!HloOpTy::template hasTrait<OpTrait::ZeroRegions>()) {
      if (failed(convertRegions(rewriter, this->getTypeConverter(), hloOp,
                                stablehloOp))) {
        return failure();
      }
    }

    rewriter.replaceOp(hloOp, stablehloOp);
    return success();
  }

  bool allowExperimentalFeatures;
  bool allowXlaFeatures;
};

// Deprecated ops.
template <>
class HloToStablehloOpConverter<stablehlo::UnaryEinsumOp>
    : public OpConversionPattern<stablehlo::UnaryEinsumOp> {
 public:
  HloToStablehloOpConverter(TypeConverter& converter, MLIRContext* context,
                            bool /*allowExperimentalFeatures*/,
                            bool /*allowXlaFeatures*/)
      : OpConversionPattern<stablehlo::UnaryEinsumOp>::OpConversionPattern(
            converter, context) {}
  LogicalResult matchAndRewrite(stablehlo::UnaryEinsumOp stablehloOp,
                                typename stablehlo::UnaryEinsumOp::Adaptor,
                                ConversionPatternRewriter&) const final {
    return stablehloOp.emitError(
        "UnaryEinsumOp is deprecated and not supported in MHLO");
  }
};

template <typename... StablehloOpTypes>
void populateHloToStablehloPatterns(RewritePatternSet* patterns,
                                    TypeConverter* converter,
                                    MLIRContext* context,
                                    bool allowExperimentalFeatures,
                                    bool allowXlaFeatures) {
  patterns->add<HloToStablehloOpConverter<StablehloOpTypes>...>(
      *converter, context, allowExperimentalFeatures, allowXlaFeatures);
}

template <typename... HloOpTypes>
void populateHloToStablehloCustomCallPatterns(RewritePatternSet* patterns,
                                              TypeConverter* converter,
                                              MLIRContext* context,
                                              bool allowExperimentalFeatures) {
  patterns->add<HloToStablehloCustomCallOpConverter<HloOpTypes>...>(
      *converter, context, allowExperimentalFeatures);
}

}  // namespace

void populateHloToStablehloPatterns(RewritePatternSet* patterns,
                                    TypeConverter* converter,
                                    MLIRContext* context,
                                    bool allowExperimentalFeatures,
                                    bool allowXlaFeatures) {
  // Populate conversion patterns for all StableHLO ops.
  // Our guiding principle is to support all StableHLO functionality in MHLO.
  // The inverse is not necessarily true - some MHLO ops are missing from
  // StableHLO (either deliberately or haven't yet been proposed to StableHLO).
  // As a result, these MHLO ops will not be added to these patterns and
  // will fail the conversion.
  populateHloToStablehloPatterns<
#define GET_OP_LIST
#include "stablehlo/dialect/StablehloOps.cpp.inc"
      >(patterns, converter, context, allowExperimentalFeatures,
        allowXlaFeatures);

  populateHloToStablehloPatterns<mhlo::AddDependencyOp, mhlo::AsyncStartOp,
                                 mhlo::AsyncUpdateOp, mhlo::AsyncDoneOp>(
      patterns, converter, context, allowExperimentalFeatures,
      allowXlaFeatures);

  populateHloToStablehloCustomCallPatterns<
      mhlo::AcosOp, mhlo::AcoshOp, mhlo::AsinOp, mhlo::AsinhOp, mhlo::AtanhOp,
      mhlo::CoshOp, mhlo::SinhOp, mhlo::ErfOp, mhlo::TopKOp, mhlo::MulhiOp,
      mhlo::ScanOp>(patterns, converter, context, allowExperimentalFeatures);
}

}  // namespace stablehlo
}  // namespace mlir
