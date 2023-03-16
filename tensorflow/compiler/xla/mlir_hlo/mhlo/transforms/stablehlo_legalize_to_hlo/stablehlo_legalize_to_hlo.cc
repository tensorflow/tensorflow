/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/map_stablehlo_to_hlo_op.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo {
namespace {

#define RETURN_CONVERTED_ENUM_ATTR(Name)                             \
  auto stablehloValue = stablehlo::stringify##Name(attr.getValue()); \
  auto hloValue = mhlo::symbolize##Name(stablehloValue);             \
  if (!hloValue.has_value()) return {};                              \
  return mhlo::Name##Attr::get(attr.getContext(), hloValue.value())

Attribute convertAttr(Attribute stablehloAttr) {
  // Handle StableHLO attributes.
  // The logic that handles attributes from other dialects (e.g. builtin
  // attributes) lives below.
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::ChannelHandleAttr>()) {
    return mhlo::ChannelHandleAttr::get(attr.getContext(), attr.getHandle(),
                                        attr.getType());
  }
  if (auto attr =
          stablehloAttr.dyn_cast<stablehlo::ComparisonDirectionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonDirection);
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::ComparisonTypeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonType);
  }
  if (auto attr =
          stablehloAttr.dyn_cast<stablehlo::ConvDimensionNumbersAttr>()) {
    return mhlo::ConvDimensionNumbersAttr::get(
        attr.getContext(), attr.getInputBatchDimension(),
        attr.getInputFeatureDimension(), attr.getInputSpatialDimensions(),
        attr.getKernelInputFeatureDimension(),
        attr.getKernelOutputFeatureDimension(),
        attr.getKernelSpatialDimensions(), attr.getOutputBatchDimension(),
        attr.getOutputFeatureDimension(), attr.getOutputSpatialDimensions());
  }
  if (auto attr =
          stablehloAttr.dyn_cast<stablehlo::CustomCallApiVersionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(CustomCallApiVersion);
  }
  if (auto attr =
          stablehloAttr.dyn_cast<stablehlo::DotDimensionNumbersAttr>()) {
    return mhlo::DotDimensionNumbersAttr::get(
        attr.getContext(), attr.getLhsBatchingDimensions(),
        attr.getRhsBatchingDimensions(), attr.getLhsContractingDimensions(),
        attr.getRhsContractingDimensions());
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::FftTypeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(FftType);
  }
  if (auto attr =
          stablehloAttr.dyn_cast<stablehlo::GatherDimensionNumbersAttr>()) {
    return mhlo::GatherDimensionNumbersAttr::get(
        attr.getContext(), attr.getOffsetDims(), attr.getCollapsedSliceDims(),
        attr.getStartIndexMap(), attr.getIndexVectorDim());
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::OutputOperandAliasAttr>()) {
    return mhlo::OutputOperandAliasAttr::get(
        attr.getContext(), attr.getOutputTupleIndices(), attr.getOperandIndex(),
        attr.getOperandTupleIndices());
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::PrecisionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(Precision);
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::RngAlgorithmAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(RngAlgorithm);
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::RngDistributionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(RngDistribution);
  }
  if (auto attr =
          stablehloAttr.dyn_cast<stablehlo::ScatterDimensionNumbersAttr>()) {
    return mhlo::ScatterDimensionNumbersAttr::get(
        attr.getContext(), attr.getUpdateWindowDims(),
        attr.getInsertedWindowDims(), attr.getScatterDimsToOperandDims(),
        attr.getIndexVectorDim());
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::TransposeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(Transpose);
  }
  if (stablehloAttr.getDialect().getNamespace() ==
      stablehlo::StablehloDialect::getDialectNamespace()) {
    // Our guiding principle is to support all StableHLO functionality in MHLO.
    // This check is here only for exceptional situations, e.g. when we added
    // a new StableHLO attribute and forgot to update the code above.
    return {};
  }

  // Handle non-StableHLO attributes.
  // If an attribute is not defined in StableHLO, then it is unchanged,
  // with the exception of ArrayAttr which is converted recursively.
  if (auto stablehloAttrs = stablehloAttr.dyn_cast<ArrayAttr>()) {
    SmallVector<Attribute> hloAttrs;
    for (auto stablehloAttr : stablehloAttrs) {
      auto hloAttr = convertAttr(stablehloAttr);
      if (!hloAttr) return {};
      hloAttrs.push_back(hloAttr);
    }
    return ArrayAttr::get(stablehloAttrs.getContext(), hloAttrs);
  }
  return stablehloAttr;
}

#undef RETURN_CONVERTED_ENUM_ATTR

// Convert array of enum strings to array of enum attrs
//   ["PACKED_NIBBLE"] --> [#mhlo<precision PACKED_NIBBLE>]
Attribute decodePrecisionConfig(Attribute stablehloAttr) {
  auto arrayAttr = stablehloAttr.dyn_cast<ArrayAttr>();
  if (!arrayAttr) return {};
  SmallVector<Attribute> hloAttrs;
  for (auto attr : arrayAttr) {
    auto precisionStr = attr.dyn_cast<StringAttr>();
    if (!precisionStr) return {};
    auto precisionOpt = mhlo::symbolizePrecision(precisionStr.getValue());
    if (!precisionOpt.has_value()) return {};
    hloAttrs.push_back(mhlo::PrecisionAttr::get(stablehloAttr.getContext(),
                                                precisionOpt.value()));
  }
  return ArrayAttr::get(stablehloAttr.getContext(), hloAttrs);
}

// Experimental and public ops in MHLO that do not exist yet in StableHLO can be
// encoded as a StableHLO CustomCallOp to allow round-tripping between dialects.
//
// Example:
//  %0 = stablehlo.custom_call @mhlo.dot {
//    mhlo.attributes = {precision_config = ["PACKED_NIBBLE"]}}
//  ==>
//   %0 = "mhlo.dot"(%arg0, %arg1) {
//     precision_config = [#mhlo<precision PACKED_NIBBLE>] } ...
LogicalResult rewriteCustomCallAsMhloOp(stablehlo::CustomCallOp stablehloOp,
                                        ConversionPatternRewriter& rewriter,
                                        SmallVector<Type>& hloTypes,
                                        ValueRange hloOperands) {
  // Only call_target_name, backend_config, and mhlo.attributes are compatible
  // with the extensibility protocol.
  auto isSupportedAttrName = [](NamedAttribute attr) {
    auto name = attr.getName();
    return name == "call_target_name" || name == "backend_config" ||
           name == "mhlo.attributes" || name == "mhlo.version";
  };
  if (!llvm::all_of(stablehloOp->getAttrs(), isSupportedAttrName) ||
      !stablehloOp.getBackendConfig().empty()) {
    return failure();
  }

  auto stablehloConvertedAttrs = stablehloOp->getAttr("mhlo.attributes")
                                     .dyn_cast_or_null<DictionaryAttr>();
  if (!stablehloConvertedAttrs) {
    return failure();
  }

  // Convert Attributes back to MHLO
  SmallVector<NamedAttribute> hloConvertedAttrs;
  for (NamedAttribute stablehloAttr : stablehloConvertedAttrs.getValue()) {
    Attribute hloAttr;
    if (stablehloAttr.getName() == "precision_config") {
      hloAttr = decodePrecisionConfig(stablehloAttr.getValue());
    } else {
      hloAttr = convertAttr(stablehloAttr.getValue());
    }
    if (!hloAttr) return failure();
    hloConvertedAttrs.push_back({stablehloAttr.getName(), hloAttr});
  }

  // Dynamically create the corresponding MHLO op using call_target_name
  // and converted attributes. (It is quite neat that we have an API for this!).
  OperationState hloOpState(stablehloOp.getLoc(),
                            stablehloOp.getCallTargetName());
  hloOpState.addOperands(hloOperands);
  hloOpState.addTypes(hloTypes);
  hloOpState.addAttributes(hloConvertedAttrs);
  Operation* hloOp = rewriter.create(hloOpState);
  rewriter.replaceOp(stablehloOp, hloOp->getResults());
  return success();
}

template <typename StablehloOpTy>
class StablehloToHloOpConverter : public OpConversionPattern<StablehloOpTy> {
 public:
  using OpConversionPattern<StablehloOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      StablehloOpTy stablehloOp, typename StablehloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // Convert StableHLO types to HLO equivalents.
    // If a type is not defined in StableHLO, then it is unchanged,
    // with the exception of RankedTensorType and TupleType which are
    // converted recursively.
    // See `StablehloToHloTypeConverter` for more information on when this
    // conversion will succeed or fail.
    SmallVector<Type> hloTypes;
    if (failed(this->getTypeConverter()->convertTypes(
            stablehloOp->getResultTypes(), hloTypes)))
      return failure();

    // These operands have already been converted to MHLO by
    // the dialect conversion infrastructure.
    ValueRange hloOperands = adaptor.getOperands();

    // Extensibility protocol for public MHLO features that are not yet
    // supported in StableHLO. See hlo_legalize_to_stablehlo.cc for details.
    if constexpr (std::is_same<StablehloOpTy, stablehlo::CustomCallOp>::value) {
      if (stablehloOp.getCallTargetName().starts_with("mhlo.")) {
        return rewriteCustomCallAsMhloOp(stablehloOp, rewriter, hloTypes,
                                         hloOperands);
      }
    }

    // Convert StableHLO attributes to MHLO equivalents.
    // If an attribute is not defined in StableHLO, then it is unchanged,
    // with the exception of ArrayAttr which is converted recursively.
    SmallVector<NamedAttribute> hloAttrs;
    for (NamedAttribute stablehloAttr : stablehloOp->getAttrs()) {
      auto hloAttr = convertAttr(stablehloAttr.getValue());
      if (!hloAttr) return failure();
      hloAttrs.push_back({stablehloAttr.getName(), hloAttr});
    }

    // Convert the MHLO operation to a StableHLO equivalent.
    // This can almost be done in a generic fashion, except for mhlo.case
    // that uses a variadic number of regions which means an additional argument
    // for the generic builder.
    StablehloToHloOp<StablehloOpTy> hloOp;
    if constexpr (std::is_same<StablehloOpTy, stablehlo::CaseOp>::value) {
      hloOp = rewriter.replaceOpWithNewOp<mhlo::CaseOp>(
          stablehloOp, hloTypes, hloOperands, hloAttrs,
          stablehloOp.getBranches().size());
    } else {
      hloOp = rewriter.replaceOpWithNewOp<StablehloToHloOp<StablehloOpTy>>(
          stablehloOp, hloTypes, hloOperands, hloAttrs);
    }

    // Finally, populate the regions while converting argument types
    // and nested operations.
    for (auto [stablehloRegion, hloRegion] :
         llvm::zip(stablehloOp->getRegions(), hloOp->getRegions())) {
      rewriter.inlineRegionBefore(stablehloRegion, hloRegion, hloRegion.end());
      if (failed(rewriter.convertRegionTypes(&hloRegion,
                                             *this->getTypeConverter(),
                                             /*entryConversion=*/nullptr)))
        return failure();
    }
    return success();
  }
};

template <typename... StablehloOpTypes>
void populateStablehloToHloPatterns(RewritePatternSet* patterns,
                                    TypeConverter* converter,
                                    MLIRContext* context) {
  patterns->add<StablehloToHloOpConverter<StablehloOpTypes>...>(*converter,
                                                                context);
}

}  // namespace

void populateStablehloToHloPatterns(RewritePatternSet* patterns,
                                    TypeConverter* converter,
                                    MLIRContext* context) {
  // Populate conversion patterns for all StableHLO ops.
  // Our guiding principle is to support all StableHLO functionality in MHLO.
  populateStablehloToHloPatterns<
#define GET_OP_LIST
#include "stablehlo/dialect/StablehloOps.cpp.inc"
      >(patterns, converter, context);
}

}  // namespace stablehlo
}  // namespace mlir
