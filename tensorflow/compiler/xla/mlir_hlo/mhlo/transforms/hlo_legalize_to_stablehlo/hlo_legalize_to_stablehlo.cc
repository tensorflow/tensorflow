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

#include <string>

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
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo {
namespace {

#define RETURN_CONVERTED_ENUM_ATTR(Name)                      \
  auto hloValue = mhlo::stringify##Name(attr.getValue());     \
  auto stablehloValue = stablehlo::symbolize##Name(hloValue); \
  if (!stablehloValue.has_value()) return {};                 \
  return stablehlo::Name##Attr::get(attr.getContext(), stablehloValue.value())

Attribute convertAttr(Attribute hloAttr) {
  // Handle MHLO attributes.
  // The logic that handles attributes from other dialects (e.g. builtin
  // attributes) lives below.
  if (auto attr = hloAttr.dyn_cast<mhlo::ChannelHandleAttr>()) {
    return stablehlo::ChannelHandleAttr::get(attr.getContext(),
                                             attr.getHandle(), attr.getType());
  }
  if (auto attr = hloAttr.dyn_cast<mhlo::ComparisonDirectionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonDirection);
  }
  if (auto attr = hloAttr.dyn_cast<mhlo::ComparisonTypeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonType);
  }
  if (auto attr = hloAttr.dyn_cast<mhlo::ConvDimensionNumbersAttr>()) {
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
  if (auto attr = hloAttr.dyn_cast<mhlo::DotDimensionNumbersAttr>()) {
    return stablehlo::DotDimensionNumbersAttr::get(
        attr.getContext(), attr.getLhsBatchingDimensions(),
        attr.getRhsBatchingDimensions(), attr.getLhsContractingDimensions(),
        attr.getRhsContractingDimensions());
  }
  if (auto attr = hloAttr.dyn_cast<mhlo::FftTypeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(FftType);
  }
  if (auto attr = hloAttr.dyn_cast<mhlo::GatherDimensionNumbersAttr>()) {
    return stablehlo::GatherDimensionNumbersAttr::get(
        attr.getContext(), attr.getOffsetDims(), attr.getCollapsedSliceDims(),
        attr.getStartIndexMap(), attr.getIndexVectorDim());
  }
  if (auto attr = hloAttr.dyn_cast<mhlo::OutputOperandAliasAttr>()) {
    return stablehlo::OutputOperandAliasAttr::get(
        attr.getContext(), attr.getOutputTupleIndices(), attr.getOperandIndex(),
        attr.getOperandTupleIndices());
  }
  if (auto attr = hloAttr.dyn_cast<mhlo::PrecisionAttr>()) {
    // This precision value is used to experiment with int4 support.
    // Needs more experimental data before we decide whether or not to propose
    // it to StableHLO.
    if (attr.getValue() == mhlo::Precision::PACKED_NIBBLE) return {};
    RETURN_CONVERTED_ENUM_ATTR(Precision);
  }
  if (auto attr = hloAttr.dyn_cast<mhlo::RngAlgorithmAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(RngAlgorithm);
  }
  if (auto attr = hloAttr.dyn_cast<mhlo::RngDistributionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(RngDistribution);
  }
  if (auto attr = hloAttr.dyn_cast<mhlo::ScatterDimensionNumbersAttr>()) {
    return stablehlo::ScatterDimensionNumbersAttr::get(
        attr.getContext(), attr.getUpdateWindowDims(),
        attr.getInsertedWindowDims(), attr.getScatterDimsToOperandDims(),
        attr.getIndexVectorDim());
  }
  if (auto attr = hloAttr.dyn_cast<mhlo::TransposeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(Transpose);
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
  if (auto hloAttrs = hloAttr.dyn_cast<ArrayAttr>()) {
    SmallVector<Attribute> stablehloAttrs;
    for (auto hloAttr : hloAttrs) {
      auto stablehloAttr = convertAttr(hloAttr);
      if (!stablehloAttr) return {};
      stablehloAttrs.push_back(stablehloAttr);
    }
    return ArrayAttr::get(hloAttrs.getContext(), stablehloAttrs);
  }
  return hloAttr;
}

#undef RETURN_CONVERTED_ENUM_ATTR

template <typename HloOpTy>
class HloToStablehloOpConverter : public OpConversionPattern<HloOpTy> {
 public:
  using OpConversionPattern<HloOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      HloOpTy hloOp, typename HloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // Most MHLO ops which end up here are fully supported by StableHLO.
    // However, some of these ops are supported only partially because they
    // have features that either haven't been proposed to StableHLO yet
    // or aren't planned to be proposed to StableHLO.
    // The check below makes sure we only proceed for supported ops.
    if constexpr (std::is_same<HloOpTy, mhlo::ConvolutionOp>::value) {
      // StableHLO convolution doesn't support "unknown" dimensions.
      // This is an esoteric feature of MHLO convolutions, and it's different
      // from the notion of dynamic dimensions. For more context, here's the
      // commit which introduced it:
      // https://github.com/tensorflow/mlir-hlo/commit/4d6dc3163c1c9289d86455d9f4de5711465c50fb
      // This feature isn't supported in HLO and doesn't have documentation, so
      // we may end up removing it from MHLO as well.
      auto dimensionNumbers = debugString(hloOp.getDimensionNumbers());
      if (dimensionNumbers.find('?') != std::string::npos) return failure();
    }

    if constexpr (std::is_same<HloOpTy, mhlo::AllToAllOp>::value) {
      // StableHLO AllToAll doesn't support the tuple form yet.
      // Proposal: https://github.com/openxla/stablehlo/issues/574.
      if (hloOp.getNumOperands() != 1) return failure();
    }

    if constexpr (std::is_same<HloOpTy, mhlo::CustomCallOp>::value) {
      // StableHLO CustomCall doesn't support dictionary backend config.
      // Proposal: https://github.com/openxla/stablehlo/issues/637
      auto backendConfig = hloOp.getBackendConfig();
      if (backendConfig && !backendConfig->template isa<mlir::StringAttr>())
        return failure();
    }

    // Convert MHLO types to StableHLO equivalents.
    // If a type is not defined in MHLO, then it is unchanged,
    // with the exception of RankedTensorType and TupleType which are
    // converted recursively.
    // See `HloToStablehloTypeConverter` for more information on when this
    // conversion will succeed or fail.
    SmallVector<Type> stablehloTypes;
    if (failed(this->getTypeConverter()->convertTypes(hloOp->getResultTypes(),
                                                      stablehloTypes)))
      return failure();

    // These operands have already been converted to StableHLO by
    // the dialect conversion infrastructure.
    ValueRange stablehloOperands = adaptor.getOperands();

    // Convert MHLO attributes to StableHLO equivalents.
    // If an attribute is not defined in MHLO, then it is unchanged,
    // with the exception of ArrayAttr which is converted recursively.
    SmallVector<NamedAttribute> stablehloAttrs;
    for (NamedAttribute hloAttr : hloOp->getAttrs()) {
      if constexpr (std::is_same<HloOpTy, mhlo::CustomCallOp>::value) {
        if (hloAttr.getName() == "api_version") {
          // StableHLO CustomCall doesn't support API_VERSION_TYPED_FFI yet.
          // Proposal: https://github.com/openxla/stablehlo/issues/637.
          auto attr = hloAttr.getValue().cast<mhlo::CustomCallApiVersionAttr>();
          if (attr.getValue() ==
              mhlo::CustomCallApiVersion::API_VERSION_TYPED_FFI)
            return failure();
        }
      }
      auto stablehloAttr = convertAttr(hloAttr.getValue());
      if (!stablehloAttr) return failure();
      stablehloAttrs.push_back({hloAttr.getName(), stablehloAttr});
    }

    // Convert the MHLO operation to a StableHLO equivalent.
    // This can almost be done in a generic fashion, except for stablehlo.case
    // that uses a variadic number of regions which means an additional argument
    // for the generic builder.
    HloToStablehloOp<HloOpTy> stablehloOp;
    if constexpr (std::is_same<HloOpTy, mhlo::CaseOp>::value) {
      stablehloOp = rewriter.replaceOpWithNewOp<stablehlo::CaseOp>(
          hloOp, stablehloTypes, stablehloOperands, stablehloAttrs,
          hloOp.getBranches().size());
    } else {
      stablehloOp = rewriter.replaceOpWithNewOp<HloToStablehloOp<HloOpTy>>(
          hloOp, stablehloTypes, stablehloOperands, stablehloAttrs);
    }

    // Finally, populate the regions while converting argument types
    // and nested operations.
    for (auto [hloRegion, stablehloRegion] :
         llvm::zip(hloOp->getRegions(), stablehloOp->getRegions())) {
      rewriter.inlineRegionBefore(hloRegion, stablehloRegion,
                                  stablehloRegion.end());
    }
    return success();
  }
};

template <typename... StablehloOpTypes>
void populateHloToStablehloPatterns(RewritePatternSet* patterns,
                                    TypeConverter* converter,
                                    MLIRContext* context) {
  patterns
      ->add<HloToStablehloOpConverter<StablehloToHloOp<StablehloOpTypes>>...>(
          *converter, context);
}

}  // namespace

void populateHloToStablehloPatterns(RewritePatternSet* patterns,
                                    TypeConverter* converter,
                                    MLIRContext* context) {
  // Populate conversion patterns for all StableHLO ops.
  // Our guiding principle is to support all StableHLO functionality in MHLO.
  // The inverse is not necessarily true - some MHLO ops are missing from
  // StableHLO (either deliberately or haven't yet been proposed to StableHLO).
  // As a result, these MHLO ops will not be added to these patterns and
  // will fail the conversion.
  populateHloToStablehloPatterns<
#define GET_OP_LIST
#include "stablehlo/dialect/StablehloOps.cpp.inc"
      >(patterns, converter, context);
}

}  // namespace stablehlo
}  // namespace mlir
