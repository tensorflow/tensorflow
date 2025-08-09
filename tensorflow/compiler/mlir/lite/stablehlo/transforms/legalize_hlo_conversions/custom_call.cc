/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/custom_call.h"

#include <optional>

#include "mlir/AsmParser/AsmParser.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {
namespace {

class ConvertCustomCallOp : public OpConversionPattern<mhlo::CustomCallOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::CustomCallOp mhlo_custom_call, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};

// TFL op on StableHLO CustomCall carrier must serialize its attributes in
// the CustomCallOp's backend_config StringAttr, following MLIR
// DictionaryAttr serialization format. If no attributes are specified,
// the backend_config should be the serialized empty DictionaryAttr.
mlir::DictionaryAttr ParseSerializedTFLOpAttributes(
    std::optional<mlir::Attribute> backend_config, MLIRContext* ctx) {
  if (!backend_config) {
    return nullptr;
  }

  auto serialized_attributes =
      mlir::dyn_cast_or_null<mlir::StringAttr>(*backend_config);
  if (!serialized_attributes) {
    return nullptr;
  }

  auto dict_attribute = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
      parseAttribute(serialized_attributes.getValue(), ctx));
  return dict_attribute;
}

LogicalResult ConvertCustomCallOp::matchAndRewrite(
    mhlo::CustomCallOp mhlo_custom_call, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto call_target_name = mhlo_custom_call.getCallTargetName();
  if (call_target_name.starts_with("tfl.")) {
    auto bc = mhlo_custom_call.getBackendConfig();
    if (mlir::DictionaryAttr attributes =
            ParseSerializedTFLOpAttributes(bc, getContext())) {
      // Short-cut: TFL direct lowering on StableHLO CustomCall carrier.
      mlir::OperationState new_op(mhlo_custom_call.getLoc(), call_target_name,
                                  mhlo_custom_call.getOperands(),
                                  mhlo_custom_call.getResultTypes(),
                                  attributes.getValue());
      rewriter.replaceOp(mhlo_custom_call, rewriter.create(new_op));
      return success();
    }
  }

  if (!call_target_name.starts_with("custom_call.")) {
    return failure();
  }
  auto tfl_custom = rewriter.create<TFL::CustomOp>(
      mhlo_custom_call.getLoc(), mhlo_custom_call.getResultTypes(),
      mhlo_custom_call.getInputs());
  tfl_custom.setCustomCodeAttr(rewriter.getStringAttr(call_target_name));

  if (auto bc = mhlo_custom_call.getBackendConfig()) {
    if (auto stringattr = mlir::dyn_cast_or_null<mlir::StringAttr>(*bc)) {
      tfl_custom.setCustomOptionAttr(
          TFL::ConstBytesAttr::get(rewriter.getContext(), stringattr));
    }
  } else {
    tfl_custom.setCustomOptionAttr(
        TFL::ConstBytesAttr::get(rewriter.getContext(), ""));
  }

  rewriter.replaceOp(mhlo_custom_call, tfl_custom);
  return success();
}

// Removes the `mhlo.custom_call @shape_assertion` custom call which represents
// an assertion that the first operand (`assert_what`) evaluates to `true`.
// This is a temporary workaround for unblocking dynamic model conversion
// because starting from version 7, in presence of shape polymorphism JAX will
// emit stablehlo.custom_call @shape_assertion to verify at compile time that
// the code is used with compatible actual shapes.
// TFLite runtime kernels support shape checking and shape inference to some
// extent, it is okay to remove the shape assertion in most scenarios. However
// this is not always the case, JAX may trace the program differently based on
// the shape polymorphism specification, for example, if the program contains
// a conditional on "x.shape[0] % 2 == 0" that conditional would evaluate to
// True with x specified as (2*b, ...) and False otherwise. We can revisit
// this when need arises. See b/295316438 for details.
class RemoveCustomCallWithShapeAssertion
    : public OpRewritePattern<mhlo::CustomCallOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::CustomCallOp op,
                                PatternRewriter& rewriter) const final;
};

LogicalResult RemoveCustomCallWithShapeAssertion::matchAndRewrite(
    mhlo::CustomCallOp op, PatternRewriter& rewriter) const {
  if (op.getCallTargetName() != "shape_assertion") {
    return mlir::failure();
  }
  rewriter.eraseOp(op);
  return success();
}

std::optional<bool> IsCustomCallLegal(mhlo::CustomCallOp op) {
  auto call_target_name = op.getCallTargetName();
  if (call_target_name.starts_with("custom_call.")) {
    auto bc = op.getBackendConfig();
    if (!bc || mlir::isa<mlir::StringAttr>(*bc)) {
      return false;
    }
  }
  if (call_target_name.starts_with("tfl.")) {
    auto bc = op.getBackendConfig();
    if (!bc || mlir::isa<mlir::DictionaryAttr, mlir::StringAttr>(*bc)) {
      return false;
    }
  }

  return true;
}
}  // namespace

void PopulateCustomCallPatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                                ConversionTarget& target) {
  patterns.add<ConvertCustomCallOp>(ctx);
  target.addDynamicallyLegalOp<mhlo::CustomCallOp>(IsCustomCallLegal);
}

void PopulateCustomCallPreparePatterns(MLIRContext* ctx,
                                       RewritePatternSet& patterns) {
  patterns.add<RemoveCustomCallWithShapeAssertion>(ctx);
}

}  // namespace odml
}  // namespace mlir
