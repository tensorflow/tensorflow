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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/AsmParser/AsmParser.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/transforms/lower_quant_annotations_helper.h"
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

// Quantization parameters carried in the backend_config of a `tfl.quantize` /
// `tfl.dequantize` custom call.
struct QuantParams {
  SmallVector<double, 4> scales;
  SmallVector<int64_t, 4> zero_points;
  int num_bits = 8;
  bool is_signed = true;
  bool narrow_range = false;
  int32_t quantized_dimension = 0;
};

QuantParams ParseQuantParams(DictionaryAttr attributes) {
  QuantParams params;

  if (auto scale_attr = attributes.getAs<mlir::FloatAttr>("scale")) {
    params.scales.push_back(scale_attr.getValueAsDouble());
  } else if (auto scale_array = attributes.getAs<mlir::ArrayAttr>("scale")) {
    for (auto elem : scale_array) {
      if (auto f_attr = mlir::dyn_cast<mlir::FloatAttr>(elem)) {
        params.scales.push_back(f_attr.getValueAsDouble());
      } else if (auto i_attr = mlir::dyn_cast<mlir::IntegerAttr>(elem)) {
        params.scales.push_back(static_cast<double>(i_attr.getInt()));
      }
    }
  } else if (auto scale_dense =
                 attributes.getAs<mlir::DenseFPElementsAttr>("scale")) {
    for (auto val : scale_dense.getValues<APFloat>()) {
      params.scales.push_back(val.convertToDouble());
    }
  } else if (auto scale_int = attributes.getAs<mlir::IntegerAttr>("scale")) {
    params.scales.push_back(static_cast<double>(scale_int.getInt()));
  }

  if (auto zp_attr = attributes.getAs<mlir::IntegerAttr>("zero_point")) {
    params.zero_points.push_back(zp_attr.getInt());
  } else if (auto zp_array = attributes.getAs<mlir::ArrayAttr>("zero_point")) {
    for (auto elem : zp_array) {
      if (auto i_attr = mlir::dyn_cast<mlir::IntegerAttr>(elem)) {
        params.zero_points.push_back(i_attr.getInt());
      }
    }
  } else if (auto zp_dense =
                 attributes.getAs<mlir::DenseIntElementsAttr>("zero_point")) {
    for (auto val : zp_dense.getValues<APInt>()) {
      params.zero_points.push_back(val.getSExtValue());
    }
  }

  if (auto num_bits_attr = attributes.getAs<mlir::IntegerAttr>("num_bits")) {
    params.num_bits = num_bits_attr.getInt();
  }
  if (auto is_signed_attr = attributes.getAs<mlir::BoolAttr>("is_signed")) {
    params.is_signed = is_signed_attr.getValue();
  }
  if (auto narrow_range_attr =
          attributes.getAs<mlir::BoolAttr>("narrow_range")) {
    params.narrow_range = narrow_range_attr.getValue();
  }
  if (auto dim_attr =
          attributes.getAs<mlir::IntegerAttr>("quantized_dimension")) {
    params.quantized_dimension = dim_attr.getInt();
  }

  if (params.scales.empty()) {
    params.scales.push_back(1.0);
  }
  if (params.zero_points.empty()) {
    params.zero_points.push_back(0);
  }
  return params;
}

// Builds the quantized tensor type of `shape` described by `params`. Returns
// null if `params.quantized_dimension` is out of range for `shape`.
RankedTensorType BuildQuantizedTensorType(Builder& builder,
                                          const QuantParams& params,
                                          ArrayRef<int64_t> shape,
                                          Type expressed_type, Location loc) {
  Type quantized_element_type;
  if (params.scales.size() == 1) {
    quantized_element_type = TFL::GetPerTensorQuantizedTensorType(
        builder, params.scales[0], params.zero_points[0], expressed_type,
        params.num_bits, loc, params.narrow_range, params.is_signed);
  } else {
    const int64_t rank = static_cast<int64_t>(shape.size());
    int64_t quant_dim = params.quantized_dimension;
    if (quant_dim < 0) {
      quant_dim += rank;
    }
    if (quant_dim < 0 || quant_dim >= rank) {
      return nullptr;
    }
    quantized_element_type = TFL::GetPerAxisQuantizedTensorType(
        builder, params.scales, params.zero_points,
        static_cast<int32_t>(quant_dim), expressed_type, params.num_bits, loc,
        params.narrow_range, params.is_signed);
  }
  if (!quantized_element_type) {
    return nullptr;
  }
  return RankedTensorType::get(shape, quantized_element_type);
}

bool ApproxEqual(double a, double b) {
  return std::abs(a - b) <= 1e-6 * std::max(std::abs(a), std::abs(b));
}

// Returns true if `qtype` has the scales and zero points in `params`.
bool MatchesScalesAndZeroPoints(quant::QuantizedType qtype,
                                const QuantParams& params) {
  if (auto uniform = mlir::dyn_cast<quant::UniformQuantizedType>(qtype)) {
    return params.scales.size() == 1 &&
           ApproxEqual(uniform.getScale(), params.scales[0]) &&
           uniform.getZeroPoint() == params.zero_points[0];
  }
  if (auto per_axis =
          mlir::dyn_cast<quant::UniformQuantizedPerAxisType>(qtype)) {
    ArrayRef<double> scales = per_axis.getScales();
    ArrayRef<int64_t> zero_points = per_axis.getZeroPoints();
    if (scales.size() != params.scales.size() ||
        zero_points.size() != params.zero_points.size()) {
      return false;
    }
    for (size_t i = 0; i < scales.size(); ++i) {
      if (!ApproxEqual(scales[i], params.scales[i]) ||
          zero_points[i] != params.zero_points[i]) {
        return false;
      }
    }
    return true;
  }
  return false;
}

// Rewrites a `tfl.quantize` custom call on `input` into `tfl.quantize`.
LogicalResult RewriteQuantizeCustomCall(mhlo::CustomCallOp op, Value input,
                                        PatternRewriter& rewriter) {
  DictionaryAttr attributes =
      ParseSerializedTFLOpAttributes(op.getBackendConfig(), op.getContext());
  if (!attributes) {
    return failure();
  }
  QuantParams params = ParseQuantParams(attributes);

  ShapedType input_shaped_type = mlir::cast<ShapedType>(input.getType());
  RankedTensorType output_type = BuildQuantizedTensorType(
      rewriter, params, input_shaped_type.getShape(),
      input_shaped_type.getElementType(), op.getLoc());
  if (!output_type) {
    return op.emitError("invalid quantization parameters for tfl.quantize");
  }
  auto quant_op = rewriter.create<TFL::QuantizeOp>(
      op.getLoc(), output_type, input, TypeAttr::get(output_type));
  rewriter.replaceOp(op, quant_op.getOutput());
  return success();
}

// Rewrites a `tfl.dequantize` custom call on `input` into `tfl.dequantize`.
// - An already-quantized input is dequantized with its own type, which must
//   match the scales and zero points in the backend_config.
// - An integer constant input becomes a `tfl.pseudo_qconst` with the quantized
//   type described by the backend_config.
// Any other input (e.g. a runtime integer tensor) can't be given a quantized
// type and is rejected.
LogicalResult RewriteDequantizeCustomCall(mhlo::CustomCallOp op, Value input,
                                          PatternRewriter& rewriter) {
  DictionaryAttr attributes =
      ParseSerializedTFLOpAttributes(op.getBackendConfig(), op.getContext());
  if (!attributes) {
    return failure();
  }

  if (auto call_op = input.getDefiningOp<mhlo::CustomCallOp>()) {
    if (call_op.getCallTargetName() == "tfl.quantize") {
      return failure();
    }
  }
  ShapedType input_shaped_type = mlir::cast<ShapedType>(input.getType());
  Type output_type = op.getResultTypes().front();
  Type expressed_type = mlir::cast<ShapedType>(output_type).getElementType();
  QuantParams params = ParseQuantParams(attributes);

  Value tfl_dequant_input = input;
  if (auto qtype = mlir::dyn_cast<quant::QuantizedType>(
          input_shaped_type.getElementType())) {
    if (!MatchesScalesAndZeroPoints(qtype, params)) {
      return op.emitError(
          "tfl.dequantize scale/zero_point don't match the quantized type of "
          "its input");
    }
  } else {
    ElementsAttr const_value;
    Location const_loc = op.getLoc();
    if (auto const_op = input.getDefiningOp<arith::ConstantOp>()) {
      const_value = mlir::dyn_cast<ElementsAttr>(const_op.getValue());
      const_loc = const_op.getLoc();
    } else if (auto const_op = input.getDefiningOp<mhlo::ConstantOp>()) {
      const_value = mlir::dyn_cast<ElementsAttr>(const_op.getValue());
      const_loc = const_op.getLoc();
    }
    if (!const_value ||
        !mlir::isa<mlir::IntegerType>(const_value.getElementType())) {
      return rewriter.notifyMatchFailure(
          op, "tfl.dequantize input must be quantized or an integer constant");
    }
    RankedTensorType qtensor_type =
        BuildQuantizedTensorType(rewriter, params, input_shaped_type.getShape(),
                                 expressed_type, op.getLoc());
    if (!qtensor_type) {
      return op.emitError("invalid quantization parameters for tfl.dequantize");
    }
    auto qconst_op = rewriter.create<TFL::QConstOp>(
        const_loc, qtensor_type, TypeAttr::get(qtensor_type), const_value);
    tfl_dequant_input = qconst_op.getResult();
  }

  auto dequant_op = rewriter.create<TFL::DequantizeOp>(
      op.getLoc(), output_type, tfl_dequant_input);
  rewriter.replaceOp(op, dequant_op.getOutput());
  return success();
}

LogicalResult ConvertCustomCallOp::matchAndRewrite(
    mhlo::CustomCallOp mhlo_custom_call, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto call_target_name = mhlo_custom_call.getCallTargetName();

  if (call_target_name == "tfl.quantize") {
    return RewriteQuantizeCustomCall(
        mhlo_custom_call, adaptor.getOperands().front(), rewriter);
  }
  if (call_target_name == "tfl.dequantize") {
    return RewriteDequantizeCustomCall(
        mhlo_custom_call, adaptor.getOperands().front(), rewriter);
  }

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
  auto tfl_custom = TFL::CustomOp::create(rewriter, mhlo_custom_call.getLoc(),
                                          mhlo_custom_call.getResultTypes(),
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

class RewriteQuantizeCustomCallOp
    : public OpRewritePattern<mhlo::CustomCallOp> {
 public:
  using OpRewritePattern<mhlo::CustomCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getCallTargetName() != "tfl.quantize") {
      return failure();
    }
    return RewriteQuantizeCustomCall(op, op.getOperand(0), rewriter);
  }
};

class RewriteDequantizeCustomCallOp
    : public OpRewritePattern<mhlo::CustomCallOp> {
 public:
  using OpRewritePattern<mhlo::CustomCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getCallTargetName() != "tfl.dequantize") {
      return failure();
    }
    return RewriteDequantizeCustomCall(op, op.getOperand(0), rewriter);
  }
};

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

// TFL ops whose quantized output uses the same scale and zero point as their
// first operand: they only move, select or replicate elements.
constexpr llvm::StringLiteral kScalePreservingTflOps[] = {
    "tfl.broadcast_to", "tfl.depth_to_space", "tfl.expand_dims",
    "tfl.gather",       "tfl.gather_nd",      "tfl.reshape",
    "tfl.reverse_v2",   "tfl.slice",          "tfl.space_to_depth",
    "tfl.squeeze",      "tfl.strided_slice",  "tfl.tile",
    "tfl.transpose",
};

// Gives a scale-preserving `tfl.*` custom call the quantized type of its first
// operand, so that e.g. `reshape(quantize(x))` stays quantized. Ops that
// compute new values (FC, BMM, add, ...) need their own output scale and are
// deliberately not handled here.
class PropagateQuantizedTypeToCustomCallOp
    : public OpRewritePattern<mhlo::CustomCallOp> {
 public:
  using OpRewritePattern<mhlo::CustomCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    if (!llvm::is_contained(kScalePreservingTflOps, op.getCallTargetName())) {
      return failure();
    }
    if (op.getNumOperands() == 0 || op.getNumResults() != 1) {
      return failure();
    }
    auto operand_type = mlir::dyn_cast<ShapedType>(op.getOperand(0).getType());
    if (!operand_type) {
      return failure();
    }
    auto qtype =
        mlir::dyn_cast<quant::QuantizedType>(operand_type.getElementType());
    if (!qtype) {
      return failure();
    }
    auto res_shaped_type =
        mlir::dyn_cast<ShapedType>(op.getResult(0).getType());
    if (!res_shaped_type) {
      return failure();
    }
    // Only retype results that already hold the storage type, e.g. i8 -> !quant
    // with i8 storage.
    auto res_int_type =
        mlir::dyn_cast<mlir::IntegerType>(res_shaped_type.getElementType());
    if (!res_int_type ||
        res_int_type.getWidth() != qtype.getStorageTypeIntegralWidth()) {
      return failure();
    }
    rewriter.modifyOpInPlace(op, [&] {
      op.getResult(0).setType(res_shaped_type.clone(qtype));
    });
    return success();
  }
};

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
  patterns.add<RewriteQuantizeCustomCallOp>(ctx);
  patterns.add<RewriteDequantizeCustomCallOp>(ctx);
  patterns.add<PropagateQuantizedTypeToCustomCallOp>(ctx);
}

}  // namespace odml
}  // namespace mlir
