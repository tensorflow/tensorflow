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
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/tf_to_uniform_attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/core/ir/importexport/convert_tensor.h"

namespace mlir {
namespace quant {
namespace {

constexpr StringRef kQuantizeFuncName = "quantize_i8";
constexpr StringRef kDequantizeFuncName = "dequantize_i8";
constexpr StringRef kAttrMapAttribute = "attr_map";
constexpr StringRef kQuantizedOpsAttribute = "tf_quant.quantized_ops";
constexpr StringRef kCompositeFuncPrefix = "composite_";
constexpr StringRef kQuantizedFuncPrefix = "quantized_";
constexpr StringRef kFloatOutputFuncPrefix = "_float_output_fn";

class QuantizeCompositeFunctionsPass
    : public mlir::PassWrapper<QuantizeCompositeFunctionsPass,
                               OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizeCompositeFunctionsPass)

  explicit QuantizeCompositeFunctionsPass() = default;

  explicit QuantizeCompositeFunctionsPass(
      QuantizationMethod quantization_method, OpSet target_opset,
      bool enable_per_channel_quantization) {
    quantization_method_ = quantization_method;
    target_opset_ = target_opset;
    enable_per_channel_quantization_ = enable_per_channel_quantization;
  }

  QuantizeCompositeFunctionsPass(const QuantizeCompositeFunctionsPass& other) {
    quantization_method_ = other.quantization_method_;
    target_opset_ = other.target_opset_;
    enable_per_channel_quantization_ = other.enable_per_channel_quantization_;
  }

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-quantize-composite-functions";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Quantize composite functions with QDQ input/outputs.";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect, quant::QuantizationDialect,
                    quantfork::QuantizationForkDialect>();
  }

 private:
  void runOnOperation() override;

  // These flags are only used for testing purpose.
  Option<QuantizationMethod> quantization_method_{
      *this, "quantization-method",
      llvm::cl::init(QuantizationMethod::kPostTrainingQuantization),
      llvm::cl::desc("Choose quantization method."),
      llvm::cl::values(
          clEnumValN(QuantizationMethod::kPostTrainingQuantization, "ptq",
                     "Post-training static-range quantization"),
          clEnumValN(QuantizationMethod::kDynamicRangeQuantization, "drq",
                     "Post-training dynamic-range quantizaiton"))};
  Option<OpSet> target_opset_{
      *this, "target-opset", llvm::cl::init(OpSet::TF),
      llvm::cl::desc("Choose target opset."),
      llvm::cl::values(
          clEnumValN(OpSet::TF, "TF",
                     "Uses TF ops that mimic quantization behavior"),
          clEnumValN(OpSet::XLA, "XLA", "Uses TF XLA ops"),
          clEnumValN(OpSet::UNIFORM_QUANTIZED, "UNIFORM_QUANTIZED",
                     "Uses TF Uniform Quantized ops"))};

  Option<bool> enable_per_channel_quantization_{
      *this, "enable-per-channel-quantization", llvm::cl::init(false),
      llvm::cl::desc("Whether enable per-channel quantized weights.")};
};

LogicalResult CreateUniformQuantizedTypeParams(UniformQuantizedType qtype,
                                               Location loc,
                                               PatternRewriter& rewriter,
                                               Value& scale,
                                               Value& zero_point) {
  TensorType scale_type = RankedTensorType::get({}, rewriter.getF32Type());
  TensorType zero_point_type = scale_type.clone(rewriter.getI32Type());
  scale = rewriter.create<TF::ConstOp>(
      loc, scale_type,
      DenseFPElementsAttr::get(scale_type,
                               {static_cast<float>(qtype.getScale())}));
  zero_point = rewriter.create<TF::ConstOp>(
      loc, zero_point_type,
      DenseIntElementsAttr::get(zero_point_type,
                                {static_cast<int32_t>(qtype.getZeroPoint())}));
  return success(scale && zero_point);
}

LogicalResult CreateUniformQuantizedPerAxisTypeParams(
    quant::UniformQuantizedPerAxisType qtype, Location loc,
    PatternRewriter& rewriter, Value& scale, Value& zero_point) {
  // Consuming op should already know about Quantized channel information,
  // so not passing it during conversion. This design might change if needed.
  ArrayRef<double> scales = qtype.getScales();
  ArrayRef<int64_t> zero_points = qtype.getZeroPoints();
  const int num_channels = scales.size();
  TensorType scale_type = RankedTensorType::get(
      {static_cast<int64_t>(num_channels)}, rewriter.getF32Type());
  TensorType zero_point_type = scale_type.clone(rewriter.getI32Type());

  llvm::SmallVector<float, 4> float_scales;
  llvm::SmallVector<int32_t, 4> int32_zero_points;
  float_scales.reserve(num_channels);
  int32_zero_points.reserve(num_channels);
  for (int i = 0; i < num_channels; ++i) {
    float_scales.push_back(scales[i]);
    int32_zero_points.push_back(zero_points[i]);
  }
  scale = rewriter.create<TF::ConstOp>(
      loc, scale_type, DenseFPElementsAttr::get(scale_type, float_scales));
  zero_point = rewriter.create<TF::ConstOp>(
      loc, zero_point_type,
      DenseIntElementsAttr::get(zero_point_type, int32_zero_points));
  return success(scale && zero_point);
}

LogicalResult CreateQuantizationParams(QuantizedType elem_type, Location loc,
                                       PatternRewriter& rewriter, Value& scale,
                                       Value& zero_point) {
  if (!elem_type) {
    return failure();
  }
  if (auto qtype = elem_type.dyn_cast<UniformQuantizedType>()) {
    return CreateUniformQuantizedTypeParams(qtype, loc, rewriter, scale,
                                            zero_point);
  } else if (auto qtype =
                 elem_type.dyn_cast<quant::UniformQuantizedPerAxisType>()) {
    return CreateUniformQuantizedPerAxisTypeParams(qtype, loc, rewriter, scale,
                                                   zero_point);
  }
  return failure();
}

// Replaces quant.qcast op to composite quantize_i8 function.
class ReplaceQuantizePattern
    : public mlir::OpRewritePattern<quantfork::QuantizeCastOp> {
 public:
  explicit ReplaceQuantizePattern(MLIRContext* context)
      : OpRewritePattern<quantfork::QuantizeCastOp>(context) {}

 private:
  LogicalResult matchAndRewrite(quantfork::QuantizeCastOp q_op,
                                PatternRewriter& rewriter) const override {
    auto output_type = q_op.getType().cast<TensorType>();
    auto elem_type = output_type.getElementType().dyn_cast<QuantizedType>();
    const Location loc = q_op->getLoc();
    Value scale, zero_point;

    if (failed(CreateQuantizationParams(elem_type, loc, rewriter, scale,
                                        zero_point))) {
      return failure();
    }

    SmallVector<Type> output_types = {
        output_type.clone(elem_type.getStorageType())};
    SmallVector<Value> args = {q_op.getArg(), scale, zero_point};
    FlatSymbolRefAttr func_name =
        FlatSymbolRefAttr::get(rewriter.getStringAttr(kQuantizeFuncName));

    auto quantize_call = rewriter.create<TF::PartitionedCallOp>(
        loc, output_types, args, func_name,
        /*config=*/"", /*config_proto=*/"", /*executor_type=*/"");
    auto scast_op = rewriter.create<quantfork::StorageCastOp>(
        loc, output_type, quantize_call->getResult(0));
    q_op->replaceAllUsesWith(scast_op);
    return success();
  }
};

// Replaces quant.dcast op to composite dequantize_i8 function.
class ReplaceDequantizePattern
    : public mlir::OpRewritePattern<quantfork::DequantizeCastOp> {
 public:
  explicit ReplaceDequantizePattern(MLIRContext* context)
      : OpRewritePattern<quantfork::DequantizeCastOp>(context) {}

 private:
  LogicalResult matchAndRewrite(quantfork::DequantizeCastOp dq_op,
                                PatternRewriter& rewriter) const override {
    auto input_type = dq_op.getArg().getType().cast<TensorType>();
    auto elem_type = input_type.getElementType().dyn_cast<QuantizedType>();
    const Location loc = dq_op->getLoc();

    Value scale, zero_point;
    if (failed(CreateQuantizationParams(elem_type, loc, rewriter, scale,
                                        zero_point))) {
      return failure();
    }

    TensorType output_type = input_type.clone(elem_type.getStorageType());
    auto scast_op = rewriter.create<quantfork::StorageCastOp>(loc, output_type,
                                                              dq_op.getArg());

    FlatSymbolRefAttr func_name =
        FlatSymbolRefAttr::get(rewriter.getStringAttr(kDequantizeFuncName));
    SmallVector<Value> args = {scast_op->getResult(0), scale, zero_point};
    auto dequantize_call = rewriter.create<TF::PartitionedCallOp>(
        loc, dq_op.getResult().getType(), args, func_name,
        /*config=*/"", /*config_proto=*/"", /*executor_type=*/"");
    dq_op->replaceAllUsesWith(dequantize_call);
    return success();
  }
};

// Checks if input weights are quantized only. For now, weight index is only at
// the first index(rhs). Later this can be replaced to use a map that has weight
// index information for each op.
bool IsQuantizedCallforDynamicRange(TF::PartitionedCallOp call_op) {
  bool has_quantized_types_for_weights = false;
  for (int32_t cur_idx = 0; cur_idx < call_op.getArgs().size(); cur_idx++) {
    // Check if the only the weight index has QuantizeCastOp.
    auto cur_op = dyn_cast_or_null<quantfork::QuantizeCastOp>(
        call_op.getArgs()[cur_idx].getDefiningOp());
    if ((!cur_op && cur_idx == 1) || (cur_op && cur_idx != 1)) {
      return false;
    } else if (cur_op) {
      // Check if the QuantizeCastOp has element type of quantized type.
      if (!getElementTypeOrSelf(cur_op.getResult().getType())
               .isa<QuantizedType>()) {
        return false;
      }
      // Satisfies the input condition.
      has_quantized_types_for_weights = true;
    }
  }
  for (Value output : call_op.getOutput()) {
    if (auto type = output.getType().dyn_cast<TensorType>()) {
      if (type.getElementType().isa<QuantizedType>()) {
        return false;
      }
    }
  }
  return has_quantized_types_for_weights;
}

// Checks if all the inputs are quantized.
bool IsQuantizedCallforStaticRange(TF::PartitionedCallOp call_op) {
  bool has_quantized_types = false;
  for (Value input : call_op.getArgs()) {
    if (auto type = input.getType().dyn_cast<TensorType>()) {
      if (type.getElementType().isa<FloatType>()) {
        return false;
      }
      if (type.getElementType().isa<QuantizedType>()) {
        has_quantized_types = true;
      }
    }
  }
  for (Value output : call_op.getOutput()) {
    if (auto type = output.getType().dyn_cast<TensorType>()) {
      if (type.getElementType().isa<FloatType>()) {
        return false;
      }
      if (type.getElementType().isa<QuantizedType>()) {
        has_quantized_types = true;
      }
    }
  }
  return has_quantized_types;
}

// Converts the element type of the input tensor to the corresponding quantized
// version. Supports only int8 for now and returns nullptr if the input type is
// not supported.
ShapedType ConvertIntToQint(ShapedType input_type, MLIRContext* ctx) {
  int bit_width;
  bool is_signed;

  Type ele_type = input_type.getElementType();
  if (ele_type.isIntOrFloat()) {
    bit_width = ele_type.getIntOrFloatBitWidth();
    is_signed = ele_type.isSignlessIntOrFloat() || ele_type.isSignedInteger();
  } else if (QuantizedType qtype = ele_type.dyn_cast<QuantizedType>()) {
    bit_width = qtype.getStorageTypeIntegralWidth();
    is_signed = qtype.isSigned();
  } else {
    return input_type;
  }

  Type new_storage_type;
  if (is_signed) {
    switch (bit_width) {
      case 8:
        new_storage_type = mlir::TF::Qint8Type::get(ctx);
        break;
      default:
        return nullptr;  // Not yet supported
    }
  } else {
    return nullptr;  // Not yet supported
  }

  input_type = input_type.clone(new_storage_type);
  return input_type;
}

// Transfers the attributes of the corresponding ops from the float function to
// the quantized function using the attr_map attribute. In the quantized
// function, this map (map1) is in {attr_name_1: attr_identifier} format; and in
// the float function, this map (map2) is in {attr_identifier: attr_name_2}
// format. Where, the attribute identifiers should match between two maps,
// attr_name_1 is the name of the of the attribute needs to be set in the
// quantized function, attr_name_2 is the name of the attribute corresponding to
// the attribute identifier in the float function.
LogicalResult TransferTFAttributesToTFUniformAttributes(
    PatternRewriter& rewriter, func::FuncOp float_func,
    func::FuncOp quantized_func, QuantizationMethod quantization_method,
    bool enable_per_channel_quantization) {
  // A map to find an attribute from its identifier.
  llvm::StringMap<Attribute> identifier_to_attr;

  for (Operation& inner_op : float_func.getBody().front().getOperations()) {
    if (!inner_op.hasAttr(kAttrMapAttribute)) continue;
    // Insert quantization related attribute if they exists. Quantization
    // attributes are generated in the prepare pass so the attr_map doesn't
    // contain the attribute names.
    // TransferQuantizationAttributes(rewriter, inner_op, attrs);
    std::string attr_map_str =
        inner_op.getAttrOfType<StringAttr>(kAttrMapAttribute).str();
    for (absl::string_view element_str : absl::StrSplit(attr_map_str, ',')) {
      std::vector<absl::string_view> key_and_value_pair =
          absl::StrSplit(element_str, ':');
      if (key_and_value_pair.size() != 2) {
        float_func.emitError("The attr_map attribute is malformed");
        return failure();
      }
      identifier_to_attr.insert(
          {llvm::StringRef(std::string(key_and_value_pair[1])),
           inner_op.getAttr(
               llvm::StringRef(std::string(key_and_value_pair[1])))});
    }
  }

  // Set the attributes for ops with the attr_map attribute.
  for (Operation& inner_op : quantized_func.getBody().front().getOperations()) {
    if (auto uniform_op =
            llvm::dyn_cast<TF::UniformQuantizedConvolutionHybridOp>(inner_op)) {
      if (failed(FillAttributesForUniformQuantizedConvolutionOp(
              rewriter, uniform_op, identifier_to_attr, quantization_method,
              enable_per_channel_quantization)))
        return failure();
    } else if (auto uniform_op =
                   llvm::dyn_cast<TF::UniformQuantizedDotHybridOp>(inner_op)) {
      if (failed(FillAttributesForUniformQuantizedDotOp(
              rewriter, uniform_op, identifier_to_attr, quantization_method,
              enable_per_channel_quantization)))
        return failure();
    }
  }
  return success();
}

// Transfers the attributes of the corresponding ops from the float function to
// the quantized function using the attr_map attribute. In the quantized
// function, this map (map1) is in {attr_name_1: attr_identifier} format; and in
// the float function, this map (map2) is in {attr_identifier: attr_name_2}
// format. Where, the attribute identifiers should match between two maps,
// attr_name_1 is the name of the of the attribute needs to be set in the
// quantized function, attr_name_2 is the name of the attribute corresponding to
// the attribute identifier in the float function.
LogicalResult TransferAttributes(func::FuncOp float_func,
                                 func::FuncOp quantized_func) {
  // A map to find an attribute from its identifier.
  llvm::StringMap<Attribute> identifier_to_attr;
  for (Operation& inner_op : float_func.getBody().front().getOperations()) {
    if (!inner_op.hasAttr(kAttrMapAttribute)) continue;
    std::string attr_map_str =
        inner_op.getAttrOfType<StringAttr>(kAttrMapAttribute).str();
    for (absl::string_view element_str : absl::StrSplit(attr_map_str, ',')) {
      std::vector<absl::string_view> key_and_value_pair =
          absl::StrSplit(element_str, ':');
      if (key_and_value_pair.size() != 2) {
        float_func.emitError("The attr_map attribute is malformed");
        return failure();
      }
      identifier_to_attr.insert(
          {llvm::StringRef(std::string(key_and_value_pair[0])),
           inner_op.getAttr(
               llvm::StringRef(std::string(key_and_value_pair[1])))});
    }
  }

  // Set the attributes for ops with the attr_map attribute.
  for (Operation& inner_op : quantized_func.getBody().front().getOperations()) {
    if (!inner_op.hasAttr(kAttrMapAttribute)) continue;

    std::string attr_map_str =
        inner_op.getAttrOfType<StringAttr>(kAttrMapAttribute).str();
    for (absl::string_view element_str : absl::StrSplit(attr_map_str, ',')) {
      std::vector<absl::string_view> key_and_value_pair =
          absl::StrSplit(element_str, ':');
      if (key_and_value_pair.size() != 2) {
        float_func.emitError("The attr_map attribute is malformed");
        return failure();
      }
      if (identifier_to_attr.count(
              llvm::StringRef(std::string(key_and_value_pair[1]))) == 0) {
        float_func.emitWarning(absl::StrCat("Using the default value for the '",
                                            key_and_value_pair[0],
                                            "' attribute"));
        continue;
      }
      inner_op.setAttr(llvm::StringRef(std::string(key_and_value_pair[0])),
                       identifier_to_attr[llvm::StringRef(
                           std::string(key_and_value_pair[1]))]);
    }
    inner_op.removeAttr(kAttrMapAttribute);
  }
  return success();
}

// Get the corresponding quantized function name from the given function name.
std::string GetQuantizedFunctionName(StringRef func_name) {
  if (func_name.startswith(kQuantizedFuncPrefix)) return func_name.str();
  if (!func_name.startswith(kCompositeFuncPrefix)) return "";

  return llvm::Twine(kQuantizedFuncPrefix)
      .concat(llvm::Twine(
          func_name.substr(kCompositeFuncPrefix.size()).rsplit("_fn").first))
      .concat("_fn")
      .str();
}

// Unwraps quantization parameters of PartitionedCall ops with quantized
// input/outputs that are created from QuantizePass.
class QuantizeFunctionPattern
    : public mlir::OpRewritePattern<TF::PartitionedCallOp> {
 public:
  explicit QuantizeFunctionPattern(MLIRContext* context,
                                   QuantizationMethod quantization_method,
                                   OpSet target_opset,
                                   bool enable_per_channel_quantization)
      : OpRewritePattern<TF::PartitionedCallOp>(context),
        quantization_method_(quantization_method),
        target_opset_(target_opset),
        enable_per_channel_quantization_(enable_per_channel_quantization) {}

 private:
  QuantizationMethod quantization_method_ =
      QuantizationMethod::kPostTrainingQuantization;
  OpSet target_opset_ = OpSet::TF;
  bool enable_per_channel_quantization_;

  LogicalResult matchAndRewrite(TF::PartitionedCallOp call_op,
                                PatternRewriter& rewriter) const override {
    const auto f_attr = call_op.getFAttr().dyn_cast<FlatSymbolRefAttr>();
    // removeAttr will return nullptr if no attribute was removed.
    if (!call_op->removeAttr(kQuantTraitAttrName) || !f_attr) {
      return failure();
    }

    // Determines if all required float input/outputs are now quantized.
    bool has_quantized_types = false;
    if (quantization_method_ == QuantizationMethod::kDynamicRangeQuantization) {
      has_quantized_types = IsQuantizedCallforDynamicRange(call_op);
      if (f_attr.getValue().startswith(kCompositeFuncPrefix) &&
          !has_quantized_types) {
        call_op->emitError(
            "Only quantizable ops need to be in composite function for dynamic"
            "-range PTQ case.");
        return failure();
      }
    } else {
      has_quantized_types = IsQuantizedCallforStaticRange(call_op);
    }

    if (!f_attr.getValue().startswith(kCompositeFuncPrefix) ||
        !has_quantized_types) {
      return failure();
    }

    SmallVector<Value, 4> args;
    SmallVector<Value, 4> qparam_args;
    for (Value arg : call_op.getArgs()) {
      if (const auto arg_type = arg.getType().dyn_cast<TensorType>()) {
        QuantizedType qtype =
            arg_type.getElementType().dyn_cast<QuantizedType>();
        if (!qtype) continue;
        if (!qtype.isa<UniformQuantizedType,
                       quant::UniformQuantizedPerAxisType>()) {
          return failure();
        }
        Value scale, zero_point;
        if (failed(CreateQuantizationParams(qtype, arg.getLoc(), rewriter,
                                            scale, zero_point))) {
          // As the quantized types are already checked, this is unexpected.
          call_op->emitError(
              "Failed to create quantization parameter for an argument.");
          return failure();
        }
        qparam_args.push_back(scale);
        qparam_args.push_back(zero_point);
      }
    }

    for (Value result : call_op->getResults()) {
      if (auto result_type = result.getType().dyn_cast<TensorType>()) {
        QuantizedType qtype =
            result_type.getElementType().dyn_cast<QuantizedType>();
        if (!qtype) continue;
        if (!qtype.isa<UniformQuantizedType,
                       quant::UniformQuantizedPerAxisType>()) {
          return failure();
        }
        Value scale, zero_point;
        if (failed(CreateQuantizationParams(qtype, result.getLoc(), rewriter,
                                            scale, zero_point))) {
          // As the quantized types are already checked, this is unexpected.
          call_op->emitError(
              "Failed to create quantization parameter for a result.");
          return failure();
        }
        qparam_args.push_back(scale);
        qparam_args.push_back(zero_point);
      }
    }

    rewriter.setInsertionPoint(call_op);

    for (Value arg : call_op.getArgs()) {
      TensorType arg_type = arg.getType().dyn_cast<TensorType>();
      if (!arg_type) {
        args.push_back(arg);
        continue;
      }
      QuantizedType qtype = arg_type.getElementType().dyn_cast<QuantizedType>();
      if (!qtype) {
        args.push_back(arg);
        continue;
      }

      quantfork::StorageCastOp scast_op;
      if (target_opset_ == OpSet::UNIFORM_QUANTIZED) {
        ShapedType new_arg_type = ConvertIntToQint(arg_type.cast<ShapedType>(),
                                                   rewriter.getContext());
        if (!new_arg_type) {
          call_op->emitError(
              "Failed to convert the type to the corresponding qtype.");
          return failure();
        }
        scast_op = rewriter.create<quantfork::StorageCastOp>(
            arg.getLoc(), new_arg_type.cast<TensorType>(), arg);
      } else {
        scast_op = rewriter.create<quantfork::StorageCastOp>(
            arg.getLoc(), arg_type.clone(qtype.getStorageType()), arg);
      }
      args.push_back(scast_op.getResult());
    }
    args.insert(args.end(), qparam_args.begin(), qparam_args.end());
    // For XLA opset, try to merge quantized functions with following Dequantize
    // for optimization.
    if (target_opset_ == OpSet::XLA) {
      if (failed(mergeDequantizeOpFollowingQuantizedFunction(call_op, args,
                                                             rewriter))) {
        return failure();
      }
    }
    if (call_op->use_empty()) return success();

    DenseMap<Value, quantfork::StorageCastOp> replace_map;
    rewriter.setInsertionPointAfter(call_op);

    SmallVector<Type, 4> result_types;
    for (Value result : call_op->getResults()) {
      TensorType result_type = result.getType().dyn_cast<TensorType>();
      if (!result_type) {
        result_types.push_back(result.getType());
        continue;
      }
      QuantizedType qtype =
          result_type.getElementType().dyn_cast<QuantizedType>();
      if (!qtype) {
        result_types.push_back(result_type);
        continue;
      }
      auto scast_op = rewriter.create<quantfork::StorageCastOp>(
          call_op.getLoc(), result_type, result);
      replace_map.insert(std::make_pair(result, scast_op));

      result_types.push_back(result_type.clone(qtype.getStorageType()));
    }

    for (auto replace_pair : replace_map) {
      Value result = replace_pair.first;
      quantfork::StorageCastOp scast_op = replace_pair.second;
      result.replaceAllUsesExcept(scast_op, scast_op);
    }

    // Make a copy of the quantized function.
    auto module = call_op->getParentOfType<ModuleOp>();
    SymbolTable symbol_table(module);

    mlir::func::FuncOp float_func =
        dyn_cast<func::FuncOp>(symbol_table.lookup(f_attr.getValue()));
    rewriter.setInsertionPointAfter(float_func);

    const std::string quantized_function_name =
        GetQuantizedFunctionName(f_attr.getValue());
    const mlir::func::FuncOp quantized_func =
        dyn_cast<func::FuncOp>(symbol_table.lookup(quantized_function_name));
    mlir::func::FuncOp new_quantized_func =
        dyn_cast<func::FuncOp>(quantized_func->clone());
    if (new_quantized_func == nullptr) {
      return failure();
    }

    new_quantized_func.setType(
        FunctionType::get(getContext(), TypeRange{ValueRange{args}},
                          new_quantized_func.getResultTypes()));
    for (auto [partitioned_call_arg, new_quantized_func_arg] :
         llvm::zip_first(args, new_quantized_func.getArguments())) {
      new_quantized_func_arg.setType(partitioned_call_arg.getType());
    }

    // Set the attributes for ops with the attr_map attribute.
    if (target_opset_ == OpSet::UNIFORM_QUANTIZED) {
      if (failed(TransferTFAttributesToTFUniformAttributes(
              rewriter, float_func, new_quantized_func, quantization_method_,
              enable_per_channel_quantization_))) {
        return failure();
      }
    } else {
      if (failed(TransferAttributes(float_func, new_quantized_func))) {
        return failure();
      }
    }

    rewriter.setInsertionPoint(call_op);

    const StringAttr new_quant_func_name =
        symbol_table.insert(new_quantized_func);
    rewriter.replaceOpWithNewOp<TF::PartitionedCallOp>(
        call_op, result_types, args,
        FlatSymbolRefAttr::get(new_quant_func_name));

    return success();
  }

  // For composite functions followed by Dequantize ops, merges the Dequantize
  // op into the functions by creating quantized functions with float output.
  LogicalResult mergeDequantizeOpFollowingQuantizedFunction(
      TF::PartitionedCallOp call_op, const SmallVector<Value, 4>& args,
      PatternRewriter& rewriter) const {
    bool followed_by_dequantize = false;
    for (Operation* user : call_op->getUsers()) {
      if (llvm::isa<quantfork::DequantizeCastOp>(user)) {
        followed_by_dequantize = true;
        break;
      }
    }
    if (!followed_by_dequantize) return success();

    rewriter.setInsertionPointAfter(call_op);
    SmallVector<Type, 4> result_types;
    for (Value result : call_op->getResults()) {
      TensorType result_type = result.getType().dyn_cast<TensorType>();
      if (!result_type) {
        result_types.push_back(result.getType());
        continue;
      }
      QuantizedType qtype =
          result_type.getElementType().dyn_cast<QuantizedType>();
      if (!qtype) {
        result_types.push_back(result_type);
        continue;
      }

      result_types.push_back(result_type.clone(qtype.getExpressedType()));
    }

    // Make a copy of the quantized function.
    auto module = call_op->getParentOfType<ModuleOp>();
    SymbolTable symbol_table(module);

    const auto f_attr = call_op.getFAttr().dyn_cast<FlatSymbolRefAttr>();
    const auto float_func =
        dyn_cast<func::FuncOp>(symbol_table.lookup(f_attr.getValue()));
    rewriter.setInsertionPointAfter(float_func);

    // the length of the "_fn" suffix.
    const size_t fn_suffix_length = 3;
    std::string quantized_function_name =
        GetQuantizedFunctionName(f_attr.getValue());
    quantized_function_name.replace(
        quantized_function_name.size() - fn_suffix_length, fn_suffix_length,
        kFloatOutputFuncPrefix);
    const auto quantized_func =
        dyn_cast<func::FuncOp>(symbol_table.lookup(quantized_function_name));
    auto new_quantized_func = dyn_cast<func::FuncOp>(quantized_func->clone());
    if (new_quantized_func == nullptr) {
      return failure();
    }
    new_quantized_func.setType(
        FunctionType::get(getContext(), TypeRange{ValueRange{args}},
                          new_quantized_func.getResultTypes()));
    for (auto [partitioned_call_arg, new_quantized_func_arg] :
         llvm::zip_first(args, new_quantized_func.getArguments())) {
      new_quantized_func_arg.setType(partitioned_call_arg.getType());
    }

    // Set the attributes for ops with the attr_map attribute.
    if (target_opset_ == OpSet::UNIFORM_QUANTIZED) {
      if (failed(TransferTFAttributesToTFUniformAttributes(
              rewriter, float_func, new_quantized_func, quantization_method_,
              enable_per_channel_quantization_))) {
        return failure();
      }
    } else {
      if (failed(TransferAttributes(float_func, new_quantized_func))) {
        return failure();
      }
    }

    rewriter.setInsertionPoint(call_op);
    const StringAttr new_quant_func_name =
        symbol_table.insert(new_quantized_func);
    auto quantized_call_op = rewriter.create<TF::PartitionedCallOp>(
        call_op.getLoc(), result_types, args,
        FlatSymbolRefAttr::get(new_quant_func_name));

    for (int result_idx : llvm::seq<int>(0, call_op->getNumResults())) {
      Value result = call_op->getResult(result_idx);
      for (Operation* user : result.getUsers()) {
        if (auto dequant_op =
                llvm::dyn_cast<quantfork::DequantizeCastOp>(user)) {
          dequant_op.getResult().replaceAllUsesWith(
              quantized_call_op->getResult(result_idx));
        }
      }
    }

    return success();
  }
};

// Converts const -> quant.qcast pattern to quantized constant, after
// quantization parameters are safely included to each quantize composite
// functions.
class QuantizeConstPattern
    : public OpRewritePattern<quantfork::QuantizeCastOp> {
 public:
  // This pattern should have larger benefit than ReplaceQuantizePattern
  explicit QuantizeConstPattern(MLIRContext* context, OpSet target_opset)
      : OpRewritePattern<quantfork::QuantizeCastOp>(context, /*benefit=*/10),
        target_opset_(target_opset) {}

 private:
  QuantizationMethod quantization_method_ =
      QuantizationMethod::kPostTrainingQuantization;
  LogicalResult matchAndRewrite(quantfork::QuantizeCastOp q_op,
                                PatternRewriter& rewriter) const override {
    DenseFPElementsAttr attr;
    if (!matchPattern(q_op.getArg(), m_Constant(&attr))) {
      return failure();
    }

    ShapedType tensor_qtype = q_op.getResult().getType().cast<ShapedType>();
    Attribute tensor_proto_attr = Quantize(attr, tensor_qtype);
    if (!tensor_proto_attr) {
      return failure();
    }

    Type storage_type =
        tensor_qtype.getElementType().cast<QuantizedType>().getStorageType();
    ShapedType new_type = tensor_qtype.clone(storage_type);
    Location loc = q_op.getArg().getLoc();

    if (target_opset_ == OpSet::UNIFORM_QUANTIZED) {
      new_type = ConvertIntToQint(new_type, rewriter.getContext());
      tensor_qtype = ConvertIntToQint(tensor_qtype, rewriter.getContext());

      // TODO(b/225793355): It adds TensorProtoAttr to the constant as a
      // workaround.
      tensorflow::TensorProto tensor_proto;
      if (!mlir::tfg::ConvertToTensorProto(tensor_proto_attr, &tensor_proto)
               .ok()) {
        return failure();
      }

      tensor_proto.set_dtype(tensorflow::DT_QINT8);

      tensor_proto_attr = ElementsAttr(TF::TensorProtoAttr::get(
          new_type, tensorflow::mangling_util::MangleTensor(tensor_proto)));
    }
    auto const_op =
        rewriter.create<TF::ConstOp>(loc, new_type, tensor_proto_attr);
    // Add scast op to match quantize -> composition pattern. The added scast
    // is then removed by canonicalization. ([scast - scast] -> [])
    auto scast_op = rewriter.create<quantfork::StorageCastOp>(
        loc, tensor_qtype, const_op.getOutput());
    q_op->replaceAllUsesWith(scast_op);
    return success();
  }

  OpSet target_opset_;
};

// Get the representative name attribute value of a composite function.
FailureOr<StringRef> GetRepresentativeName(const SymbolTable& symbol_table,
                                           StringRef func_name) {
  std::string quantized_func_name = GetQuantizedFunctionName(func_name);

  func::FuncOp quantized_func =
      dyn_cast<func::FuncOp>(symbol_table.lookup(quantized_func_name));
  if (!quantized_func->hasAttrOfType<ArrayAttr>(kQuantizedOpsAttribute)) {
    quantized_func->emitError()
        << "Missing " << kQuantizedOpsAttribute
        << " attribute in the quantized composite function.";
    return failure();
  }

  auto quantized_ops =
      quantized_func->getAttrOfType<ArrayAttr>(kQuantizedOpsAttribute)
          .getValue();
  if (quantized_ops.empty()) {
    quantized_func->emitError() << "At least one op is expected in the "
                                << kQuantizedOpsAttribute << " attribute.";
    return failure();
  }

  // Use the first op as the representative name.
  return quantized_ops.front().cast<StringAttr>().getValue();
}

// Prints a summary about the quantization results.
void PrintQuantizationSummary(ModuleOp module) {
  llvm::StringMap<int32_t> quantized_func_count;
  llvm::StringMap<int32_t> composite_func_count;
  int32_t total_quantized_func_count = 0, float_output_func_count = 0,
          quantize_func_count = 0, dequantize_func_count = 0;

  SymbolTable symbol_table(module);
  module.walk([&](Operation* op) {
    if (auto call_op = llvm::dyn_cast_or_null<TF::PartitionedCallOp>(op)) {
      const auto f_attr = call_op.getFAttr().dyn_cast<FlatSymbolRefAttr>();
      if (!f_attr) return;
      StringRef func_name = f_attr.getValue();

      if (func_name.startswith(kQuantizedFuncPrefix)) {
        auto representative_name =
            GetRepresentativeName(symbol_table, func_name);
        if (failed(representative_name)) return;

        quantized_func_count[representative_name.value()]++;
        total_quantized_func_count++;
        if (func_name.contains(kFloatOutputFuncPrefix)) {
          float_output_func_count++;
        }
      } else if (func_name.startswith(kCompositeFuncPrefix)) {
        auto representative_name =
            GetRepresentativeName(symbol_table, func_name);
        if (failed(representative_name)) return;

        composite_func_count[representative_name.value()]++;
      } else if (func_name.startswith("quantize_i")) {
        quantize_func_count++;
      } else if (func_name.startswith("dequantize_i")) {
        dequantize_func_count++;
      }
    }
  });

  // Pad string to a certain size to format the table. Space is preferred to
  // Tab since it is easier to check the format in the mlir tests.
  auto pad_string = [](StringRef s, int32_t width) -> std::string {
    return llvm::Twine(s).concat(std::string(width - s.size(), ' ')).str();
  };

  // Generate a quantization report.
  size_t name_col_width = 5;
  absl::c_for_each(quantized_func_count.keys(),
                   [&name_col_width](const auto& key) {
                     name_col_width = std::max(name_col_width, key.size() + 1);
                   });

  std::vector<std::string> lines;
  lines.push_back("-------- Quantization Summary --------");
  lines.push_back("Number of quantized layers in the model");
  lines.push_back("--------------------------------");
  lines.push_back(
      absl::StrFormat("%s Count/Total", pad_string("Name", name_col_width)));
  lines.push_back("================================");
  for (StringRef op_name : quantized_func_count.keys()) {
    const int32_t quantized_count = quantized_func_count[op_name];
    const int32_t total_count = quantized_count + composite_func_count[op_name];
    lines.push_back(absl::StrFormat("%s %d/%d",
                                    pad_string(op_name, name_col_width),
                                    quantized_count, total_count));
  }
  lines.push_back("");
  lines.push_back(absl::StrFormat(
      "Number of quantized layers with quantized outputs: %d/%d",
      total_quantized_func_count - float_output_func_count,
      total_quantized_func_count));
  lines.push_back(absl::StrFormat("Number of quantize layers added: %d",
                                  quantize_func_count));
  lines.push_back(absl::StrFormat("Number of dequantize layers added: %d",
                                  dequantize_func_count));
  lines.push_back("");

  // Make the report visible by default.
  const std::string log_message =
      absl::StrJoin(lines.begin(), lines.end(), /*separator=*/"\n");
  llvm::errs() << log_message;

  // Create a FuncOp and attach the quantization summary to it. This is a
  // a hack to check the summary in mlir tests. This function will be
  // automatically removed since this pass is always followed by the Symbol
  // DCE pass.
  OpBuilder builder(module);
  builder.setInsertionPointToEnd(&module.getBodyRegion().back());
  const auto func_type = builder.getFunctionType(/*inputs=*/{}, /*results=*/{});
  auto summary_func = builder.create<func::FuncOp>(
      builder.getUnknownLoc(), /*sym_name=*/"summary", func_type);
  summary_func.setPrivate();
  summary_func->setAttr("quantization_summary",
                        builder.getStringAttr(log_message));
}

static PassRegistration<QuantizeCompositeFunctionsPass> pass;

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.inc"

void QuantizeCompositeFunctionsPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  ModuleOp module = getOperation();

  PassManager pm(ctx);
  // Intermediate output from QuantizePass will have PartitionedCall ops with
  // quantized input and output types, which are not allowed in TF dialect.
  // This can be removed when the composite call supports quantized types.
  pm.enableVerifier(false);

  QuantizationSpecs quant_specs;
  if (quantization_method_ == QuantizationMethod::kDynamicRangeQuantization) {
    quant_specs.weight_quantization = true;
    quant_specs.inference_type = tensorflow::DT_QINT8;
    quant_specs.disable_per_channel = !enable_per_channel_quantization_;
    pm.addPass(CreatePrepareQuantizeDRQPass(quant_specs, target_opset_));
  } else {
    pm.addNestedPass<func::FuncOp>(
        CreatePrepareQuantizePass(quantization_method_));
  }
  pm.addNestedPass<func::FuncOp>(
      CreateQuantizePass(quant_specs, target_opset_));

  pm.addNestedPass<func::FuncOp>(CreatePostQuantizePass());
  if (failed(pm.run(module))) {
    signalPassFailure();
  }

  RewritePatternSet patterns(ctx);
  patterns.add<QuantizeFunctionPattern>(ctx, quantization_method_,
                                        target_opset_,
                                        enable_per_channel_quantization_);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }

  // Constant quantization is a lossy transformation, so they are applied only
  // after all the other patterns have been aplied.
  RewritePatternSet patterns_2(ctx);
  populateWithGenerated(patterns_2);
  patterns_2.add<ReplaceQuantizePattern, ReplaceDequantizePattern>(ctx);
  patterns_2.add<QuantizeConstPattern>(ctx, target_opset_);
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns_2))) ||
      failed(verify(module))) {
    signalPassFailure();
  }

  PrintQuantizationSummary(module);
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateQuantizeCompositeFunctionsPass(
    QuantizationMethod quantization_method, OpSet target_opset,
    bool enable_per_channel_quantization) {
  return std::make_unique<QuantizeCompositeFunctionsPass>(
      quantization_method, target_opset, enable_per_channel_quantization);
}

}  // namespace quant
}  // namespace mlir
