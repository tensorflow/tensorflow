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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/run_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/tf_to_uniform_attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/core/ir/importexport/convert_tensor.h"

namespace mlir {
namespace quant {
namespace {

using QuantMethod = tensorflow::quantization::QuantizationMethod::PresetMethod;
using ::mlir::quant::ir::DequantizeCastOp;
using ::mlir::quant::ir::QuantizeCastOp;
using ::mlir::quant::ir::StorageCastOp;
using ::tensorflow::quantization::OpSet;

constexpr absl::string_view kQuantizeCompositeFunctionsStepName =
    "_quantize_composite_functions";
constexpr StringRef kQuantizeFuncName = "quantize_i8";
constexpr StringRef kDequantizeFuncName = "dequantize_i8";
constexpr StringRef kAttrMapAttribute = "attr_map";
constexpr StringRef kQuantizedOpsAttribute = "tf_quant.quantized_ops";
constexpr StringRef kCompositeFuncPrefix = "composite_";
constexpr StringRef kQuantizedFuncPrefix = "quantized_";
constexpr StringRef kFloatOutputFuncSuffix = "_float_output_fn";
constexpr StringRef kHybridFuncSuffix = "_hybrid_fn";

class QuantizeCompositeFunctionsPass
    : public mlir::PassWrapper<QuantizeCompositeFunctionsPass,
                               OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizeCompositeFunctionsPass)

  explicit QuantizeCompositeFunctionsPass() = default;

  explicit QuantizeCompositeFunctionsPass(
      const QuantMethod quantization_method, const OpSet target_opset,
      const bool enable_per_channel_quantization,
      const int min_num_elements_for_weights,
      const bool enable_legacy_weight_only,
      std::optional<const std::string> mlir_dump_file_name)
      : enable_legacy_weight_only_(enable_legacy_weight_only),
        min_num_elements_for_weights_(min_num_elements_for_weights),
        mlir_dump_file_name_(std::move(mlir_dump_file_name)) {
    quantization_method_ = quantization_method;
    target_opset_ = target_opset;
    enable_per_channel_quantization_ = enable_per_channel_quantization;
  }

  QuantizeCompositeFunctionsPass(const QuantizeCompositeFunctionsPass& other) {
    quantization_method_ = other.quantization_method_;
    target_opset_ = other.target_opset_;
    enable_per_channel_quantization_ = other.enable_per_channel_quantization_;
    min_num_elements_for_weights_ = other.min_num_elements_for_weights_;
    enable_legacy_weight_only_ = other.enable_legacy_weight_only_;
    mlir_dump_file_name_ = other.mlir_dump_file_name_;
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
    registry.insert<TF::TensorFlowDialect, quant::QuantDialect,
                    mlir::quant::ir::TFQuantDialect>();
  }

 private:
  void runOnOperation() override;

  bool enable_legacy_weight_only_;
  int min_num_elements_for_weights_;
  std::optional<std::string> mlir_dump_file_name_;

  // These flags are only used for testing purpose.
  Option<QuantMethod> quantization_method_{
      *this, "quantization-method",
      llvm::cl::init(tensorflow::quantization::QuantizationMethod::
                         METHOD_STATIC_RANGE_INT8),
      llvm::cl::desc("Choose quantization method."),
      llvm::cl::values(
          clEnumValN(tensorflow::quantization::QuantizationMethod::
                         METHOD_STATIC_RANGE_INT8,
                     "ptq", "Post-training static-range quantization"),
          clEnumValN(tensorflow::quantization::QuantizationMethod::
                         METHOD_DYNAMIC_RANGE_INT8,
                     "drq", "Post-training dynamic-range quantizaiton"),
          clEnumValN(tensorflow::quantization::QuantizationMethod::
                         METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8,
                     "weight_only", "Post-training weight-only quantization"))};

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
  scale = TF::ConstOp::create(
      rewriter, loc, scale_type,
      DenseFPElementsAttr::get(scale_type,
                               {static_cast<float>(qtype.getScale())}));
  zero_point = TF::ConstOp::create(
      rewriter, loc, zero_point_type,
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
  scale =
      TF::ConstOp::create(rewriter, loc, scale_type,
                          DenseFPElementsAttr::get(scale_type, float_scales));
  zero_point = TF::ConstOp::create(
      rewriter, loc, zero_point_type,
      DenseIntElementsAttr::get(zero_point_type, int32_zero_points));
  return success(scale && zero_point);
}

LogicalResult CreateQuantizationParams(QuantizedType elem_type, Location loc,
                                       PatternRewriter& rewriter, Value& scale,
                                       Value& zero_point) {
  if (!elem_type) {
    return failure();
  }
  if (auto qtype = mlir::dyn_cast<UniformQuantizedType>(elem_type)) {
    return CreateUniformQuantizedTypeParams(qtype, loc, rewriter, scale,
                                            zero_point);
  } else if (auto qtype = mlir::dyn_cast<quant::UniformQuantizedPerAxisType>(
                 elem_type)) {
    return CreateUniformQuantizedPerAxisTypeParams(qtype, loc, rewriter, scale,
                                                   zero_point);
  }
  return failure();
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
  } else if (QuantizedType qtype = mlir::dyn_cast<QuantizedType>(ele_type)) {
    bit_width = qtype.getStorageTypeIntegralWidth();
    is_signed = qtype.isSigned();
  } else {
    return input_type;
  }

  Type new_storage_type;
  if (is_signed) {
    switch (bit_width) {
      case 8:
        new_storage_type = TF::Qint8Type::get(ctx);
        break;
      case 32:
        new_storage_type = TF::Qint32Type::get(ctx);
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

// Replaces quant.qcast op to composite quantize_i8 function.
class ReplaceQuantizePattern
    : public mlir::OpRewritePattern<mlir::quant::ir::QuantizeCastOp> {
 public:
  explicit ReplaceQuantizePattern(MLIRContext* context, OpSet target_opset)
      : OpRewritePattern<mlir::quant::ir::QuantizeCastOp>(context),
        target_opset_(target_opset) {}

 private:
  OpSet target_opset_ = OpSet::TF;

  LogicalResult matchAndRewrite(mlir::quant::ir::QuantizeCastOp q_op,
                                PatternRewriter& rewriter) const override {
    auto output_type = mlir::cast<TensorType>(q_op.getType());
    auto elem_type =
        mlir::dyn_cast<QuantizedType>(output_type.getElementType());
    const Location loc = q_op->getLoc();
    Value scale, zero_point;

    if (failed(CreateQuantizationParams(elem_type, loc, rewriter, scale,
                                        zero_point))) {
      return failure();
    }

    SmallVector<Type> output_types;

    if (target_opset_ == OpSet::UNIFORM_QUANTIZED) {
      ShapedType new_output_type = ConvertIntToQint(
          mlir::cast<ShapedType>(output_type), rewriter.getContext());
      if (!new_output_type) {
        q_op->emitError(
            "Failed to convert the type to the corresponding qtype.");
        return failure();
      }
      output_types = {new_output_type};
    } else {
      output_types = {output_type.clone(elem_type.getStorageType())};
    }

    SmallVector<Value> args = {q_op.getArg(), scale, zero_point};
    FlatSymbolRefAttr func_name =
        FlatSymbolRefAttr::get(rewriter.getStringAttr(kQuantizeFuncName));

    auto quantize_call = TF::PartitionedCallOp::create(
        rewriter, loc, output_types, args, /*args_attrs=*/nullptr,
        /*res_attrs=*/nullptr, func_name,
        /*config=*/"", /*config_proto=*/"", /*executor_type=*/"");
    auto scast_op = StorageCastOp::create(rewriter, loc, output_type,
                                          quantize_call->getResult(0));
    q_op->replaceAllUsesWith(scast_op);
    return success();
  }
};

// Replaces quant.dcast op to composite dequantize_i8 function.
class ReplaceDequantizePattern
    : public mlir::OpRewritePattern<mlir::quant::ir::DequantizeCastOp> {
 public:
  explicit ReplaceDequantizePattern(MLIRContext* context, OpSet target_opset)
      : OpRewritePattern<mlir::quant::ir::DequantizeCastOp>(context),
        target_opset_(target_opset) {}

 private:
  OpSet target_opset_ = OpSet::TF;

  LogicalResult matchAndRewrite(mlir::quant::ir::DequantizeCastOp dq_op,
                                PatternRewriter& rewriter) const override {
    auto input_type = mlir::cast<TensorType>(dq_op.getArg().getType());
    auto elem_type = mlir::dyn_cast<QuantizedType>(input_type.getElementType());
    const Location loc = dq_op->getLoc();

    Value scale, zero_point;
    if (failed(CreateQuantizationParams(elem_type, loc, rewriter, scale,
                                        zero_point))) {
      return failure();
    }

    TensorType output_type = input_type.clone(elem_type.getStorageType());
    if (target_opset_ == OpSet::UNIFORM_QUANTIZED) {
      ShapedType new_output_type = ConvertIntToQint(
          mlir::cast<ShapedType>(output_type), rewriter.getContext());
      if (!new_output_type) {
        dq_op->emitError(
            "Failed to convert the type to the corresponding qtype.");
        return failure();
      }
      output_type = mlir::cast<TensorType>(new_output_type);
    }

    auto scast_op =
        StorageCastOp::create(rewriter, loc, output_type, dq_op.getArg());

    FlatSymbolRefAttr func_name =
        FlatSymbolRefAttr::get(rewriter.getStringAttr(kDequantizeFuncName));
    SmallVector<Value> args = {scast_op->getResult(0), scale, zero_point};
    auto dequantize_call = TF::PartitionedCallOp::create(
        rewriter, loc, dq_op.getResult().getType(), args,
        /*args_attrs=*/nullptr,
        /*res_attrs=*/nullptr, func_name,
        /*config=*/"", /*config_proto=*/"", /*executor_type=*/"");
    dq_op->replaceAllUsesWith(dequantize_call);
    return success();
  }
};

// Checks if input weights are quantized only.
bool IsQuantizedCallforDynamicRange(TF::PartitionedCallOp call_op) {
  bool has_quantized_types_for_weights = false;
  std::unique_ptr<OpQuantSpec> spec = GetTFOpQuantSpec(call_op);

  for (int32_t cur_idx = 0; cur_idx < call_op.getArgs().size(); cur_idx++) {
    // Check if the only the weight index has QuantizeCastOp.
    auto cur_op = dyn_cast_or_null<mlir::quant::ir::QuantizeCastOp>(
        call_op.getArgs()[cur_idx].getDefiningOp());
    if (!cur_op && spec->quantizable_operands.contains(cur_idx)) {
      return false;
    } else if (cur_op) {
      // Check if the QuantizeCastOp has element type of quantized type.
      if (!mlir::isa<QuantizedType>(
              getElementTypeOrSelf(cur_op.getResult().getType()))) {
        return false;
      }
      // Satisfies the input condition.
      has_quantized_types_for_weights = true;
    }
  }
  for (Value output : call_op.getOutput()) {
    if (auto type = mlir::dyn_cast<TensorType>(output.getType())) {
      if (mlir::isa<QuantizedType>(type.getElementType())) {
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
    if (auto type = mlir::dyn_cast<TensorType>(input.getType())) {
      if (mlir::isa<QuantizedType>(type.getElementType())) {
        has_quantized_types = true;
      }
    }
  }
  for (Value output : call_op.getOutput()) {
    if (auto type = mlir::dyn_cast<TensorType>(output.getType())) {
      if (mlir::isa<QuantizedType>(type.getElementType())) {
        has_quantized_types = true;
      }
    }
  }
  return has_quantized_types;
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
    func::FuncOp quantized_func, QuantMethod quantization_method,
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
            llvm::dyn_cast<TF::UniformQuantizedConvolutionHybridOp>(inner_op);
        uniform_op != nullptr) {
      if (failed(FillAttributesForUniformQuantizedConvolutionOp(
              rewriter, uniform_op, identifier_to_attr, quantization_method,
              enable_per_channel_quantization)))
        return failure();
    } else if (auto uniform_op =
                   llvm::dyn_cast<TF::UniformQuantizedConvolutionOp>(inner_op);
               uniform_op != nullptr) {
      if (failed(FillAttributesForUniformQuantizedConvolutionOp(
              rewriter, uniform_op, identifier_to_attr, quantization_method,
              enable_per_channel_quantization)))
        return failure();
    } else if (auto uniform_op =
                   llvm::dyn_cast<TF::UniformQuantizedDotHybridOp>(inner_op);
               uniform_op != nullptr) {
      if (failed(FillAttributesForUniformQuantizedDotOp(
              rewriter, uniform_op, identifier_to_attr, quantization_method,
              enable_per_channel_quantization)))
        return failure();
    } else if (auto uniform_op =
                   llvm::dyn_cast<TF::UniformQuantizedAddOp>(inner_op);
               uniform_op != nullptr) {
      if (failed(FillAttributesForUniformQuantizedAddOp(
              rewriter, uniform_op, identifier_to_attr, quantization_method,
              enable_per_channel_quantization)))
        return failure();
    } else if (auto uniform_op =
                   llvm::dyn_cast<TF::UniformQuantizedClipByValueOp>(inner_op);
               uniform_op != nullptr) {
      if (failed(FillAttributesForUniformQuantizedClipByValueOp(
              rewriter, uniform_op, identifier_to_attr, quantization_method,
              enable_per_channel_quantization)))
        return failure();
    } else if (auto uniform_op =
                   llvm::dyn_cast<TF::UniformRequantizeOp>(inner_op);
               uniform_op != nullptr) {
      if (failed(FillAttributesForUniformRequantizeOp(
              rewriter, uniform_op, identifier_to_attr, quantization_method,
              enable_per_channel_quantization)))
        return failure();
    } else if (auto uniform_op =
                   llvm::dyn_cast<TF::UniformQuantizeOp>(inner_op);
               uniform_op != nullptr) {
      if (failed(FillAttributesForUniformQuantizeOp(
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

// Transfers the location of the main op in float function to ops with
// `attr_map` attributes in quantized function.
LogicalResult TransferLocation(func::FuncOp float_func,
                               func::FuncOp quantized_func) {
  Operation* main_op = nullptr;
  for (Operation& inner_op : float_func.getBody().front().getOperations()) {
    // Expect only one quantizable op in the composite function.
    if (IsOpWithQuantizableTrait(&inner_op)) {
      main_op = &inner_op;
      break;
    }
  }
  if (!main_op) {
    float_func.emitError() << "No quantizable ops found in the function.";
    return failure();
  }

  for (Operation& inner_op : quantized_func.getBody().front().getOperations()) {
    if (!inner_op.hasAttr(kAttrMapAttribute)) continue;
    inner_op.setLoc(main_op->getLoc());
  }
  return success();
}

// Get the corresponding quantized function name from the given function name.
std::string GetQuantizedFunctionName(StringRef func_name,
                                     const bool merged_with_dequantize,
                                     const bool is_hybrid) {
  if (func_name.starts_with(kQuantizedFuncPrefix)) return func_name.str();
  if (!func_name.starts_with(kCompositeFuncPrefix)) return "";

  auto base_function_name =
      llvm::Twine(kQuantizedFuncPrefix)
          .concat(llvm::Twine(func_name.substr(kCompositeFuncPrefix.size())
                                  .rsplit("_fn")
                                  .first));

  if (merged_with_dequantize) {
    return base_function_name.concat("_float_output_fn").str();
  }

  if (is_hybrid) {
    return base_function_name.concat("_hybrid_fn").str();
  }

  return base_function_name.concat("_fn").str();
}

bool ContainsFloatResultType(ArrayRef<Type> result_types) {
  for (auto current_type : result_types) {
    if (mlir::dyn_cast<TensorType>(current_type).getElementType().isF32())
      return true;
  }
  return false;
}

// Unwraps quantization parameters of PartitionedCall ops with quantized
// input/outputs that are created from QuantizePass.
class QuantizeFunctionPattern
    : public mlir::OpRewritePattern<TF::PartitionedCallOp> {
 public:
  explicit QuantizeFunctionPattern(MLIRContext* context,
                                   const QuantMethod quantization_method,
                                   const OpSet target_opset,
                                   const bool enable_per_channel_quantization)
      : OpRewritePattern<TF::PartitionedCallOp>(context),
        quantization_method_(quantization_method),
        target_opset_(target_opset),
        enable_per_channel_quantization_(enable_per_channel_quantization) {}

 private:
  QuantMethod quantization_method_ =
      tensorflow::quantization::QuantizationMethod::METHOD_STATIC_RANGE_INT8;
  OpSet target_opset_ = OpSet::TF;
  bool enable_per_channel_quantization_;

  LogicalResult matchAndRewrite(TF::PartitionedCallOp call_op,
                                PatternRewriter& rewriter) const override {
    const auto f_attr = mlir::dyn_cast<FlatSymbolRefAttr>(call_op.getFAttr());
    // removeAttr will return nullptr if no attribute was removed.
    if (!call_op->removeAttr(kQuantTraitAttrName) || !f_attr) {
      return failure();
    }
    if (!f_attr.getValue().starts_with(kCompositeFuncPrefix)) {
      return failure();
    }

    bool has_quantized_types = false;
    if (quantization_method_ == tensorflow::quantization::QuantizationMethod::
                                    METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8) {
      // Skipping input type check for weight-only quantization as it can be
      // dequantized beforehand for the legacy scheme.
      has_quantized_types = true;
    } else {
      // Determines if all required float input/outputs are now quantized.
      // Either one of the criteria needs to meet.
      has_quantized_types |= IsQuantizedCallforDynamicRange(call_op);
      has_quantized_types |= IsQuantizedCallforStaticRange(call_op);
    }

    if (!has_quantized_types) return failure();

    SmallVector<Value, 4> args;
    SmallVector<Value, 4> qparam_args;
    for (Value arg : call_op.getArgs()) {
      if (const auto arg_type = mlir::dyn_cast<TensorType>(arg.getType())) {
        QuantizedType qtype =
            mlir::dyn_cast<QuantizedType>(arg_type.getElementType());
        if (!qtype) continue;
        if (!mlir::isa<UniformQuantizedType,
                       quant::UniformQuantizedPerAxisType>(qtype)) {
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
      if (auto result_type = mlir::dyn_cast<TensorType>(result.getType())) {
        QuantizedType qtype =
            mlir::dyn_cast<QuantizedType>(result_type.getElementType());
        if (!qtype) continue;
        if (!mlir::isa<UniformQuantizedType,
                       quant::UniformQuantizedPerAxisType>(qtype)) {
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
      TensorType arg_type = mlir::dyn_cast<TensorType>(arg.getType());
      if (!arg_type) {
        args.push_back(arg);
        continue;
      }
      QuantizedType qtype =
          mlir::dyn_cast<QuantizedType>(arg_type.getElementType());
      if (!qtype) {
        args.push_back(arg);
        continue;
      }

      StorageCastOp scast_op;
      if (target_opset_ == OpSet::UNIFORM_QUANTIZED) {
        ShapedType new_arg_type = ConvertIntToQint(
            mlir::cast<ShapedType>(arg_type), rewriter.getContext());
        if (!new_arg_type) {
          call_op->emitError(
              "Failed to convert the type to the corresponding qtype.");
          return failure();
        }
        scast_op = StorageCastOp::create(
            rewriter, arg.getLoc(), mlir::cast<TensorType>(new_arg_type), arg);
      } else {
        scast_op =
            StorageCastOp::create(rewriter, arg.getLoc(),
                                  arg_type.clone(qtype.getStorageType()), arg);
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

    DenseMap<Value, StorageCastOp> replace_map;
    rewriter.setInsertionPointAfter(call_op);

    SmallVector<Type, 4> result_types;
    for (Value result : call_op->getResults()) {
      TensorType result_type = mlir::dyn_cast<TensorType>(result.getType());
      if (!result_type) {
        result_types.push_back(result.getType());
        continue;
      }
      QuantizedType qtype =
          mlir::dyn_cast<QuantizedType>(result_type.getElementType());
      if (!qtype) {
        result_types.push_back(result_type);
        continue;
      }
      if (target_opset_ == OpSet::UNIFORM_QUANTIZED) {
        ShapedType new_result_type = ConvertIntToQint(
            mlir::cast<ShapedType>(result_type), rewriter.getContext());
        result_types.push_back(new_result_type);
      } else {
        result_types.push_back(result_type.clone(qtype.getStorageType()));
      }
      auto scast_op = StorageCastOp::create(rewriter, call_op.getLoc(),
                                            result_type, result);
      replace_map.insert(std::make_pair(result, scast_op));
    }

    for (auto replace_pair : replace_map) {
      Value result = replace_pair.first;
      StorageCastOp scast_op = replace_pair.second;
      result.replaceAllUsesExcept(scast_op, scast_op);
    }

    // Make a copy of the quantized function.
    auto module = call_op->getParentOfType<ModuleOp>();
    SymbolTable symbol_table(module);

    mlir::func::FuncOp float_func =
        dyn_cast<func::FuncOp>(symbol_table.lookup(f_attr.getValue()));
    rewriter.setInsertionPointAfter(float_func);

    // Applies only for hybrid ops in SRQ.
    const bool is_hybrid =
        ContainsFloatResultType(result_types) &&
        (quantization_method_ == tensorflow::quantization::QuantizationMethod::
                                     METHOD_STATIC_RANGE_INT8);
    const std::string quantized_function_name = GetQuantizedFunctionName(
        f_attr.getValue(), /*merged_with_dequantize=*/false,
        /*is_hybrid=*/is_hybrid);

    const mlir::func::FuncOp quantized_func = dyn_cast_or_null<func::FuncOp>(
        symbol_table.lookup(quantized_function_name));
    if (quantized_func == nullptr) {
      call_op->emitError("Failed to find the quantized function: " +
                         quantized_function_name);
      return failure();
    }
    mlir::func::FuncOp new_quantized_func =
        dyn_cast<func::FuncOp>(quantized_func->clone());

    new_quantized_func.setType(
        FunctionType::get(getContext(), TypeRange{ValueRange{args}},
                          new_quantized_func.getResultTypes()));
    for (auto [partitioned_call_arg, new_quantized_func_arg] :
         llvm::zip_equal(args, new_quantized_func.getArguments())) {
      new_quantized_func_arg.setType(partitioned_call_arg.getType());
    }

    // Set the location for ops so the op name is preserved.
    if (failed(TransferLocation(float_func, new_quantized_func))) {
      return failure();
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
        call_op, result_types, args, call_op.getArgAttrsAttr(),
        call_op.getResAttrsAttr(), FlatSymbolRefAttr::get(new_quant_func_name));

    return success();
  }

  // For composite functions followed by Dequantize ops, merges the Dequantize
  // op into the functions by creating quantized functions with float output.
  LogicalResult mergeDequantizeOpFollowingQuantizedFunction(
      TF::PartitionedCallOp call_op, const SmallVector<Value, 4>& args,
      PatternRewriter& rewriter) const {
    bool followed_by_dequantize = false;
    for (Operation* user : call_op->getUsers()) {
      if (llvm::isa<DequantizeCastOp>(user)) {
        followed_by_dequantize = true;
        break;
      }
    }
    if (!followed_by_dequantize) return success();

    rewriter.setInsertionPointAfter(call_op);
    SmallVector<Type, 4> result_types;
    for (Value result : call_op->getResults()) {
      TensorType result_type = mlir::dyn_cast<TensorType>(result.getType());
      if (!result_type) {
        result_types.push_back(result.getType());
        continue;
      }
      QuantizedType qtype =
          mlir::dyn_cast<QuantizedType>(result_type.getElementType());
      if (!qtype) {
        result_types.push_back(result_type);
        continue;
      }

      result_types.push_back(result_type.clone(qtype.getExpressedType()));
    }

    // Make a copy of the quantized function.
    auto module = call_op->getParentOfType<ModuleOp>();
    SymbolTable symbol_table(module);

    const auto f_attr = mlir::dyn_cast<FlatSymbolRefAttr>(call_op.getFAttr());
    const auto float_func =
        dyn_cast<func::FuncOp>(symbol_table.lookup(f_attr.getValue()));
    rewriter.setInsertionPointAfter(float_func);

    const std::string quantized_function_name = GetQuantizedFunctionName(
        f_attr.getValue(), /*merged_with_dequantize=*/true,
        /*is_hybrid=*/false);
    const auto quantized_func = dyn_cast_or_null<func::FuncOp>(
        symbol_table.lookup(quantized_function_name));
    if (quantized_func == nullptr) {
      call_op->emitError("Failed to find the quantized function: " +
                         quantized_function_name);
      return failure();
    }
    auto new_quantized_func = dyn_cast<func::FuncOp>(quantized_func->clone());
    new_quantized_func.setType(
        FunctionType::get(getContext(), TypeRange{ValueRange{args}},
                          new_quantized_func.getResultTypes()));
    for (auto [partitioned_call_arg, new_quantized_func_arg] :
         llvm::zip_first(args, new_quantized_func.getArguments())) {
      new_quantized_func_arg.setType(partitioned_call_arg.getType());
    }

    // Set the location for ops so the op name is preserved.
    if (failed(TransferLocation(float_func, new_quantized_func))) {
      return failure();
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
    auto quantized_call_op = TF::PartitionedCallOp::create(
        rewriter, call_op.getLoc(), result_types, args,
        call_op.getArgAttrsAttr(), call_op.getResAttrsAttr(),
        FlatSymbolRefAttr::get(new_quant_func_name));

    for (int result_idx : llvm::seq<int>(0, call_op->getNumResults())) {
      Value result = call_op->getResult(result_idx);
      for (Operation* user : result.getUsers()) {
        if (auto dequant_op = llvm::dyn_cast<DequantizeCastOp>(user)) {
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
class QuantizeConstPattern : public OpRewritePattern<QuantizeCastOp> {
 public:
  // This pattern should have larger benefit than ReplaceQuantizePattern
  explicit QuantizeConstPattern(MLIRContext* context, OpSet target_opset)
      : OpRewritePattern<QuantizeCastOp>(context, /*benefit=*/10),
        target_opset_(target_opset) {}

 private:
  LogicalResult matchAndRewrite(QuantizeCastOp q_op,
                                PatternRewriter& rewriter) const override {
    DenseFPElementsAttr attr;
    if (!matchPattern(q_op.getArg(), m_Constant(&attr))) {
      return failure();
    }

    ShapedType tensor_qtype =
        mlir::cast<ShapedType>(q_op.getResult().getType());
    Attribute tensor_proto_attr = Quantize(attr, tensor_qtype);
    if (!tensor_proto_attr) {
      return failure();
    }

    Type storage_type = mlir::cast<QuantizedType>(tensor_qtype.getElementType())
                            .getStorageType();
    ShapedType new_type = tensor_qtype.clone(storage_type);
    Location loc = q_op.getArg().getLoc();

    if (target_opset_ == OpSet::UNIFORM_QUANTIZED) {
      new_type = ConvertIntToQint(new_type, rewriter.getContext());

      // TODO(b/225793355): It adds TensorProtoAttr to the constant as a
      // workaround.
      tensorflow::TensorProto tensor_proto;
      if (!mlir::tfg::ConvertToTensorProto(
               mlir::cast<ElementsAttr>(tensor_proto_attr), &tensor_proto)
               .ok()) {
        return failure();
      }

      const int bit_width =
          mlir::dyn_cast<QuantizedType>(tensor_qtype.getElementType())
              .getStorageTypeIntegralWidth();

      tensor_proto.set_dtype((bit_width == 8) ? tensorflow::DT_QINT8
                                              : tensorflow::DT_QINT32);

      tensor_proto_attr = ElementsAttr(TF::TensorProtoAttr::get(
          new_type, tensorflow::mangling_util::MangleTensor(tensor_proto)));
    }
    auto const_op =
        TF::ConstOp::create(rewriter, loc, new_type, tensor_proto_attr);
    // Add scast op to match quantize -> composition pattern. The added scast
    // is then removed by canonicalization. ([scast - scast] -> [])
    auto scast_op = StorageCastOp::create(rewriter, loc, tensor_qtype,
                                          const_op.getOutput());
    q_op->replaceAllUsesWith(scast_op);
    return success();
  }

  OpSet target_opset_;
};

// To calculate per-channel scale and offset, weight of depthwise was reshaped
// to [H, W, 1, InxMul]. After scale and offset has been calculated, this
// pattern gets called and restores the weight of depthwise back
// into [H, W, In, Mul]
class RestoreWeightShapePattern
    : public OpRewritePattern<TF::PartitionedCallOp> {
  using OpRewritePattern<TF::PartitionedCallOp>::OpRewritePattern;

 private:
  LogicalResult addReshapeOpToDepthwiseWeight(TF::PartitionedCallOp op,
                                              PatternRewriter& rewriter) const {
    int weight_operand_idx = 1;
    Operation* weight_op = op.getOperand(weight_operand_idx).getDefiningOp();

    auto weight_type =
        mlir::dyn_cast<ShapedType>(weight_op->getResult(0).getType());
    auto input_type = mlir::dyn_cast<ShapedType>(op.getOperand(0).getType());

    llvm::ArrayRef<int64_t> weight_shape = weight_type.getShape();
    llvm::ArrayRef<int64_t> input_shape = input_type.getShape();

    // If weight_shape[2] != 1, it means weight shape was already restored.
    if (weight_shape[2] != 1) return failure();

    // Weight was reshaped into [H, W, 1, InxMul].
    // Since we know in_channels from input_shape, we can derive multiplier.
    int64_t in_channels = input_shape[3];
    // If in_channels is 1, there is no need to restore weight shape.
    if (in_channels == 1) return failure();
    int64_t multiplier = weight_shape[3] / in_channels;

    TensorType new_shape = RankedTensorType::get(
        {weight_shape[0], weight_shape[1], in_channels, multiplier},
        weight_type.getElementType());

    int cur_rank = weight_type.getRank();

    // Inserts a reshape op.
    auto shape_spec_type =
        RankedTensorType::get({cur_rank}, rewriter.getIntegerType(64));
    auto new_shape_const_attr =
        DenseElementsAttr::get(shape_spec_type, new_shape.getShape());
    rewriter.setInsertionPointAfter(weight_op);
    auto new_shape_const = TF::ConstOp::create(
        rewriter, weight_op->getLoc(), shape_spec_type, new_shape_const_attr);
    auto reshape_op =
        TF::ReshapeOp::create(rewriter, weight_op->getLoc(), new_shape,
                              weight_op->getResult(0), new_shape_const);
    op->setOperand(weight_operand_idx, reshape_op);

    return success();
  }

  LogicalResult matchAndRewrite(TF::PartitionedCallOp call_op,
                                PatternRewriter& rewriter) const override {
    const auto f_attr = mlir::dyn_cast<FlatSymbolRefAttr>(call_op.getFAttr());
    StringRef function_name = f_attr.getValue();
    // TODO(b/228928859): Improve the getter function to match attributes rather
    // than function name.
    // If enable_legacy_weight_only is enabled, QuantizeFunctionsPattern
    // does not get called and function remains as composite
    if (!function_name.starts_with("quantized_") &&
        !function_name.starts_with("composite_")) {
      return failure();
    }

    if (function_name.contains("depthwise_conv2d")) {
      return addReshapeOpToDepthwiseWeight(call_op, rewriter);
    }

    return failure();
  }
};

// Prints a summary about the quantization results.
class QuantizationSummary {
 public:
  explicit QuantizationSummary(ModuleOp module)
      : module_(module), symbol_table_(module) {}

  void Print() {
    llvm::StringMap<OpCountItem> func_count_map;
    int32_t total_quantized_func_count = 0, float_output_func_count = 0,
            quantize_func_count = 0, dequantize_func_count = 0,
            weight_only_count = 0;

    module_.walk([&](Operation* op) {
      if (auto call_op = llvm::dyn_cast_or_null<TF::PartitionedCallOp>(op)) {
        const auto f_attr =
            mlir::dyn_cast<FlatSymbolRefAttr>(call_op.getFAttr());
        if (!f_attr) return;
        StringRef func_name = f_attr.getValue();
        if (func_name.starts_with(kQuantizedFuncPrefix)) {
          auto representative_name = GetRepresentativeName(func_name);
          if (failed(representative_name)) return;

          func_count_map[representative_name.value()].num_quant++;
          total_quantized_func_count++;
          if (func_name.contains(kFloatOutputFuncSuffix) ||
              func_name.contains(kHybridFuncSuffix)) {
            float_output_func_count++;
          }
        } else if (func_name.starts_with(kCompositeFuncPrefix)) {
          auto representative_name = GetRepresentativeName(func_name);
          if (failed(representative_name)) {
            // TODO(b/264507511): Print quantization summary for weight-only.
            weight_only_count++;
          } else {
            func_count_map[representative_name.value()].num_float++;
          }
        } else if (func_name.starts_with("quantize_i")) {
          quantize_func_count++;
        } else if (func_name.starts_with("dequantize_i")) {
          dequantize_func_count++;
        }
      } else if (auto einsum = llvm::isa<TF::EinsumOp>(op)) {
        if (IsInCompsiteFunction(op)) return;
        // Leftover Einsum ops are always non-quantized.
        auto op_name = op->getName().stripDialect();
        func_count_map[op_name].num_float++;
      }
    });

    // Pad string to a certain size to format the table. Space is preferred to
    // Tab since it is easier to check the format in the mlir tests.
    auto pad_string = [](StringRef s, int32_t width) -> std::string {
      return llvm::Twine(s).concat(std::string(width - s.size(), ' ')).str();
    };

    // Generate a quantization report.
    size_t name_col_width = 5;
    absl::c_for_each(func_count_map.keys(), [&name_col_width](const auto& key) {
      name_col_width = std::max(name_col_width, key.size() + 1);
    });

    std::vector<std::string> lines;
    lines.push_back("-------- Quantization Summary --------");
    lines.push_back("Number of quantized layers in the model");
    lines.push_back("--------------------------------");
    lines.push_back(
        absl::StrFormat("%s Count/Total", pad_string("Name", name_col_width)));
    lines.push_back("================================");
    for (StringRef op_name : func_count_map.keys()) {
      const int32_t quantized_count = func_count_map[op_name].num_quant;
      const int32_t total_count =
          quantized_count + func_count_map[op_name].num_float;
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
    OpBuilder builder(module_);
    builder.setInsertionPointToEnd(&module_.getBodyRegion().back());
    const auto func_type =
        builder.getFunctionType(/*inputs=*/{}, /*results=*/{});
    auto summary_func = func::FuncOp::create(builder, builder.getUnknownLoc(),
                                             /*sym_name=*/"summary", func_type);
    summary_func.setPrivate();
    summary_func->setAttr("quantization_summary",
                          builder.getStringAttr(log_message));
  }

 private:
  // Structs used to count quantized and non-quantized ops.
  struct OpCountItem {
    int32_t num_quant = 0;
    int32_t num_float = 0;
  };

  // Get the representative name attribute value of a composite function.
  FailureOr<StringRef> GetRepresentativeName(StringRef func_name) {
    std::string quantized_func_name = GetQuantizedFunctionName(
        func_name, /*merged_with_dequantize=*/false, /*is_hybrid=*/false);
    auto quantized_func = dyn_cast_or_null<func::FuncOp>(
        symbol_table_.lookup(quantized_func_name));
    // Quantized function does not exist for weight-only case.
    if (!quantized_func ||
        !quantized_func->hasAttrOfType<ArrayAttr>(kQuantizedOpsAttribute)) {
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
    return mlir::cast<StringAttr>(quantized_ops.front()).getValue();
  }

  bool IsInCompsiteFunction(Operation* op) {
    func::FuncOp parent = op->getParentOfType<func::FuncOp>();
    if (!parent) return false;

    StringRef sym_name = parent.getSymName();
    return sym_name.starts_with(kQuantizedFuncPrefix) ||
           sym_name.starts_with(kCompositeFuncPrefix);
  }

  ModuleOp module_;
  SymbolTable symbol_table_;
};

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
  quant_specs.inference_type = tensorflow::DT_QINT8;
  quant_specs.disable_per_channel = !enable_per_channel_quantization_;

  pm.addPass(CreatePreprocessOpPass(target_opset_, quantization_method_,
                                    enable_per_channel_quantization_));

  // Apply activation-weight quantization.
  if (quantization_method_ ==
      tensorflow::quantization::QuantizationMethod::METHOD_STATIC_RANGE_INT8) {
    // For XLA case, weight quantization will be applied for the remaining f32
    // weights even in SRQ.
    pm.addNestedPass<func::FuncOp>(
        CreatePrepareQuantizePass(quant_specs, quantization_method_));
    pm.addNestedPass<func::FuncOp>(
        CreateQuantizePass(quant_specs, target_opset_));
    pm.addNestedPass<func::FuncOp>(CreatePostQuantizePass());
  } else {
    // Apply weight quantization.
    quant_specs.minimum_elements_for_weights = min_num_elements_for_weights_;
    quant_specs.weight_quantization = true;
    quant_specs.weight_only_quantization = enable_legacy_weight_only_;
    pm.addPass(CreatePrepareQuantizeDRQPass(quant_specs, target_opset_));
    pm.addNestedPass<func::FuncOp>(
        CreateQuantizePass(quant_specs, target_opset_));
    pm.addNestedPass<func::FuncOp>(CreatePostQuantizePass());
  }

  absl::Status pm_run_status = tensorflow::quantization::RunPassesOnModuleOp(
      mlir_dump_file_name_, pm, module);
  if (!pm_run_status.ok()) {
    signalPassFailure();
  }

  // Legacy weight-only does not require quantized ops.
  if (!enable_legacy_weight_only_) {
    RewritePatternSet patterns(ctx);
    patterns.add<QuantizeFunctionPattern>(ctx, quantization_method_,
                                          target_opset_,
                                          enable_per_channel_quantization_);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  // Constant quantization is a lossy transformation, so they are applied only
  // after all the other patterns have been applied.
  RewritePatternSet patterns_2(ctx);
  populateWithGenerated(patterns_2);
  patterns_2.add<ReplaceQuantizePattern, ReplaceDequantizePattern>(
      ctx, target_opset_);
  patterns_2.add<QuantizeConstPattern>(ctx, target_opset_);

  if (target_opset_ == OpSet::XLA && enable_per_channel_quantization_) {
    patterns_2.add<RestoreWeightShapePattern>(ctx);
  }

  if (failed(applyPatternsGreedily(module, std::move(patterns_2))) ||
      failed(verify(module))) {
    signalPassFailure();
  }
  QuantizationSummary(module).Print();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateQuantizeCompositeFunctionsPass(
    const QuantMethod quantization_method, const OpSet target_opset,
    const bool enable_per_channel_quantization,
    const int min_num_elements_for_weights,
    const bool enable_legacy_weight_only,
    std::optional<const absl::string_view> mlir_dump_file_prefix) {
  std::optional<std::string> mlir_dump_file_name;
  if (mlir_dump_file_prefix) {
    mlir_dump_file_name = absl::StrCat(mlir_dump_file_prefix.value(),
                                       kQuantizeCompositeFunctionsStepName);
  }
  return std::make_unique<QuantizeCompositeFunctionsPass>(
      quantization_method, target_opset, enable_per_channel_quantization,
      min_num_elements_for_weights, enable_legacy_weight_only,
      mlir_dump_file_name);
}

}  // namespace quant
}  // namespace mlir
