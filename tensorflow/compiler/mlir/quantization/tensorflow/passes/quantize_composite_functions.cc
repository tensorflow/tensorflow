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
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {
namespace {

constexpr char kQuantizeFuncName[] = "quantize_i8";
constexpr char kDequantizeFuncName[] = "dequantize_i8";
constexpr char kAttrMapAttribute[] = "attr_map";

class QuantizeCompositeFunctionsPass
    : public mlir::PassWrapper<QuantizeCompositeFunctionsPass,
                               OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizeCompositeFunctionsPass)

  explicit QuantizeCompositeFunctionsPass() {}
  explicit QuantizeCompositeFunctionsPass(
      QuantizationMethod quantization_method)
      : quantization_method_(quantization_method) {}

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
    registry.insert<TF::TensorFlowDialect, QuantizationDialect>();
  }

 private:
  void runOnOperation() override;

  QuantizationMethod quantization_method_ =
      QuantizationMethod::kQuantizationAwareTraining;
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
    UniformQuantizedPerAxisType qtype, Location loc, PatternRewriter& rewriter,
    Value& scale, Value& zero_point) {
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
  } else if (auto qtype = elem_type.dyn_cast<UniformQuantizedPerAxisType>()) {
    return CreateUniformQuantizedPerAxisTypeParams(qtype, loc, rewriter, scale,
                                                   zero_point);
  }
  return failure();
}

// Replaces quant.qcast op to composite quantize_i8 function.
class ReplaceQuantizePattern : public mlir::OpRewritePattern<QuantizeCastOp> {
 public:
  explicit ReplaceQuantizePattern(MLIRContext* context)
      : OpRewritePattern<QuantizeCastOp>(context) {}

 private:
  LogicalResult matchAndRewrite(QuantizeCastOp q_op,
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
    SmallVector<Value> args = {q_op.arg(), scale, zero_point};
    FlatSymbolRefAttr func_name =
        FlatSymbolRefAttr::get(rewriter.getStringAttr(kQuantizeFuncName));

    auto quantize_call = rewriter.create<TF::PartitionedCallOp>(
        loc, output_types, args, func_name,
        /*config=*/"", /*config_proto=*/"", /*executor_type=*/"");
    auto scast_op = rewriter.create<quant::StorageCastOp>(
        loc, output_type, quantize_call->getResult(0));
    q_op->replaceAllUsesWith(scast_op);
    return success();
  }
};

// Replaces quant.dcast op to composite dequantize_i8 function.
class ReplaceDequantizePattern
    : public mlir::OpRewritePattern<DequantizeCastOp> {
 public:
  explicit ReplaceDequantizePattern(MLIRContext* context)
      : OpRewritePattern<DequantizeCastOp>(context) {}

 private:
  LogicalResult matchAndRewrite(DequantizeCastOp dq_op,
                                PatternRewriter& rewriter) const override {
    auto input_type = dq_op.arg().getType().cast<TensorType>();
    auto elem_type = input_type.getElementType().dyn_cast<QuantizedType>();
    const Location loc = dq_op->getLoc();

    Value scale, zero_point;
    if (failed(CreateQuantizationParams(elem_type, loc, rewriter, scale,
                                        zero_point))) {
      return failure();
    }

    TensorType output_type = input_type.clone(elem_type.getStorageType());
    auto scast_op =
        rewriter.create<quant::StorageCastOp>(loc, output_type, dq_op.arg());

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
  for (int32_t cur_idx = 0; cur_idx < call_op.args().size(); cur_idx++) {
    // Check if the only the weight index has QuantizeCastOp.
    auto cur_op = dyn_cast_or_null<QuantizeCastOp>(
        call_op.args()[cur_idx].getDefiningOp());
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
  for (Value output : call_op.output()) {
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
  for (Value input : call_op.args()) {
    if (auto type = input.getType().dyn_cast<TensorType>()) {
      if (type.getElementType().isa<FloatType>()) {
        return false;
      }
      if (type.getElementType().isa<QuantizedType>()) {
        has_quantized_types = true;
      }
    }
  }
  for (Value output : call_op.output()) {
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

// Unwraps quantization parameters of PartitionedCall ops with quantized
// input/outputs that are created from QuantizePass.
class QuantizeFunctionPattern
    : public mlir::OpRewritePattern<TF::PartitionedCallOp> {
 public:
  explicit QuantizeFunctionPattern(MLIRContext* context,
                                   QuantizationMethod quantization_method)
      : OpRewritePattern<TF::PartitionedCallOp>(context),
        quantization_method_(quantization_method) {}

 private:
  QuantizationMethod quantization_method_ =
      QuantizationMethod::kPostTrainingQuantization;

  LogicalResult matchAndRewrite(TF::PartitionedCallOp call_op,
                                PatternRewriter& rewriter) const override {
    const auto f_attr = call_op.fAttr().dyn_cast<FlatSymbolRefAttr>();
    // removeAttr will return nullptr if no attribute was removed.
    if (!call_op->removeAttr(kQuantTraitAttrName) || !f_attr) {
      return failure();
    }

    // Determines if all required float input/outputs are now quantized.
    bool has_quantized_types = false;
    if (quantization_method_ == QuantizationMethod::kDynamicRangeQuantization) {
      has_quantized_types = IsQuantizedCallforDynamicRange(call_op);
    } else {
      has_quantized_types = IsQuantizedCallforStaticRange(call_op);
    }

    if (!f_attr.getValue().startswith("composite_") || !has_quantized_types) {
      return failure();
    }

    SmallVector<Value, 4> args;
    for (Value arg : call_op.args()) {
      if (const auto arg_type = arg.getType().dyn_cast<TensorType>()) {
        QuantizedType qtype =
            arg_type.getElementType().dyn_cast<QuantizedType>();
        if (qtype &&
            !qtype.isa<UniformQuantizedType, UniformQuantizedPerAxisType>()) {
          return failure();
        }
      }
    }

    for (Value result : call_op->getResults()) {
      if (auto result_type = result.getType().dyn_cast<TensorType>()) {
        QuantizedType qtype =
            result_type.getElementType().dyn_cast<QuantizedType>();
        if (qtype &&
            !qtype.isa<UniformQuantizedType, UniformQuantizedPerAxisType>()) {
          return failure();
        }
      }
    }

    rewriter.setInsertionPoint(call_op);

    SmallVector<Value, 4> qparam_args;
    for (Value arg : call_op.args()) {
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
      Value scale, zero_point;
      if (failed(CreateQuantizationParams(qtype, arg.getLoc(), rewriter, scale,
                                          zero_point))) {
        // As the quantized types are already checked, this is unexpected.
        call_op->emitError(
            "Failed to create quantization parameter for an argument.");
        return failure();
      }
      auto scast_op = rewriter.create<StorageCastOp>(
          arg.getLoc(), arg_type.clone(qtype.getStorageType()), arg);
      args.push_back(scast_op.getResult());
      qparam_args.push_back(scale);
      qparam_args.push_back(zero_point);
    }

    DenseMap<Value, StorageCastOp> replace_map;
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
      Value scale, zero_point;
      if (failed(CreateQuantizationParams(qtype, result.getLoc(), rewriter,
                                          scale, zero_point))) {
        // As the quantized types are already checked, this is unexpected.
        call_op->emitError(
            "Failed to create quantization parameter for a result.");
        return failure();
      }
      auto scast_op =
          rewriter.create<StorageCastOp>(call_op.getLoc(), result_type, result);
      replace_map.insert(std::make_pair(result, scast_op));

      result_types.push_back(result_type.clone(qtype.getStorageType()));
      qparam_args.push_back(scale);
      qparam_args.push_back(zero_point);
    }

    for (auto replace_pair : replace_map) {
      Value result = replace_pair.first;
      StorageCastOp scast_op = replace_pair.second;
      result.replaceAllUsesExcept(scast_op, scast_op);
    }

    args.insert(args.end(), qparam_args.begin(), qparam_args.end());

    // Make a copy of the quantized function.
    auto module = call_op->getParentOfType<ModuleOp>();
    SymbolTable symbol_table(module);

    func::FuncOp float_func =
        dyn_cast<func::FuncOp>(symbol_table.lookup(f_attr.getValue()));
    rewriter.setInsertionPointAfter(float_func);

    // substr(10) == strip the "composite_" prefix.
    const llvm::Twine quantized_function_name = llvm::Twine(
        "quantized_", f_attr.getValue().substr(10).rsplit('_').first);
    const func::FuncOp quantized_func = dyn_cast<func::FuncOp>(
        symbol_table.lookup(quantized_function_name.str()));
    func::FuncOp new_quantized_func =
        dyn_cast<func::FuncOp>(quantized_func->clone());
    if (new_quantized_func == nullptr) {
      return failure();
    }
    new_quantized_func.setType(
        FunctionType::get(getContext(), TypeRange(ArrayRef<Value>(args)),
                          new_quantized_func.getResultTypes()));
    for (auto pair : llvm::zip_first(args, new_quantized_func.getArguments())) {
      auto new_quantized_func_arg = std::get<1>(pair);
      auto partitioned_call_arg = std::get<0>(pair);
      new_quantized_func_arg.setType(partitioned_call_arg.getType());
    }

    // Set the attributes for ops with the attr_map attribute.
    if (failed(TransferAttributes(float_func, new_quantized_func))) {
      return failure();
    }

    rewriter.setInsertionPoint(call_op);

    const StringAttr new_quant_func_name =
        symbol_table.insert(new_quantized_func);
    rewriter.replaceOpWithNewOp<TF::PartitionedCallOp>(
        call_op, result_types, args,
        FlatSymbolRefAttr::get(new_quant_func_name));

    return success();
  }
};

// Converts const -> quant.qcast pattern to quantized constant, after
// quantization parameters are safely included to each quantize composite
// functions.
class QuantizeConstPattern : public OpRewritePattern<QuantizeCastOp> {
 public:
  // This pattern should have larger benefit than ReplaceQuantizePattern
  explicit QuantizeConstPattern(MLIRContext* context)
      : OpRewritePattern<QuantizeCastOp>(context, /*benefit=*/10) {}
  LogicalResult matchAndRewrite(QuantizeCastOp q_op,
                                PatternRewriter& rewriter) const override {
    DenseFPElementsAttr attr;
    if (!matchPattern(q_op.arg(), m_Constant(&attr))) {
      return failure();
    }

    ShapedType tensor_qtype = q_op.getResult().getType().cast<ShapedType>();
    Attribute quantized_attr = Quantize(attr, tensor_qtype);
    if (!quantized_attr) {
      return failure();
    }

    Type storage_type =
        tensor_qtype.getElementType().cast<QuantizedType>().getStorageType();
    ShapedType new_type = tensor_qtype.clone(storage_type);
    Location loc = q_op.arg().getLoc();
    auto const_op = rewriter.create<TF::ConstOp>(loc, new_type, quantized_attr);
    // Add scast op to match quantize -> composition pattern. The added scast
    // is then removed by canonicalization. ([scast - scast] -> [])
    auto scast_op = rewriter.create<quant::StorageCastOp>(loc, tensor_qtype,
                                                          const_op.output());
    q_op->replaceAllUsesWith(scast_op);
    return success();
  }
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
  if (quantization_method_ == QuantizationMethod::kDynamicRangeQuantization) {
    quant_specs.weight_quantization = true;
    quant_specs.inference_type = tensorflow::DT_QINT8;
    pm.addNestedPass<func::FuncOp>(CreatePrepareQuantizeDRQPass());
  } else {
    pm.addNestedPass<func::FuncOp>(
        CreatePrepareQuantizePass(quantization_method_));
  }
  pm.addNestedPass<func::FuncOp>(CreateQuantizePass(quant_specs));

  pm.addNestedPass<func::FuncOp>(CreatePostQuantizePass());
  if (failed(pm.run(module))) {
    signalPassFailure();
  }

  RewritePatternSet patterns(ctx);
  patterns.add<QuantizeFunctionPattern>(ctx, quantization_method_);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }

  // Constant quantization is a lossy transformation, so they are applied only
  // after all the other patterns have been aplied.
  RewritePatternSet patterns_2(ctx);
  populateWithGenerated(patterns_2);
  patterns_2.add<ReplaceQuantizePattern, ReplaceDequantizePattern,
                 QuantizeConstPattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns_2))) ||
      failed(verify(module))) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateQuantizeCompositeFunctionsPass(
    QuantizationMethod quantization_method) {
  return std::make_unique<QuantizeCompositeFunctionsPass>(quantization_method);
}

}  // namespace quant
}  // namespace mlir
