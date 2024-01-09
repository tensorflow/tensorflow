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

#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/quantization_patterns.h"

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/algorithm/container.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BlockSupport.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/uniform_quantized_types.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

#define DEBUG_TYPE "populate-quantization-patterns"

namespace mlir::quant::stablehlo {

namespace {

using ::mlir::quant::TryCast;
using ::mlir::stablehlo::AddOp;
using ::mlir::stablehlo::ConcatenateOp;
using ::mlir::stablehlo::ConvolutionOp;
using ::mlir::stablehlo::DotGeneralOp;
using ::mlir::stablehlo::DynamicBroadcastInDimOp;
using ::mlir::stablehlo::GetDimensionSizeOp;
using ::mlir::stablehlo::ReshapeOp;
using ::mlir::stablehlo::UniformQuantizeOp;

constexpr StringRef kCompositeFuncPrefix = "composite_";
constexpr StringRef kQuantizedFuncPrefix = "quantized_";
constexpr StringRef kEntryFuncAttrName = "_entry_function";

// Returns true if `type` is a TensorType with quantized elements.
bool IsQuantizedTensorType(const Type type) {
  return type.isa<TensorType>() &&
         type.cast<TensorType>().getElementType().isa<QuantizedType>();
}

// Returns dynamically broadcasted user op of an input op. Returns null if
// the op is not dynamically broadcasted or not the intended type.
// Dynamic shapes usually has the following pattern. In the example below,
// the input operand would be stablehlo.convolution op, and return value would
// be stablehlo.add op.
// Note that the patterns below differ from lifted patterns as
// ShapeLegalizeToHloPass is ran prior to running this pass.
//
// ```
// %0 = stablehlo.constant dense<3>
// %1 = stablehlo.constant dense<4>
// %2 = stablehlo.constant dense<2>
// %3 = stablehlo.convolution(%%arg0, %%arg1) :
//          (tensor<?x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<?x3x4x2xf32>
// %4 = stablehlo.get_dimension_size %3, dim = 0 :
//          (tensor<?x3x4x2xf32>) -> tensor<i32>
// %5 = stablehlo.reshape %4 :
//          (tensor<i32>) -> tensor<1xi32>
// %6 = stablehlo.concatenate %5, %0, %1, %2, dim = 0 :
//          (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>)
//            -> tensor<4xi32>
// %7 = stablehlo.dynamic_broadcast_in_dims %arg2, %6
// %8 = stablehlo.add %3, %7
// ```
template <typename T>
Operation* GetDynamicallyBroadcastedUserOp(Operation* op) {
  FailureOr<GetDimensionSizeOp> get_dimension_size_op =
      TryCast<GetDimensionSizeOp>(op->getNextNode(),
                                  /*name=*/"get_dimension_size_op");
  if (failed(get_dimension_size_op)) {
    return nullptr;
  }
  auto reshape_op = TryCast<ReshapeOp>((*get_dimension_size_op)->getNextNode(),
                                       /*name=*/"reshape_op");
  if (failed(reshape_op)) {
    return nullptr;
  }
  auto concatenate_op = TryCast<ConcatenateOp>((*reshape_op)->getNextNode(),
                                               /*name=*/"concatenate_op");
  if (failed(concatenate_op)) {
    return nullptr;
  }
  auto dynamic_broadcast_in_dim_op =
      TryCast<DynamicBroadcastInDimOp>((*concatenate_op)->getNextNode(),
                                       /*name=*/"dynamic_broadcast_in_dim_op");
  if (failed(dynamic_broadcast_in_dim_op)) {
    return nullptr;
  }
  auto target_op = TryCast<T>((*dynamic_broadcast_in_dim_op)->getNextNode(),
                              /*name=*/"target_op");
  if (failed(target_op)) {
    return nullptr;
  }
  return *target_op;
}

// Checks if all inputs and outputs are quantized.
bool HasQuantizedOperandOrOutput(Operation* call_op) {
  SmallVector<Type> arg_types;
  for (const Value arg : call_op->getOperands()) {
    arg_types.push_back(arg.getType());
  }

  SmallVector<Type> output_types;
  for (const Value output : call_op->getResults()) {
    output_types.push_back(output.getType());
  }

  return absl::c_all_of(arg_types, IsQuantizedTensorType) &&
         absl::c_all_of(output_types, IsQuantizedTensorType);
}

// Gets the corresponding quantized function name from the given function name.
// Example: "composite_dot_general_fn_1" => "quantized_dot_general_fn"
std::string GetQuantizedFunctionName(const StringRef func_name) {
  return Twine(kQuantizedFuncPrefix)
      .concat(func_name.rsplit(kCompositeFuncPrefix).second)
      .str();
}

// Returns true if `xla_call_module_op` is quantized. To be considered
// quantized, it should meet three conditions:
// 1. At least one of the inputs or outputs should be a uniform quantized type.
// 2. `xla_call_module_op` should have the `kQuantTraitAttrName` attribute.
// 3. It should also have the `kEntryFuncAttrName` attribute, which points to
//    the function that `xla_call_module_op` represents.
bool IsQuantizedXlaCallModuleOp(TF::XlaCallModuleOp xla_call_module_op) {
  return HasQuantizedOperandOrOutput(xla_call_module_op) &&
         xla_call_module_op->hasAttr(kQuantTraitAttrName) &&
         xla_call_module_op->hasAttr(kEntryFuncAttrName);
}

// Returns the entry function, i.e. the callee of `xla_call_module_op`.
func::FuncOp GetEntryFuncOp(TF::XlaCallModuleOp xla_call_module_op,
                            SymbolTable symbol_table) {
  const auto entry_function_symbol_ref =
      xla_call_module_op->getAttrOfType<FlatSymbolRefAttr>(kEntryFuncAttrName);

  return dyn_cast_or_null<func::FuncOp>(
      symbol_table.lookup(entry_function_symbol_ref.getValue()));
}

// Replaces the function type of `entry_func_op` to a quantized one, matching
// the input and output types of `xla_call_module_op`.
void SetQuantizedFunctionType(PatternRewriter& rewriter,
                              func::FuncOp entry_func_op,
                              TF::XlaCallModuleOp xla_call_module_op) {
  SmallVector<Type> arg_types;
  SmallVector<Location> arg_locs;
  for (const Value arg : xla_call_module_op.getArgs()) {
    arg_types.push_back(arg.getType());
    arg_locs.push_back(arg.getLoc());
  }

  SmallVector<Type> output_types;
  for (const Value output : xla_call_module_op.getOutput()) {
    output_types.push_back(output.getType());
  }

  entry_func_op.setFunctionType(
      rewriter.getFunctionType(arg_types, output_types));

  // Replace argument types and locs.
  Block& entry = entry_func_op->getRegion(0).front();
  for (auto [arg, arg_type, arg_loc] :
       llvm::zip_equal(entry.getArguments(), arg_types, arg_locs)) {
    arg.setType(arg_type);
    arg.setLoc(arg_loc);
  }
}

// Creates a UniformQuantize op and sets it as return op.
void CreateAndReturnUniformQuantizeOp(PatternRewriter& rewriter, Operation& op,
                                      func::FuncOp entry_func_op,
                                      const Type func_result_type) {
  // Add i32 -> i8 requantization.
  UniformQuantizeOp uniform_quant_op = rewriter.create<UniformQuantizeOp>(
      op.getLoc(), func_result_type, op.getResults());
  cast<func::ReturnOp>(entry_func_op.getBody().front().getTerminator())
      .setOperand(0, uniform_quant_op);
}

template <typename GemmStyleOp>
// Creates a quantized bias pattern for static and dynamic shape case
// and sets the quantized bias as the return op.
void CreateAndReturnQuantizedBiasPattern(
    Operation* op, PatternRewriter& rewriter, func::FuncOp entry_func_op,
    const Type func_result_type, const Type gemm_style_quantized_element_type,
    GemmStyleOp gemm_style_op, double result_scale) {
  Value bias_op = op->getOperand(1);
  Value add_op_result = op->getResult(0);
  // For bias add with dynamic shape, quantize the broadcasted bias.
  if (auto dynamic_bcast_op =
          cast_or_null<DynamicBroadcastInDimOp>(bias_op.getDefiningOp())) {
    const UniformQuantizedType dynamic_bcast_quantized_element_type =
        CreateI32F32UniformQuantizedType(gemm_style_op->getLoc(),
                                         *rewriter.getContext(), result_scale,
                                         /*zero_point=*/0);

    Value dynamic_bcast_op_result = dynamic_bcast_op->getResult(0);
    auto dynamic_bcast_op_result_type =
        dynamic_bcast_op_result.getType().cast<RankedTensorType>();
    const ArrayRef<int64_t> dynamic_bcast_shape =
        dynamic_bcast_op_result_type.getShape();

    const TensorType new_dynamic_bcast_op_result_type =
        dynamic_bcast_op_result_type.cloneWith(
            dynamic_bcast_shape, gemm_style_quantized_element_type);
    dynamic_bcast_op_result.setType(new_dynamic_bcast_op_result_type);
  }
  const auto add_op_result_type =
      add_op_result.getType().cast<RankedTensorType>();
  const ArrayRef<int64_t> add_op_shape = add_op_result_type.getShape();
  // For quantized bias add case, lhs, rhs, and result have the same types.
  const TensorType new_add_op_result_type = add_op_result_type.cloneWith(
      add_op_shape, gemm_style_quantized_element_type);
  add_op_result.setType(new_add_op_result_type);

  AddOp bias_add_op =
      rewriter.create<AddOp>(gemm_style_op->getLoc(), gemm_style_op, bias_op);

  CreateAndReturnUniformQuantizeOp(rewriter, *bias_add_op, entry_func_op,
                                   func_result_type);
}

// An interface representing patterns that quantizes an entry function's body.
// The entry function's signatures should have already been quantized at the
// point of rewriting.
class EntryFuncBodyQuantizationPattern {
 public:
  virtual ~EntryFuncBodyQuantizationPattern() = default;

  // Returns `success()` if `entry_func_op`'s body is eligible for rewriting. At
  // this point `entry_func_op`'s signature has not been reset with quantized
  // types.
  virtual LogicalResult match(func::FuncOp entry_func_op) const = 0;

  // Rewrites the `entry_func_op`'s body.
  virtual void rewrite(func::FuncOp entry_func_op,
                       PatternRewriter& rewriter) const = 0;
};

// Gemm Style Op: glossary/gemm.
template <typename GemmStyleOp>
// Match for all gemm_style op and check for possible fusions.
LogicalResult MatchGemmStyleOp(func::FuncOp entry_func_op) {
  auto op_iterator_range = entry_func_op.getOps<GemmStyleOp>();
  if (op_iterator_range.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Function does not have GemmStyle op.\n");
    return failure();
  }
  if (!isa<RankedTensorType>(
          (*op_iterator_range.begin()).getResult().getType())) {
    LLVM_DEBUG(llvm::dbgs() << "GemmStyle op must have ranked tensor type.\n");
    return failure();
  }

  MutableArrayRef<BlockArgument> operands =
      entry_func_op.getBody().getArguments();
  // Function must have input, filter, and optionally bias.
  if (operands.size() != 2 && operands.size() != 3) {
    LLVM_DEBUG(llvm::dbgs()
               << "GemmStyle op function should have 2 or 3 operands.\n");
    return failure();
  }
  return success();
}

// Gemm Style Op: glossary/gemm.
template <typename GemmStyleOp>
void RewriteGemmStyleOp(func::FuncOp entry_func_op, PatternRewriter& rewriter) {
  // Update the output type of the gemm_style op.
  GemmStyleOp gemm_style_op = *entry_func_op.getOps<GemmStyleOp>().begin();

  const Type input_type = entry_func_op.getArgumentTypes()[0];
  const Type filter_type = entry_func_op.getArgumentTypes()[1];
  const Type func_result_type = entry_func_op.getResultTypes()[0];

  const double input_scale =
      getElementTypeOrSelf(input_type).cast<UniformQuantizedType>().getScale();
  const double filter_scale =
      getElementTypeOrSelf(filter_type).cast<UniformQuantizedType>().getScale();
  const double result_scale = input_scale * filter_scale;

  // Define the intermediate output type, which is an i32 quantized type.
  // This is intermediate because the final output type of the entry_func_op
  // should be an i8 quantized type.
  const UniformQuantizedType gemm_style_quantized_element_type =
      CreateI32F32UniformQuantizedType(gemm_style_op->getLoc(),
                                       *rewriter.getContext(), result_scale,
                                       /*zero_point=*/0);

  Value gemm_style_op_result = gemm_style_op->getResult(0);
  auto gemm_style_op_result_type =
      gemm_style_op_result.getType().cast<RankedTensorType>();
  const ArrayRef<int64_t> gemm_style_shape =
      gemm_style_op_result_type.getShape();

  const TensorType new_gemm_style_op_result_type =
      gemm_style_op_result_type.cloneWith(gemm_style_shape,
                                          gemm_style_quantized_element_type);
  gemm_style_op_result.setType(new_gemm_style_op_result_type);

  rewriter.setInsertionPointAfter(gemm_style_op);

  Operation* next_op = gemm_style_op->getNextNode();

  if (isa<AddOp>(next_op) && gemm_style_op->hasOneUse()) {
    // bias fusion
    CreateAndReturnQuantizedBiasPattern(
        next_op, rewriter, entry_func_op, func_result_type,
        gemm_style_quantized_element_type, gemm_style_op, result_scale);
  } else if (auto add_op = cast_or_null<AddOp>(
                 GetDynamicallyBroadcastedUserOp<AddOp>(gemm_style_op))) {
    // dynamic bias fusion
    rewriter.setInsertionPointAfter(add_op);
    CreateAndReturnQuantizedBiasPattern(
        add_op, rewriter, entry_func_op, func_result_type,
        gemm_style_quantized_element_type, gemm_style_op, result_scale);
  } else {
    // Non fusible op
    // If an op is used multiple times and is not a dynamic shape case, do not
    // apply quantization of fused patterns to prevent removal of dependee ops.
    CreateAndReturnUniformQuantizeOp(rewriter, *gemm_style_op, entry_func_op,
                                     func_result_type);
  }
}

// Quantizes the entry function's body containing a `DotGeneralOp`.
class QuantizeDotGeneralOpPattern : public EntryFuncBodyQuantizationPattern {
 public:
  explicit QuantizeDotGeneralOpPattern() = default;

  LogicalResult match(func::FuncOp entry_func_op) const override {
    return MatchGemmStyleOp<DotGeneralOp>(entry_func_op);
  }

  void rewrite(func::FuncOp entry_func_op,
               PatternRewriter& rewriter) const override {
    RewriteGemmStyleOp<DotGeneralOp>(entry_func_op, rewriter);
  }
};

// Quantizes the entry function's body containing a `ConvolutionOp`.
class QuantizeConvolutionOpPattern : public EntryFuncBodyQuantizationPattern {
 public:
  explicit QuantizeConvolutionOpPattern() = default;

  LogicalResult match(func::FuncOp entry_func_op) const override {
    return MatchGemmStyleOp<ConvolutionOp>(entry_func_op);
  }

  void rewrite(func::FuncOp entry_func_op,
               PatternRewriter& rewriter) const override {
    RewriteGemmStyleOp<ConvolutionOp>(entry_func_op, rewriter);
  }
};

// Converts `entry_func_op` to be quantized according to the respective
// inputs and outputs of `xla_call_module_op` that are possibly quantized. It
// signature (type) is reset to match that of `xla_call_module_op`.
// `entry_func_body_quantization_pattern` rewrites the function's body, based on
// the new signature.
void QuantizeEntryFuncOp(
    MLIRContext& ctx, PatternRewriter& rewriter,
    TF::XlaCallModuleOp xla_call_module_op, func::FuncOp entry_func_op,
    const EntryFuncBodyQuantizationPattern& body_rewrite_pattern) {
  SetQuantizedFunctionType(rewriter, entry_func_op, xla_call_module_op);

  body_rewrite_pattern.rewrite(entry_func_op, rewriter);

  // Rename the function to be clear that the function has been quantized.
  const std::string quantized_function_name =
      GetQuantizedFunctionName(entry_func_op.getSymName());
  entry_func_op.setSymName(quantized_function_name);
}

// Replaces a quantized `xla_call_module_op` with a `func::CallOp`. The callee
// is expected to remain unquantized (thus having a signature mismatch), and it
// is also quantized accordingly.
void ReplaceQuantizedXlaCallModuleOpWithQuantizedCallOp(
    MLIRContext& ctx, PatternRewriter& rewriter,
    TF::XlaCallModuleOp xla_call_module_op,
    const EntryFuncBodyQuantizationPattern& body_rewrite_pattern) {
  ModuleOp module_op = xla_call_module_op->getParentOfType<ModuleOp>();
  SymbolTable symbol_table(module_op);

  func::FuncOp entry_func_op = GetEntryFuncOp(xla_call_module_op, symbol_table);
  QuantizeEntryFuncOp(ctx, rewriter, xla_call_module_op, entry_func_op,
                      body_rewrite_pattern);

  // Replace the XlaCallModuleOp with a new CallOp.
  rewriter.setInsertionPoint(xla_call_module_op);
  rewriter.replaceOpWithNewOp<func::CallOp>(xla_call_module_op, entry_func_op,
                                            xla_call_module_op.getArgs());
}

// Pattern that mainly does two things:
//
//   1. Replaces quantized `TF::XlaCallModuleOp` with a `func::CallOp`.
//   2. Quantizes the callee function.
//
// The inputs of this pattern assumes an invalid IR, where even if a
// `TF::XlaCallModuleOp` is quantized the callee remains unquantized. Step (2)
// not only replaces the input and output tensor types into quantized ones, but
// also rewrites the body with a quantized equivalent.
//
// `FuncBodyRewritePatternT` defines how a function body is quantized and
// rewritten.
template <typename FuncBodyRewritePatternT,
          typename = std::enable_if_t<std::is_base_of_v<
              EntryFuncBodyQuantizationPattern, FuncBodyRewritePatternT>>>
class XlaCallModuleOpToCallOp : public OpRewritePattern<TF::XlaCallModuleOp> {
 public:
  explicit XlaCallModuleOpToCallOp(MLIRContext& ctx)
      : OpRewritePattern<TF::XlaCallModuleOp>(&ctx) {}

  LogicalResult match(TF::XlaCallModuleOp op) const override {
    ModuleOp module_op = op->getParentOfType<ModuleOp>();
    SymbolTable symbol_table(module_op);

    // Ignore unquantized ops.
    if (!IsQuantizedXlaCallModuleOp(op)) return failure();

    func::FuncOp entry_func_op = GetEntryFuncOp(op, symbol_table);
    if (!entry_func_op) {
      op->emitError("Failed to find a valid entry function.");
      return failure();
    }
    return FuncBodyRewritePatternT().match(entry_func_op);
  }

  void rewrite(TF::XlaCallModuleOp xla_call_module_op,
               PatternRewriter& rewriter) const override {
    ReplaceQuantizedXlaCallModuleOpWithQuantizedCallOp(
        *rewriter.getContext(), rewriter, xla_call_module_op,
        FuncBodyRewritePatternT());
  }
};

}  // namespace

// TODO: b/307620428 - Increase fused op coverage for static range quantization.
void PopulateFusedGemmStylePatterns(MLIRContext& ctx,
                                    RewritePatternSet& patterns) {
  patterns.add<XlaCallModuleOpToCallOp<QuantizeDotGeneralOpPattern>,
               XlaCallModuleOpToCallOp<QuantizeConvolutionOpPattern>>(ctx);
}

}  // namespace mlir::quant::stablehlo
