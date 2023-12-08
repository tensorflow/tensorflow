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
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
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
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/uniform_quantized_types.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/run_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

#define DEBUG_TYPE "quantize-composite-functions"

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_QUANTIZECOMPOSITEFUNCTIONSPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

using QuantMethod = tensorflow::quantization::QuantizationMethod::PresetMethod;
using ::mlir::stablehlo::AddOp;
using ::mlir::stablehlo::ConvolutionOp;
using ::mlir::stablehlo::DotGeneralOp;
using ::mlir::stablehlo::DynamicBroadcastInDimOp;
using ::mlir::stablehlo::UniformQuantizeOp;
using ::tensorflow::quantization::RunPassesOnModuleOp;

constexpr StringRef kCompositeFuncPrefix = "composite_";
constexpr StringRef kQuantizedFuncPrefix = "quantized_";
constexpr StringRef kEntryFuncAttrName = "_entry_function";

class QuantizeCompositeFunctionsPass
    : public impl::QuantizeCompositeFunctionsPassBase<
          QuantizeCompositeFunctionsPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizeCompositeFunctionsPass)

  using impl::QuantizeCompositeFunctionsPassBase<
      QuantizeCompositeFunctionsPass>::QuantizeCompositeFunctionsPassBase;

 private:
  void runOnOperation() override;
};

// Returns true if `type` is a TensorType with quantized elements.
bool IsQuantizedTensorType(const Type type) {
  return type.isa<TensorType>() &&
         type.cast<TensorType>().getElementType().isa<QuantizedType>();
}

// Returns true if an op has adjacent bias or activation that can be fused
// together into the quantization function.
// TODO: b/307620428 - Consider using matchAndRewrite to check and apply
// patterns at the same time. Also add check for fusible activation or
// fusible patterns with dynamic shape.
bool HasFusibleQuantizationPattern(Operation& op) {
  if (isa<AddOp>(op.getNextNode())) {
    return true;
  }
  return false;
}

// Returns dynamically broadcasted user op of an input op. Returns null if
// the op is used multiple times or the user op is not dynamically broadcasted.
// Dynamic shapes usually has the following pattern. In the example below,
// the input operand would be stablehlo.gemm_style op, and return value would
// be stablehlo.add op.
//
// ```
// %2 = stablehlo.gemm_style(%0, %1)
// %3 = shape.shape_of %2
// %4 = stablehlo.dynamic_broadcast_in_dims %cst, %3
// %5 = stablehlo.add %2, %4
// ```
Operation* GetDynamicallyBroadcastedUserOp(Operation& op) {
  if (!op.hasOneUse()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Target op is used multiple times and will not be checked "
                  "for dynamic shape case.\n");
    return nullptr;
  }
  Operation& shapeof_op = *op.getNextNode();
  if (!isa<shape::ShapeOfOp>(shapeof_op)) {
    return nullptr;
  }
  Operation& broadcast_in_dims_op = *shapeof_op.getNextNode();
  if (!isa<DynamicBroadcastInDimOp>(broadcast_in_dims_op)) {
    return nullptr;
  }
  return broadcast_in_dims_op.getNextNode();
}

// Checks if all inputs and outputs are quantized.
bool HasQuantizedOperandOrOutput(Operation& call_op) {
  SmallVector<Type> arg_types;
  for (const Value arg : call_op.getOperands()) {
    arg_types.push_back(arg.getType());
  }

  SmallVector<Type> output_types;
  for (const Value output : call_op.getResults()) {
    output_types.push_back(output.getType());
  }

  return absl::c_all_of(arg_types, IsQuantizedTensorType) &&
         absl::c_all_of(output_types, IsQuantizedTensorType);
}

// Get the corresponding quantized function name from the given function name.
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
  return HasQuantizedOperandOrOutput(*xla_call_module_op) &&
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
  // function must have input, filter, and optionally bias.
  auto& operations = entry_func_op.getBody().front().getOperations();
  if (operations.size() != 2 && operations.size() != 3) {
    return failure();
  }
  if (!isa<GemmStyleOp>(operations.front())) {
    return failure();
  } else if (GetDynamicallyBroadcastedUserOp(operations.front())) {
    LLVM_DEBUG(llvm::dbgs()
               << "Currently gemm style ops quantization only supports static "
                  " shapes.\n");
    return failure();
  } else if (!isa<RankedTensorType>(
                 operations.front().getResult(0).getType())) {
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

  Operation& next_op = *gemm_style_op->getNextNode();
  // If an op is used multiple times, do not apply quantization of fused
  // patterns to prevent removal of dependee ops.
  const bool should_quantize_without_fusion =
      HasFusibleQuantizationPattern(*gemm_style_op.getOperation()) &&
      !gemm_style_op->hasOneUse();

  // TODO: b/307620428 - Add support for dynamic shapes.
  if (should_quantize_without_fusion || !isa<AddOp>(next_op)) {
    // no bias
    CreateAndReturnUniformQuantizeOp(rewriter, *gemm_style_op, entry_func_op,
                                     func_result_type);
    return;
  }
  // bias fusion
  Value bias_op = next_op.getOperand(1);
  Value add_op_result = next_op.getResult(0);
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

void QuantizeCompositeFunctionsPass::runOnOperation() {
  MLIRContext& ctx = getContext();

  QuantizationSpecs quant_specs;
  quant_specs.inference_type = tensorflow::DT_QINT8;

  PassManager pm(&ctx);
  // Intermediate output from QuantizePass will have quantized ops
  // (XlaCallModuleOps) with quantized input and output types, which are not
  // allowed in the TF dialect.
  pm.enableVerifier(false);

  pm.addNestedPass<func::FuncOp>(CreatePrepareQuantizePass());
  pm.addNestedPass<func::FuncOp>(CreateQuantizePass(quant_specs));
  pm.addNestedPass<func::FuncOp>(createPostQuantizePass());

  ModuleOp module_op = getOperation();
  if (const absl::Status pm_run_status =
          RunPassesOnModuleOp(mlir_dump_file_name_, pm, module_op);
      !pm_run_status.ok()) {
    signalPassFailure();
  }

  // TODO - b/307839649: Move this as a separate pass.
  RewritePatternSet patterns(&ctx);
  patterns.add<XlaCallModuleOpToCallOp<QuantizeDotGeneralOpPattern>,
               XlaCallModuleOpToCallOp<QuantizeConvolutionOpPattern>>(ctx);

  if (failed(applyPatternsAndFoldGreedily(module_op, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

}  // namespace mlir::quant::stablehlo
