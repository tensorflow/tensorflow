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
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_quantize_op.h"

#include <functional>
#include <optional>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/tf_quantize_op_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {
namespace {
constexpr StringRef kDequantizeFunctionName = "composite_dequantize";
constexpr StringRef kUniformQuantizationFunctionName = "uniform";

// Pre-actions before adding quantization logics. It creates a function with the
// func_name where input_val is an input and result_type is a result.
func::FuncOp PrepareFunctionRegister(PatternRewriter& rewriter, Value input_val,
                                     ShapedType result_type,
                                     StringRef func_name,
                                     Value& func_input_arg) {
  Operation* input_op = input_val.getDefiningOp();

  Operation* insertion_point = input_op->getParentOfType<func::FuncOp>();
  if (!insertion_point) insertion_point = input_op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPointAfter(insertion_point);

  UnrankedTensorType create_unknown_input_shape =
      CreateUnknownShapeFromElementType(input_val.getType());
  UnrankedTensorType create_unknown_output_shape =
      CreateUnknownShapeFromElementType(result_type);

  FunctionType func_type =
      FunctionType::get(rewriter.getContext(), {create_unknown_input_shape},
                        {create_unknown_output_shape});

  func::FuncOp quantization_func =
      rewriter.create<func::FuncOp>(input_op->getLoc(), func_name, func_type);

  OpBuilder::InsertionGuard guard = OpBuilder::InsertionGuard(rewriter);
  ArrayRef<Type> inputs = quantization_func.getFunctionType().getInputs();
  Block* block = rewriter.createBlock(
      &quantization_func.getBody(), quantization_func.begin(), inputs,
      SmallVector<Location>(inputs.size(), quantization_func.getLoc()));
  func_input_arg = block->getArgument(0);
  return quantization_func;
}

// Post-actions after adding quantization logics. Post-actions include
// 1) Adding the created function in the symbol table
// 2) Creating a PartitionedCallOp in the main graph that calls the created
//    function.
TF::PartitionedCallOp FinalizeFunctionRegister(
    PatternRewriter& rewriter, Value input, Value output,
    func::FuncOp& quantization_func, Operation* quantized_op,
    StringRef func_name, IRRewriter::InsertPoint original_point,
    Type quantize_result_type) {
  rewriter.create<func::ReturnOp>(input.getLoc(), ArrayRef<Value>({output}));

  quantization_func.setVisibility(func::FuncOp::Visibility::Private);
  SymbolTable symbol_table(quantized_op->getParentOfType<ModuleOp>());

  symbol_table.insert(quantization_func);

  FlatSymbolRefAttr func_name_attr =
      FlatSymbolRefAttr::get(rewriter.getStringAttr(func_name));

  rewriter.restoreInsertionPoint(original_point);

  auto quantize_call = rewriter.create<TF::PartitionedCallOp>(
      quantized_op->getLoc(), quantize_result_type, input,
      /*args_attrs=*/nullptr, /*res_attrs=*/nullptr, func_name_attr,
      /*config=*/"", /*config_proto=*/"", /*executor_type=*/"");
  return quantize_call;
}

// Acts as a register of a function where the body has a sequence of operations
// required to execute certain quantization scheme's quant/dequantization
// logics.
std::optional<TF::PartitionedCallOp> RegisterOperationsInFuncOp(
    StringRef func_name, PatternRewriter& rewriter, QuantizedType quant_type,
    Value input_val, ShapedType result_type,
    std::function<Operation*(PatternRewriter&, Operation*, Value, ShapedType,
                             QuantizedType)>
        quantization_operations_func) {
  Operation* input_op = input_val.getDefiningOp();
  auto original_point = rewriter.saveInsertionPoint();

  auto unique_func_name = func_name.str();
  SymbolTable symbol_table(input_op->getParentOfType<ModuleOp>());
  while (symbol_table.lookup(unique_func_name)) {
    absl::StrAppend(&unique_func_name, "_");
  }

  Value func_input_arg;
  // Creates a function.
  func::FuncOp func_op = PrepareFunctionRegister(
      rewriter, input_val, result_type, unique_func_name, func_input_arg);

  // Fills the body.
  Operation* last_op_in_func =
      quantization_operations_func(rewriter, func_op.getOperation(),
                                   func_input_arg, result_type, quant_type);

  // Connect the function in the existing graph.
  auto end_call_op = FinalizeFunctionRegister(
      rewriter, input_val, last_op_in_func->getResult(0), func_op, input_op,
      unique_func_name, original_point, result_type);
  return end_call_op;
}

QuantizedType CalculateUniformQuantParams(
    PatternRewriter& rewriter, TF::ConstOp op,
    tensorflow::quantization::QuantizationComponentSpec& weight_spec) {
  // TODO - b/278949920: Enable Per-Channel Quantization for XLA Opset
  // Currently, support symmetric, per-tensor, signed int8
  const bool kIsNarrowRange = true;
  const bool kIsSigned = true;
  const int kBitWidth = 8;

  DenseFPElementsAttr attr;
  if (!matchPattern(op->getResult(0), m_Constant(&attr))) return nullptr;

  QuantizedType quant_type = mlir::dyn_cast<quant::QuantizedType>(
      quant::GetUniformQuantizedTypeForWeight(
          attr, /*symmetric=*/kIsNarrowRange && kIsSigned, kBitWidth, kIsSigned,
          kIsNarrowRange, /*is_legacy_float*/ false));

  return quant_type;
}

// Add uniform quantization's quantization logic.
std::optional<Value> AddUniformQuantizeOps(PatternRewriter& rewriter,
                                           TF::ConstOp op,
                                           QuantizedType quant_type) {
  DenseFPElementsAttr attr;
  if (!matchPattern(op->getResult(0), m_Constant(&attr))) {
    return nullptr;
  }
  Type expressed_type = op.getResult().getType();
  Type quantized_type = quant_type.castFromExpressedType(expressed_type);
  ShapedType shaped_quantized_type = mlir::cast<ShapedType>(quantized_type);
  DenseElementsAttr tensor_proto_attr =
      mlir::dyn_cast<DenseElementsAttr>(Quantize(attr, shaped_quantized_type));
  if (!tensor_proto_attr) {
    return nullptr;
  }

  Type storage_type =
      mlir::cast<QuantizedType>(shaped_quantized_type.getElementType())
          .getStorageType();
  ShapedType new_type = shaped_quantized_type.clone(storage_type);

  rewriter.setInsertionPointAfter(op);
  auto const_op =
      rewriter.create<TF::ConstOp>(op.getLoc(), new_type, tensor_proto_attr);
  auto new_identity_op = rewriter.create<TF::IdentityOp>(
      op->getLoc(), const_op.getType(), const_op);
  return new_identity_op.getResult();
}

Operation* LogicsForUniformDequanization(PatternRewriter& rewriter,
                                         Operation* func_op, Value input_val,
                                         ShapedType original_input_tensor_type,
                                         QuantizedType quant_type) {
  auto loc = input_val.getLoc();
  rewriter.setInsertionPointToStart(
      &(cast<func::FuncOp>(func_op)).getBody().front());

  UnrankedTensorType create_unknown_input_shape =
      CreateUnknownShapeFromElementType(original_input_tensor_type);
  auto new_cast_op =
      rewriter.create<TF::CastOp>(loc, create_unknown_input_shape, input_val);
  // TODO - b/278949920: Enable Per-Channel Quantization for XLA Opset
  auto qtype = mlir::dyn_cast<UniformQuantizedType>(quant_type);
  TensorType scale_type = RankedTensorType::get({}, rewriter.getF32Type());
  Value scale_op = rewriter.create<TF::ConstOp>(
      loc, scale_type,
      DenseFPElementsAttr::get(scale_type,
                               {static_cast<float>(qtype.getScale())}));

  if (original_input_tensor_type.getElementType().isBF16()) {
    // Add bf16 cast op after scale to match with the next op's data
    // type.
    scale_op = rewriter.create<TF::CastOp>(
        loc, UnrankedTensorType::get(rewriter.getBF16Type()), scale_op);
  }

  auto mul_op = rewriter.create<TF::MulOp>(loc, new_cast_op.getType(), scale_op,
                                           new_cast_op);
  return mul_op;
}

// Add uniform quantization's dequantization logic.
std::optional<TF::PartitionedCallOp> AddUniformDequantizeOps(
    PatternRewriter& rewriter, QuantizedType quant_type,
    Value val_to_dequantize, ShapedType result_type) {
  auto func_name = absl::StrJoin(
      {kDequantizeFunctionName, kUniformQuantizationFunctionName}, "_");

  std::optional<TF::PartitionedCallOp> dequant_op = RegisterOperationsInFuncOp(
      func_name, rewriter, quant_type, val_to_dequantize, result_type,
      LogicsForUniformDequanization);

  return dequant_op;
}
}  // namespace

// Generate quantize and dequantize functions with uniform quantization.
std::optional<TF::PartitionedCallOp> ApplyUniformQuantization(
    PatternRewriter& rewriter, TF::ConstOp op,
    tensorflow::quantization::QuantizationComponentSpec& weight_spec) {
  QuantizedType quant_type =
      CalculateUniformQuantParams(rewriter, op, weight_spec);
  if (!quant_type) return nullptr;

  std::optional<Value> quantized_val =
      AddUniformQuantizeOps(rewriter, op, quant_type);
  if (!quantized_val.has_value()) return std::nullopt;

  std::optional<TF::PartitionedCallOp> dequantized_val =
      AddUniformDequantizeOps(rewriter, quant_type, quantized_val.value(),
                              mlir::cast<ShapedType>(op.getType()));

  return dequantized_val;
}

}  // namespace quant
}  // namespace mlir
