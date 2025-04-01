/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This pass converts a TFLite uint8 graph to the int8 domain, with adaptors at
// input and output tensors. This is needed because TOSA precision is
// implemented in the int8 domain. This pass does:
// 1. match TFL::QConst with uint8, generate TFL::QConst with int8 with value
// remapped.
// 2. insert tosa.RESCALE uint8 -> int8 if block argument (placeholder of graph)
// is uint8 typed.
// 3. insert tosa.RESCALE int8 -> uint8 if original returned tensor is uint8
// typed.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"                // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"                // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"          // from @llvm-project
#include "mlir/IR/Builders.h"                            // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"                   // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                        // from @llvm-project
#include "mlir/IR/PatternMatch.h"                        // from @llvm-project
#include "mlir/Pass/PassRegistry.h"                      // from @llvm-project
#include "mlir/Support/LLVM.h"                           // from @llvm-project
#include "mlir/Support/LogicalResult.h"                  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_common.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_utils.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"

#define PASS_NAME "tosa-convert-tfl-uint8"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace tosa {
namespace {

#define GEN_PASS_DEF_TOSACONVERTTFLUINT8PASS
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

// Performs lowering to TOSA dialect.
class ConvertUint8ToInt8
    : public impl::TosaConvertTFLUint8PassBase<ConvertUint8ToInt8> {
 public:
  explicit ConvertUint8ToInt8() = default;
  void runOnOperation() override;
};

struct ConvertUint8QConstOp : public RewritePattern {
  explicit ConvertUint8QConstOp(MLIRContext *context)
      : RewritePattern(TFL::QConstOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &builder) const override {
    auto tfl_qconst_op = cast<TFL::QConstOp>(op);

    // Skip if it's not ranked tensor type.
    auto output_type =
        dyn_cast<mlir::RankedTensorType>(tfl_qconst_op.getResult().getType());
    if (!output_type)
      return builder.notifyMatchFailure(op, "not ranked tensor");

    // Skip if output is not per-tensor quantized type.
    auto output_element_type = dyn_cast<mlir::quant::UniformQuantizedType>(
        output_type.getElementType());
    if (!output_element_type) return failure();

    // Skip if output is not uint8.
    if (output_element_type.isSigned() ||
        output_element_type.getStorageTypeIntegralWidth() != 8) {
      return failure();
    }

    mlir::DenseElementsAttr src_dense_attr =
        mlir::cast<DenseElementsAttr>(tfl_qconst_op.getValue());

    double type_range_min =
        static_cast<double>(output_element_type.getStorageTypeMin() -
                            output_element_type.getZeroPoint()) *
        output_element_type.getScale();
    double type_range_max =
        static_cast<double>(output_element_type.getStorageTypeMax() -
                            output_element_type.getZeroPoint()) *
        output_element_type.getScale();
    bool narrow_range =
        output_element_type.getStorageTypeMin() == 1 ? true : false;

    auto dst_qconst_type = TypeAttr::get(RankedTensorType::get(
        output_type.getShape(),
        buildQTypeFromMinMax(
            builder, output_element_type.getExpressedType(),
            builder.getF64FloatAttr(type_range_min),
            builder.getF64FloatAttr(type_range_max),
            builder.getI32IntegerAttr(
                output_element_type.getStorageTypeIntegralWidth()),
            0, true /* signed */, builder.getBoolAttr(narrow_range))));

    Type dst_dense_element_type = builder.getIntegerType(8);

    auto dst_dense_attr = src_dense_attr.mapValues(
        dst_dense_element_type, [](const APInt &in) -> APInt {
          int64_t in_i64 = in.getLimitedValue();
          int64_t out_i64 = in_i64 - 128;
          return APInt(8, out_i64, true);
        });

    builder.replaceOpWithNewOp<TFL::QConstOp>(op, dst_qconst_type,
                                              dst_dense_attr);

    return success();
  }
};

namespace {

// returns true iff @a type is a shaped type with element type that is uint8
// if it is, then return the rescaled type, uint8_zp, and output_zp to use to
// rescale type to signed type with adjusted zero point.
bool IsShapedUint8Type(OpBuilder &builder, const Type type, Type &rescaled_type,
                       int32_t &uint8_zp, int32_t &output_zp) {
  auto uint8_type = dyn_cast<mlir::ShapedType>(type);
  if (!uint8_type) return false;

  auto element_type = uint8_type.getElementType();
  auto uint8_element_quant_type =
      dyn_cast<mlir::quant::UniformQuantizedType>(element_type);
  bool is_uint8_element_quant_type =
      uint8_element_quant_type && !uint8_element_quant_type.isSigned() &&
      uint8_element_quant_type.getStorageTypeIntegralWidth() == 8;
  bool is_uint8_element_type = element_type.isUnsignedInteger(8);
  if (!is_uint8_element_quant_type && !is_uint8_element_type) return false;

  // type has uint8 element type
  if (is_uint8_element_quant_type) {
    double type_range_min =
        static_cast<double>(uint8_element_quant_type.getStorageTypeMin() -
                            uint8_element_quant_type.getZeroPoint()) *
        uint8_element_quant_type.getScale();
    double type_range_max =
        static_cast<double>(uint8_element_quant_type.getStorageTypeMax() -
                            uint8_element_quant_type.getZeroPoint()) *
        uint8_element_quant_type.getScale();
    bool narrow_range =
        uint8_element_quant_type.getStorageTypeMin() == 1 ? true : false;

    rescaled_type = uint8_type.clone(buildQTypeFromMinMax(
        builder, uint8_element_quant_type.getExpressedType(),
        builder.getF64FloatAttr(type_range_min),
        builder.getF64FloatAttr(type_range_max),
        builder.getI32IntegerAttr(
            uint8_element_quant_type.getStorageTypeIntegralWidth()),
        0, true /* signed */, builder.getBoolAttr(narrow_range)));
    uint8_zp = uint8_element_quant_type.getZeroPoint();
  } else {
    // convert ui8 to i8 with zp=-128
    rescaled_type = uint8_type.clone(quant::UniformQuantizedType::getChecked(
        builder.getUnknownLoc(), quant::QuantizationFlags::Signed,
        builder.getI8Type(), builder.getF32Type(),
        /* scale = */ 1.0,
        /* zeroPoint = */ -128,
        /* storagTypeMin = */ -128,
        /* storageTypeMax = */ 127));
    uint8_zp = 0;
  }
  output_zp = uint8_zp - 128;
  return true;
}

}  // namespace

LogicalResult convert_graph_uint8_tensor(mlir::MLIRContext &context,
                                         mlir::func::FuncOp &function) {
  size_t num_blocks_in_main = 0;
  mlir::Region *region = function.getCallableRegion();
  OpBuilder builder(&context);
  auto loc = function.getLoc();

  auto tmp_const_type = RankedTensorType::get({1}, builder.getIntegerType(8));
  auto tmp_const_attr =
      DenseElementsAttr::get(tmp_const_type, {static_cast<uint8_t>(0)});

  for (mlir::Block &bb : region->getBlocks()) {
    // Always have one block for each region right now.
    num_blocks_in_main++;
    if (num_blocks_in_main > 1) {
      return function.emitError("Invalid MLIR: multiple blocks in a region");
    }

    if (!bb.isEntryBlock()) {
      return function.emitError("Invalid MLIR: block must be entry block");
    }

    auto multiplier = tosa::getConstTensorInt<int32_t>(builder, loc, {1 << 30});
    auto shift = tosa::getConstTensorInt<int8_t>(builder, loc, {30});

    // Insert rescale uint8->int8 after placeholders.
    for (Value arg : bb.getArguments()) {
      auto shaped_type = dyn_cast<ShapedType>(arg.getType());
      if (!shaped_type) continue;
      Type rescaled_type;
      int32_t rescale_input_zp_val, rescale_output_zp_val;
      if (!IsShapedUint8Type(builder, arg.getType(), rescaled_type,
                             rescale_input_zp_val, rescale_output_zp_val))
        continue;

      // Keep original input_val use with tmp_val.
      Value tmp_val =
          builder.create<TFL::ConstOp>(loc, tmp_const_type, tmp_const_attr);
      arg.replaceAllUsesWith(tmp_val);
      // mlir::quant::UniformQuantizedType uses signless storage type.
      // For example, tensor<1x!quant.uniform<u8:...>> has the same storage type
      // as tensor<1xi8>.
      auto rescale_input_zp = tosa::getConstTensorInt<int8_t>(
          builder, loc, {static_cast<int8_t>(rescale_input_zp_val)});
      auto rescale_output_zp = tosa::getConstTensorInt<int8_t>(
          builder, loc, {static_cast<int8_t>(rescale_output_zp_val)});

      auto rescale_op = builder.create<tosa::RescaleOp>(
          loc, rescaled_type, arg, multiplier, shift, rescale_input_zp,
          rescale_output_zp,
          /* scale32 = */ builder.getBoolAttr(true),
          /* rounding_mode = */ builder.getStringAttr("SINGLE_ROUND"),
          /* per_channel = */ builder.getBoolAttr(false),
          /* input_unsigned = */ builder.getBoolAttr(true),     // uint8_t ->
          /* output_unsigned = */ builder.getBoolAttr(false));  // int8_t

      Operation *op_rescale_op = static_cast<Operation *>(rescale_op);
      bb.push_front(op_rescale_op);
      tmp_val.replaceAllUsesWith(rescale_op.getResult());
      tmp_val.getDefiningOp()->erase();
      bb.push_front(rescale_output_zp.getDefiningOp());
      bb.push_front(rescale_input_zp.getDefiningOp());
    }

    bb.push_front(shift.getDefiningOp());
    bb.push_front(multiplier.getDefiningOp());

    // Record types of original graph output before we convert intermediate
    // tensor.
    auto terminator = bb.getTerminator();
    SmallVector<Type, 4> output_types;
    for (Value val : terminator->getOperands()) {
      output_types.push_back(val.getType());
    }

    // Convert intermediate tensor.
    for (auto &op : bb) {
      if (llvm::dyn_cast<tosa::ConstOp>(&op)) {
        continue;  // Skip if the operation is a tosa::ConstOp
      }

      for (Value output_val : op.getResults()) {
        auto shaped_type = dyn_cast<ShapedType>(output_val.getType());
        if (!shaped_type) continue;
        Type new_type;
        int32_t unused_input_zp, unused_output_zp;
        if (IsShapedUint8Type(builder, output_val.getType(), new_type,
                              unused_input_zp, unused_output_zp)) {
          output_val.setType(new_type);
        }
      }
    }

    if (terminator->getNumOperands() != output_types.size()) {
      return function.emitError(
          "Terminator's operand mismatch with number of outputs in graph");
    }

    // Insert int8->uint8 rescale before all terminator's operand.
    for (int32_t i = 0; i < terminator->getNumOperands(); i++) {
      auto defining_op = terminator->getOperand(i).getDefiningOp();
      // skip if operand of terminator is block arg (nullptr in this case) or
      // not
      if (!defining_op) continue;
      Value input_val = defining_op->getResult(0);

      // Check if graph output is uint8 type.
      auto uint8_output_type = dyn_cast<mlir::ShapedType>(output_types[i]);
      if (!uint8_output_type) continue;

      // Check if graph output is uint8 type.
      Type rescaled_type;
      int32_t uint8_zp_val, unused_output_zp_val;
      if (!IsShapedUint8Type(builder, output_types[i], rescaled_type,
                             uint8_zp_val, unused_output_zp_val))
        continue;

      // convert terminator operand type back to original output_type.
      auto terminator_operand_type =
          dyn_cast<mlir::ShapedType>(terminator->getOperand(i).getType());
      if (!terminator_operand_type) continue;
      int operand_zp_val = 0;
      auto quantized_type = dyn_cast<mlir::quant::UniformQuantizedType>(
          terminator_operand_type.getElementType());
      if (quantized_type) {
        operand_zp_val = quantized_type.getZeroPoint();
      }

      // Keep original input_val use with tmp_val.
      Value tmp_val =
          builder.create<TFL::ConstOp>(loc, tmp_const_type, tmp_const_attr);
      input_val.replaceUsesWithIf(tmp_val, [&terminator](OpOperand &use) {
        return use.getOwner() == terminator;
      });

      auto rescale_input_zp = tosa::getConstTensorInt<int8_t>(
          builder, loc, {static_cast<int8_t>(operand_zp_val)});
      auto rescale_output_zp = tosa::getConstTensorInt<int8_t>(
          builder, loc, {static_cast<int8_t>(uint8_zp_val)});

      auto rescale_op = builder.create<tosa::RescaleOp>(
          loc, uint8_output_type, input_val, multiplier, shift,
          rescale_input_zp, rescale_output_zp,
          /* scale32 = */ builder.getBoolAttr(true),
          /* rounding_mode = */ builder.getStringAttr("SINGLE_ROUND"),
          /* per_channel = */ builder.getBoolAttr(false),
          /* input_unsigned = */ builder.getBoolAttr(false),   // int8_t ->
          /* output_unsigned = */ builder.getBoolAttr(true));  // uint8_t

      Operation *op_rescale_op = static_cast<Operation *>(rescale_op);
      bb.push_back(op_rescale_op);
      op_rescale_op->moveBefore(terminator);
      tmp_val.replaceAllUsesWith(rescale_op.getResult());
      tmp_val.getDefiningOp()->erase();
      bb.push_front(rescale_output_zp.getDefiningOp());
      bb.push_front(rescale_input_zp.getDefiningOp());
    }
  }

  return success();
}

void ConvertUint8ToInt8::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto &ctx = getContext();
  mlir::func::FuncOp func = getOperation();

  // Convert uint8 const tensor. const needs to be handled specifically.
  patterns.add<ConvertUint8QConstOp>(&ctx);
  (void)applyPatternsGreedily(func, std::move(patterns));

  // Replace uint8 tensor in the graph and insert rescale as needed.
  (void)convert_graph_uint8_tensor(ctx, func);
}

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>> createConvertTFLUint8Pass() {
  return std::make_unique<ConvertUint8ToInt8>();
}

}  // namespace tosa

}  // namespace mlir
