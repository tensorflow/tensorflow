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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
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

LogicalResult convert_graph_uint8_tensor(mlir::MLIRContext &context,
                                         mlir::func::FuncOp &function) {
  size_t num_blocks_in_main = 0;
  mlir::Region *region = function.getCallableRegion();
  OpBuilder builder(&context);

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

    // Insert rescale uint8->int8 after placeholders.
    for (Value arg : bb.getArguments()) {
      auto uint8_type = dyn_cast<mlir::ShapedType>(arg.getType());
      if (!uint8_type) continue;

      auto uint8_element_type = dyn_cast<mlir::quant::UniformQuantizedType>(
          uint8_type.getElementType());
      if (!uint8_element_type) continue;

      if (uint8_element_type.isSigned() ||
          uint8_element_type.getStorageTypeIntegralWidth() != 8)
        continue;

      double type_range_min =
          static_cast<double>(uint8_element_type.getStorageTypeMin() -
                              uint8_element_type.getZeroPoint()) *
          uint8_element_type.getScale();
      double type_range_max =
          static_cast<double>(uint8_element_type.getStorageTypeMax() -
                              uint8_element_type.getZeroPoint()) *
          uint8_element_type.getScale();
      bool narrow_range =
          uint8_element_type.getStorageTypeMin() == 1 ? true : false;

      Type int8_type = uint8_type.clone(buildQTypeFromMinMax(
          builder, uint8_element_type.getExpressedType(),
          builder.getF64FloatAttr(type_range_min),
          builder.getF64FloatAttr(type_range_max),
          builder.getI32IntegerAttr(
              uint8_element_type.getStorageTypeIntegralWidth()),
          0, true /* signed */, builder.getBoolAttr(narrow_range)));

      int32_t uint8_zp = uint8_element_type.getZeroPoint();
      int32_t int8_zp = uint8_zp - 128;

      // Keep original input_val use with tmp_val.
      Value tmp_val = builder.create<TFL::ConstOp>(
          function.getLoc(), tmp_const_type, tmp_const_attr);
      arg.replaceAllUsesWith(tmp_val);
      auto rescale_op = builder.create<tosa::RescaleOp>(
          function.getLoc(), int8_type, arg,
          builder.getI32IntegerAttr(uint8_zp),
          builder.getI32IntegerAttr(int8_zp),
          builder.getDenseI32ArrayAttr({1 << 30}),
          builder.getDenseI8ArrayAttr({30}), builder.getBoolAttr(true),
          builder.getBoolAttr(false), builder.getBoolAttr(false));

      Operation *op_rescale_op = static_cast<Operation *>(rescale_op);
      bb.push_front(op_rescale_op);
      tmp_val.replaceAllUsesWith(rescale_op.getResult());
      tmp_val.getDefiningOp()->erase();
    }

    // Record types of original graph output before we convert intermediate
    // tensor.
    auto terminator = bb.getTerminator();
    SmallVector<Type, 4> output_types;
    for (Value val : terminator->getOperands()) {
      output_types.push_back(val.getType());
    }

    // Convert intermediate tensor.
    for (auto &op : bb) {
      for (Value output_val : op.getResults()) {
        // Skip if output value is not RankedTensorType.
        auto output_type = dyn_cast<mlir::ShapedType>(output_val.getType());
        if (!output_type) continue;

        // Skip if output value is not per-tensor quantized element type.
        auto output_element_type = dyn_cast<mlir::quant::UniformQuantizedType>(
            output_type.getElementType());
        if (!output_element_type) continue;

        // Skip if output is not uint8.
        if (output_element_type.isSigned() ||
            output_element_type.getStorageTypeIntegralWidth() != 8)
          continue;

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

        Type new_type = output_type.clone(buildQTypeFromMinMax(
            builder, output_element_type.getExpressedType(),
            builder.getF64FloatAttr(type_range_min),
            builder.getF64FloatAttr(type_range_max),
            builder.getI32IntegerAttr(
                output_element_type.getStorageTypeIntegralWidth()),
            0, true /* signed */, builder.getBoolAttr(narrow_range)));

        output_val.setType(new_type);
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

      auto uint8_output_element_type =
          dyn_cast<mlir::quant::UniformQuantizedType>(
              uint8_output_type.getElementType());
      if (!uint8_output_element_type) continue;

      if (uint8_output_element_type.isSigned() ||
          uint8_output_element_type.getStorageTypeIntegralWidth() != 8)
        continue;

      // Check if output coming into terminator is int8 type.
      auto int8_output_type =
          dyn_cast<mlir::ShapedType>(terminator->getOperand(i).getType());
      if (!int8_output_type) continue;

      auto int8_output_element_type =
          dyn_cast<mlir::quant::UniformQuantizedType>(
              int8_output_type.getElementType());
      if (!int8_output_element_type) continue;

      if (!int8_output_element_type.isSigned() ||
          int8_output_element_type.getStorageTypeIntegralWidth() != 8)
        continue;

      int32_t int8_zp = int8_output_element_type.getZeroPoint();
      int32_t uint8_zp = uint8_output_element_type.getZeroPoint();

      // Sanity check if uint8/int8's scale and zeropoint match.
      if (((uint8_zp - int8_zp) != 128) ||
          (int8_output_element_type.getScale() !=
           uint8_output_element_type.getScale())) {
        return terminator->emitError(
            "convert_uint8_to_int8: scale mismatch at the output tensors");
      }

      // Keep original input_val use with tmp_val.
      Value tmp_val = builder.create<TFL::ConstOp>(
          function.getLoc(), tmp_const_type, tmp_const_attr);
      input_val.replaceAllUsesWith(tmp_val);
      auto rescale_op = builder.create<tosa::RescaleOp>(
          function.getLoc(), uint8_output_type, input_val,
          builder.getI32IntegerAttr(int8_zp),
          builder.getI32IntegerAttr(uint8_zp),
          builder.getDenseI32ArrayAttr({1 << 30}),
          builder.getDenseI8ArrayAttr({30}), builder.getBoolAttr(true),
          builder.getBoolAttr(false), builder.getBoolAttr(false));

      Operation *op_rescale_op = static_cast<Operation *>(rescale_op);
      bb.push_back(op_rescale_op);
      op_rescale_op->moveBefore(terminator);
      tmp_val.replaceAllUsesWith(rescale_op.getResult());
      tmp_val.getDefiningOp()->erase();
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
