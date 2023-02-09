/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/utils.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/rewriters.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace mlir {
namespace mhlo {
namespace {

#define GEN_PASS_DEF_CONVERTMHLOQUANTTOINT
#include "tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf_passes.h.inc"

FailureOr<TensorType> GetSameShapeTensorType(Operation *op,
                                             TensorType tensor_type,
                                             Type element_type,
                                             PatternRewriter &rewriter) {
  if (auto ranked_ty = tensor_type.dyn_cast_or_null<RankedTensorType>()) {
    Attribute encoding = ranked_ty.getEncoding();
    if (!(!encoding || encoding.isa<TypeExtensionsAttr>() ||
          encoding.isa<sparse_tensor::SparseTensorEncodingAttr>())) {
      return rewriter.notifyMatchFailure(
          op,
          "Ranked tensor encoding must be either null, TypeExtensionsAttr, or "
          "SparseTensorEncodingAttr.");
    }
    return RankedTensorType::get(ranked_ty.getShape(), element_type, encoding);
  }
  if (auto unranked_ty = tensor_type.dyn_cast_or_null<UnrankedTensorType>()) {
    return UnrankedTensorType::get(element_type);
  }
  llvm_unreachable("unhandled type");
}

class ConvertMHLOQuantToInt
    : public impl::ConvertMHLOQuantToIntBase<ConvertMHLOQuantToInt> {
 public:
  // Performs conversion of MHLO quant ops to primitive ops.
  void runOnOperation() override;
};

class ConvertUniformQuantizeOp
    : public OpConversionPattern<mhlo::UniformQuantizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::UniformQuantizeOp op, UniformQuantizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto element_type = getElementTypeOrSelf(op.getResult().getType())
                            .dyn_cast<quant::UniformQuantizedType>();
    // Currently for activation, PTQ supports per-tensor quantization only, and
    // UniformQuantize op is only for activation.
    if (!element_type) {
      return rewriter.notifyMatchFailure(
          op, "Legalization supports only per-tensor quantization.");
    }
    Value scale = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getF32FloatAttr(element_type.getScale()));
    Value zero_point = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(
                          static_cast<int32_t>(element_type.getZeroPoint())));
    Value half = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getF32FloatAttr(0.5f));
    Value quantization_min = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(
                          element_type.getStorageTypeMin())));
    Value quantization_max = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(
                          element_type.getStorageTypeMax())));

    auto scalar_broadcast_dims = GetI64ElementsAttr({}, &rewriter);
    auto res_float_tensor_type_or =
        GetSameShapeTensorType(op, op.getOperand().getType().cast<TensorType>(),
                               rewriter.getF32Type(), rewriter);
    if (failed(res_float_tensor_type_or)) {
      return failure();
    }
    Value res_float = rewriter.create<chlo::BroadcastDivOp>(
        op->getLoc(), *res_float_tensor_type_or, adaptor.getOperand(), scale,
        scalar_broadcast_dims);
    res_float = rewriter.create<chlo::BroadcastAddOp>(
        op->getLoc(), *res_float_tensor_type_or, res_float, half,
        scalar_broadcast_dims);
    res_float = rewriter.create<mhlo::FloorOp>(op->getLoc(), res_float);
    auto res_int32_tensor_type_or =
        GetSameShapeTensorType(op, res_float.getType().cast<TensorType>(),
                               rewriter.getI32Type(), rewriter);
    if (failed(res_int32_tensor_type_or)) {
      return failure();
    }
    Value res_int32 = rewriter.create<mhlo::ConvertOp>(
        op->getLoc(), *res_int32_tensor_type_or, res_float);
    res_int32 = rewriter.create<chlo::BroadcastAddOp>(
        op->getLoc(), *res_int32_tensor_type_or, res_int32, zero_point,
        scalar_broadcast_dims);
    res_int32 = rewriter.create<chlo::BroadcastMaxOp>(
        op->getLoc(), *res_int32_tensor_type_or, res_int32, quantization_min,
        scalar_broadcast_dims);
    res_int32 = rewriter.create<chlo::BroadcastMinOp>(
        op->getLoc(), *res_int32_tensor_type_or, res_int32, quantization_max,
        scalar_broadcast_dims);
    auto res_final_tensor_type_or =
        GetSameShapeTensorType(op, res_int32.getType().cast<TensorType>(),
                               rewriter.getI8Type(), rewriter);
    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(op, *res_final_tensor_type_or,
                                                 res_int32);
    return success();
  }
};

class ConvertUniformDequantizeOp
    : public OpConversionPattern<mhlo::UniformDequantizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::UniformDequantizeOp op, UniformDequantizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto element_type = getElementTypeOrSelf(op.getOperand().getType())
                            .dyn_cast<quant::UniformQuantizedType>();
    // Currently for activation, PTQ supports per-tensor quantization only, and
    // UniformQuantize op is only for activation.
    if (!element_type) {
      return rewriter.notifyMatchFailure(
          op, "Legalization supports only per-tensor quantization.");
    }
    Value scale = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getF32FloatAttr(element_type.getScale()));
    Value zero_point = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(
                          static_cast<int32_t>(element_type.getZeroPoint())));

    Value input = adaptor.getOperand();
    auto scalar_broadcast_dims = GetI64ElementsAttr({}, &rewriter);
    auto res_int32_tensor_type_or =
        GetSameShapeTensorType(op, input.getType().cast<TensorType>(),
                               rewriter.getI32Type(), rewriter);
    if (failed(res_int32_tensor_type_or)) {
      return failure();
    }
    Value res_int32 = rewriter.create<mhlo::ConvertOp>(
        op->getLoc(), *res_int32_tensor_type_or, input);
    res_int32 = rewriter.create<chlo::BroadcastSubOp>(
        op->getLoc(), *res_int32_tensor_type_or, res_int32, zero_point,
        scalar_broadcast_dims);
    auto res_float_tensor_type_or =
        GetSameShapeTensorType(op, res_int32.getType().cast<TensorType>(),
                               rewriter.getF32Type(), rewriter);
    if (failed(res_float_tensor_type_or)) {
      return failure();
    }
    Value res_float = rewriter.create<mhlo::ConvertOp>(
        op->getLoc(), *res_float_tensor_type_or, res_int32);
    res_float = rewriter.replaceOpWithNewOp<chlo::BroadcastMulOp>(
        op, *res_float_tensor_type_or, res_float, scale, scalar_broadcast_dims);
    return success();
  }
};

// Performs conversion of MHLO quant ops to primitive ops.
void ConvertMHLOQuantToInt::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *context = op->getContext();
  RewritePatternSet patterns(context);

  // Populate MHLO quant ops conversion patterns.
  patterns.add<ConvertUniformQuantizeOp, ConvertUniformDequantizeOp>(context);

  ConversionTarget target(*op->getContext());
  auto is_legal = [](Operation *op) {
    auto is_not_quant = [](Type type) {
      return !getElementTypeOrSelf(type).isa<quant::UniformQuantizedType>();
    };
    return llvm::all_of(op->getOperandTypes(), is_not_quant) &&
           llvm::all_of(op->getResultTypes(), is_not_quant);
  };
  target.addDynamicallyLegalDialect<MhloDialect>(is_legal);
  target.addDynamicallyLegalDialect<chlo::ChloDialect>(is_legal);

  LogicalResult result =
      applyPartialConversion(op, target, std::move(patterns));
  if (failed(result)) {
    signalPassFailure();
  }
}

}  // end namespace

std::unique_ptr<OperationPass<func::FuncOp>> createConvertMHLOQuantToIntPass() {
  return std::make_unique<ConvertMHLOQuantToInt>();
}

}  // end namespace mhlo
}  // end namespace mlir
