/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/iota.h"

#include <cstdint>
#include <tuple>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/op_util_common.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {
namespace {

class LegalizeIota : public OpConversionPattern<mhlo::IotaOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::IotaOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};

std::tuple<DenseElementsAttr, DenseElementsAttr, DenseElementsAttr>
BuildRangeParams(Type e_type, int64_t iota_dim_size, OpBuilder& b) {
  if (e_type.isInteger()) {
    return std::tuple(BuildScalarDense<int>(e_type, 0),
                      BuildScalarDense<int>(e_type, iota_dim_size),
                      BuildScalarDense<int>(e_type, 1));
  }
  return std::tuple(BuildScalarDense<float>(e_type, 0.0),
                    BuildScalarDense<float>(e_type, iota_dim_size),
                    BuildScalarDense<float>(e_type, 1.0));
}

LogicalResult LegalizeIota::matchAndRewrite(
    mhlo::IotaOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto type = llvm::cast<ShapedType>(op.getType());
  if (!type.getElementType().isIntOrFloat()) {
    return rewriter.notifyMatchFailure(op, "Must be int or float");
  }

  auto e_type = type.getElementType();
  const int64_t iota_dim_size = type.getDimSize(op.getIotaDimension());

  auto [start, limit, delta] =
      BuildRangeParams(e_type, iota_dim_size, rewriter);

  auto start_op = rewriter.create<arith::ConstantOp>(op->getLoc(), start);
  auto limit_op = rewriter.create<arith::ConstantOp>(op->getLoc(), limit);
  auto delta_op = rewriter.create<arith::ConstantOp>(op->getLoc(), delta);

  auto range_type = RankedTensorType::get({iota_dim_size}, e_type);
  auto range_op = rewriter.create<TFL::RangeOp>(op->getLoc(), range_type,
                                                start_op, limit_op, delta_op);

  if (type.getRank() == 1) {
    rewriter.replaceOp(op, range_op);
    return success();
  }

  // mhlo.iota allows filling ND tensors iota-style. Reshape and broadcast
  // tfl 1D range output.

  llvm::SmallVector<int64_t> reshape_shape(type.getRank(), 1);
  reshape_shape[op.getIotaDimension()] = iota_dim_size;
  Value reshape_shape_cst = rewriter.create<arith::ConstantOp>(
      op->getLoc(), rewriter.getI64TensorAttr(reshape_shape));
  reshape_shape_cst = rewriter.create<TFL::CastOp>(
      op->getLoc(),
      llvm::cast<ShapedType>(reshape_shape_cst.getType())
          .clone(rewriter.getI32Type()),
      reshape_shape_cst);

  auto reshape_type = RankedTensorType::get(reshape_shape, e_type);
  auto reshape_op = rewriter.create<TFL::ReshapeOp>(
      op->getLoc(), reshape_type, range_op, reshape_shape_cst);

  auto broad_cast_shape_cst = rewriter.create<arith::ConstantOp>(
      op->getLoc(), rewriter.getI64TensorAttr(type.getShape()));

  rewriter.replaceOpWithNewOp<TFL::BroadcastToOp>(op, type, reshape_op,
                                                  broad_cast_shape_cst);

  return success();
}

}  // namespace

void PopulateIotaPatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                          ConversionTarget& target) {
  patterns.add<LegalizeIota>(ctx);
  target.addIllegalOp<mhlo::IotaOp>();
}

}  // namespace mlir::odml
