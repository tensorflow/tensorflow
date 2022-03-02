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

// This file implements logic for lowering HLO dialect to LHLO dialect.

#include <functional>
#include <memory>
#include <utility>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir-hlo/Dialect/mhlo/transforms/type_conversion.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {
namespace {

struct HloToMemrefReshapeUnrankedConverter
    : public OpConversionPattern<mhlo::ReshapeOp> {
  HloToMemrefReshapeUnrankedConverter(TypeConverter& type_converter,
                                      MLIRContext* ctx)
      : OpConversionPattern<mhlo::ReshapeOp>(type_converter, ctx) {}

  LogicalResult matchAndRewrite(
      mhlo::ReshapeOp op, mhlo::ReshapeOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto unranked_operand_type =
        adaptor.operand().getType().dyn_cast<UnrankedMemRefType>();
    if (unranked_operand_type == nullptr) return failure();
    auto result_type = op.getType().cast<RankedTensorType>();

    rewriter.replaceOpWithNewOp<memref::CastOp>(
        op,
        MemRefType::get(result_type.getShape(), result_type.getElementType()),
        adaptor.operand());
    return success();
  }
};

struct HloToMemrefDynamicReshapeConverter
    : public OpConversionPattern<mhlo::DynamicReshapeOp> {
  HloToMemrefDynamicReshapeConverter(TypeConverter& type_converter,
                                     MLIRContext* ctx)
      : OpConversionPattern<mhlo::DynamicReshapeOp>(type_converter, ctx) {}

  LogicalResult matchAndRewrite(
      mhlo::DynamicReshapeOp op, mhlo::DynamicReshapeOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Type op_result_type = op.getType();

    ShapedType result_type;
    if (auto ranked_type = op_result_type.dyn_cast<RankedTensorType>()) {
      result_type =
          MemRefType::get(ranked_type.getShape(), ranked_type.getElementType());
    } else if (auto unranked_type =
                   op_result_type.dyn_cast<UnrankedTensorType>()) {
      result_type = UnrankedMemRefType::get(unranked_type.getElementType(), 0);
    } else {
      return failure();
    }
    rewriter.replaceOpWithNewOp<memref::ReshapeOp>(
        op, result_type, adaptor.operand(), adaptor.output_shape());
    return success();
  }
};

// TODO(b/175670649) Fix this to no longer access original tensor operands.
struct HloToMemrefDynamicBroadcastInDimOpConverter
    : public OpConversionPattern<mhlo::DynamicBroadcastInDimOp> {
  HloToMemrefDynamicBroadcastInDimOpConverter(
      TypeConverter& converter, MLIRContext* ctx,
      std::function<bool(Operation*)> enforce_identity_maps)
      : OpConversionPattern<mhlo::DynamicBroadcastInDimOp>(converter, ctx),
        enforce_identity_maps(std::move(enforce_identity_maps)) {}

  LogicalResult matchAndRewrite(
      mhlo::DynamicBroadcastInDimOp op,
      mhlo::DynamicBroadcastInDimOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto result_type = op.getType().dyn_cast<RankedTensorType>();
    if (!result_type) return failure();

    Value result =
        InsertDynamicMemrefCastOp(op, adaptor.getOperands().front(), &rewriter);

    if (enforce_identity_maps(op)) {
      result = CreateCopy(op, result, &rewriter);
    }
    rewriter.replaceOp(op, {result});
    return success();
  }

 private:
  // Inserts dynamic memref to change the layout of the memref to put 0-stride
  // and size of the target dimension if size-1 dimension expansion is
  // necessary.
  memref::ReinterpretCastOp InsertDynamicMemrefCastOp(
      mhlo::DynamicBroadcastInDimOp op, Value operand, OpBuilder* b) const {
    auto loc = op.getLoc();
    auto operand_type = operand.getType().cast<MemRefType>();
    auto operand_shape = operand_type.getShape();
    auto operand_rank = operand_type.getRank();

    auto result_type = op.getType().cast<RankedTensorType>();
    auto result_rank = result_type.getRank();

    Value zero = b->create<arith::ConstantIndexOp>(loc, 0);
    Value one = b->create<arith::ConstantIndexOp>(loc, 1);

    // Compute a reversed scan product. Compute the stride for the dimensions so
    // far, working from minor to major dimensions. Additionally, save the
    // operand shape Values to use in the next loop.
    SmallVector<Value, 2> operand_strides(operand_rank, one);
    SmallVector<Value, 2> operand_sizes(operand_rank, one);
    Value stride_so_far = one;
    for (int i = operand_rank - 1; i >= 0; --i) {
      Value operand_dim_size =
          ShapedType::isDynamic(operand_shape[i])
              ? b->create<memref::DimOp>(loc, operand, i).getResult()
              : b->create<arith::ConstantIndexOp>(loc, operand_shape[i])
                    .getResult();
      operand_sizes[i] = operand_dim_size;

      operand_strides[i] = stride_so_far;
      if (i > 0) {
        stride_so_far =
            b->create<arith::MulIOp>(loc, stride_so_far, operand_dim_size);
      }
    }

    SmallVector<OpFoldResult, 2> sizes, strides;
    sizes.reserve(result_rank);
    strides.reserve(result_rank);

    DenseMap<int, int> output_to_input_dim;
    for (const auto& dim : llvm::enumerate(op.broadcast_dimensions())) {
      output_to_input_dim[dim.value().getSExtValue()] = dim.index();
    }
    for (int i = 0; i < result_rank; ++i) {
      Value i_val = b->create<arith::ConstantIndexOp>(loc, i);
      Value result_dim_size =
          b->create<tensor::ExtractOp>(loc, op.output_dimensions(), i_val);
      if (!result_dim_size.getType().isIndex()) {
        result_dim_size = b->create<arith::IndexCastOp>(loc, b->getIndexType(),
                                                        result_dim_size);
      }
      if (result_type.isDynamicDim(i)) {
        sizes.push_back(result_dim_size);
      } else {
        sizes.push_back(b->getIndexAttr(result_type.getDimSize(i)));
      }

      auto it = output_to_input_dim.find(i);
      // If the rank of the output is greater than the rank of the input, i.e.
      // there was no output dimension in the inverse broadcast_dimensions map
      // we also set stride to 0 to emulate padding of the shape with 1s and the
      // corresponding expansion.
      if (it == output_to_input_dim.end()) {
        strides.push_back(zero);
        continue;
      }

      // There can be two cases:
      // 1) Operand dim == result dim => expansion is not needed
      //    => stride flattened buffer stride
      // 2) Operand dim < result dim => expansion is needed => stride := 0.
      int dim = it->second;
      Value is_expansion = b->create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, operand_sizes[dim], result_dim_size);
      Value select = b->create<mlir::arith::SelectOp>(loc, is_expansion, zero,
                                                      operand_strides[dim]);
      strides.push_back(select);
    }

    // Type-erased memref type with static rank and dynamic strides.
    SmallVector<int64_t, 2> dynamic_layout(result_rank,
                                           ShapedType::kDynamicStrideOrOffset);
    auto type_erased_memref_type = MemRefType::get(
        result_type.getShape(), operand_type.getElementType(),
        makeStridedLinearLayoutMap(dynamic_layout,
                                   /*offset=*/0, b->getContext()));

    auto transformed_operand = b->create<memref::ReinterpretCastOp>(
        loc, type_erased_memref_type, operand,
        /*offset=*/b->getI64IntegerAttr(0), sizes, strides);
    return transformed_operand;
  }

  Value CreateCopy(mhlo::DynamicBroadcastInDimOp op, Value broadcasted,
                   OpBuilder* b) const {
    MemRefType result_type = broadcasted.getType().cast<MemRefType>();
    auto loc = op.getLoc();
    SmallVector<Value, 4> dynamic_operands;
    for (int i = 0; i < result_type.getRank(); ++i) {
      if (!result_type.isDynamicDim(i)) continue;
      auto index = b->createOrFold<arith::ConstantIndexOp>(loc, i);
      Value size =
          b->create<tensor::ExtractOp>(loc, op.output_dimensions(), index);
      if (!size.getType().isIndex()) {
        size = b->create<arith::IndexCastOp>(loc, b->getIndexType(), size);
      }
      dynamic_operands.push_back(size);
    }
    auto identity_map_memref =
        MemRefType::get(result_type.getShape(), result_type.getElementType());
    auto copy = b->create<memref::AllocOp>(op.getLoc(), identity_map_memref,
                                           dynamic_operands);
    b->create<memref::CopyOp>(loc, broadcasted, copy);

    return copy;
  }

  std::function<bool(Operation*)> enforce_identity_maps;
};

struct HloLegalizeToMemrefPass
    : public HloLegalizeToMemrefPassBase<HloLegalizeToMemrefPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                    tensor::TensorDialect>();
  }

 public:
  void runOnOperation() override {
    auto& context = getContext();
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);

    bufferization::BufferizeTypeConverter converter;

    populateHLOToMemrefConversionPattern(&converter, &patterns);

    target.addIllegalOp<DynamicReshapeOp, DynamicBroadcastInDimOp>();
    target.addLegalDialect<arith::ArithmeticDialect,
                           bufferization::BufferizationDialect, BuiltinDialect,
                           memref::MemRefDialect, tensor::TensorDialect>();

    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

void populateHLOToMemrefConversionPattern(
    bufferization::BufferizeTypeConverter* converter,
    RewritePatternSet* patterns,
    const std::function<bool(Operation*)>& enforce_identity_maps) {
  MLIRContext* context = patterns->getContext();
  patterns->add<HloToMemrefDynamicBroadcastInDimOpConverter>(
      *converter, context, enforce_identity_maps);
  patterns->add<HloToMemrefDynamicReshapeConverter,
                HloToMemrefReshapeUnrankedConverter>(*converter, context);
}

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToMemrefPass() {
  return std::make_unique<HloLegalizeToMemrefPass>();
}

}  // namespace mhlo
}  // namespace mlir
