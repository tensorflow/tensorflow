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

#include <utility>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir-hlo/Dialect/mhlo/transforms/type_conversion.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {
namespace {

template <typename T>
class SignlessOpConversion : public OpConversionPattern<T> {
 public:
  SignlessOpConversion(TypeConverter& type_converter,
                       RemoveSignTypeConverter* remove_sign_converter,
                       MLIRContext* ctx)
      : OpConversionPattern<T>(type_converter, ctx),
        remove_sign_converter_(remove_sign_converter) {}

  LogicalResult matchAndRewrite(
      T op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    // Sign-convert operands and result type.
    SmallVector<Value> converted_operands;
    for (auto operand : operands) {
      Type original = operand.getType();
      Type converted = remove_sign_converter_->convertType(original);
      if (converted == original) {
        converted_operands.push_back(operand);
      } else {
        converted_operands.push_back(
            rewriter
                .create<UnrealizedConversionCastOp>(loc, converted, operand)
                ->getResult(0));
      }
    }
    Type op_result_type = remove_sign_converter_->convertType(op.getType());
    // Perform actual rewrite.
    Value result =
        signlessRewrite(op, converted_operands, op_result_type, rewriter);
    if (!result) return failure();

    // If the element type of the original op and the returned value differ,
    // do a conversion cast to fix it up.
    auto expected_element_type =
        op.getType().template cast<ShapedType>().getElementType();
    auto result_type = result.getType().cast<BaseMemRefType>();
    auto actual_element_type = result_type.getElementType();
    if (expected_element_type != actual_element_type) {
      assert(remove_sign_converter_->convertType(expected_element_type) ==
             actual_element_type);
      Type new_type;
      if (auto ranked = result_type.dyn_cast<MemRefType>()) {
        new_type =
            MemRefType::get(ranked.getShape(), expected_element_type,
                            ranked.getAffineMaps(), ranked.getMemorySpace());
      } else {
        new_type = UnrankedMemRefType::get(expected_element_type,
                                           result_type.getMemorySpace());
      }
      result =
          rewriter.create<UnrealizedConversionCastOp>(loc, new_type, result)
              .getResult(0);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

 protected:
  virtual Value signlessRewrite(T op, ArrayRef<Value> operands,
                                Type result_type,
                                ConversionPatternRewriter& rewriter) const = 0;

 private:
  RemoveSignTypeConverter* remove_sign_converter_;
};

template <typename T>
using BaseOpConversion = SignlessOpConversion<T>;

class HloToMemrefReshapeUnrankedConverter
    : public BaseOpConversion<mhlo::ReshapeOp> {
 public:
  using BaseOpConversion<mhlo::ReshapeOp>::BaseOpConversion;

  Value signlessRewrite(mhlo::ReshapeOp op, ArrayRef<Value> operands,
                        Type op_result_type,
                        ConversionPatternRewriter& rewriter) const final {
    mhlo::ReshapeOp::Adaptor adaptor(operands);
    auto unranked_operand_type =
        adaptor.operand().getType().dyn_cast<UnrankedMemRefType>();
    if (unranked_operand_type == nullptr) return {};
    auto loc = op->getLoc();
    auto result_type = op_result_type.cast<RankedTensorType>();
    auto cast = rewriter.create<memref::CastOp>(
        loc, adaptor.operand(),
        MemRefType::get(result_type.getShape(), result_type.getElementType()));

    return cast;
  }
};

class HloToMemrefDynamicReshapeConverter
    : public BaseOpConversion<mhlo::DynamicReshapeOp> {
 public:
  using BaseOpConversion<mhlo::DynamicReshapeOp>::BaseOpConversion;

  Value signlessRewrite(mhlo::DynamicReshapeOp op, ArrayRef<Value> operands,
                        Type op_result_type,
                        ConversionPatternRewriter& rewriter) const final {
    ShapedType result_type;
    if (auto ranked_type = op_result_type.dyn_cast<RankedTensorType>()) {
      result_type =
          MemRefType::get(ranked_type.getShape(), ranked_type.getElementType());
    } else if (auto unranked_type =
                   op_result_type.dyn_cast<UnrankedTensorType>()) {
      result_type = UnrankedMemRefType::get(unranked_type.getElementType(), 0);
    } else {
      return {};
    }
    mhlo::DynamicReshapeOp::Adaptor adaptor(operands);
    auto reshape = rewriter.create<memref::ReshapeOp>(
        op.getLoc(), result_type, adaptor.operand(), adaptor.output_shape());
    return reshape;
  }
};

// TODO(b/175670649) Fix this to no longer access original tensor operands.
class HloToMemrefDynamicBroadcastInDimOpConverter
    : public BaseOpConversion<mhlo::DynamicBroadcastInDimOp> {
 public:
  HloToMemrefDynamicBroadcastInDimOpConverter(
      TypeConverter& converter, RemoveSignTypeConverter* sign_converter,
      MLIRContext* ctx, bool enforce_identity_maps)
      : BaseOpConversion<mhlo::DynamicBroadcastInDimOp>(converter,
                                                        sign_converter, ctx),
        enforce_identity_maps_(enforce_identity_maps) {}

  Value signlessRewrite(mhlo::DynamicBroadcastInDimOp op,
                        ArrayRef<Value> operands, Type op_result_type,
                        ConversionPatternRewriter& rewriter) const final {
    auto result_type = op_result_type.dyn_cast<RankedTensorType>();
    if (!result_type) return {};
    Value result = InsertDynamicMemrefCastOp(op, operands.front(), &rewriter);

    if (enforce_identity_maps_) {
      result = CreateCopy(op, result, &rewriter);
    }

    return result;
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

    Value zero = b->create<ConstantIndexOp>(loc, 0);
    Value one = b->create<ConstantIndexOp>(loc, 1);

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
              : b->create<ConstantIndexOp>(loc, operand_shape[i]).getResult();
      operand_sizes[i] = operand_dim_size;

      operand_strides[i] = stride_so_far;
      if (i > 0) {
        stride_so_far = b->create<MulIOp>(loc, stride_so_far, operand_dim_size);
      }
    }

    SmallVector<OpFoldResult, 2> sizes, strides;
    sizes.reserve(result_rank);
    strides.reserve(result_rank);

    DenseMap<int, int> output_to_input_dim;
    for (auto dim : llvm::enumerate(op.broadcast_dimensions())) {
      output_to_input_dim[dim.value().getSExtValue()] = dim.index();
    }
    for (int i = 0; i < result_rank; ++i) {
      Value i_val = b->create<ConstantIndexOp>(loc, i);
      Value result_dim_size =
          b->create<tensor::ExtractOp>(loc, op.output_dimensions(), i_val);
      if (!result_dim_size.getType().isIndex()) {
        result_dim_size =
            b->create<IndexCastOp>(loc, result_dim_size, b->getIndexType());
      }
      sizes.push_back(result_dim_size);

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
      Value is_expansion = b->create<CmpIOp>(
          loc, CmpIPredicate::slt, operand_sizes[dim], result_dim_size);
      Value select = b->create<mlir::SelectOp>(loc, is_expansion, zero,
                                               operand_strides[dim]);
      strides.push_back(select);
    }

    // Type-erased memref type with static rank, dynamic sizes and strides.
    SmallVector<int64_t, 2> dynamic_layout(result_rank,
                                           MemRefType::kDynamicStrideOrOffset);
    SmallVector<int64_t, 2> dynamic_shape(result_rank,
                                          MemRefType::kDynamicSize);
    auto type_erased_memref_type = MemRefType::get(
        dynamic_shape, operand_type.getElementType(),
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
      auto index = b->createOrFold<ConstantIndexOp>(loc, i);
      Value size =
          b->create<tensor::ExtractOp>(loc, op.output_dimensions(), index);
      if (!size.getType().isIndex()) {
        size = b->create<IndexCastOp>(loc, size, b->getIndexType());
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

  bool enforce_identity_maps_;
};

struct HloLegalizeToMemrefPass
    : public HloLegalizeToMemrefPassBase<HloLegalizeToMemrefPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<memref::MemRefDialect, tensor::TensorDialect>();
  }

 public:
  void runOnFunction() override {
    auto& context = getContext();
    OwningRewritePatternList patterns(&context);
    ConversionTarget target(context);

    BufferizeTypeConverter converter;
    RemoveSignTypeConverter sign_converter;

    populateHLOToMemrefConversionPattern(&converter, &sign_converter,
                                         &patterns);

    target.addIllegalOp<DynamicReshapeOp, DynamicBroadcastInDimOp>();
    target.addLegalDialect<BuiltinDialect, memref::MemRefDialect,
                           StandardOpsDialect, tensor::TensorDialect>();

    auto func = getFunction();
    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

void populateHLOToMemrefConversionPattern(
    BufferizeTypeConverter* converter, RemoveSignTypeConverter* sign_converter,
    OwningRewritePatternList* patterns, bool enforce_identity_maps) {
  MLIRContext* context = patterns->getContext();
  patterns->insert<HloToMemrefDynamicBroadcastInDimOpConverter>(
      *converter, sign_converter, context, enforce_identity_maps);
  patterns->insert<HloToMemrefDynamicReshapeConverter,
                   HloToMemrefReshapeUnrankedConverter>(
      *converter, sign_converter, context);
}

std::unique_ptr<FunctionPass> createLegalizeToMemrefPass() {
  return std::make_unique<HloLegalizeToMemrefPass>();
}

}  // namespace mhlo
}  // namespace mlir
