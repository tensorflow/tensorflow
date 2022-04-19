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

// This file implements logic for lowering HLO dialect to LHLO dialect.

#include <memory>
#include <utility>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {
namespace {

struct RngGetAndUpdateStatePattern
    : public OpConversionPattern<mhlo::XlaRngGetAndUpdateStateOp> {
  using OpConversionPattern<
      mhlo::XlaRngGetAndUpdateStateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::XlaRngGetAndUpdateStateOp op,
      XlaRngGetAndUpdateStateOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // Get various type related information
    auto loc = op->getLoc();

    const auto global_name = rewriter.getStringAttr("rng_state");
    constexpr auto initial_seed = 0x7012395ull;
    auto seed_type = rewriter.getIntegerType(128);
    auto memref_type = MemRefType::get({}, seed_type);

    auto result_type = op.getType();
    auto word_size = result_type.getElementType().getIntOrFloatBitWidth();
    auto smaller_int_type = rewriter.getIntegerType(word_size);
    auto num_elements = result_type.getNumElements();

    // Get or define the global variable
    auto* global_op =
        mlir::SymbolTable::lookupNearestSymbolFrom(op, global_name);
    if (!global_op) {
      auto* parent = mlir::SymbolTable::getNearestSymbolTable(op);
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&parent->getRegions().front().front());

      const auto priv = rewriter.getStringAttr("private");
      auto initial_value = mlir::DenseElementsAttr::get(
          mlir::RankedTensorType::get({}, seed_type),
          rewriter.getIntegerAttr(seed_type, initial_seed));
      global_op =
          rewriter.create<memref::GlobalOp>(loc, global_name, priv, memref_type,
                                            initial_value, /*constant=*/false,
                                            /*alignment=*/IntegerAttr());
    }
    assert(isa<memref::GlobalOp>(global_op) &&
           "rng_state was defined somewhere else, not as a global op");

    // Get and update
    Value rng_state =
        rewriter.create<memref::GetGlobalOp>(loc, memref_type, global_name);
    Value old_val = rewriter.create<memref::LoadOp>(loc, rng_state);
    Value delta = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(seed_type,
                                     static_cast<int64_t>(adaptor.delta())));
    Value new_val = rewriter.create<arith::AddIOp>(loc, old_val, delta);
    (void)rewriter.create<memref::StoreOp>(loc, new_val, rng_state);

    // Create the proper return type by packing the old seed into a tensor
    SmallVector<Value> pieces;
    for (int i = (num_elements - 1) * word_size; i >= 0; i -= word_size) {
      Value shift_distance = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(seed_type, i));
      pieces.push_back(rewriter.create<arith::TruncIOp>(
          loc, smaller_int_type,
          rewriter.create<arith::ShRUIOp>(loc, old_val, shift_distance)));
    }

    // Obtain a tensor with the correct shape and bit widths but the incorrect
    // integer signedness, then cast the tensor to the correct signedness to
    // ensure that unrealized casts will successfully lower later.
    Value result_tensor = rewriter.create<tensor::FromElementsOp>(
        loc,
        mlir::RankedTensorType::get(result_type.getShape(), smaller_int_type),
        pieces);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, result_type,
                                                            result_tensor);
    return success();
  }
};

struct HloLegalizeToArithmeticPass
    : public HloLegalizeToArithmeticPassBase<HloLegalizeToArithmeticPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<arith::ArithmeticDialect, memref::MemRefDialect,
                    tensor::TensorDialect>();
  }

 public:
  void runOnOperation() override {
    auto& context = getContext();
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);

    populateHLOToArithmeticConversionPatterns(&patterns);

    target.addIllegalOp<XlaRngGetAndUpdateStateOp>();
    target.addLegalDialect<arith::ArithmeticDialect, BuiltinDialect,
                           memref::MemRefDialect, tensor::TensorDialect>();

    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

void populateHLOToArithmeticConversionPatterns(RewritePatternSet* patterns) {
  patterns->add<RngGetAndUpdateStatePattern>(patterns->getContext());
}

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToArithmeticPass() {
  return std::make_unique<HloLegalizeToArithmeticPass>();
}

}  // namespace mhlo
}  // namespace mlir
