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

#include <memory>
#include <utility>

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/fusion_interface.h"
#include "mlir-hlo/Dialect/gml_st/transforms/fusion_interface_impl.h"
#include "mlir-hlo/Dialect/gml_st/transforms/pass_detail.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir-hlo/Dialect/thlo/IR/thlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

// TODO(frgossen): Move this to the shape reification pass.
struct DimOpFissionPattern : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp extract,
                                PatternRewriter& rewriter) const override {
    auto shapeDef = llvm::dyn_cast_or_null<shape::ShapeOfOp>(
        extract.getTensor().getDefiningOp());
    if (!shapeDef || extract.getIndices().size() != 1) return failure();
    rewriter.replaceOpWithNewOp<tensor::DimOp>(extract, shapeDef.getArg(),
                                               extract.getIndices().front());
    return success();
  }
};

// TODO(frgossen): Implement this through the shape reification interface and
// move this pattern to the shape reification pass.
struct DimOpReificationPattern : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter& rewriter) const override {
    Operation* def = op.getSource().getDefiningOp();
    if (!def) return failure();

    // Case MaterializeOp.
    if (auto materializeOp = llvm::dyn_cast<MaterializeOp>(def)) {
      assert(materializeOp->getNumResults() == 1 && "assume single result");
      Value set = materializeOp.set();
      if (!set.getType().isa<TileType>()) return failure();
      rewriter.replaceOpWithNewOp<gml_st::SizeOp>(op, set, op.getIndex());
      return success();
    }

    // Case GenericOp.
    if (auto genericOp = llvm::dyn_cast<linalg::GenericOp>(def)) {
      if (genericOp.getNumResults() != 1 || !genericOp.hasTensorSemantics()) {
        return failure();
      }
      Value outputOperand = genericOp.getOutputOperand(0)->get();
      rewriter.replaceOpWithNewOp<tensor::DimOp>(op, outputOperand,
                                                 op.getIndex());
      return success();
    }

    // Case InitTensorOp.
    if (auto initTensorOp = llvm::dyn_cast<linalg::InitTensorOp>(def)) {
      if (auto indexConstantOp = llvm::dyn_cast_or_null<arith::ConstantOp>(
              op.getIndex().getDefiningOp())) {
        int64_t idx =
            indexConstantOp.getValue().dyn_cast<IntegerAttr>().getInt();
        OpFoldResult dim = initTensorOp.getMixedSizes()[idx];
        Value dimValue;
        if (dim.is<Value>()) {
          dimValue = dim.get<Value>();
        } else {
          assert(dim.is<Attribute>() && "expected Value or Attribute");
          int64_t dimInt = dim.get<Attribute>().cast<IntegerAttr>().getInt();
          dimValue =
              rewriter.create<arith::ConstantIndexOp>(op.getLoc(), dimInt);
        }
        assert(dimValue);
        rewriter.replaceOp(op, ValueRange{dimValue});
        return success();
      }
    }

    // Case DynamicBroadcastInDimOp.
    if (auto bcast = llvm::dyn_cast<thlo::DynamicBroadcastInDimOp>(def)) {
      rewriter.replaceOpWithNewOp<tensor::DimOp>(op, bcast.init(),
                                                 op.getIndex());
      return success();
    }

    return failure();
  }
};

struct FusionPattern : public OpRewritePattern<MaterializeOp> {
  using OpRewritePattern<MaterializeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MaterializeOp op,
                                PatternRewriter& rewriter) const override {
    Operation* def = op.source().getDefiningOp();
    if (!def) return failure();

    auto iface = llvm::dyn_cast<FusionInterface>(def);
    if (!iface) return failure();

    Value fused = iface.fuse(op.getLoc(), op.set(), rewriter);
    if (!fused) return failure();

    rewriter.replaceOp(op, fused);
    return success();
  }
};

class FusionPass : public FusionPassBase<FusionPass> {
  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<scf::SCFDialect>();
    registerFusionInterfaceExternalModels(registry);
  }

  void runOnOperation() final {
    MLIRContext* ctx = &getContext();

    // Populate patterns.
    RewritePatternSet patterns(ctx);
    // clang-format off
    patterns.insert<
        DimOpFissionPattern,
        DimOpReificationPattern,
        FusionPattern>(ctx);
    // clang-format on

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createFusionPass() {
  return std::make_unique<FusionPass>();
}

}  // namespace gml_st
}  // namespace mlir
