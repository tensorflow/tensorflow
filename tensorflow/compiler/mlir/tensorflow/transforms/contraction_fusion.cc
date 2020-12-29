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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TF {
namespace {

// -------------------------------------------------------------------------- //
// Fuse ContractionFusableInterface operations into contraction operation.
// -------------------------------------------------------------------------- //

template <typename BaseOp, typename FusedOp>
class FuseIntoContractionOp : public RewritePattern {
 public:
  FuseIntoContractionOp()
      : RewritePattern(PatternBenefit(1), MatchAnyOpTypeTag()) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto fusable = dyn_cast<ContractionFusableInterface>(op);
    if (!fusable) return failure();

    auto failed = [&](Twine message) -> LogicalResult {
      return rewriter.notifyMatchFailure(op, message);
    };

    // Check if the operation can be fused.
    Optional<ContractionFusion> fusion = fusable.GetContractionFusion();
    if (!fusion.hasValue()) {
      return failed("returned empty contraction fusion specification");
    }

    // Check if preceeding operation is a BaseOp or FusedOp that we can use for
    // fusion.
    Operation *fuse_into = nullptr;
    Value operand = op->getOperand(0);

    if (BaseOp base_op = operand.getDefiningOp<BaseOp>()) {
      fuse_into = base_op.getOperation();
    } else if (FusedOp fused_op = operand.getDefiningOp<FusedOp>()) {
      fuse_into = fused_op.getOperation();
    } else {
      return failed("input to the fusable op must be a " +
                    BaseOp::getOperationName() + " or a " +
                    FusedOp::getOperationName());
    }

    // Operand result must have one use, because we do not want to compute
    // tensor contraction twice.
    if (!fuse_into->getResult(0).hasOneUse()) {
      return failed("fused into op result must have one use");
    }

    MLIRContext *ctx = op->getContext();

    // Build a fused MatMul operation from a base MatMul and a fusion.
    SmallVector<Location, 3> locations = {fuse_into->getLoc(), op->getLoc()};
    Location loc = rewriter.getFusedLoc(locations);

    // Fusion can't change the type of a fused operation.
    Type result_ty = fuse_into->getResult(0).getType();

    // Copy all operands from a base op and add additional fusion arguments.
    SmallVector<Value, 3> operands(fuse_into->getOperands());
    for (int idx : fusion->additional_arguments) {
      operands.push_back(op->getOperand(idx));
    }

    // Copy attributes from a base op that we fuse into (e.g. copy all
    // MatMul or Conv attributes to the fused operation).
    SmallVector<NamedAttribute, 4> attrs(fuse_into->getAttrs().begin(),
                                         fuse_into->getAttrs().end());

    // Add fusion specific additional attributes.
    for (auto attr : fusion->additional_attributes) {
      attrs.push_back(attr);
    }

    // Add a fused output kernel name to the list of fusions.
    Identifier fusion_id = Identifier::get("fusion", ctx);
    StringAttr fusion_name = StringAttr::get(fusion->output_kernel, ctx);

    auto is_fusion = [&](const NamedAttribute &attr) -> bool {
      return attr.first == fusion_id;
    };

    if (isa<BaseOp>(fuse_into)) {
      NamedAttribute fusion_attr(fusion_id, ArrayAttr::get({fusion_name}, ctx));
      attrs.push_back(fusion_attr);

    } else {
      ArrayAttr arr =
          llvm::find_if(attrs, is_fusion)->second.template cast<ArrayAttr>();
      llvm::erase_if(attrs, is_fusion);

      auto rng = arr.getAsRange<Attribute>();
      SmallVector<Attribute, 4> updated(rng.begin(), rng.end());
      updated.push_back(fusion_name);

      attrs.push_back(NamedAttribute(fusion_id, ArrayAttr::get(updated, ctx)));
    }

    // Update all uses of a fusable op with a new fused operation.
    Value fused = rewriter.create<FusedOp>(loc, result_ty, operands, attrs);
    rewriter.replaceOp(op, {fused});

    return failure();
  }
};

// -------------------------------------------------------------------------- //

using FuseIntoMatMulOp = FuseIntoContractionOp<MatMulOp, _JitFusedMatMulOp>;

struct ContractionFusionPass
    : public PassWrapper<ContractionFusionPass, FunctionPass> {
  void runOnFunction() override;
};

void ContractionFusionPass::runOnFunction() {
  FuncOp func = getFunction();

  OwningRewritePatternList patterns;
  patterns.insert<FuseIntoMatMulOp>();
  applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateContractionFusionPass() {
  return std::make_unique<ContractionFusionPass>();
}

static PassRegistration<ContractionFusionPass> pass(
    "tf-contraction-fusion",
    "Fuses operations implementing ContractionFusionInterface into the "
    "contraction operations");

}  // namespace TF
}  // namespace mlir
