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

#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

constexpr llvm::StringRef
    mlir::kernel_gen::tf_framework ::JITCompileFromStrOp::kJITEntryFunctionName;

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

static constexpr StringRef kEmitCInterfaceAttrName = "llvm.emit_c_interface";

bool IsTFOperation(Operation *op) {
  return op != nullptr &&
         op->getDialect() ==
             op->getContext()->getLoadedDialect<TF::TensorFlowDialect>();
}

struct TFToJITInvocationsPattern : public RewritePattern {
  explicit TFToJITInvocationsPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Apply to all TF ops except those that are already in a JIT-compiled
    // region.
    if (!IsTFOperation(op) || op->getParentOfType<tf_framework::JITCompileOp>())
      return failure();

    // Find last TF op.
    while (IsTFOperation(op->getNextNode())) op = op->getNextNode();

    // Find JIT compile region operands and results.
    SmallVector<Operation *, 16> cluster;
    llvm::SmallPtrSet<Value, 16> operand_set, result_set;
    Operation *it = op;
    while (IsTFOperation(it)) {
      // Find results that escape the JIT compile region.
      for (auto &use : it->getUses()) {
        if (!llvm::is_contained(cluster, use.getOwner()))
          result_set.insert(use.get());
      }

      // Update JIT region operands and results.
      for (Value v : it->getResults()) operand_set.erase(v);
      for (Value v : it->getOperands()) operand_set.insert(v);

      cluster.push_back(it);
      it = it->getPrevNode();
    }

    // Introduce order to the operands and results.
    auto operands = llvm::to_vector<16>(operand_set);
    auto results = llvm::to_vector<16>(result_set);
    auto operand_types = llvm::to_vector<16>(
        llvm::map_range(operands, [](Value v) { return v.getType(); }));
    auto result_types = llvm::to_vector<16>(
        llvm::map_range(results, [](Value v) { return v.getType(); }));

    // Create the JIT compile op.
    auto loc = op->getLoc();
    auto jit_compile_op = rewriter.create<tf_framework::JITCompileOp>(
        loc, rewriter.getType<tf_framework::JITCallableType>(), llvm::None);

    // Move the TF operations into the new op's body.
    BlockAndValueMapping bvm;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *block =
          rewriter.createBlock(&jit_compile_op.body(), {}, operand_types);
      for (auto it : llvm::zip(operands, block->getArguments()))
        bvm.map(std::get<0>(it), std::get<1>(it));
      rewriter.setInsertionPointToStart(block);
      for (Operation *it : llvm::reverse(cluster)) rewriter.clone(*it, bvm);
      auto mapped_results = llvm::to_vector<16>(
          llvm::map_range(results, [&](Value v) { return bvm.lookup(v); }));
      rewriter.create<tf_framework::JITCompileYieldOp>(loc, TypeRange{},
                                                       mapped_results);
    }

    // Create JIT execute op.
    auto jit_execute_op = rewriter.create<tf_framework::JITExecuteOp>(
        loc, result_types, Value(), jit_compile_op.result(), operands);

    // Replace old TF ops with the new results.
    for (auto it : llvm::zip(results, jit_execute_op.results()))
      bvm.map(std::get<0>(it), std::get<1>(it));
    for (Operation *it : cluster) {
      if (it->getUses().empty()) {
        rewriter.eraseOp(it);
        continue;
      }
      auto replacements = llvm::to_vector<16>(llvm::map_range(
          it->getResults(), [&](Value v) { return bvm.lookup(v); }));
      rewriter.replaceOp(it, replacements);
    }

    return success();
  }
};

struct PackJITCompileOpPattern
    : public OpRewritePattern<tf_framework::JITCompileOp> {
  using OpRewritePattern<tf_framework::JITCompileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tf_framework::JITCompileOp op,
                                PatternRewriter &rewriter) const override {
    Block *body = op.getBody();
    auto yield_op =
        llvm::cast<tf_framework::JITCompileYieldOp>(body->getTerminator());

    // Temporarily, build the module that would be JIT-compiled. This is only to
    // obtain the serialized code attribute.
    auto loc = op->getLoc();
    OpBuilder tmp_module_builder(getContext(), rewriter.getListener());
    auto jit_module = tmp_module_builder.create<ModuleOp>(loc);
    tmp_module_builder.setInsertionPointToStart(jit_module.getBody());
    auto jit_function = tmp_module_builder.create<FuncOp>(
        loc, tf_framework::JITCompileFromStrOp::kJITEntryFunctionName,
        tmp_module_builder.getFunctionType(body->getArgumentTypes(),
                                           yield_op->getOperandTypes()));
    jit_function->setAttr(tf_framework::TFFrameworkDialect::kTFEntryAttrName,
                          tmp_module_builder.getUnitAttr());
    jit_function->setAttr(kEmitCInterfaceAttrName,
                          tmp_module_builder.getUnitAttr());
    jit_function.getBody().takeBody(op.getBodyRegion());
    tmp_module_builder.setInsertionPointToEnd(&jit_function.getBody().front());
    tmp_module_builder.create<ReturnOp>(loc, yield_op.result());
    rewriter.eraseOp(yield_op);

    // Serialize JIT module.
    std::string code;
    llvm::raw_string_ostream ss(code);
    jit_module.print(ss);

    // Finally, create the new JIT compile op.
    rewriter.replaceOpWithNewOp<tf_framework::JITCompileFromStrOp>(
        op, op->getResultTypes(), op.ctx(), rewriter.getStringAttr(code));

    return success();
  }
};

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct TFToJITInvocationPass
    : public TFToJITInvocationPassBase<TFToJITInvocationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::kernel_gen::tf_framework::TFFrameworkDialect>();
  }

  void runOnFunction() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    PopulateTFToJITInvocationPatterns(ctx, &patterns);
    if (failed(
            applyPatternsAndFoldGreedily(getFunction(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void PopulateTFToJITInvocationPatterns(MLIRContext *ctx,
                                       RewritePatternSet *patterns) {
  // clang-format off
  patterns->insert<
      PackJITCompileOpPattern,
      TFToJITInvocationsPattern>(ctx);
  // clang-format on
}

std::unique_ptr<FunctionPass> CreateTFToJITInvocationPass() {
  return std::make_unique<TFToJITInvocationPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
