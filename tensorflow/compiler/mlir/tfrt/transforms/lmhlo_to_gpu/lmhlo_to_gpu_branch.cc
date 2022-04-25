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

// This file implements logic to convert lmhlo branch operations to tfrt.

#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"  // from @tf_runtime

namespace tensorflow {

namespace {

class ConvertLmhloToTfrtBranchPass
    : public PassWrapper<ConvertLmhloToTfrtBranchPass,
                         OperationPass<ModuleOp>> {
 public:
  ConvertLmhloToTfrtBranchPass() = default;
  ConvertLmhloToTfrtBranchPass(const ConvertLmhloToTfrtBranchPass&) {}

  llvm::StringRef getArgument() const override {
    return "lmhlo-to-tfrt-branch";
  }
  llvm::StringRef getDescription() const override {
    return "Convert lmhlo branch operations to tfrt.";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<memref::MemRefDialect, tfrt::compiler::TFRTDialect,
                    tfrt::gpu::GpuDialect>();
  }

  void runOnOperation() override;
};

// Replaces lmhlo.while ops within a module with tfrt.while ops.
//
//   "lmhlo.while"(%cond) ({
//     <cond_ops>
//     return
//   }, {
//     <body_ops>
//     return
//   }) : (memref<i1>) -> ()
//
// is rewritten to:
//
//   func.func @while_cond(%cond, <captures>) : (memref<i1>, ...) -> i1 {
//     <cond_ops>
//     %value = memref.load %cond : memref<i1>
//     return %value : i1
//   }
//   func.func @while_body(%cond, <captures>)
//       : (memref<i1>, ...) -> (memref<i1>, ..., i1) {
//     <body_ops>
//     %value = tfrt.call @while_cond(%cond, <captures>)
//     return %cond, <captures>, %value
//   }
//   %value = tfrt.call @while_cond(%cond, <captures>)
//   %results:N = tfrt.while %value @while_body(%cond, <captures>)
//
struct WhilePattern : OpRewritePattern<lmhlo::WhileOp> {
  WhilePattern(MLIRContext* context, SymbolTable& symbol_table)
      : OpRewritePattern(context), symbol_table(symbol_table) {}

  LogicalResult matchAndRewrite(lmhlo::WhileOp while_op,
                                PatternRewriter& rewriter) const override;

  SymbolTable& symbol_table;
};

LogicalResult WhilePattern::matchAndRewrite(lmhlo::WhileOp while_op,
                                            PatternRewriter& rewriter) const {
  if (while_op->getNumOperands() != 1)
    return rewriter.notifyMatchFailure(while_op, "expected single condition");
  if (while_op.trip_count())
    return rewriter.notifyMatchFailure(while_op, "trip count not supported");

  // Collect condition value and implicit captures.
  llvm::SetVector<Value> while_args;
  while_args.insert(while_op.cond_val().front());
  getUsedValuesDefinedAbove(while_op.getOperation()->getRegions(), while_args);
  auto return_types = llvm::to_vector<4>(TypeRange(while_args.getArrayRef()));
  auto i1_type = rewriter.getI1Type();
  return_types.push_back(i1_type);
  auto argument_types = TypeRange(return_types).drop_back();

  // Clones single-block 'region' into 'func' and returns the `func.return` op.
  auto clone_region = [&](Region& region, func::FuncOp func) {
    Block* block = func.addEntryBlock();
    BlockAndValueMapping mapping;
    for (auto pair : llvm::zip_first(while_args, block->getArguments()))
      mapping.map(std::get<0>(pair), std::get<1>(pair));
    rewriter.cloneRegionBefore(region, func.getRegion(), func.end(), mapping);
    // Merge cloned block into entry block.
    rewriter.mergeBlocks(&func.back(), block);
    rewriter.setInsertionPointToEnd(block);
    Operation* terminator = block->getTerminator();
    return rewriter.replaceOpWithNewOp<func::ReturnOp>(terminator);
  };

  // Insert while_cond function.
  rewriter.setInsertionPoint(while_op->getParentOfType<func::FuncOp>());
  auto cond_func_type = rewriter.getFunctionType(argument_types, i1_type);
  auto cond_func = rewriter.create<func::FuncOp>(while_op.cond().getLoc(),
                                                 "while_cond", cond_func_type);
  symbol_table.insert(cond_func);
  auto cond_return = clone_region(while_op.cond(), cond_func);
  rewriter.setInsertionPoint(cond_return);
  Value cond_result = rewriter.create<memref::LoadOp>(cond_return.getLoc(),
                                                      cond_func.getArgument(0));
  cond_return->setOperands(cond_result);

  // Insert while_body function.
  rewriter.setInsertionPointAfter(cond_func);
  auto body_func_type = rewriter.getFunctionType(argument_types, return_types);
  auto body_func = rewriter.create<func::FuncOp>(while_op.body().getLoc(),
                                                 "while_body", body_func_type);
  symbol_table.insert(body_func);
  auto body_return = clone_region(while_op.body(), body_func);
  rewriter.setInsertionPoint(body_return);
  auto body_call = rewriter.create<tfrt::compiler::CallOp>(
      body_return.getLoc(), i1_type, cond_func.getSymName(),
      body_func.getArguments());
  body_return->setOperands(body_call.getResults());
  body_return->insertOperands(0, body_func.getArguments());

  // Replace lmhlo.while with calls to cond and body functions.
  rewriter.setInsertionPoint(while_op);
  auto while_call = rewriter.create<tfrt::compiler::CallOp>(
      while_op.getLoc(), i1_type, cond_func.getSymName(),
      while_args.getArrayRef());
  rewriter.create<tfrt::compiler::WhileOp>(
      while_op.getLoc(), argument_types, while_call.getResult(0),
      while_args.getArrayRef(), body_func.getSymName());
  rewriter.eraseOp(while_op);
  return success();
}

}  // namespace

void ConvertLmhloToTfrtBranchPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  SymbolTable symbol_table(getOperation());
  patterns.add<WhilePattern>(&getContext(), symbol_table);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> createConvertLmhloToGpuBranchPass() {
  return std::make_unique<ConvertLmhloToTfrtBranchPass>();
}

void registerConvertLmhloToGpuBranchPass() {
  ::registerPass([] { return createConvertLmhloToGpuBranchPass(); });
}

}  // namespace tensorflow
