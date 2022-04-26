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
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLmhloToTfrtBranchPass)

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

// Replaces lmhlo.case ops within a module with tfrt.case ops.
//
//   "lmhlo.case"(%index_memref) ({
//     region-0
//   }, {
//   ...
//     region-N-1
//   }) : (memref<value_type>) -> ()
//
// is rewritten to
//
//   func.func @case_func_0(<captures>) : (capture_types) -> () {
//   }
//   ...
//   func.func @case_func_N-1(<captures>) : (capture_types) -> () {
//   }
//   %index_value = memref.load %index_memref
//
// and (when value_type == i32)
//   tfrt.case %index_value [@case_func_0, ... @case_func_N-1] <captures>
//             : (capture_types) -> ()
//
// or (when value_type == i1)
//   tfrt.cond %index_value @case_func_1, @case_func_0, <captures>
//             : (capture_types) -> ()
//
struct CasePattern : OpRewritePattern<lmhlo::CaseOp> {
  CasePattern(MLIRContext* context, SymbolTable& symbol_table)
      : OpRewritePattern(context), symbol_table(symbol_table) {}

  LogicalResult matchAndRewrite(lmhlo::CaseOp case_op,
                                PatternRewriter& rewriter) const override;

  SymbolTable& symbol_table;
};

}  // namespace

// Clones region to func, with values in args replaced by the corresponding
// arguments of the function. Returns the terminator of the function, which is
// a ReturnOp without any value.
//
static func::ReturnOp CloneRegionToFunc(PatternRewriter& rewriter,
                                        Region& region, func::FuncOp func,
                                        const llvm::SetVector<Value>& args) {
  Block* block = func.addEntryBlock();
  BlockAndValueMapping mapping;
  for (auto pair : llvm::zip_first(args, block->getArguments()))
    mapping.map(std::get<0>(pair), std::get<1>(pair));
  rewriter.cloneRegionBefore(region, func.getRegion(), func.end(), mapping);
  // Merge cloned block into entry block.
  rewriter.mergeBlocks(&func.back(), block);
  rewriter.setInsertionPointToEnd(block);
  Operation* terminator = block->getTerminator();
  return rewriter.replaceOpWithNewOp<func::ReturnOp>(terminator);
}

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

  // Insert while_cond function.
  rewriter.setInsertionPoint(while_op->getParentOfType<func::FuncOp>());
  auto cond_func_type = rewriter.getFunctionType(argument_types, i1_type);
  auto cond_func = rewriter.create<func::FuncOp>(while_op.cond().getLoc(),
                                                 "while_cond", cond_func_type);
  symbol_table.insert(cond_func);
  auto cond_return =
      CloneRegionToFunc(rewriter, while_op.cond(), cond_func, while_args);
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
  auto body_return =
      CloneRegionToFunc(rewriter, while_op.body(), body_func, while_args);
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

LogicalResult CasePattern::matchAndRewrite(lmhlo::CaseOp case_op,
                                           PatternRewriter& rewriter) const {
  mlir::Value index_memref = case_op.index();
  auto int_type = index_memref.getType()
                      .cast<mlir::ShapedType>()
                      .getElementType()
                      .dyn_cast<IntegerType>();

  if (!int_type)
    return rewriter.notifyMatchFailure(case_op, "expected integer index value");

  int bitwidth = int_type.getWidth();
  // TODO(b/230355363): Support other integer types through enhancing tfrt.case.
  if (bitwidth != 1 && bitwidth != 32)
    return rewriter.notifyMatchFailure(case_op,
                                       "expected i1 or i32 index value");

  Location loc = case_op.getLoc();

  // Load the index value.
  rewriter.setInsertionPoint(case_op);
  mlir::Value index_value = rewriter.create<memref::LoadOp>(loc, index_memref);

  // Collect implicit captures for the branching regions.
  llvm::SetVector<Value> case_args;
  getUsedValuesDefinedAbove(case_op.getOperation()->getRegions(), case_args);
  ArrayRef<Value> args = case_args.getArrayRef();
  auto arg_types = llvm::to_vector<4>(TypeRange(args));

  // Outline each branching region as a function and record the FuncOp.
  size_t branch_count = case_op.branches().size();
  SmallVector<func::FuncOp, 4> branch_funcs;
  branch_funcs.reserve(branch_count);
  auto func_type = rewriter.getFunctionType(arg_types, {});
  for (size_t i = 0; i < branch_count; ++i) {
    Region* region = &case_op.branches()[i];
    rewriter.setInsertionPoint(case_op->getParentOfType<func::FuncOp>());
    auto func =
        rewriter.create<func::FuncOp>(region->getLoc(), "case_func", func_type);
    branch_funcs.push_back(func);
    symbol_table.insert(func);
    CloneRegionToFunc(rewriter, *region, func, case_args);
  }

  // Convert the operation to either a tfrt.cond or a tfrt.case.
  rewriter.setInsertionPoint(case_op);
  MLIRContext* context = case_op.getContext();
  if (bitwidth == 1) {
    auto else_branch =
        mlir::SymbolRefAttr::get(context, branch_funcs[0].getSymName());
    auto then_branch =
        mlir::SymbolRefAttr::get(context, branch_funcs[1].getSymName());
    rewriter.create<tfrt::compiler::CondOp>(loc, llvm::None, index_value,
                                            then_branch, else_branch, args);
  } else {
    SmallVector<Attribute, 4> branch_attrs;
    branch_attrs.reserve(branch_count);
    llvm::transform(branch_funcs, std::back_inserter(branch_attrs),
                    [&](func::FuncOp func) { return func.getSymNameAttr(); });
    auto branches = ArrayAttr::get(context, branch_attrs);
    rewriter.create<tfrt::compiler::CaseOp>(loc, llvm::None, index_value,
                                            branches, args);
  }

  rewriter.eraseOp(case_op);
  return success();
}

void ConvertLmhloToTfrtBranchPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  SymbolTable symbol_table(getOperation());
  patterns.add<WhilePattern>(&getContext(), symbol_table);
  patterns.add<CasePattern>(&getContext(), symbol_table);
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
