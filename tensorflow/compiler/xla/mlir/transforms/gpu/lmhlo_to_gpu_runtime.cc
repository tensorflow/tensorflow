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

#include <iterator>
#include <memory>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/transforms/gpu/custom_calls.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"

namespace xla {
namespace gpu {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/xla/mlir/transforms/gpu/passes.h.inc"

using namespace mlir;  // NOLINT

using mlir::gpu::MemcpyOp;

using mlir::lmhlo::CaseOp;
using mlir::lmhlo::CustomCallOp;
using mlir::lmhlo::InfeedOp;
using mlir::lmhlo::OutfeedOp;
using mlir::lmhlo::TerminatorOp;
using mlir::lmhlo::WhileOp;

class ConvertLmhloToGpuRuntimePass
    : public ConvertLmhloToGpuRuntimePassBase<ConvertLmhloToGpuRuntimePass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry
        .insert<arith::ArithmeticDialect, cf::ControlFlowDialect,
                func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  }
};

//===----------------------------------------------------------------------===//

class TerminatorOpLowering : public OpRewritePattern<TerminatorOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TerminatorOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//

template <typename IoFeedOp>
class IoFeedOpLowering : public OpRewritePattern<IoFeedOp> {
  static StringRef Target(InfeedOp) { return "xla.gpu.infeed"; }
  static StringRef Target(OutfeedOp) { return "xla.gpu.outfeed"; }

 public:
  IoFeedOpLowering(MLIRContext* ctx, CustomCalls& custom_calls)
      : OpRewritePattern<IoFeedOp>(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(IoFeedOp op,
                                PatternRewriter& rewriter) const override {
    // Get or create a custom call function declaration.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    func::FuncOp callee = custom_calls_.GetOrCreate(b, Target(op), op);

    llvm::SmallVector<NamedAttribute> custom_call_attrs = {
        {b.getStringAttr("config"), op.getConfigAttr()}};

    // Call the runtime intrinsic with the original operands.
    auto call = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee.getName(), TypeRange(), op.getOperands());
    AppendCustomCallAttrs(call, custom_call_attrs);

    return success();
  }

 private:
  CustomCalls& custom_calls_;
};

class InfeedOpLowering : public IoFeedOpLowering<InfeedOp> {
 public:
  using IoFeedOpLowering::IoFeedOpLowering;
};

class OutfeedOpLowering : public IoFeedOpLowering<OutfeedOp> {
 public:
  using IoFeedOpLowering::IoFeedOpLowering;
};

//===----------------------------------------------------------------------===//

class CustomCallOpLowering : public OpRewritePattern<CustomCallOp> {
 private:
  static constexpr const char kCustomCallTarget[] = "xla.gpu.custom_call";

 public:
  CustomCallOpLowering(MLIRContext* ctx, CustomCalls& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // By default all operands passed to the custom call handler.
    llvm::SmallVector<Value> operands = op.getOperands();

    // If custom call has target arguments mapping, then we need to pass empty
    // memrefs in place of holes.
    if (op.getTargetArgMapping().has_value()) {
      auto mapping = *op.getTargetArgMapping();
      int64_t num_args = mapping.getNumArgs();
      int64_t num_results = mapping.getNumResults();

      // We represent holes as empty i8 memrefs.
      Value hole =
          b.create<memref::AllocaOp>(MemRefType::get({0}, b.getI8Type()));
      operands = llvm::SmallVector<Value>(num_args + num_results, hole);

      // Update operands to mapped custom call arguments.
      auto args = mapping.getArgsToTargetArgs();
      for (const auto& indexed : llvm::enumerate(args))
        operands[indexed.value()] = op.getArgs()[indexed.index()];

      // Update operands to mapped custom call results.
      auto res = mapping.getResultsToTargetResults();
      for (const auto& indexed : llvm::enumerate(res))
        operands[num_args + indexed.value()] = op.getOutput()[indexed.index()];
    }

    // Create a custom call function declaration.
    func::FuncOp callee = custom_calls_.GetOrCreate(
        b, kCustomCallTarget, TypeRange(ValueRange(operands)), TypeRange());

    llvm::SmallVector<NamedAttribute> custom_call_attrs = {
        {b.getStringAttr("api_version"), op.getApiVersionAttr()},
        {b.getStringAttr("backend_config"), op.getBackendConfigAttr()},
        {b.getStringAttr("call_target_name"), op.getCallTargetNameAttr()}};

    // Call the runtime intrinsic with the original operands.
    auto call = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee.getName(), TypeRange(), operands);
    AppendCustomCallAttrs(call, custom_call_attrs);

    return success();
  }

 private:
  CustomCalls& custom_calls_;
};

//===----------------------------------------------------------------------===//

class CaseOpLowering : public OpRewritePattern<CaseOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CaseOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Copy index buffer to the host ...
    auto index_type = op.getIndex().getType().dyn_cast<MemRefType>();

    // TODO(ezhulenev): We need to make sure that `alloca` is placed in the
    // parent function entry block.
    // https://llvm.org/docs/Frontend/PerformanceTips.html#use-of-allocas
    Value index_on_host = b.create<memref::AllocaOp>(index_type);
    b.create<MemcpyOp>(TypeRange(), ValueRange({index_on_host, op.getIndex()}));

    // Get the index value from the buffer.
    Value index = b.create<memref::LoadOp>(index_type.getElementType(),
                                           index_on_host, ValueRange());

    bool is_predicate = index_type.getElementType().isInteger(1);

    // For binary index (predicate) convert i1 to i32 index.
    if (is_predicate) {
      Value c0 = b.create<arith::ConstantOp>(b.getI32IntegerAttr(0));
      Value c1 = b.create<arith::ConstantOp>(b.getI32IntegerAttr(1));
      index = b.create<arith::SelectOp>(index, c0, c1);
    }

    // For integer index make sure that it is within range.
    if (!is_predicate) {
      unsigned n = op.getNumRegions() - 1;
      Value c0 = b.create<arith::ConstantOp>(b.getI32IntegerAttr(0));
      Value cN = b.create<arith::ConstantOp>(b.getI32IntegerAttr(n));

      Value too_small = b.create<arith::CmpIOp>(
          b.getI1Type(), arith::CmpIPredicate::slt, index, c0);
      Value too_large = b.create<arith::CmpIOp>(
          b.getI1Type(), arith::CmpIPredicate::sgt, index, cN);

      Value out_of_range = b.create<arith::OrIOp>(too_small, too_large);
      index = b.create<arith::SelectOp>(out_of_range, cN, index);
    }

    // Split block right at the case operation.
    Block* cont = rewriter.splitBlock(op->getBlock(), op->getIterator());
    Block* orig = cont->getPrevNode();

    // Prepare case destinations for the `scf.switch` operation.
    llvm::SmallVector<llvm::APInt> case_values;
    llvm::SmallVector<Block*> case_blocks;
    llvm::SmallVector<ValueRange> case_operands;

    // Create blocks from each of the case regions.
    for (Region& region : op->getRegions()) {
      // Move `lmhlo.case` block before the continuation.
      Block& block = region.front();
      block.moveBefore(cont);

      // Erase original `lmhlo.terminator`.
      rewriter.eraseOp(block.getTerminator());

      // Branch into the continuation block.
      b.setInsertionPointToEnd(&block);
      b.create<cf::BranchOp>(cont);

      // Add a `cf.switch` case.
      int32_t idx = case_blocks.size();
      case_values.push_back(b.getI32IntegerAttr(idx).getValue());
      case_blocks.push_back(&block);
      case_operands.push_back({});
    }

    // Replace `lmhlo.case` with a `cf.switch` operation on the host.
    b.setInsertionPointToEnd(orig);
    b.create<cf::SwitchOp>(index, cont, ValueRange(), case_values, case_blocks,
                           case_operands);

    // Erase the original case operation.
    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//

class WhileOpLowering : public OpRewritePattern<WhileOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Create an `scf.while` loop in place of `lmhlo.while` loop.
    auto loop = b.create<scf::WhileOp>(TypeRange(), ValueRange());

    // Predicate buffer placed on the device.
    assert(op.getNumOperands() == 1 && "expected single cond operand");
    Value pred = op.getOperand(0);

    // Clone condition and body blocks into the new loop operation.
    BlockAndValueMapping mapping;
    op.getCond().cloneInto(&loop.getBefore(), mapping);
    op.getBody().cloneInto(&loop.getAfter(), mapping);

    {  // Replace loop condition terminator.
      auto* terminator = loop.getBefore().back().getTerminator();
      b.setInsertionPointAfter(terminator);

      // Copy predicate buffer to the host ...
      auto i1 = b.getI1Type();
      Value pred_on_host = b.create<memref::AllocaOp>(MemRefType::get({}, i1));
      b.create<gpu::MemcpyOp>(TypeRange(), ValueRange({pred_on_host, pred}));

      // .. and check if we need to continue loop iteration.
      Value cond = b.create<memref::LoadOp>(i1, pred_on_host, ValueRange());
      b.create<scf::ConditionOp>(cond, ValueRange());
      rewriter.eraseOp(terminator);
    }

    {  // Replace loop body terminator.
      auto* terminator = loop.getAfter().back().getTerminator();
      b.setInsertionPointAfter(terminator);
      b.create<scf::YieldOp>(TypeRange(), ValueRange());
      rewriter.eraseOp(terminator);
    }

    // Erase the original while loop.
    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//

void ConvertLmhloToGpuRuntimePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* ctx = module.getContext();

  // Keep track of the custom calls created from the lowered operations.
  SymbolTable sym_table(module);
  CustomCalls custom_calls(std::move(sym_table));

  // Convert lmhlo operations to XLA gpu runtime custom calls.
  RewritePatternSet patterns(ctx);
  patterns.insert<TerminatorOpLowering, CaseOpLowering, WhileOpLowering>(ctx);
  patterns.insert<InfeedOpLowering, OutfeedOpLowering, CustomCallOpLowering>(
      ctx, custom_calls);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertLmhloToGpuRuntimePass() {
  return std::make_unique<ConvertLmhloToGpuRuntimePass>();
}

}  // namespace gpu
}  // namespace xla
