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

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"

constexpr llvm::StringRef
    mlir::kernel_gen::tf_framework ::JITCompileFromStrOp::kJITEntryFunctionName;

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

constexpr int64_t i32Limit = 2147483647;
using func::FuncOp;
using shape::ShapeOfOp;

LogicalResult RewriteToFullJit(func::FuncOp op) {
  // Rewrite all functions that have a single result.
  if (op.getResultTypes().size() != 1) return failure();

  IRRewriter rewriter(op.getContext());

  // Insert a new block at the front of the function.
  Block *old_body = &op.getFunctionBody().front();
  Location loc = op.getLoc();
  llvm::SmallVector<Location> locs(old_body->getNumArguments(), loc);
  Block *new_body = rewriter.createBlock(&op.getBody(), op.getBody().begin(),
                                         old_body->getArgumentTypes(), locs);

  // Create the JIT compile op.
  auto jit_compile_op = rewriter.create<tf_framework::JITCompileOp>(
      loc, rewriter.getType<tf_framework::JITCallableType>(),
      /*ctx=*/std::nullopt);

  // Move the original functions operations into the body.
  {
    OpBuilder::InsertionGuard guard(rewriter);
    Block *jit_block = rewriter.createBlock(&jit_compile_op.getBody(), {},
                                            old_body->getArgumentTypes(), locs);

    rewriter.inlineBlockBefore(old_body, jit_block, jit_block->begin(),
                               jit_block->getArguments());

    Operation *terminator = jit_block->getTerminator();
    rewriter.setInsertionPointAfter(terminator);
    rewriter.create<tf_framework::JITCompileYieldOp>(
        loc, terminator->getOperands().front());
    terminator->erase();
  }

  // Create JIT execute op.
  auto execute = rewriter.create<tf_framework::JITExecuteOp>(
      loc, op.getResultTypes().front(), /*ctx=*/Value(),
      jit_compile_op.getResult(), new_body->getArguments());

  // Create a return.
  rewriter.create<func::ReturnOp>(loc, execute.getResult());
  return success();
}

LogicalResult RewriteToLargeSizeJit(FuncOp op) {
  // Rewrite all functions that have at most two arguments and a single result.
  if (op.getArgumentTypes().size() > 2 || op.getResultTypes().size() != 1)
    return failure();

  IRRewriter rewriter(op.getContext());
  Location loc = op.getLoc();

  // Insert a new block at the front of the function.
  Block *old_body = &op.getFunctionBody().front();
  llvm::SmallVector<Location> locs(old_body->getNumArguments(), loc);
  Block *new_body = rewriter.createBlock(&op.getBody(), op.getBody().begin(),
                                         old_body->getArgumentTypes(), locs);

  // Create large argument condition.
  auto arg_1 = new_body->getArgument(0);
  auto shape_1 = rewriter.create<shape::ShapeOfOp>(loc, arg_1);
  auto num_elems_1 = rewriter.create<shape::NumElementsOp>(loc, shape_1);
  Value cst_i32_limit = rewriter.create<arith::ConstantIndexOp>(loc, i32Limit);
  Value large_tensor_predicate = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sgt, num_elems_1, cst_i32_limit);
  if (new_body->getNumArguments() > 1) {
    auto arg_2 = new_body->getArgument(1);
    auto shape_2 = rewriter.create<shape::ShapeOfOp>(loc, arg_2);
    auto num_elems_2 = rewriter.create<shape::NumElementsOp>(loc, shape_2);
    large_tensor_predicate = rewriter.create<arith::OrIOp>(
        loc, large_tensor_predicate,
        // Compare op to check size of the second op
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                       num_elems_2, cst_i32_limit));
  }

  // Create dispatch code.
  auto jit_body_builder_fn = [&](OpBuilder &b, Location loc) {
    // Create JIT compile op.
    auto callable_ty = b.getType<tf_framework::JITCallableType>();
    auto jit_compile_op =
        b.create<tf_framework::JITCompileOp>(loc, callable_ty, /*ctx=*/Value());
    {
      OpBuilder::InsertionGuard g(b);
      Block *block = b.createBlock(
          &jit_compile_op.getBody(), {}, new_body->getArgumentTypes(),
          SmallVector<Location>(new_body->getNumArguments(), loc));
      b.setInsertionPointToStart(block);
      IRMapping bvm;
      bvm.map(old_body->getArguments(), block->getArguments());
      for (auto &op : old_body->without_terminator()) {
        b.clone(op, bvm);
      }
      b.create<tf_framework::JITCompileYieldOp>(
          loc, block->back().getResults().front());
    }

    // Create JIT execute op.
    auto jit_execute_op = b.create<tf_framework::JITExecuteOp>(
        loc, op.getResultTypes().front(), /*ctx=*/Value(),
        jit_compile_op.getResult(), new_body->getArguments());
    b.create<scf::YieldOp>(loc, jit_execute_op.getResult());
  };
  auto aot_body_builder_fn = [&](OpBuilder &b, Location loc) {
    IRMapping bvm;
    bvm.map(old_body->getArguments(), new_body->getArguments());
    Operation *last_clone;
    for (auto &op : old_body->without_terminator()) {
      last_clone = b.clone(op, bvm);
    }
    b.create<scf::YieldOp>(loc, last_clone->getResults().front());
  };

  // Create the conditional and return operation.
  auto ifOp = rewriter.create<scf::IfOp>(
      loc, large_tensor_predicate, jit_body_builder_fn, aot_body_builder_fn);
  rewriter.create<func::ReturnOp>(loc, ifOp.getResults().front());

  // Remove the old body.
  rewriter.eraseBlock(old_body);
  return success();
}

void PackJITCompileOp(tf_framework::JITCompileOp op,
                      llvm::ArrayRef<int64_t> tile_sizes,
                      llvm::ArrayRef<int64_t> unroll_factors, bool enable_ftz,
                      bool index_64bit, bool cpu_codegen) {
  IRRewriter rewriter(op.getContext());
  Block *body = op.SingleBlock::getBody();
  auto yield_op =
      llvm::cast<tf_framework::JITCompileYieldOp>(body->getTerminator());

  // Temporarily, build the module that would be JIT-compiled. This is only to
  // obtain the serialized code attribute.
  auto loc = op->getLoc();
  auto jit_module = rewriter.create<ModuleOp>(loc);
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(jit_module.SingleBlock::getBody());
    auto jit_function = rewriter.create<func::FuncOp>(
        loc, tf_framework::JITCompileFromStrOp::kJITEntryFunctionName,
        rewriter.getFunctionType(body->getArgumentTypes(),
                                 yield_op->getOperandTypes()));
    jit_function->setAttr(tf_framework::TFFrameworkDialect::kTFEntryAttrName,
                          rewriter.getUnitAttr());
    jit_function.getBody().takeBody(op.getBodyRegion());
    rewriter.setInsertionPointToEnd(&jit_function.getBody().front());
    rewriter.create<func::ReturnOp>(loc, yield_op.getResult());
    rewriter.eraseOp(yield_op);
  }

  // Serialize JIT module.
  std::string code;
  llvm::raw_string_ostream ss(code);
  assert(succeeded(jit_module.verify()));
  mlir::OpPrintingFlags flags;
  jit_module.print(ss, flags.assumeVerified());

  // Remove temporary module.
  rewriter.eraseOp(jit_module);

  // Finally, create the new JIT compile op.
  rewriter.setInsertionPointAfter(op);
  rewriter.replaceOpWithNewOp<tf_framework::JITCompileFromStrOp>(
      op, op->getResultTypes(), op.getCtx(), rewriter.getStringAttr(code),
      rewriter.getI64ArrayAttr(tile_sizes),
      rewriter.getI64ArrayAttr(unroll_factors),
      rewriter.getBoolAttr(enable_ftz), rewriter.getBoolAttr(index_64bit),
      rewriter.getBoolAttr(cpu_codegen));
}

#define GEN_PASS_DEF_FUNCTOJITINVOCATIONPASS
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct FuncToJITInvocationPass
    : public impl::FuncToJITInvocationPassBase<FuncToJITInvocationPass> {
  explicit FuncToJITInvocationPass(llvm::ArrayRef<int64_t> tile_sizes,
                                   llvm::ArrayRef<int64_t> unroll_factors,
                                   bool enable_ftz, bool index_64bit,
                                   bool cpu_codegen,
                                   bool jit_i64_indexed_for_large_tensors) {
    tile_sizes_ = tile_sizes;
    unroll_factors_ = unroll_factors;
    enable_ftz_ = enable_ftz;
    index_64bit_ = index_64bit;
    cpu_codegen_ = cpu_codegen;
    jit_i64_indexed_for_large_tensors_ = jit_i64_indexed_for_large_tensors;
  }

  void runOnOperation() override {
    if (jit_i64_indexed_for_large_tensors_) {
      if (failed(RewriteToLargeSizeJit(getOperation()))) {
        return signalPassFailure();
      }
    } else {
      if (failed(RewriteToFullJit(getOperation()))) {
        return signalPassFailure();
      }
    }

    getOperation().walk([&](tf_framework::JITCompileOp op) {
      PackJITCompileOp(op, tile_sizes_, unroll_factors_, enable_ftz_,
                       index_64bit_ || jit_i64_indexed_for_large_tensors_,
                       cpu_codegen_);
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateFuncToJITInvocationPass(
    llvm::ArrayRef<int64_t> tile_sizes, llvm::ArrayRef<int64_t> unroll_factors,
    bool enable_ftz, bool index_64bit, bool cpu_codegen,
    bool jit_i64_indexed_for_large_tensors) {
  return std::make_unique<FuncToJITInvocationPass>(
      tile_sizes, unroll_factors, enable_ftz, index_64bit, cpu_codegen,
      jit_i64_indexed_for_large_tensors);
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
