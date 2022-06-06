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

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

namespace {

// Replaces flattened memref arguments (base, aligned, offset, sizes, strides)
// with base and constants if the corresponding launch_func ops argument has
// static shape. Removes all arguments but base.
class PropagateStaticShapesPattern : public OpRewritePattern<LLVM::LLVMFuncOp> {
 public:
  explicit PropagateStaticShapesPattern(MLIRContext* ctx,
                                        SymbolTable& symbol_table,
                                        Type pointer_type)
      : OpRewritePattern<LLVM::LLVMFuncOp>(ctx),
        symbol_table_(symbol_table),
        pointer_type_(pointer_type) {}

 private:
  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp func_op,
                                PatternRewriter& rewriter) const final;

  SymbolTable& symbol_table_;
  Type pointer_type_;
};

class PropagateStaticShapesToKernelPass
    : public PropagateStaticShapesToKernelPassBase<
          PropagateStaticShapesToKernelPass> {
 public:
  explicit PropagateStaticShapesToKernelPass(Type pointer_type)
      : pointer_type_(pointer_type) {}

 private:
  void runOnOperation() override;

  Type pointer_type_;
};

}  // namespace

// Replaces 'arguments' (containing 'base', 'align', 'offset', 'sizes[rank]',
// 'strides[rank]') corresponding to statically shaped 'memref' with the base
// pointer and constants. The base pointer is changed to 'pointer_type' if
// provided.
static void ReplaceStaticMemRefArguments(ArrayRef<BlockArgument> arguments,
                                         MemRefType memref, Type pointer_type,
                                         PatternRewriter& rewriter) {
  assert(arguments.size() >= 3 && "expected at least 3 arguments");
  Value base = arguments[0];
  if (pointer_type) {
    // Change base to given type, replace with bitcast back to original type.
    Type type = base.getType();
    base.setType(pointer_type);
    auto cast = rewriter.create<LLVM::BitcastOp>(base.getLoc(), type, base);
    base.replaceAllUsesExcept(/*newValue=*/cast, /*exceptedUser=*/cast);
    base = cast.getResult();
  }

  // Replace uses of 'aligned' with 'base'.
  arguments[1].replaceAllUsesWith(base);
  // Replace uses of 'offset' with constant.
  arguments[2].replaceAllUsesWith(rewriter.create<LLVM::ConstantOp>(
      arguments[2].getLoc(), arguments[2].getType(),
      rewriter.getIntegerAttr(arguments[2].getType(), 0)));
  auto replace = [&](ArrayRef<int64_t> values,
                     ArrayRef<BlockArgument> arguments) {
    for (auto val_and_arg : llvm::zip_first(values, arguments)) {
      auto argument = std::get<1>(val_and_arg);
      argument.replaceAllUsesWith(rewriter.create<LLVM::ConstantOp>(
          argument.getLoc(), argument.getType(),
          rewriter.getIntegerAttr(argument.getType(),
                                  std::get<0>(val_and_arg))));
    }
  };
  // Replace 'sizes' and 'strides' with constants.
  replace(memref.getShape(), arguments.drop_front(3));
  auto strides = llvm::to_vector<4>(memref.getShape());
  std::partial_sum(strides.rbegin(), strides.rend(), strides.rbegin(),
                   std::multiplies<int64_t>());
  strides.push_back(1);
  replace(llvm::makeArrayRef(strides).drop_front(),
          arguments.drop_front(3 + memref.getRank()));
}

LogicalResult PropagateStaticShapesPattern::matchAndRewrite(
    LLVM::LLVMFuncOp func_op, PatternRewriter& rewriter) const {
  if (func_op.isExternal())
    return rewriter.notifyMatchFailure(func_op, "external");
  if (!func_op->getAttrOfType<UnitAttr>(
          gpu::GPUDialect::getKernelFuncAttrName())) {
    return rewriter.notifyMatchFailure(func_op, "missing gpu.kernel");
  }

  // Collect gpu.launch_func ops which launch the func_op kernel.
  Optional<SymbolTable::UseRange> sym_uses =
      symbol_table_.getSymbolUses(func_op, symbol_table_.getOp());
  if (!sym_uses)
    return rewriter.notifyMatchFailure(func_op, "failed to find symbol uses");
  auto mapper = [](SymbolTable::SymbolUse sym_use) {
    return dyn_cast<gpu::LaunchFuncOp>(sym_use.getUser());
  };
  auto filter = [](gpu::LaunchFuncOp op) -> bool { return op; };
  auto launch_ops = llvm::to_vector(
      llvm::make_filter_range(llvm::map_range(*sym_uses, mapper), filter));
  if (launch_ops.empty())
    return rewriter.notifyMatchFailure(func_op, "no gpu.launch_func uses");
  OperandRange operands = launch_ops.begin()->operands();
  if (llvm::any_of(launch_ops, [&](gpu::LaunchFuncOp op) {
        return op.operands().getTypes() != operands.getTypes();
      })) {
    return rewriter.notifyMatchFailure(func_op, "operand types mismatch");
  }

  rewriter.setInsertionPointToStart(&func_op.front());
  BitVector args_to_drop(func_op.getNumArguments());
  // Loop over the launch_op's 'operands' containing scalars and memrefs and the
  // func_ops's 'arguments' containing scalars and flattened memrefs. When an
  // operand is a staticlly shaped memref, replace the range of arguments
  // corresponding to the flattened memref with just the 'base' pointer.
  for (auto arguments = func_op.getArguments(); !arguments.empty();
       operands = operands.drop_front()) {
    auto memref = operands.getTypes().front().dyn_cast<MemRefType>();
    if (!memref) {
      // Scalar argument, advance by one.
      arguments = arguments.drop_front();
      continue;
    }
    if (!memref.hasRank()) break;  // Bail out if unranked.
    // memref is flattened to base, align, offset, strides and sizes.
    int64_t num_args = 3 + memref.getRank() * 2;
    auto is_ptr = [](BlockArgument arg) {
      return arg.getType().isa<LLVM::LLVMPointerType>();
    };
    auto is_int = [](BlockArgument arg) {
      return arg.getType().isa<IntegerType>();
    };
    // Bail out if the next num_args are not the expected type.
    if (arguments.size() < num_args) break;
    ArrayRef<BlockArgument> memref_args = arguments.take_front(num_args);
    if (!llvm::all_of(memref_args.take_front(2), is_ptr)) break;
    if (!llvm::all_of(memref_args.drop_front(2), is_int)) break;
    // Replace memref_args with just memref_args[0] if memref has static shape.
    if (memref.hasStaticShape() && memref.getLayout().isIdentity()) {
      ReplaceStaticMemRefArguments(memref_args, memref, pointer_type_,
                                   rewriter);
      unsigned arg_number = arguments.front().getArgNumber();
      // Drop all but 'base' from the flattened memref arguments.
      args_to_drop.set(arg_number + 1, arg_number + num_args);
    }
    arguments = arguments.drop_front(num_args);
  }
  if (args_to_drop.none()) {
    return rewriter.notifyMatchFailure(func_op, "no static shapes");
  }
  rewriter.updateRootInPlace(func_op, [&] {
    func_op.eraseArguments(args_to_drop);
    auto arg_types = llvm::to_vector(TypeRange(func_op.getArguments()));
    func_op.setType(LLVM::LLVMFunctionType::get(
        func_op.getFunctionType().getReturnType(), arg_types));
  });
  return success();
}

void PropagateStaticShapesToKernelPass::runOnOperation() {
  MLIRContext* ctx = getOperation().getContext();
  auto pointer_type = [&]() -> FailureOr<Type> {
    if (ptr_type_opt.empty()) return pointer_type_;
    Type type = parseType(ptr_type_opt, ctx);
    if (!type)
      return emitError(UnknownLoc::get(ctx), "invalid convert_pointer_args");
    return type;
  }();
  if (failed(pointer_type)) return signalPassFailure();
  SymbolTable symbol_table(getOperation());
  RewritePatternSet patterns(ctx);
  patterns.add<PropagateStaticShapesPattern>(ctx, symbol_table, *pointer_type);
  FrozenRewritePatternSet frozen(std::move(patterns));
  auto callback = [&](gpu::GPUModuleOp gpu_module) -> WalkResult {
    return applyPatternsAndFoldGreedily(gpu_module, frozen);
  };
  if (getOperation()->walk(callback).wasInterrupted())
    return signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
CreatePropagateStaticShapesToKernelPass(Type pointer_type) {
  return std::make_unique<PropagateStaticShapesToKernelPass>(pointer_type);
}

}  // namespace mlir
