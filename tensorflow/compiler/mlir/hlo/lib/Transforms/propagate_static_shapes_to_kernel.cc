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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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
                                        SymbolTable& symbolTable,
                                        Type pointerType)
      : OpRewritePattern<LLVM::LLVMFuncOp>(ctx),
        symbolTable(symbolTable),
        pointerType(pointerType) {}

 private:
  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp funcOp,
                                PatternRewriter& rewriter) const final;

  SymbolTable& symbolTable;
  Type pointerType;
};

class PropagateStaticShapesToKernelPass
    : public PropagateStaticShapesToKernelPassBase<
          PropagateStaticShapesToKernelPass> {
 public:
  explicit PropagateStaticShapesToKernelPass(Type pointerType)
      : pointerType(pointerType) {}

 private:
  void runOnOperation() override;

  Type pointerType;
};

}  // namespace

// Replaces 'arguments' (containing 'base', 'align', 'offset', 'sizes[rank]',
// 'strides[rank]') corresponding to statically shaped 'memref' with the base
// pointer and constants. The base pointer is changed to 'pointer_type' if
// provided.
static void replaceStaticMemRefArguments(ArrayRef<BlockArgument> arguments,
                                         MemRefType memref, Type pointerType,
                                         PatternRewriter& rewriter) {
  assert(arguments.size() >= 3 && "expected at least 3 arguments");
  Value base = arguments[0];
  if (pointerType) {
    // Change base to given type, replace with bitcast back to original type.
    Type type = base.getType();
    base.setType(pointerType);
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
    for (auto valAndArg : llvm::zip_first(values, arguments)) {
      auto argument = std::get<1>(valAndArg);
      argument.replaceAllUsesWith(rewriter.create<LLVM::ConstantOp>(
          argument.getLoc(), argument.getType(),
          rewriter.getIntegerAttr(argument.getType(), std::get<0>(valAndArg))));
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
    LLVM::LLVMFuncOp funcOp, PatternRewriter& rewriter) const {
  if (funcOp.isExternal())
    return rewriter.notifyMatchFailure(funcOp, "external");
  if (!funcOp->getAttrOfType<UnitAttr>(
          gpu::GPUDialect::getKernelFuncAttrName())) {
    return rewriter.notifyMatchFailure(funcOp, "missing gpu.kernel");
  }

  // Collect gpu.launch_func ops which launch the func_op kernel.
  Optional<SymbolTable::UseRange> symUses =
      symbolTable.getSymbolUses(funcOp, symbolTable.getOp());
  if (!symUses)
    return rewriter.notifyMatchFailure(funcOp, "failed to find symbol uses");
  auto mapper = [](SymbolTable::SymbolUse symUse) {
    return dyn_cast<gpu::LaunchFuncOp>(symUse.getUser());
  };
  auto filter = [](gpu::LaunchFuncOp op) -> bool { return op; };
  auto launchOps = llvm::to_vector(
      llvm::make_filter_range(llvm::map_range(*symUses, mapper), filter));
  if (launchOps.empty())
    return rewriter.notifyMatchFailure(funcOp, "no gpu.launch_func uses");
  OperandRange operands = launchOps.begin()->operands();
  if (llvm::any_of(launchOps, [&](gpu::LaunchFuncOp op) {
        return op.operands().getTypes() != operands.getTypes();
      })) {
    return rewriter.notifyMatchFailure(funcOp, "operand types mismatch");
  }

  rewriter.setInsertionPointToStart(&funcOp.front());
  BitVector argsToDrop(funcOp.getNumArguments());
  // Loop over the launch_op's 'operands' containing scalars and memrefs and the
  // func_ops's 'arguments' containing scalars and flattened memrefs. When an
  // operand is a staticlly shaped memref, replace the range of arguments
  // corresponding to the flattened memref with just the 'base' pointer.
  for (auto arguments = funcOp.getArguments(); !arguments.empty();
       operands = operands.drop_front()) {
    auto memref = operands.getTypes().front().dyn_cast<MemRefType>();
    if (!memref) {
      // Scalar argument, advance by one.
      arguments = arguments.drop_front();
      continue;
    }
    if (!memref.hasRank()) break;  // Bail out if unranked.
    // memref is flattened to base, align, offset, strides and sizes.
    int64_t numArgs = 3 + memref.getRank() * 2;
    auto isPtr = [](BlockArgument arg) {
      return arg.getType().isa<LLVM::LLVMPointerType>();
    };
    auto isInt = [](BlockArgument arg) {
      return arg.getType().isa<IntegerType>();
    };
    // Bail out if the next num_args are not the expected type.
    if (arguments.size() < numArgs) break;
    ArrayRef<BlockArgument> memrefArgs = arguments.take_front(numArgs);
    if (!llvm::all_of(memrefArgs.take_front(2), isPtr)) break;
    if (!llvm::all_of(memrefArgs.drop_front(2), isInt)) break;
    // Replace memref_args with just memref_args[0] if memref has static shape.
    if (memref.hasStaticShape() && memref.getLayout().isIdentity()) {
      replaceStaticMemRefArguments(memrefArgs, memref, pointerType, rewriter);
      unsigned argNumber = arguments.front().getArgNumber();
      // Drop all but 'base' from the flattened memref arguments.
      argsToDrop.set(argNumber + 1, argNumber + numArgs);
    }
    arguments = arguments.drop_front(numArgs);
  }
  if (argsToDrop.none()) {
    return rewriter.notifyMatchFailure(funcOp, "no static shapes");
  }
  rewriter.updateRootInPlace(funcOp, [&] {
    funcOp.eraseArguments(argsToDrop);
    auto argTypes = llvm::to_vector(TypeRange(funcOp.getArguments()));
    funcOp.setType(LLVM::LLVMFunctionType::get(
        funcOp.getFunctionType().getReturnType(), argTypes));
  });
  return success();
}

void PropagateStaticShapesToKernelPass::runOnOperation() {
  MLIRContext* ctx = getOperation().getContext();
  auto pointerType = [&]() -> FailureOr<Type> {
    if (ptr_type_opt.empty()) return this->pointerType;
    Type type = parseType(ptr_type_opt, ctx);
    if (!type)
      return emitError(UnknownLoc::get(ctx), "invalid convert_pointer_args");
    return type;
  }();
  if (failed(pointerType)) return signalPassFailure();
  SymbolTable symbolTable(getOperation());
  RewritePatternSet patterns(ctx);
  patterns.add<PropagateStaticShapesPattern>(ctx, symbolTable, *pointerType);
  FrozenRewritePatternSet frozen(std::move(patterns));
  auto callback = [&](gpu::GPUModuleOp gpuModule) -> WalkResult {
    return applyPatternsAndFoldGreedily(gpuModule, frozen);
  };
  if (getOperation()->walk(callback).wasInterrupted())
    return signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
createPropagateStaticShapesToKernelPass(Type pointerType) {
  return std::make_unique<PropagateStaticShapesToKernelPass>(pointerType);
}

}  // namespace mlir
