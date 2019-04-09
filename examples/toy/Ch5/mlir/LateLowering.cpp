//====- LateLowering.cpp - Lowering from Toy+Linalg to LLVM -===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements late lowering of IR mixing Toy and Linalg to LLVM.
// It involves intemerdiate steps:
// -
// - a mix of affine and standard dialect.
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"

#include "linalg1/Intrinsics.h"
#include "linalg1/ViewOp.h"
#include "linalg3/ConvertToLLVMDialect.h"
#include "linalg3/TensorOps.h"
#include "linalg3/Transforms.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"

#include <algorithm>

using namespace mlir;

namespace {
/// Utility function for type casting: this is making the type checker happy,
/// while delaying the actual work involved to convert the type. Most of the
/// time both side of the cast (producer and consumer) will be lowered to a
/// dialect like LLVM and end up with the same LLVM representation, at which
/// point this becomes a no-op and is eliminated.
Value *typeCast(FuncBuilder &builder, Value *val, Type destTy) {
  if (val->getType() == destTy)
    return val;
  return builder.create<toy::TypeCastOp>(val->getLoc(), val, destTy)
      .getResult();
}

/// Create a type cast to turn a toy.array into a memref. The Toy Array will be
/// lowered to a memref during buffer allocation, at which point the type cast
/// becomes useless.
Value *memRefTypeCast(FuncBuilder &builder, Value *val) {
  if (val->getType().isa<MemRefType>())
    return val;
  auto toyArrayTy = val->getType().dyn_cast<toy::ToyArrayType>();
  if (!toyArrayTy)
    return val;
  return typeCast(builder, val, toyArrayTy.toMemref());
}

/// Lower a toy.add to an affine loop nest.
///
/// This class inherit from `DialectOpConversion` and override `rewrite`,
/// similarly to the PatternRewriter introduced in the previous chapter.
/// It will be called by the DialectConversion framework (see `LateLowering`
/// class below).
class AddOpConversion : public DialectOpConversion {
public:
  explicit AddOpConversion(MLIRContext *context)
      : DialectOpConversion(toy::AddOp::getOperationName(), 1, context) {}

  /// Lower the `op` by generating IR using the `rewriter` builder. The builder
  /// is setup with a new function, the `operands` array has been populated with
  /// the rewritten operands for `op` in the new function.
  /// The results created by the new IR with the builder are returned, and their
  /// number must match the number of result of `op`.
  SmallVector<Value *, 4> rewrite(Operation *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    auto add = op->cast<toy::AddOp>();
    auto loc = add.getLoc();
    // Create a `toy.alloc` operation to allocate the output buffer for this op.
    Value *result = memRefTypeCast(
        rewriter, rewriter.create<toy::AllocOp>(loc, add.getResult()->getType())
                      .getResult());
    Value *lhs = memRefTypeCast(rewriter, operands[0]);
    Value *rhs = memRefTypeCast(rewriter, operands[1]);

    using namespace edsc;
    ScopedContext scope(rewriter, loc);
    ValueHandle zero = intrinsics::constant_index(0);
    MemRefView vRes(result), vLHS(lhs), vRHS(rhs);
    IndexedValue iRes(result), iLHS(lhs), iRHS(rhs);
    IndexHandle i, j, M(vRes.ub(0));
    if (vRes.rank() == 1) {
      LoopNestBuilder({&i}, {zero}, {M}, {1})({iRes(i) = iLHS(i) + iRHS(i)});
    } else {
      assert(vRes.rank() == 2 && "only rank 1 and 2 are supported right now");
      IndexHandle N(vRes.ub(1));
      LoopNestBuilder({&i, &j}, {zero, zero}, {M, N},
                      {1, 1})({iRes(i, j) = iLHS(i, j) + iRHS(i, j)});
    }

    // Return the newly allocated buffer, with a type.cast to preserve the
    // consumers.
    return {typeCast(rewriter, result, add.getType())};
  }
};

/// Lowers `toy.print` to a loop nest calling `printf` on every individual
/// elements of the array.
class PrintOpConversion : public DialectOpConversion {
public:
  explicit PrintOpConversion(MLIRContext *context)
      : DialectOpConversion(toy::PrintOp::getOperationName(), 1, context) {}

  SmallVector<Value *, 4> rewrite(Operation *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    // Get or create the declaration of the printf function in the module.
    Function *printfFunc = getPrintf(*op->getFunction()->getModule());

    auto print = op->cast<toy::PrintOp>();
    auto loc = print.getLoc();
    // We will operate on a MemRef abstraction, we use a type.cast to get one
    // if our operand is still a Toy array.
    Value *operand = memRefTypeCast(rewriter, operands[0]);
    Type retTy = printfFunc->getType().getResult(0);

    // Create our loop nest now
    using namespace edsc;
    using llvmCall = intrinsics::ValueBuilder<LLVM::CallOp>;
    ScopedContext scope(rewriter, loc);
    ValueHandle zero = intrinsics::constant_index(0);
    ValueHandle fmtCst(getConstantCharBuffer(rewriter, loc, "%f "));
    MemRefView vOp(operand);
    IndexedValue iOp(operand);
    IndexHandle i, j, M(vOp.ub(0));

    ValueHandle fmtEol(getConstantCharBuffer(rewriter, loc, "\n"));
    if (vOp.rank() == 1) {
      // clang-format off
      LoopBuilder(&i, zero, M, 1)({
        llvmCall(retTy,
                 rewriter.getFunctionAttr(printfFunc),
                 {fmtCst, iOp(i)})
      });
      llvmCall(retTy, rewriter.getFunctionAttr(printfFunc), {fmtEol});
      // clang-format on
    } else {
      IndexHandle N(vOp.ub(1));
      // clang-format off
      LoopBuilder(&i, zero, M, 1)({
        LoopBuilder(&j, zero, N, 1)({
          llvmCall(retTy,
                   rewriter.getFunctionAttr(printfFunc),
                   {fmtCst, iOp(i, j)})
        }),
        llvmCall(retTy, rewriter.getFunctionAttr(printfFunc), {fmtEol})
      });
      // clang-format on
    }
    return {};
  }

private:
  // Turn a string into a toy.alloc (malloc/free abstraction) and a sequence
  // of stores into the buffer, and return a MemRef into the buffer.
  Value *getConstantCharBuffer(FuncBuilder &builder, Location loc,
                               StringRef data) const {
    auto retTy =
        builder.getMemRefType(data.size() + 1, builder.getIntegerType(8));
    Value *result = builder.create<toy::AllocOp>(loc, retTy).getResult();
    using namespace edsc;
    using intrinsics::constant_index;
    using intrinsics::constant_int;
    ScopedContext scope(builder, loc);
    MemRefView vOp(result);
    IndexedValue iOp(result);
    for (uint64_t i = 0; i < data.size(); ++i) {
      iOp(constant_index(i)) = constant_int(data[i], 8);
    }
    iOp(constant_index(data.size())) = constant_int(0, 8);
    return result;
  }

  /// Return the prototype declaration for printf in the module, create it if
  /// necessary.
  Function *getPrintf(Module &module) const {
    auto *printfFunc = module.getNamedFunction("printf");
    if (printfFunc)
      return printfFunc;

    // Create a function declaration for printf, signature is `i32 (i8*, ...)`
    Builder builder(&module);
    MLIRContext *context = module.getContext();
    LLVM::LLVMDialect *llvmDialect = static_cast<LLVM::LLVMDialect *>(
        module.getContext()->getRegisteredDialect("llvm"));
    auto &llvmModule = llvmDialect->getLLVMModule();
    llvm::IRBuilder<> llvmBuilder(llvmModule.getContext());

    auto llvmI32Ty = LLVM::LLVMType::get(context, llvmBuilder.getIntNTy(32));
    auto llvmI8PtrTy =
        LLVM::LLVMType::get(context, llvmBuilder.getIntNTy(8)->getPointerTo());
    auto printfTy = builder.getFunctionType({llvmI8PtrTy}, {llvmI32Ty});
    printfFunc = new Function(builder.getUnknownLoc(), "printf", printfTy);
    // It should be variadic, but we don't support it fully just yet.
    printfFunc->setAttr("std.varargs", builder.getBoolAttr(true));
    module.getFunctions().push_back(printfFunc);
    return printfFunc;
  }
};

/// Lowers constant to a sequence of store in a buffer.
class ConstantOpConversion : public DialectOpConversion {
public:
  explicit ConstantOpConversion(MLIRContext *context)
      : DialectOpConversion(toy::ConstantOp::getOperationName(), 1, context) {}

  SmallVector<Value *, 4> rewrite(Operation *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    toy::ConstantOp cstOp = op->cast<toy::ConstantOp>();
    auto loc = cstOp.getLoc();
    auto retTy = cstOp.getResult()->getType().cast<toy::ToyArrayType>();
    auto shape = retTy.getShape();
    Value *result = memRefTypeCast(
        rewriter, rewriter.create<toy::AllocOp>(loc, retTy).getResult());

    auto cstValue = cstOp.getValue();
    auto f64Ty = rewriter.getF64Type();
    using namespace edsc;
    using intrinsics::constant_float;
    using intrinsics::constant_index;
    ScopedContext scope(rewriter, loc);
    MemRefView vOp(result);
    IndexedValue iOp(result);
    for (uint64_t i = 0; i < shape[0]; ++i) {
      if (shape.size() == 1) {
        auto value = cstValue.getValue(ArrayRef<uint64_t>{i})
                         .cast<FloatAttr>()
                         .getValue();
        iOp(constant_index(i)) = constant_float(value, f64Ty);
        continue;
      }
      for (uint64_t j = 0; j < shape[1]; ++j) {
        auto value = cstValue.getValue(ArrayRef<uint64_t>{i, j})
                         .cast<FloatAttr>()
                         .getValue();
        iOp(constant_index(i), constant_index(j)) =
            constant_float(value, f64Ty);
      }
    }
    return {result};
  }
};

/// Lower transpose operation to an affine loop nest.
class TransposeOpConversion : public DialectOpConversion {
public:
  explicit TransposeOpConversion(MLIRContext *context)
      : DialectOpConversion(toy::TransposeOp::getOperationName(), 1, context) {}

  SmallVector<Value *, 4> rewrite(Operation *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    auto transpose = op->cast<toy::TransposeOp>();
    auto loc = transpose.getLoc();
    Value *result = memRefTypeCast(
        rewriter,
        rewriter.create<toy::AllocOp>(loc, transpose.getResult()->getType())
            .getResult());
    Value *operand = memRefTypeCast(rewriter, operands[0]);

    using namespace edsc;
    ScopedContext scope(rewriter, loc);
    ValueHandle zero = intrinsics::constant_index(0);
    MemRefView vRes(result), vOperand(operand);
    IndexedValue iRes(result), iOperand(operand);
    IndexHandle i, j, M(vRes.ub(0)), N(vRes.ub(1));
    // clang-format off
    LoopNestBuilder({&i, &j}, {zero, zero}, {M, N}, {1, 1})({
      iRes(i, j) = iOperand(j, i)
    });
    // clang-format on

    return {typeCast(rewriter, result, transpose.getType())};
  }
};

// Lower toy.return to standard return operation.
class ReturnOpConversion : public DialectOpConversion {
public:
  explicit ReturnOpConversion(MLIRContext *context)
      : DialectOpConversion(toy::ReturnOp::getOperationName(), 1, context) {}

  SmallVector<Value *, 4> rewrite(Operation *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    auto retOp = op->cast<toy::ReturnOp>();
    using namespace edsc;
    auto loc = retOp.getLoc();
    // Argument is optional, handle both cases.
    if (retOp.getNumOperands())
      rewriter.create<ReturnOp>(loc, operands[0]);
    else
      rewriter.create<ReturnOp>(loc);
    return {};
  }
};

/// This is the main class registering our individual converter classes with
/// the DialectConversion framework in MLIR.
class LateLowering : public DialectConversion {
protected:
  /// Initialize the list of converters.
  llvm::DenseSet<DialectOpConversion *>
  initConverters(MLIRContext *context) override {
    return ConversionListBuilder<AddOpConversion, PrintOpConversion,
                                 ConstantOpConversion, TransposeOpConversion,
                                 ReturnOpConversion>::build(&allocator,
                                                            context);
  }

  /// Convert a Toy type, this gets called for block and region arguments, and
  /// attributes.
  Type convertType(Type t) override {
    if (auto array = t.cast<toy::ToyArrayType>()) {
      return array.toMemref();
    }
    return t;
  }

private:
  llvm::BumpPtrAllocator allocator;
};

/// This is lowering to Linalg the parts that can be (matmul and add on arrays)
/// and is targeting LLVM otherwise.
struct LateLoweringPass : public ModulePass<LateLoweringPass> {

  void runOnModule() override {
    // Perform Toy specific lowering
    if (failed(LateLowering().convert(&getModule()))) {
      getModule().getContext()->emitError(
          UnknownLoc::get(getModule().getContext()), "Error lowering Toy\n");
      signalPassFailure();
    }
    // At this point the IR is almost using only standard and affine dialects.
    // A few things remain before we emit LLVM IR. First to reuse as much of
    // MLIR as possible we will try to lower everything to the standard and/or
    // affine dialect: they already include conversion to the LLVM dialect.

    // First patch calls type to return memref instead of ToyArray
    for (auto &function : getModule()) {
      function.walk([&](Operation *op) {
        auto callOp = op->dyn_cast<CallOp>();
        if (!callOp)
          return;
        if (!callOp.getNumResults())
          return;
        auto retToyTy =
            callOp.getResult(0)->getType().dyn_cast<toy::ToyArrayType>();
        if (!retToyTy)
          return;
        callOp.getResult(0)->setType(retToyTy.toMemref());
      });
    }

    for (auto &function : getModule()) {
      function.walk([&](Operation *op) {
        // Turns toy.alloc into sequence of alloc/dealloc (later malloc/free).
        if (auto allocOp = op->dyn_cast<toy::AllocOp>()) {
          auto result = allocTensor(allocOp);
          allocOp.replaceAllUsesWith(result);
          allocOp.erase();
          return;
        }
        // Eliminate all type.cast before lowering to LLVM.
        if (auto typeCastOp = op->dyn_cast<toy::TypeCastOp>()) {
          typeCastOp.replaceAllUsesWith(typeCastOp.getOperand());
          typeCastOp.erase();
          return;
        }
      });
    }

    // Lower Linalg to affine
    for (auto &function : getModule())
      linalg::lowerToLoops(&function);

    getModule().dump();

    // Finally convert to LLVM Dialect
    linalg::convertLinalg3ToLLVM(getModule());
  }

  /// Allocate buffers (malloc/free) for Toy operations. This can't be done as
  /// part of dialect conversion framework since we need to insert `dealloc`
  /// operations just before the return, but the conversion framework is
  /// operating in a brand new function: we don't have the return to hook the
  /// dealloc operations.
  Value *allocTensor(toy::AllocOp alloc) {
    FuncBuilder builder(alloc);
    auto retTy = alloc.getResult()->getType();

    auto memRefTy = retTy.dyn_cast<MemRefType>();
    if (!memRefTy)
      memRefTy = retTy.cast<toy::ToyArrayType>().toMemref();
    if (!memRefTy) {
      alloc.emitOpError("is expected to allocate a Toy array or a MemRef");
      llvm_unreachable("fatal error");
    }
    auto loc = alloc.getLoc();
    Value *result = builder.create<AllocOp>(loc, memRefTy).getResult();

    // Insert a `dealloc` operation right before the `return` operations, unless
    // it is returned itself in which case the caller is responsible for it.
    builder.getFunction()->walk([&](Operation *op) {
      auto returnOp = op->dyn_cast<ReturnOp>();
      if (!returnOp)
        return;
      if (returnOp.getNumOperands() && returnOp.getOperand(0) == alloc)
        return;
      builder.setInsertionPoint(returnOp);
      builder.create<DeallocOp>(alloc.getLoc(), result);
    });
    return result;
  }
};
} // end anonymous namespace

namespace toy {
Pass *createLateLoweringPass() { return new LateLoweringPass(); }

std::unique_ptr<DialectConversion> makeToyLateLowering() {
  return llvm::make_unique<LateLowering>();
}

} // namespace toy
