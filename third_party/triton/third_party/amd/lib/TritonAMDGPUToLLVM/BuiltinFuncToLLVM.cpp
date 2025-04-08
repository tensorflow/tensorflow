#include "TritonAMDGPUToLLVM/Passes.h"

#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTBUILTINFUNCTOLLVM
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

class CallOpConversion : public OpRewritePattern<LLVM::CallOp> {
public:
  CallOpConversion(mlir::MLIRContext *context, bool ftz)
      : OpRewritePattern<LLVM::CallOp>(context, 1), ftz(ftz) {}

  LogicalResult
  matchAndRewrite(LLVM::CallOp callOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (isPredicatedLoad(callOp)) {
      return convertPredicatedLoad(callOp, rewriter);
    } else if (isPredicatedStore(callOp)) {
      return convertPredicatedStore(callOp, rewriter);
    } else if (isWrappedLLVMIntrinsic(callOp)) {
      return convertToLLVMIntrinsic(callOp, rewriter);
    } else {
      return failure();
    }
  }

private:
  bool isPredicatedLoad(LLVM::CallOp callOp) const {
    return callOp.getCallee().value().contains(mlir::LLVM::AMD::predicatedLoad);
  }

  bool isPredicatedStore(LLVM::CallOp callOp) const {
    return callOp.getCallee().value().contains(
        mlir::LLVM::AMD::predicatedStore);
  }

  bool isWrappedLLVMIntrinsic(LLVM::CallOp callOp) const {
    if (std::optional<StringRef> callee = callOp.getCallee()) {
      if (callee.value().starts_with("__triton_hip_")) {
        return true;
      }
    }
    return false;
  }

  LogicalResult convertPredicatedStore(LLVM::CallOp callOp,
                                       mlir::PatternRewriter &rewriter) const {
    auto operands = callOp.getOperands();

    auto loc = callOp.getLoc();
    auto ptr = operands[0];
    auto val = operands[1];
    auto pred = operands[2];

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterStore =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterStore);
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, pred, trueBlock, afterStore);
    rewriter.setInsertionPointToStart(trueBlock);
    //               | vialatile | non-tmp | gcn instr gfx94
    // LLVM::StoreOp | 0         | 0       | (cg) global store
    //               | 0         | 1       | (cs) global store nt
    //               | 1         | 0/1     | (wt) global store sc0 sc1
    auto [volatileFlag, nonTmpFlag] =
        mlir::LLVM::AMD::getCacheModifierFlagsForPredicatedCall(callOp);
    auto storeOp = rewriter.create<LLVM::StoreOp>(
        loc, val, ptr, /*alignment=*/0, volatileFlag, nonTmpFlag);
    rewriter.create<LLVM::BrOp>(loc, afterStore);
    rewriter.setInsertionPointToStart(afterStore);
    rewriter.eraseOp(callOp);
    return mlir::success();
  }

  LogicalResult convertPredicatedLoad(LLVM::CallOp callOp,
                                      mlir::PatternRewriter &rewriter) const {
    auto operands = callOp.getOperands();
    auto result = callOp.getResult();

    auto loc = callOp.getLoc();
    auto elemTy = result.getType();
    auto ptr = operands[0];
    auto pred = operands[1];
    auto falseVal = operands[2];

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterLoad =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    afterLoad->addArgument({elemTy}, {loc});
    Block *trueBlock = rewriter.createBlock(afterLoad);
    Block *falseBlock =
        rewriter.splitBlock(trueBlock, rewriter.getInsertionPoint());
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, pred, trueBlock, falseBlock);
    rewriter.setInsertionPointToStart(trueBlock);
    //              | vialatile | non-tmp | gcn instr gfx94
    // LLVM::LoadOp | 0         | 0       | (ca) global load
    //              | 0/1       | 1       | (cg) global load nt
    //              | 1         | 0       | (cv) flat load sc0 sc1
    auto [volatileFlag, nonTmpFlag] =
        mlir::LLVM::AMD::getCacheModifierFlagsForPredicatedCall(callOp);
    auto loadOp = rewriter.create<LLVM::LoadOp>(
        loc, elemTy, ptr, /*alignment=*/0, volatileFlag, nonTmpFlag);
    rewriter.create<LLVM::BrOp>(loc, loadOp->getResult(0), afterLoad);
    rewriter.setInsertionPointToStart(falseBlock);
    rewriter.create<LLVM::BrOp>(loc, falseVal, afterLoad);
    rewriter.setInsertionPointToStart(afterLoad);
    Value loadVal = afterLoad->getArgument(0);
    rewriter.replaceOp(callOp, loadVal);
    return mlir::success();
  }

  LogicalResult convertToLLVMIntrinsic(LLVM::CallOp callOp,
                                       mlir::PatternRewriter &rewriter) const {
    StringRef calleeName = callOp.getCallee().value();

    auto operands = callOp.getOperands();
    auto result = callOp.getResult();

    LLVM::LLVMFunctionType calleeType = callOp.getCalleeFunctionType();
    Type returnType = calleeType.getReturnType();

    auto loc = callOp.getLoc();

    Operation *replacementOp = nullptr;
    if (calleeName == "__triton_hip_iabs") {
      assert(operands.size() == 1);
      replacementOp = rewriter.create<LLVM::AbsOp>(loc, returnType, operands[0],
                                                   /*is_int_min_poison=*/false);
    } else if (calleeName == "__triton_hip_fabs") {
      assert(operands.size() == 1);
      replacementOp =
          rewriter.create<LLVM::FAbsOp>(loc, returnType, operands[0]);
    } else if (calleeName == "__triton_hip_llrint") {
      assert(operands.size() == 1);
      // Note, LrintOp and LlrintOp result in a code-gen error
      Operation *op = rewriter.create<LLVM::RintOp>(loc, operands[0].getType(),
                                                    operands[0]);
      replacementOp =
          rewriter.create<LLVM::FPToSIOp>(loc, returnType, op->getResult(0));
    } else if (calleeName == "__triton_hip_fast_fdividef") {
      assert(operands.size() == 2);
      const char *intrinsic = "llvm.amdgcn.rcp.f32";
      auto rcpOp = LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic,
                                                   returnType, operands[1]);

      LLVM::FastmathFlagsAttr defaultFlags{};
      replacementOp = rewriter.create<LLVM::FMulOp>(
          loc, returnType, operands[0], rcpOp->getResult(0), defaultFlags);
    } else if (calleeName == "__triton_hip_fast_expf") {
      assert(operands.size() == 1);
      assert(operands[0].getType().getIntOrFloatBitWidth() == 32);
      const double log2e = 1.4426950408889634;
      LLVM::FastmathFlagsAttr defaultFlags{};
      auto mulOp = rewriter.create<LLVM::FMulOp>(
          loc, rewriter.getF32Type(), operands[0],
          LLVM::createConstantF32(loc, rewriter, log2e), defaultFlags);
      const char *intrinsic = ftz ? "llvm.amdgcn.exp2.f32" : "llvm.exp2.f32";

      replacementOp = LLVM::createLLVMIntrinsicCallOp(
          rewriter, loc, intrinsic, returnType, mulOp->getResult(0));
    }

    if (replacementOp) {
      rewriter.replaceOp(callOp, replacementOp);
      return mlir::success();
    }

    return mlir::failure();
  }

private:
  bool ftz;
};

struct ConvertBuiltinFuncToLLVM
    : public triton::impl::ConvertBuiltinFuncToLLVMBase<
          ConvertBuiltinFuncToLLVM> {
  explicit ConvertBuiltinFuncToLLVM(bool ftz) { this->ftz = ftz; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    GreedyRewriteConfig config;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Aggressive;

    RewritePatternSet patterns(context);
    patterns.add<CallOpConversion>(context, this->ftz);

    if (mlir::applyPatternsGreedily(mod, std::move(patterns), config)
            .failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertBuiltinFuncToLLVMPass(bool ftz) {
  return std::make_unique<ConvertBuiltinFuncToLLVM>(ftz);
}

} // namespace mlir::triton
