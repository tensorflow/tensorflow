#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_ATOMICRMWOPSEMITTER_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_ATOMICRMWOPSEMITTER_H_

#include "TargetInfo.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Analysis/Utility.h"

namespace mlir::LLVM::AMD {

class AtomicRMWEmitter {
public:
  AtomicRMWEmitter(const mlir::triton::AMD::TargetInfo &targetInfo,
                   LLVM::AtomicBinOp binOp, LLVM::AtomicOrdering memOrder,
                   StringRef scopeStr)
      : targetInfo(targetInfo), binOp(binOp), memOrder(memOrder),
        scopeStr(scopeStr) {}

  Value emitAtomicRMW(RewriterBase &rewriter, Value rmwPtr, Value valElem,
                      Value rmwMask, std::optional<Value> sharedMemBase,
                      bool enableIntraWaveReduce) const;

  Value emitPairedAtomicForEvenTID(RewriterBase &rewriter, Value rmwPtr,
                                   Value valElem, Value rmwMask,
                                   bool checkPairs = true) const;

private:
  const mlir::triton::AMD::TargetInfo &targetInfo;

  mlir::LLVM::AtomicBinOp binOp;
  mlir::LLVM::AtomicOrdering memOrder;
  std::string scopeStr;

  Value atomicIntraWaveReduce(RewriterBase &rewriter, Value rmwPtr,
                              Value operand, LLVM::AtomicBinOp opKind,
                              LLVM::AtomicOrdering memOrdering,
                              StringRef scope) const;
};

} // namespace mlir::LLVM::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_ATOMICRMWEMITTER_H_
