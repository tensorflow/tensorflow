#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOBASE_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOBASE_H

#include "triton/Conversion/MLIRTypes.h"

namespace mlir::triton {

class TargetInfoBase {
public:
  virtual bool supportMaximumMinimum() const = 0;

  virtual Value getClusterCTAId(RewriterBase &rewriter, Location loc) const = 0;

  virtual Value ballot(RewriterBase &rewriter, Location loc, Type type,
                       Value cmp) const = 0;

  // Store/load a value from shared memory, either in the same CTA or, if
  // `ctaId` is non-nullopt, in another CTA in the same group.
  //
  // A target that does not support cross-CTA transfers will assert if ctaId is
  // non-nullopt.
  //
  // Assumes the address is aligned to the width of `val`.
  virtual void storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                            std::optional<Value> ctaId, Value val,
                            Value pred) const = 0;
  virtual Value loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                            std::optional<Value> ctaId, Type elemTy,
                            Value pred) const = 0;

  void storeShared(RewriterBase &rewriter, Location loc, Value ptr, Value val,
                   Value pred) const {
    storeDShared(rewriter, loc, ptr, /*ctaId=*/std::nullopt, val, pred);
  }
  Value loadShared(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
                   Value pred) const {
    return loadDShared(rewriter, loc, ptr, /*ctaId=*/std::nullopt, elemTy,
                       pred);
  }

  virtual bool canUseStMatrix(RankedTensorType tensorTy,
                              ArrayRef<unsigned> repShape,
                              ArrayRef<unsigned> paddedRepShape,
                              ArrayRef<unsigned> order,
                              int swizzleByteSize) const = 0;

  virtual void storeMatrixShared(RewriterBase &rewriter, Location loc,
                                 Value ptr, Value val) const = 0;

  virtual Value shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                           int i) const = 0;
  virtual Value shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                          int i) const = 0;
  virtual Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                           int i) const = 0;
  virtual Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                           Value i) const = 0;

  virtual Value programId(RewriterBase &rewriter, Location loc,
                          ModuleOp moduleOp, int axis) const = 0;

  virtual bool warpReduce(RewriterBase &rewriter, Location loc,
                          SmallVector<Value> &acc, triton::ReduceOp op,
                          unsigned numLaneToReduce,
                          unsigned interleave) const = 0;

  virtual std::string getMulhiFuncName(Type resultElementTy) const = 0;
  // Emits LLVM code with |rewriter| to print a message following the given
  // format from the device. |formatStrStart| is the pointer to the start of
  // the format string global variable; |args| are the arguments to fill
  // placeholders in the format string.
  virtual void printf(RewriterBase &rewriter, Value formatStrStart,
                      int formatStrByteCount, ValueRange args,
                      ArrayRef<bool> isSigned = {}) const = 0;

  // Emits LLVM code with |rewriter| to print a message, particularly useful for
  // backend debug. |msg| is the message to print, |args| are the arguments to
  // fill placeholders in the |msg|.
  // NOTE: This function is used for backend debug. DO NOT DELETE.
  // Example use: targetInfo.printf(rewriter,"index: %d, value: %f", {index,
  // value});
  virtual void printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                      ArrayRef<bool> isSigned = {}) const = 0;

  // Emits LLVM code with |rewriter| to perform assertion failure with the given
  // |message| from the given |func| in |file|.
  virtual void assertFail(RewriterBase &rewriter, Location loc,
                          StringRef message, StringRef file, StringRef func,
                          int line) const = 0;

  virtual int getSharedAddressSpace() const = 0;

  virtual int getAddressSpace(Attribute addressSpace) const = 0;

  virtual bool supportVectorizedAtomics() const = 0;

  // Helper used by targets to annotate store operations during lowering to
  // llvm.
  virtual void storeOpAnnotation(triton::gpu::LocalStoreOp op,
                                 size_t localStoreOpCount, Type type) const {}

  virtual ~TargetInfoBase() {}
};
} // namespace mlir::triton
#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOBASE_H
