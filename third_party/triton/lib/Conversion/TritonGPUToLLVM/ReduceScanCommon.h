#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_REDUCESCANCOMMON_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_REDUCESCANCOMMON_H

// TODO: refactor so that it doesn't fail if Allocation.h
// is included after utility.h (due to conflict in `store` macro
// and <atomic>
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

//
#include "mlir/IR/TypeUtilities.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include <iterator>
#include <type_traits>

#define DEBUG_TYPE "ttgpu_to_llvm"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton {
class ReduceOp;
class ScanOp;

inline SmallVector<Value>
inlineCombineBlock(ConversionPatternRewriter &rewriter, Block &combineBlock,
                   Block *insertionBlock, Block::iterator insertionPoint,
                   ValueRange combineArgs) {
  auto returnOp = combineBlock.getTerminator();
  rewriter.inlineBlockBefore(&combineBlock, insertionBlock, insertionPoint,
                             combineArgs);

  auto results = SmallVector<Value>(returnOp->getOperands());

  // Delete the terminator, which is no longer used
  rewriter.eraseOp(returnOp);
  return results;
}

inline SmallVector<Value> applyCombineOp(Location loc,
                                         ConversionPatternRewriter &rewriter,
                                         Region &combineOp, ValueRange acc,
                                         ValueRange cur, Value pred = {}) {
  // Allows for passing an uninitialized acc and use cur as the neutral element
  if (acc.size() == 0) {
    return cur;
  }
  assert(cur.size() == acc.size());

  // Create a new copy of the combine block, and try to speculatively inline it
  Block *currentBlock = rewriter.getBlock();
  Region &parent = *currentBlock->getParent();

  rewriter.cloneRegionBefore(combineOp, parent,
                             std::next(currentBlock->getIterator()));
  Block &newCombine = *currentBlock->getNextNode();

  llvm::SmallVector<Value> combineArgs(2 * acc.size());
  for (unsigned i = 0; i < acc.size(); ++i) {
    combineArgs[i] = acc[i];
    combineArgs[acc.size() + i] = cur[i];
  }

  auto isRegionSpeculatable =
      std::all_of(newCombine.begin(), newCombine.end(),
                  [](auto &op) { return isSpeculatable(&op); });

  if (!pred || isRegionSpeculatable) {
    // Fast path, region has no side effects so we can unconditionally execute
    return inlineCombineBlock(rewriter, newCombine, currentBlock,
                              rewriter.getInsertionPoint(), combineArgs);
  }

  // Slow case, create an if to only execute region when pred is true
  // #currentBlock
  // if (pred) {
  //   #newCombine
  //   results = combineOp(cur, acc)
  //   yield results
  // } else {
  //    yield undef
  // }
  // #thenBlock
  Block *thenBlock =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

  auto returnOp = newCombine.getTerminator();
  auto results = SmallVector<Value>(returnOp->getOperands());

  rewriter.setInsertionPointToEnd(currentBlock);
  SmallVector<Value> thenBlockArgs;
  thenBlockArgs.reserve(results.size());
  for (auto result : results) {
    auto ty = result.getType();
    auto undef = rewriter.create<LLVM::UndefOp>(loc, ty);
    thenBlockArgs.push_back(undef);
    thenBlock->addArgument(ty, loc);
  }
  rewriter.create<cf::CondBranchOp>(loc, pred, &newCombine, combineArgs,
                                    thenBlock, thenBlockArgs);

  // Split a block after the call.
  rewriter.setInsertionPointToEnd(&newCombine);
  rewriter.replaceOpWithNewOp<cf::BranchOp>(returnOp, thenBlock, results);
  rewriter.setInsertionPointToStart(thenBlock);
  return SmallVector<Value>(thenBlock->getArguments());
}

} // namespace mlir::triton

template <typename SourceOp>
class ConvertTritonGPUReduceScanToLLVMPattern
    : public ConvertOpToLLVMPattern<SourceOp> {
public:
  // Make sure the class is only instantiated with Reduce and Scan
  static_assert(std::is_same_v<SourceOp, ReduceOp> ||
                std::is_same_v<SourceOp, ScanOp>);

  using ConvertOpToLLVMPattern<SourceOp>::getTypeConverter;
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;

  // Return the pointee type of the shared memory pointer for operand i.
  Type getElementType(SourceOp op, int i) const {
    auto ty = op.getInputTypes()[i].getElementType();
    return getTypeConverter()->convertType(ty);
  }

  // Helper to compute the smem bases in both reductions and scans
  SmallVector<Value> getSmemBases(SourceOp op, unsigned elems,
                                  ConversionPatternRewriter &rewriter,
                                  const TargetInfoBase &targetInfo) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // indices will store the index of the op operands in descending order
    // of their bitwidths
    std::vector<unsigned> indices(op.getNumOperands());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](unsigned i, unsigned j) {
      return op.getElementTypes()[i].getIntOrFloatBitWidth() >
             op.getElementTypes()[j].getIntOrFloatBitWidth();
    });
    // Assign base index to each operand in their order in indices
    std::map<unsigned, Value> indexToBase;
    auto basePtr =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    indexToBase[indices[0]] = basePtr;
    for (unsigned i = 1; i < op.getNumOperands(); ++i) {
      indexToBase[indices[i]] =
          b.gep(basePtr.getType(), getElementType(op, indices[i - 1]),
                indexToBase[indices[i - 1]], b.i32_val(elems));
    }
    // smemBases[k] is the base pointer for the k-th operand
    SmallVector<Value> smemBases(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      smemBases[i] = indexToBase[i];
    }
    return smemBases;
  }
};

#endif
