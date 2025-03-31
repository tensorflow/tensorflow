#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// Return true if the given funcOp is a pure matmul problem; i.e.,
// a single main loop with a single dot.
static bool isPureMatmulFunc(triton::FuncOp funcOp) {
  bool isMatmul = true;
  bool foundLoop = false;
  funcOp.walk([&](scf::ForOp forOp) -> void {
    int counter = 0;
    forOp.walk([&counter](triton::DotOp dotOp) { ++counter; });
    isMatmul = (isMatmul && (counter == 1));
    foundLoop = true;
  });
  return foundLoop && isMatmul;
}

// Return true if the given ForOp contains a pure matmul problem; i.e.,
// single dot and at least 2 glboal loads in the main loop.
static bool isPureMatmulLoop(scf::ForOp forOp) {
  int dotCounter = 0;
  int loadCounter = 0;
  forOp.walk([&](Operation *op) {
    if (isa<triton::DotOp>(op))
      ++dotCounter;
    else if (isa<triton::LoadOp>(op))
      ++loadCounter;
  });
  return dotCounter == 1 && loadCounter >= 2;
}

// Search through block to find earliest insertion point for move op. This can
// be either an atomic op or last usage of source pointer. Search ends when move
// op is encountered.
static llvm::ilist<Operation>::iterator
findEarlyInsertionPoint(Block *block, Operation *move) {
  Value src;
  if (auto ld = dyn_cast<triton::LoadOp>(move))
    src = ld.getPtr();

  auto ipnt = block->end();
  for (auto bi = block->begin(); bi != block->end(); ++bi) {
    auto *op = &*bi;
    if (op == move) // Don't move later than current location
      break;

    op->walk([&](Operation *wop) {
      if (src) {
        // Check for ops accessing src value.
        for (auto opr : wop->getOperands()) {
          if (opr == src)
            ipnt = bi;
        }
      }
      // Atomics used for global synchronization.
      if (isa<triton::AtomicRMWOp, triton::AtomicCASOp>(wop))
        ipnt = bi;
      // Break at barrier
      if (isa<gpu::BarrierOp>(wop))
        ipnt = bi;
      // Break at loops.
      if (isa<scf::ForOp, scf::WhileOp>(wop))
        ipnt = bi;
    });
  }
  return ipnt;
}

// Return the first user in the same block of the given op. If the user is in a
// nested block then return the op owning the block. Return nullptr if not
// existing.
static Operation *getFirstUseInSameBlock(Operation *op) {
  SmallVector<Operation *> usersInSameBlock;
  for (auto user : op->getUsers()) {
    if (Operation *ancestor = op->getBlock()->findAncestorOpInBlock(*user))
      usersInSameBlock.push_back(ancestor);
  }
  auto minOpIt =
      llvm::min_element(usersInSameBlock, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });
  return minOpIt != usersInSameBlock.end() ? *minOpIt : nullptr;
}

// Check if the operation opInsideLoop is inside any scf::ForOp and
// opOutsideLoop is not inside the same loop.
static bool isCrossLoopBoundary(mlir::Operation *opInsideLoop,
                                mlir::Operation *opOutsideLoop) {
  scf::ForOp parentForOp = opInsideLoop->getParentOfType<scf::ForOp>();
  return parentForOp && !parentForOp->isAncestor(opOutsideLoop);
}

//===----------------------------------------------------------------------===//
// Reorder mechanisms
//===----------------------------------------------------------------------===//

// Sink dot layout conversions into loops to decrease register pressure when
// possible.
static void sinkDotConversion(triton::FuncOp funcOp) {
  DenseMap<Operation *, Operation *> opToMove;
  funcOp.walk([&](ttg::ConvertLayoutOp op) {
    Attribute encoding = op.getType().getEncoding();
    if (!isa_and_nonnull<ttg::DotOperandEncodingAttr>(encoding))
      return;
    if (!op->hasOneUse())
      return;
    Operation *user = *op->getUsers().begin();
    if (user->getParentOfType<scf::ForOp>() ==
        op->getParentOfType<scf::ForOp>())
      return;
    opToMove[op] = user;
  });

  for (auto &kv : opToMove)
    kv.first->moveBefore(kv.second);
}

// Sink conversion after the last dealloc but before the first use in its block.
// This helps to avoid unnecessary shared memory allocation.
static void moveDownCoversion(triton::FuncOp funcOp) {
  SmallVector<ttg::ConvertLayoutOp> convertOps;
  funcOp.walk([&](ttg::ConvertLayoutOp op) { convertOps.push_back(op); });

  for (auto op : convertOps) {
    Operation *user = getFirstUseInSameBlock(op);
    for (auto it = Block::iterator(op), ie = op->getBlock()->end();
         it != ie && &*it != user; ++it)
      if (isa<ttg::LocalDeallocOp>(&*it))
        op->moveAfter(&*it);
  }
}

// Move transpositions just after their definition.
static void moveUpTranspose(triton::FuncOp funcOp) {
  SmallVector<triton::TransposeOpInterface> transOps;
  funcOp.walk([&](triton::TransposeOpInterface op) { transOps.push_back(op); });

  for (auto op : transOps)
    if (Operation *argOp = op.getSrc().getDefiningOp())
      op->moveAfter(argOp);
}

// Schedule global load and local store ops for better GEMM performance.
static void scheduleGlobalLoadLocalStore(Operation *parentOp) {
  SmallVector<Operation *> moveOps;

  // Search through the forOp initArgs to find global loads for a GEMM that
  // the pipeliner may have peeled into a loop prologue.
  if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
    SmallVector<Value> vals = forOp.getInitArgs();
    while (!vals.empty()) {
      SmallVector<Value> nextVals; // Next set of values to search via BFS.
      for (size_t i = 0; i < vals.size(); ++i) {
        Operation *defOp = vals[i].getDefiningOp();
        if (isa_and_nonnull<triton::LoadOp>(defOp)) {
          moveOps.push_back(defOp);
          continue;
        }

        // Find uses of the op that are local_store
        for (Operation *op : vals[i].getUsers()) {
          if (auto storeOp = dyn_cast<ttg::LocalStoreOp>(op)) {
            // Recurse on operands of the local_store (to find a global_load).
            nextVals.push_back(storeOp.getSrc());
          }
        }
      }
      vals.swap(nextVals);
    }
  }

  // Move local_store ops inside the loop early if dependence distance greater
  // than one iteration (i.e., num_stages > 2). For such case, better perf on
  // GEMM when local_store ops precede global loads.
  parentOp->walk([&](ttg::LocalStoreOp op) { moveOps.push_back(op); });
  // Move global_load ops inside the loop early to prefetch. This may increase
  // register pressure but it enables issuing global loads early.
  parentOp->walk([&](triton::LoadOp op) { moveOps.push_back(op); });

  for (auto op : llvm::reverse(moveOps)) {
    // Gather use-def chain in block.
    Block *block = op->getBlock();
    bool leadsToLoad = false;
    SetVector<Operation *> backwardSet;

    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.inclusive = false;
    // Slice should inlcude values flowing into op regions
    options.omitUsesFromAbove = false;
    options.filter = [&](Operation *defOp) -> bool {
      Block *defBlock = defOp->getBlock();
      if (!block->findAncestorOpInBlock(*defOp))
        return false;

      // Check for a `load` dependent path.
      leadsToLoad |= isa<triton::LoadOp>(defOp);
      // Only move ops residing in the same block.
      return defBlock == block;
    };
    mlir::getBackwardSlice(op, &backwardSet, options);
    backwardSet.insert(op);

    // Don't move a local_store if its source is a load from
    // the same iteration.
    if (isa<ttg::LocalStoreOp>(op) && leadsToLoad)
      continue;

    auto ipoint = findEarlyInsertionPoint(block, op);
    // Remove ops that already precede the insertion point. This is done
    // before moves happen to avoid `Operation::isBeforeInBlock` N^2
    // complexity.

    SmallVector<Operation *> dfg = backwardSet.takeVector();
    if (ipoint != block->end()) {
      // Move ops to insertion point.
      llvm::erase_if(
          dfg, [&](Operation *op) { return !ipoint->isBeforeInBlock(op); });
      for (auto *dfgop : llvm::reverse(dfg))
        dfgop->moveAfter(block, ipoint);
    } else {
      // Move ops to block begin.
      for (auto *dfgop : llvm::reverse(dfg))
        dfgop->moveBefore(block, block->begin());
    }
  }
}

//===-------------------------------------------------------------------===//
// Sched-load optimization for matmul kernels with large tile sizes
// The basic idea of sched-load optimization is to sink the 2nd tt.load
// after local_load so that global_load instructions can be interleaved with
// mfma's. This can help hide the issue latency of global_load instructions
// and improve performance on CDNA3.
//
// It's assumed that the IR before this optimization has the following
// structure:
// ```mlir
// scf.for ..
// {
//   tileA = tt.load a_ptr
//   tileB = tt.load b_ptr
//   opA = local_load bufferA
//   opB = local_load bufferB
//   res = tt.dot opA, opB
//   local_store tileA, bufferA
//   local_store tileB, bufferB
// }
// ```
// After this optimization, the IR is transformed to
// ```mlir
// scf.for ..
// {
//   tileA = tt.load a_ptr
//   opA = local_load bufferA
//   opB = local_load bufferB
//   tileB = tt.load b_ptr  <-- 2nd tt.load is sinked here
//   res = tt.dot opA, opB
//   local_store tileA, bufferA
//   local_store tileB, bufferB
// }
// ```
// For now, we don't have a perfect hueristic about when should this
// optimization be applied. Therefore, we implement a simple hueristic that
// this is applied when the tile size of A and B are large enough, i.e.
// nonKDim >= 128  and kDim >= 64. And also this is only applied for typical
// matmul kernels, i.e. only two tt.load's and one dotOp inside the loop. We
// are experimenting how to better control instruction scheduling and enable
// such optimizations.
//===-------------------------------------------------------------------===//
static void sinkSecondLoad(scf::ForOp forOp) {
  SetVector<triton::LoadOp> loadOps;
  triton::DotOp dotOp;
  for (Operation &op : forOp) {
    if (auto loadOp = dyn_cast<triton::LoadOp>(&op))
      loadOps.insert(loadOp);
    if (auto curOp = dyn_cast<triton::DotOp>(&op))
      dotOp = curOp;
  }
  // Only apply the optimization when there are 2 load's in the loop
  if (loadOps.size() != 2)
    return;

  auto ldAOp = loadOps[0];
  auto loadAType = dyn_cast<RankedTensorType>(ldAOp.getType());
  auto ldBOp = loadOps[1];
  auto loadBType = dyn_cast<RankedTensorType>(ldBOp.getType());
  // Only apply the optimization when loading a 2D tensor
  if (!loadAType || !loadBType)
    return;
  auto tileAShape = loadAType.getShape();
  auto tileBShape = loadBType.getShape();
  if (tileAShape.size() != 2 || tileBShape.size() != 2)
    return;
  // Only apply the optimization when tile size is large enough
  // 1. nonKDim >= 128
  // 2. kDim >= 64
  if (!(tileAShape[0] >= 128 && tileAShape[1] >= 64 && tileBShape[1] >= 128))
    return;
  // Only apply the optimization when the moving is legal
  // 1. Make sure the 2nd loadOp is before the dot
  // 2. Make sure the first user of the 2nd loadOp is after the dot.
  bool isBeforeDotOp = ldBOp->isBeforeInBlock(dotOp);
  auto firstUser = *ldBOp.getResult().getUsers().begin();
  bool firstUserAfterDotOp = dotOp->isBeforeInBlock(firstUser);
  if (isBeforeDotOp && firstUserAfterDotOp)
    // move ldBOp right before tt.dot
    ldBOp->moveBefore(dotOp);
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

namespace {
struct TritonAMDGPUReorderInstructionsPass
    : public TritonAMDGPUReorderInstructionsBase<
          TritonAMDGPUReorderInstructionsPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    for (auto funcOp : m.getOps<triton::FuncOp>()) {
      sinkDotConversion(funcOp);
      moveDownCoversion(funcOp);

      moveUpTranspose(funcOp);

      if (isPureMatmulFunc(funcOp)) {
        scheduleGlobalLoadLocalStore(funcOp);
        funcOp.walk([&](scf::ForOp forOp) -> void { sinkSecondLoad(forOp); });
      } else {
        SmallVector<scf::ForOp> leafForOps = triton::AMD::getLeafForOps(funcOp);
        for (auto forOp : leafForOps) {
          if (isPureMatmulLoop(forOp)) {
            scheduleGlobalLoadLocalStore(forOp);
            sinkSecondLoad(forOp);
          }
        }
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createTritonAMDGPUReorderInstructionsPass() {
  return std::make_unique<TritonAMDGPUReorderInstructionsPass>();
}
