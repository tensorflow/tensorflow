#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-wgmma-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define int_attr(num) builder.getI64IntegerAttr(num)

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

/// Find the minimum number of async_commit_group ops between the wait
/// and the associated async_commit_group. This can be safely used as the wait
/// number.
static int minNumInterleavedCommitOps(Operation *waitOp) {
  auto countCommitsBetween = [](Operation *op1, Operation *op2) {
    int count = 0;
    for (auto op = op1; op != op2; op = op->getNextNode()) {
      if (isa<ttg::AsyncCommitGroupOp>(op))
        count++;
      // Intentionally skip block ops' children. This will give us
      // convervatively low number of insert ops.
    }
    return count;
  };

  int minCommitNumber = INT_MAX;

  // DFS the def chain of the extract op to find the insert op. On each path
  // we calculate the number of async_commit. Then we select the minimum number
  // of async_commit ops among all the paths.
  std::function<int(Value, Operation *, int)> minOverHistories =
      [&](Value val, Operation *sinkOp, int thisHistorySum) -> int {
    if (Operation *defOp = val.getDefiningOp()) {
      thisHistorySum += countCommitsBetween(defOp->getNextNode(), sinkOp);
      minCommitNumber = std::min(minCommitNumber, thisHistorySum);
      return minCommitNumber;
    }
    if (auto arg = mlir::dyn_cast<BlockArgument>(val)) {
      Block *block = arg.getOwner();
      auto forOp = dyn_cast<scf::ForOp>(block->getParentOp());

      // Failed to track, return 0 conservatively.
      if (!forOp)
        return 0;

      Operation *firstForInst = &*forOp.getBody()->begin();
      int insertsBetween = countCommitsBetween(firstForInst, sinkOp);
      thisHistorySum += insertsBetween;
      if (thisHistorySum >= minCommitNumber)
        return minCommitNumber;

      // get the value assigned to the argument coming from outside the loop
      Value incomingVal = forOp.getInitArgs()[arg.getArgNumber() - 1];
      int min1 = minOverHistories(incomingVal, forOp, thisHistorySum);

      // get the value assigned to the argument coming from the previous
      // iteration
      Operation *yieldOp = block->getTerminator();
      Value prevVal = yieldOp->getOperand(arg.getArgNumber() - 1);
      int min2 = minOverHistories(prevVal, yieldOp, thisHistorySum);
      return std::min(std::min(min1, min2), minCommitNumber);
    }
    // Failed to track, return 0 conservatively.
    return 0;
  };

  if (waitOp->getNumOperands() != 1)
    return 0;
  Value val = waitOp->getOperand(0);
  // If the value resides in a region other than the region of the wait op, then
  // the wait op must be in some nested region. Measure the number of commits
  // between the definition value and the parent op.
  // TODO: We could measure commits in nested regions along the path if
  // necessary.
  while (waitOp->getParentRegion() != val.getParentRegion())
    waitOp = waitOp->getParentOp();
  int minCommits = minOverHistories(val, waitOp, 0);
  return minCommits;
}

// Look for consecutive wait ops and combine them into a single wait op.
static void
combineRedundantWaitOps(llvm::SmallSetVector<ttg::AsyncWaitOp, 8> &waitOps) {
  llvm::MapVector<ttg::AsyncWaitOp, ttg::AsyncWaitOp> toDelete;
  for (auto waitOp : waitOps) {
    if (toDelete.count(waitOp))
      continue;
    SmallVector<ttg::AsyncWaitOp> waitGroup = {waitOp};
    SmallVector<Value> depTokens;
    unsigned minWaitNumber = waitOp.getNum();
    Operation *next = waitOp->getNextNode();
    while (next && !isa<ttg::AsyncCommitGroupOp>(next)) {
      if (auto nextWait = dyn_cast<ttg::AsyncWaitOp>(next)) {
        waitGroup.push_back(nextWait);
        minWaitNumber = std::min(minWaitNumber, nextWait.getNum());
        depTokens.append(nextWait.getOperands().begin(),
                         nextWait.getOperands().end());
      }
      next = next->getNextNode();
    }
    if (waitGroup.size() == 1)
      continue;
    OpBuilder builder(waitGroup.front());
    auto newWaitOp = builder.create<ttg::AsyncWaitOp>(waitOp.getLoc(),
                                                      depTokens, minWaitNumber);
    for (auto waitOp : waitGroup) {
      toDelete[waitOp] = newWaitOp;
    }
  }
  for (auto waitOp : toDelete) {
    waitOp.first->replaceAllUsesWith(waitOp.second);
    waitOp.first->erase();
  }
}

/// Update wait op number by analyzing the number of async_commit_group ops
/// along all paths.
void mlir::triton::updateWaits(ModuleOp module) {
  llvm::SmallSetVector<ttg::AsyncWaitOp, 8> waitOps;
  module.walk([&](ttg::AsyncWaitOp waitOp) {
    int minNumCommits = minNumInterleavedCommitOps(waitOp);
    waitOp.setNum(minNumCommits);
    waitOps.insert(waitOp);
  });
  combineRedundantWaitOps(waitOps);
}

// Add the given values as operands of the given wait, and replace all uses of
// the values with the wait.  Also adds related MemDesc's to the wait.
//
// Threading %a through the wait transforms
//
//   %a = <...>
//   (%x', %y') = ttng.async_wait %x, %y
//   %b = fn(%a)
//
// into
//
//   %a = <...>
//   (%x', %y', %a') = ttng.async_wait %x, %y, %a
//   %b = fn(%a')
//
// The wait must dominate all uses of the elements of `values`.
//
// In addition to adding each value from `values` to the wait, this function
// also adds some MemDesc's to the wait.  The idea is that if you have
//
//   %alloc = ttg.local_alloc ...
//   %a = ttng.warp_group_dot %alloc
//   %a1 = ttng.warp_group_dot_wait %a
//
// then we want the wait to depend on %alloc as well as %a.  This extends the
// live range of %alloc, so that it won't be destroyed until after the dot is
// waited on.
//
// Specifically, this function finds all warp_group_dot ops that elements of
// `values` depend on.  Then it adds the MemDesc operands of those dots to the
// wait.
static void threadValuesThroughWait(ttng::WarpGroupDotWaitOp wait,
                                    MutableArrayRef<Value> values) {
  IRRewriter builder(wait.getContext());
  builder.setInsertionPoint(wait);

  // Operands are only added to the wait through this function, so we can have
  // the invariant that the wait has no duplicates.  This makes things a bit
  // easier below.
  size_t origNumOperands = wait.getNumOperands();
  SetVector<Value> newOperands(wait.getOperands().begin(),
                               wait.getOperands().end());
  assert(newOperands.size() == origNumOperands &&
         "Wait op has duplicate operands.");

  newOperands.insert(values.begin(), values.end());

  // Find memdefs depended on by `values` through async dot ops.
  SmallVector<ttng::WarpGroupDotOp> asyncDots;
  for (Value v : values) {
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.filter = [&](Operation *op) {
      if (auto dot = dyn_cast<ttng::WarpGroupDotOp>(op)) {
        asyncDots.push_back(dot);
        return false;
      }
      return op->getBlock() == wait->getBlock();
    };
    SetVector<Operation *> slice;
    getBackwardSlice(v, &slice, options);
  }

  for (ttng::WarpGroupDotOp dot : asyncDots) {
    for (Value operand : dot.getOperands()) {
      if (isa<ttg::MemDescType>(operand.getType())) {
        newOperands.insert(operand);
      }
    }
  }

  // We can't use replaceWithNewOp because we're changing the number of return
  // values in the operation.
  auto newWait = builder.create<ttng::WarpGroupDotWaitOp>(
      wait.getLoc(), llvm::to_vector(newOperands), wait.getPendings());

  auto dominatedByNewWait = [&](OpOperand &operand) {
    auto opInThisBlock =
        newWait->getBlock()->findAncestorOpInBlock(*operand.getOwner());
    return opInThisBlock && newWait->isBeforeInBlock(opInThisBlock);
  };
  for (int i = 0; i < origNumOperands; i++) {
    Value operand = wait.getResult(i);
    if (!isa<ttg::MemDescType>(operand.getType()))
      operand.replaceAllUsesWith(newWait.getResult(i));
  }
  for (int i = origNumOperands; i < newOperands.size(); i++) {
    Value operand = newWait.getOperand(i);
    if (!isa<ttg::MemDescType>(operand.getType()))
      operand.replaceUsesWithIf(newWait.getResult(i), dominatedByNewWait);
  }
  wait->erase();
}

// Determines whether a given MMAv3 dot op, represented as ttng.warp_group_dot,
// needs a wait immediately after it.
//
// In PTX, MMAv3 exists only as an asynchronous op.  In Triton, we can represent
// MMAv3 ops as either ttng.warp_group_dot {isAsync=True} or ttng.warp_group_dot
// {isAsync=False}.  But even if we use ttng.warp_group_dot {isAsync=True}, the
// conservative thing is to make a dot "effectively synchronous" by inserting a
// `ttng.warp_group_dot_wait {pendings=0}` right after it.
//
// We can omit the wait and create a "properly async" dot if all of the
// following are true.
//
//  1. All operands that touch shared memory are multi-buffered, i.e. can't read
//     an incomplete value while it's being written asynchronously by a load.
//     1a. If operand A is in registers, these registers cannot be updated
//     inside
//         the loop.
//         **Exception** if the operand is produced by a preceding WGMMA,
//         then this op can be properly async. Either the f16 shortcut is
//         possible and the WGMMA's can run back-to-back (see rule 3 below), or
//         elementwise truncate is needed, in which case the preceding WGMMA is
//         not async and a WarpGroupDotWait is inserted right after, which
//         guarantees exclusive access to the operand registers.
//
//  2. If the dot is used by any op in the loop, it must be used under an `if`,
//     and will be synced with a `wait 0` at the beginning of the `if` block.
//
//  3. During iteration i, between the start of the loop up until the first
//     `ttng.warp_group_dot_wait {pendings=0}` op, the result of the dot from
//     iteration i-1 is consumed only by other MMAv3 dots as the `c` operand.
//
//     This is safe because the following pseudo-PTX is valid:
//
//        %accum = warp_group_dot %a1, %b1, %c1
//        %accum = warp_group_dot %a2, %b2, %accum
//
//     That is, the second async dot can use the result of the first one without
//     an intervening wait.  However, the only operation that can legally read
//     %accum before the wait is another warp_group_dot, and this only works for
//     the `c` operand, not `a` or `b`.  See
//     https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence
//     (ttng::WarpGroupDotOp corresponds to wgmma.fence followed by one or more
//     wgmma.async ops, so our understanding is that the two
//     ttng::WarpGroupDotOps don't have to correspond to wgmma.async ops with
//     the same shapes as specified in the docs, because there's an intervening
//     fence.)
//
// If the op can be properly async, this function returns the index of the dot
// in the loop's iter_args.  (Rule (2) above ensures this is well-defined.)
//
static std::optional<int> dotCanBeProperlyAsync(ttng::WarpGroupDotOp dotOp,
                                                scf::ForOp forOp) {
  LDBG("Considering whether to make MMAv3 dot properly async: " << dotOp);

  // Rule 1: All shmem operands are multi-buffered.
  auto checkOperand = [&](Value operand) {
    if (!isa<ttg::SharedEncodingTrait>(
            cast<ttg::TensorOrMemDesc>(operand.getType()).getEncoding())) {
      // Rule 1a: Register operands must not be modified within the loop.
      // First, check for chained WGMMA as an exception.
      if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(operand.getDefiningOp())) {
        return isa<ttg::NvidiaMmaEncodingAttr>(
            cvt.getSrc().getType().getEncoding());
      }
      // And then, do a stricter-than-necessary check for now, that the operand
      // is defined outside the loop.
      return forOp.isDefinedOutsideOfLoop(operand);
    }

    // If it's a shmem operand, it must either be defined outside the loop, or
    // come from an MemDescSubview op.  Only ConvertLayout and Trans ops are
    // allowed in between.
    Value transitiveOperand = operand;
    while (isa_and_nonnull<ttg::ConvertLayoutOp, ttg::MemDescTransOp>(
               transitiveOperand.getDefiningOp()) ||
           isa<BlockArgument>(transitiveOperand)) {
      auto blockArg = dyn_cast<BlockArgument>(transitiveOperand);
      if (blockArg && blockArg.getOwner() == forOp.getBody()) {
        transitiveOperand =
            cast<scf::YieldOp>(blockArg.getOwner()->getTerminator())
                .getOperand(blockArg.getArgNumber() - 1);
      } else if (Operation *def = transitiveOperand.getDefiningOp()) {
        transitiveOperand = def->getOperand(0);
      }
    }
    return forOp.isDefinedOutsideOfLoop(transitiveOperand) ||
           transitiveOperand.getDefiningOp<ttg::MemDescSubviewOp>();
  };

  // We don't have to call checkOperand on getC() because it's always in
  // registers, never in shmem.
  assert(isa<ttg::NvidiaMmaEncodingAttr>(dotOp.getC().getType().getEncoding()));
  if (!checkOperand(dotOp.getA()) || !checkOperand(dotOp.getB())) {
    LDBG("Can't make dot async because shmem operands aren't multi-buffered");
    return std::nullopt;
  }

  // Rule 2: The dot cannot be unconditionally used by any op in the loop.
  // Uses under `if` are allowed, as can be explicitly synced with a `wait 0`.
  int iterArgIdx = -1;
  Value iterArg = nullptr;
  SmallVector<std::pair<Operation *, int>> queue;
  for (auto &use : dotOp->getUses()) {
    queue.push_back({use.getOwner(), use.getOperandNumber()});
  }
  while (!queue.empty()) {
    auto [user, argIdx] = queue.pop_back_val();
    if (user->getParentOp() == forOp) {
      if (isa<scf::YieldOp>(user)) {
        if (iterArg) {
          // The dot is used by the loop's yield, but we can't have any other
          // uses.
          LDBG("Can't make dot async because dot is used by multiple ops in "
               "the loop.");
          return std::nullopt;
        }
        iterArgIdx = argIdx;
        iterArg = forOp.getRegionIterArg(argIdx);
        continue;
      }
      LDBG("Can't make dot async because dot is unconditionally used in the "
           "loop.");
      return std::nullopt;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(user->getParentOp())) {
      if (isa<scf::YieldOp>(user)) {
        // The result is returned by the if, follow it further.
        auto uses = ifOp.getResult(argIdx).getUses();
        for (auto &use : uses) {
          queue.push_back({use.getOwner(), use.getOperandNumber()});
        }
      }
    } else {
      return std::nullopt;
    }
  }

  // Rule 3a: Are the only users of the dot's result from iteration i-1 other
  // MMAv3 dots?  If so, we're done, this dot can be properly async.
  if (llvm::all_of(iterArg.getUses(), [&](OpOperand &use) {
        return isa<ttng::WarpGroupDotOp>(use.getOwner()) &&
               use.getOperandNumber() == 2;
      })) {
    return iterArgIdx;
  }

  // Rule 3b: Are all users of the dot's result from iteration i-1 after the
  // first `warp_group_dot_wait {pendings=0}` op?  If so, the dot can be
  // properly async, but we have to thread its result from iteration i-1 through
  // the wait.
  auto waitOps = forOp.getBody()->getOps<ttng::WarpGroupDotWaitOp>();
  auto firstWaitOpIter = llvm::find_if(
      waitOps, [&](auto waitOp) { return waitOp.getPendings() == 0; });
  if (firstWaitOpIter != waitOps.end() &&
      llvm::all_of(iterArg.getUsers(), [&](Operation *user) {
        assert(forOp->isAncestor(user));
        while (user->getParentOp() != forOp) {
          user = user->getParentOp();
        }
        return (*firstWaitOpIter)->isBeforeInBlock(user);
      })) {
    LDBG("MMAv3 dot can be properly async because it follows a "
         "warp_group_dot_wait "
         "{pendings=0}.\n"
         << "  wait: " << *firstWaitOpIter << "\n"
         << "  dot: " << dotOp);
    threadValuesThroughWait(*firstWaitOpIter, {iterArg});
    return iterArgIdx;
  }

  LDBG("Can't make dot async because its result from i-1 is used by "
       "something other than another MMAv3 dot as the `c` operand.");
  return std::nullopt;
}

// If necessary, insert a dot-wait inside the loop, waiting for the results of
// the properly-async dots from iteration i-1 to complete.  (We pipeline to
// depth 2, so there are at most 2 copies of each warp_group_dot in flight at a
// time.)
//
// We can skip inserting the wait if we have a `warp_group_dot_wait
// {pendings=0}` somewhere in the loop.  To see why, consider:
//
//   warp_group_dot
//   warp_group_dot; wait 0  // synchronous dot
//   warp_group_dot
//   warp_group_dot
//
// In this example, there are three properly-async dots, so we'd normally put
// `wait 3` at the end of the loop, meaning "wait until there are 3 or fewer
// pending async dots".  But note that when this iteration of the loop
// completes, there are only *two* pending async dots from this iteration, so
// this wait would do nothing.  This is true in general, no matter where the
// `wait 0` appears.
static void insertAsyncWarpGroupDotWaitInLoop(
    scf::ForOp forOp,
    const llvm::MapVector<Operation *, int /*iterArgIdx*/> &properlyAsyncDots) {
  if (properlyAsyncDots.empty())
    return;

  if (llvm::any_of(forOp.getBody()->getOps<ttng::WarpGroupDotWaitOp>(),
                   [](auto wait) { return wait.getPendings() == 0; })) {
    return;
  }

  // Insert waits before the users of the properly async dots other than loop
  // yield.
  for (auto [asyncDot, iterArgIdx] : properlyAsyncDots) {
    SmallVector<OpOperand *> uses;
    for (auto &use : asyncDot->getUses()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(use.getOwner())) {
        continue;
      }
      uses.push_back(&use);
    }

    DenseMap<Block *, SmallVector<Value>> blockToUsers;
    for (auto use : uses) {
      auto block = use->getOwner()->getBlock();
      blockToUsers[block].push_back(use->get());
    }

    for (auto [block, users] : blockToUsers) {
      OpBuilder builder(block, block->begin());
      auto newWait = builder.create<ttng::WarpGroupDotWaitOp>(
          asyncDot->getLoc(), ArrayRef<Value>{}, 0);

      threadValuesThroughWait(newWait, users);
    }
  }

  // Add the wait right after the last properly-async dot.  This only needs to
  // wait for all properly-async dots from the i-1'th iteration to complete, IOW
  // we wait until there are most `asyncDots.size()` dots in flight.
  //
  // (You might want to put the wait at the end of the loop instead of right
  // after the last dot, but there could be a load into shmem between the last
  // async dot and the end of the loop, and that could clobber memory being used
  // by a dot.)
  IRRewriter builder(forOp.getContext());
  auto lastAsyncDot = properlyAsyncDots.back().first;
  builder.setInsertionPointAfter(lastAsyncDot);
  auto wait = builder.create<ttng::WarpGroupDotWaitOp>(
      lastAsyncDot->getLoc(),
      /*inputs=*/ArrayRef<Value>{}, properlyAsyncDots.size());

  // Thread the results of the async dots through the wait.
  SmallVector<Value> addlWaitOperands;
  for (auto [asyncDot, iterArgIdx] : properlyAsyncDots) {
    addlWaitOperands.push_back(asyncDot->getResult(0));
  }
  threadValuesThroughWait(wait, addlWaitOperands);
}

// Convert MMAv3 ttng::WarpGroupDotOps {isAsync = False} (i.e. Hopper wgmma)
// into ttng::WarpGroupDotOps {isAsync = True} and insert
// ttng::WarpGroupDotWaitOps as necessary.
//
// We assume we have space for each dot to be pipelined to depth 2, i.e. each
// dot op in the loop can have at most 2 warp_group_dot ops in flight at once.
// (Each warp_group_dot op usually corresponds to a series of wgmma.async ops.)
void triton::asyncLaunchDots(scf::ForOp forOp) {
  LDBG("Original loop:\n" << *forOp);

  // First, change every MMAv3 ttng.warp_group_dot {isAsync=false}
  // into ttng.warp_group_dot {isAsync=true}.
  // The rest of this function is concerned with inserting
  // ttng.warp_group_dot_wait ops in the appropriate places.
  //
  // We call those dots that don't need to be followed immediately by a `wait 0`
  // "properly async", or sometimes just "async".
  //
  // For each dot, determine whether it can be properly async, or if it needs a
  // sync immediately after.  If it can be properly async, we know its only use
  // is in the loop's `yield` statement; asyncDots maps the op to its index in
  // the yield op.
  IRRewriter builder(forOp.getContext());
  llvm::MapVector<Operation *, int /*iterArgIdx*/> properlyAsyncDots;
  for (auto WarpGroupDotOp : forOp.getBody()->getOps<ttng::WarpGroupDotOp>()) {
    WarpGroupDotOp.setIsAsync(true);
    if (auto iterArgIdx = dotCanBeProperlyAsync(WarpGroupDotOp, forOp)) {
      properlyAsyncDots[WarpGroupDotOp] = *iterArgIdx;
    } else {
      builder.setInsertionPointAfter(WarpGroupDotOp);
      auto wait = builder.create<ttng::WarpGroupDotWaitOp>(
          WarpGroupDotOp.getLoc(), ArrayRef<Value>{},
          /*pendings=*/0);
      SmallVector<Value> waitOperands = {WarpGroupDotOp.getResult()};
      threadValuesThroughWait(wait, waitOperands);
    }
  }

  if (properlyAsyncDots.empty()) {
    LDBG("No properly async dots.");
    return;
  }

  // Next, insert a wait inside the loop.  We pipeline to depth 2, so the third
  // iteration's set of asynchronous dots (and their corresponding async copies
  // from global to shmem) can't start until the first iteration's set has
  // completed.
  insertAsyncWarpGroupDotWaitInLoop(forOp, properlyAsyncDots);

  // Finally, insert a wait after the loop, waiting for dots from the final
  // iteration of the loop.
  SmallVector<Value> waitOperands;
  for (auto [asyncDot, iterArgIdx] : properlyAsyncDots) {
    waitOperands.push_back(forOp.getResult(iterArgIdx));
  }
  // Wait until there are 0 outstanding async dot ops.
  builder.setInsertionPointAfter(forOp);
  auto WarpGroupDotWaitAfterLoop = builder.create<ttng::WarpGroupDotWaitOp>(
      forOp.getLoc(), ArrayRef<Value>{}, 0);
  threadValuesThroughWait(WarpGroupDotWaitAfterLoop, waitOperands);
}
