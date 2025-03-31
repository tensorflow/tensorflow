#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

//===----------------------------------------------------------------------===//
// specializeLoadMMADependencies
//===----------------------------------------------------------------------===//

// Pattern match a simple `tma_load -> ... -> tl.dot` single-user chain. This
// ensures there are extraneous users of the load or intermediate values and
// that a valid partition schedule can be formed.
//
// TODO: Expand partioning scheme to support arbitrary DAG of loads and MMAs.
static LogicalResult findSingleChainToLoad(scf::ForOp loop, Value value,
                                           SmallVectorImpl<Operation *> &ops) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp || !value.hasOneUse() || defOp->getParentOp() != loop)
    return failure();

  // This only works on TMA loads because they directly use the mbarrier
  // mechanism. Since async groups are per-thread, commit groups cannot be used
  // to synchronize across warp groups. We have to wait on the async group in
  // the same partition as the loads and arrive an mbarrier to synchronize with
  // the MMA partition, and then software pipeline the load partition.
  //
  // Triple-buffered example:
  //
  //   cp.async %a_ptrs[0], %a_buf[0]
  //   cp.async %b_ptrs[0], %b_buf[0]
  //   cp.async.commit_group
  //
  //   cp.async %a_ptrs[1], %a_buf[1]
  //   cp.async %b_ptrs[1], %b_buf[1]
  //   cp.async.commit_group
  //
  //   for i in range(2, N+2):
  //     @i<N mbarrier.wait %empty_mbars[i%3]
  //     @i<N cp.async %a_ptrs[i], %a_buf[i%3]
  //     @i<N cp.async %b_ptrs[i], %b_buf[i%3]
  //     @i<N cp.async.commit_group
  //
  //     cp.async.wait_group 2 # the i-2 load group is complete
  //     mbarrier.arrive %load_mbars[(i-2)%3]
  if (isa<DescriptorLoadOp, DescriptorGatherOp>(defOp)) {
    ops.push_back(defOp);
    return success();
  }

  // See through allocations and layout conversions.
  if (isa<ttng::TMEMAllocOp, LocalAllocOp, MemDescTransOp, ConvertLayoutOp>(
          defOp)) {
    assert(llvm::is_contained({0, 1}, defOp->getNumOperands()));
    // Alloc ops have an optional source operand.
    if (defOp->getNumOperands() != 1)
      return failure();
    ops.push_back(defOp);
    return findSingleChainToLoad(loop, defOp->getOperand(0), ops);
  }

  return failure();
}

// FIXME: This won't work for persistent matmul.
static LogicalResult matchAccumulatorPattern(ttng::TMEMAllocOp alloc,
                                             ttng::TMEMLoadOp load) {
  if (!alloc.getSrc())
    return failure();
  auto arg = dyn_cast<BlockArgument>(alloc.getSrc());
  if (!arg)
    return failure();

  auto loop = cast<scf::ForOp>(alloc->getParentOp());
  if (arg.getOwner() != loop.getBody())
    return failure();

  unsigned idx = arg.getArgNumber() - 1;
  OpOperand &yielded = loop.getBody()->getTerminator()->getOpOperand(idx);
  if (yielded.get() != load.getResult() || !load->hasOneUse())
    return failure();
  return success();
}

// Create an operation inside a partition.
template <typename OpT, typename... Args>
static auto createInPartition(ImplicitLocOpBuilder &b,
                              WarpSchedule::Partition &partition,
                              Args &&...args) {
  auto op = b.create<OpT>(std::forward<Args>(args)...);
  partition.insert(op);
  return op;
}

static void lowerTMACopy(ImplicitLocOpBuilder &b,
                         WarpSchedule::Partition &partition, Operation *op,
                         Value barrier, Value view) {
  Value truePred = b.create<arith::ConstantIntOp>(true, /*width=*/1);
  if (auto load = dyn_cast<DescriptorLoadOp>(op)) {
    Value tmaPtr = createInPartition<ttng::TensorDescToTMAPtrOp>(
        b, partition, load.getDesc());
    createInPartition<ttng::AsyncTMACopyGlobalToLocalOp>(
        b, partition, tmaPtr, load.getIndices(), barrier, view, truePred);
  } else {
    auto gather = cast<DescriptorGatherOp>(op);
    Value tmaPtr = createInPartition<ttng::TensorDescToTMAPtrOp>(
        b, partition, gather.getDesc());
    createInPartition<ttng::AsyncTMAGatherOp>(
        b, partition, tmaPtr, gather.getXOffsets(), gather.getYOffset(),
        barrier, view, truePred);
  }
}

LogicalResult triton::gpu::specializeLoadMMADependencies(scf::ForOp &loop,
                                                         int defaultNumStages) {
  auto ops = llvm::to_vector(loop.getOps<ttng::MMAv5OpInterface>());
  // Support only 1 MMA op.
  if (ops.size() != 1)
    return failure();
  ttng::MMAv5OpInterface mmaOp = ops.front();
  auto dot = cast<DotOpInterface>(*mmaOp);

  SmallVector<Operation *> aChain, bChain;
  if (failed(findSingleChainToLoad(loop, dot.getA(), aChain)) ||
      failed(findSingleChainToLoad(loop, dot.getB(), bChain)))
    return failure();
  auto accOr = getTMemAllocAndLoad(mmaOp);
  if (!accOr)
    return failure();
  auto [accAlloc, accLoad] = *accOr;
  if (failed(matchAccumulatorPattern(accAlloc, accLoad)))
    return failure();

  // Pattern match succeeded. Now rewrite the loads and MMA op.
  int numStages = getNumStagesOrDefault(loop, defaultNumStages);
  int numBuffers = numStages - 1;
  WarpSchedule schedule;
  WarpSchedule::Partition *loadPartition = schedule.addPartition(0);
  WarpSchedule::Partition *mmaPartition = schedule.addPartition(numBuffers);

  ImplicitLocOpBuilder b(mmaOp.getLoc(), loop);
  auto intCst = [&](int value, unsigned width = 32) {
    return b.create<arith::ConstantIntOp>(value, width);
  };

  // Rewrite the loop to pass things through buffers.
  unsigned idx = cast<BlockArgument>(accAlloc.getSrc()).getArgNumber() - 1;
  Value newAlloc = createTMemAlloc(b, accAlloc, /*multiBufferred=*/false,
                                   /*numStages=*/1);
  Value mmaBarrier = createBarrierAlloc(loop, /*numBarriers=*/1);
  b.create<ttng::TMEMStoreOp>(newAlloc, loop.getInitArgs()[idx],
                              intCst(true, /*width=*/1));
  mmaOp.setBarrier(mmaBarrier);
  accAlloc.replaceAllUsesWith(newAlloc);
  accLoad->dropAllUses();
  accLoad->moveAfter(loop);
  loop.getResult(idx).replaceAllUsesWith(accLoad);
  accAlloc.erase();
  llvm::BitVector indices(loop.getNumResults());
  indices.set(idx);
  eraseLoopCarriedValues(loop, indices);

  // Now multi-buffer the loads.
  Operation *aLoad = aChain.back();
  Operation *bLoad = bChain.back();
  auto aType = cast<RankedTensorType>(aLoad->getResult(0).getType());
  auto bType = cast<RankedTensorType>(bLoad->getResult(0).getType());
  SharedEncodingTrait aEnc = getSharedEncoding(aChain.back());
  SharedEncodingTrait bEnc = getSharedEncoding(bChain.back());
  Value aAlloc = createAlloc(loop, aType, aLoad->getLoc(), aEnc, numBuffers);
  Value bAlloc = createAlloc(loop, bType, bLoad->getLoc(), bEnc, numBuffers);

  // Share the same set of barriers for both.
  Value emptyBars = createBarrierAlloc(loop, numBuffers);
  Value readyBars = createBarrierAlloc(loop, numBuffers);
  // Mark the empty barriers as initially ready.
  b.setInsertionPoint(loop);
  for (auto i : llvm::seq(numBuffers)) {
    Value emptyBar = createSingleBufferView(b, emptyBars, i);
    b.create<ttng::ArriveBarrierOp>(emptyBar, 1);
  }

  unsigned curArgIdx = loop.getNumRegionIterArgs();
  scf::ForOp newLoop =
      replaceForOpWithNewSignature(b, loop, {intCst(-1), intCst(0), intCst(0)});
  loop.erase();
  loop = newLoop;

  Value index = loop.getRegionIterArgs().slice(curArgIdx)[0];
  Value phase = loop.getRegionIterArgs().slice(curArgIdx)[1];
  Value mmaPhase = loop.getRegionIterArgs().slice(curArgIdx)[2];

  b.setInsertionPointToStart(loop.getBody());
  index = b.create<arith::AddIOp>(index, intCst(1));
  Value rollover = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, index,
                                           intCst(numBuffers));
  index = b.create<arith::SelectOp>(rollover, intCst(0), index);
  phase = b.create<arith::SelectOp>(
      rollover, b.create<arith::XOrIOp>(phase, intCst(1)), phase);

  Value nextMmaPhase = b.create<arith::XOrIOp>(mmaPhase, intCst(1));
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  yield->insertOperands(yield.getNumOperands(), {index, phase, nextMmaPhase});

  int loadSizeInBytes =
      product(aType.getShape()) * aType.getElementTypeBitWidth() / 8 +
      product(bType.getShape()) * bType.getElementTypeBitWidth() / 8;

  // Insert before the group of loads.
  b.setInsertionPoint(aLoad->isBeforeInBlock(bLoad) ? aLoad : bLoad);
  // Wait for the buffer to be empty and the corresponding barrier to be
  // exhausted.
  Value curEmptyBarrier = createSingleBufferView(b, emptyBars, index);
  createInPartition<ttng::WaitBarrierOp>(b, *loadPartition, curEmptyBarrier,
                                         phase);
  // Indicate the expected size of the loads.
  Value curLoadBarrier = createSingleBufferView(b, readyBars, index);
  createInPartition<ttng::BarrierExpectOp>(b, *loadPartition, curLoadBarrier,
                                           loadSizeInBytes,
                                           intCst(true, /*width=*/1));

  // Replace the loads with async copies.
  b.setInsertionPoint(aLoad);
  Value aView = createSingleBufferView(b, aAlloc, index);
  lowerTMACopy(b, *loadPartition, aLoad, curLoadBarrier, aView);
  replaceUsesAndPropagateType(b, *aLoad->user_begin(), aView);
  aLoad->user_begin()->erase();
  aLoad->erase();

  b.setInsertionPoint(bLoad);
  Value bView = createSingleBufferView(b, bAlloc, index);
  lowerTMACopy(b, *loadPartition, bLoad, curLoadBarrier, bView);
  replaceUsesAndPropagateType(b, *bLoad->user_begin(), bView);
  bLoad->user_begin()->erase();
  bLoad->erase();

  // Place the remaining users in the MMA partition. Re-acquire the use chain
  // because some ops were invalidated by `replaceUsesAndPropagateType`.
  aChain.clear();
  bChain.clear();
  aChain.push_back(mmaOp);
  (void)findSingleChainToLoad(loop, dot.getA(), aChain);
  (void)findSingleChainToLoad(loop, dot.getB(), bChain);

  // Place users in the MMA partition.
  auto allUsers = llvm::concat<Operation *>(aChain, bChain);
  for (Operation *user : allUsers)
    mmaPartition->insert(user);

  // Insert the load wait before the first user.
  auto minIt = llvm::min_element(allUsers, [](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  b.setInsertionPoint(*minIt);
  createInPartition<ttng::WaitBarrierOp>(b, *mmaPartition, curLoadBarrier,
                                         phase);

  // Insert the MMA wait after the MMA op and signal the empty barriers.
  b.setInsertionPointAfter(mmaOp);
  createInPartition<ttng::WaitBarrierOp>(b, *mmaPartition, mmaBarrier,
                                         mmaPhase);
  createInPartition<ttng::ArriveBarrierOp>(b, *mmaPartition, curEmptyBarrier,
                                           1);

  schedule.serialize(loop);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPULOADMMASPECIALIZATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct LoadMMASpecialization
    : triton::gpu::impl::TritonGPULoadMMASpecializationBase<
          LoadMMASpecialization> {
  using TritonGPULoadMMASpecializationBase::TritonGPULoadMMASpecializationBase;

  void runOnOperation() override;
};
} // namespace

void LoadMMASpecialization::runOnOperation() {
  SmallVector<scf::ForOp> loops;
  getOperation().walk([&](scf::ForOp loop) {
    if (loop->hasAttr("tt.warp_specialize"))
      loops.push_back(loop);
  });
  for (scf::ForOp loop : loops) {
    if (failed(specializeLoadMMADependencies(loop, numStages)))
      continue;
  }
}
