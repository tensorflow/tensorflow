#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#define DEBUG_TYPE "tritonamdgpu-block-pingpong"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace {

// This pass transforms a for-loop calculating a GEMM. Main purpose of the
// transform is improve the efficiency of the GPU dot instruction (mfma)
// by interleaving the execution of two warps on each SIMD. Especially it groups
// instructions into Dot and Memory clusters so they can efficiently run in
// parallel. Also this pass inserts `rocdl.s.setprio` operation and
// `amdgpu.cond_barrier` to run two parallel warps in synchronization.
// This scheduling doesn't help improving the memory latency itself but it
// relies on software-pipelining to hide the global latency. Likely to improve
// the performance of compute-bound cases.
class Pingponger {
  scf::ForOp forOp;
  SmallVector<tt::LoadOp> gLoadOps;
  SmallVector<ttg::LocalLoadOp> lLoadOps;
  SmallVector<ttg::LocalStoreOp> lStoreOps;
  SmallVector<tt::DotOp> dotOps;
  SmallVector<SmallVector<Operation *>> subViewOps;
  SmallVector<SmallVector<Operation *>> loadSliceOps;
  SmallVector<Operation *> dotSliceOps;
  SmallVector<Value> constOffsets;
  Operation *lastInsertedOp;

  // rocdl.s.setprio will be mapped to `s_setprio` instruction which set the
  // priority of the warp within a SIMD, determines which warp to occupy the
  // instruction unit when they compete on the same instruction.
  // We use this instruction in the pingpong scheduling to prevent warps from
  // entering into the dot cluster while the other warp is still busy in the dot
  // cluster. Otherwise pingpong pattern can be broken and performance drops.
  // Currently pingpong only handles two warps, we only need 0/1 priorities.
  int lowPriority = 0;
  int highPriority = 1;
  int32_t kWidth;
  int32_t numWarps;
  int32_t numStages;

public:
  Pingponger(scf::ForOp forOp, int32_t numWarps, int32_t numStages)
      : forOp(forOp), numWarps(numWarps), numStages(numStages) {}
  void getDotPingponged();

private:
  void genOffsetConstants(Location loc, OpBuilder &builder, unsigned numSlices,
                          int64_t sliceWidth);
  LogicalResult genLocalSlice(OpBuilder &builder, Value v,
                              Attribute dotEncoding, unsigned opIdx,
                              unsigned numSlices, int64_t sliceWidth);
  LogicalResult sliceDot(OpBuilder &builder, Location loc, tt::DotOp op,
                         unsigned numSlices);
  void transformOnePPClusters(OpBuilder &builder, Location loc);
  LogicalResult transformFourPPClusters(OpBuilder &builder, Location loc);
  LogicalResult transformTwoPPClusters(OpBuilder &builder, Location loc);
  void addAsymmetricSyncToLoop(OpBuilder &builder, Location loc);
  void updateOpInsertion(Operation *Op);
  void appendOp(Operation *Op);
  void moveOpAndPredecessorsUpSameBlock(Operation *Op);
  void appendSlicedLoadAB(int slice);
  void appendClusterBarrier(OpBuilder &builder, Location loc);
  void appendOpWithPrio(OpBuilder &builder, Operation *Op, Location loc);
  void determineDotMemoryOps(tt::DotOp dotOp,
                             DenseSet<tt::LoadOp> &dotGlobalLoads,
                             DenseSet<ttg::LocalLoadOp> &dotLocalLoads,
                             DenseSet<ttg::LocalStoreOp> &dotLocalStores);
  template <typename T>
  void findClosestPredOps(Value v, DenseSet<T> &matchingOps);
};

void Pingponger::updateOpInsertion(Operation *op) { lastInsertedOp = op; }
void Pingponger::appendOp(Operation *op) {
  assert(lastInsertedOp != nullptr);
  op->moveAfter(lastInsertedOp);
  lastInsertedOp = op;
}

// Move the given operations and any predecessors upon which it depends
// up in the block to the last inserted operation. This does not move
// operations that reaches the last inserted operation or
// are not in the same block. The exception is op, which is always moved
// to the new location (can move down or up).
void Pingponger::moveOpAndPredecessorsUpSameBlock(Operation *op) {
  assert(lastInsertedOp != nullptr);
  // TODO: Enable moving ops across blocks
  assert(op->getBlock() == lastInsertedOp->getBlock());
  Operation *checkedOp = lastInsertedOp;
  // Check if we are moving the op up, if so we may need to
  // move additional ops up to maintain correctness.
  if (lastInsertedOp->isBeforeInBlock(op)) {
    SetVector<Operation *> backwardSlice;
    BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    opt.filter = [&checkedOp](Operation *op) {
      return op->getBlock() == checkedOp->getBlock() &&
             checkedOp->isBeforeInBlock(op);
    };
    getBackwardSlice(op, &backwardSlice, opt);
    for (auto predOp : backwardSlice)
      appendOp(predOp);
    appendOp(op);
  } else {
    auto hasUnsafeUser = [&checkedOp](auto &&user) {
      return user != checkedOp && user->getBlock() == checkedOp->getBlock() &&
             user->isBeforeInBlock(checkedOp);
    };
    if (std::any_of(op->user_begin(), op->user_end(), hasUnsafeUser))
      LDBG("Unable to move operation "
           << op << " due to use before intended move location");
    else
      appendOp(op);
  }
}
void Pingponger::appendSlicedLoadAB(int slice) {
  appendOp(subViewOps[0][slice]);
  appendOp(loadSliceOps[0][slice]);
  appendOp(subViewOps[1][slice]);
  appendOp(loadSliceOps[1][slice]);
}
// Asymmetrically synchronized loop in the pingpong scheduling synchronizes all
// the warps at the end of each instruction cluster. Since cond_barrier
// triggered a barrier for only half of the warps in a block, at the point
// this clusterBarrier is called, half warps are at dot cluster and the others
// are at the memory cluster.
// Also, SchedBarrier with `0` is set here to tell compiler backend not to
// reorder any instruction across this point.
void Pingponger::appendClusterBarrier(OpBuilder &builder, Location loc) {
  //  MembarAnalysis can recognize gpu::BarrierOp and skip inserting additional
  //  barrier
  appendOp(builder.create<gpu::BarrierOp>(loc));
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
}
void Pingponger::appendOpWithPrio(OpBuilder &builder, Operation *op,
                                  Location loc) {
  appendOp(builder.create<ROCDL::SetPrioOp>(loc, highPriority));
  appendOp(op);
  appendOp(builder.create<ROCDL::SetPrioOp>(loc, lowPriority));
}

// Find all of the "closest" operations that are of a given type T
// in the same basic block. Here "closest" means along any path P,
// the first operation of type T that is encountered when traversing
// P from the given value v. This also includes "later" operations
// for block arguments. Note: That we find all T for every path P.
template <typename T>
void Pingponger::findClosestPredOps(Value v, DenseSet<T> &matchingOps) {
  // Create a cache so we can traverse across block arguments.
  DenseSet<Operation *> visitedOps;
  std::function<void(Value)> impl;
  impl = [&matchingOps, &visitedOps, &impl](Value v) {
    // If we encounter a block argument we only look at the terminators of the
    // current block
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      auto operandNumber = blockArg.getArgNumber();
      auto block = blockArg.getOwner();
      if (auto yield = dyn_cast<scf::YieldOp>(block->getTerminator())) {
        auto parentOp = block->getParentOp();
        // Skip the induction variables to find the yield position
        if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
          if (operandNumber < forOp.getNumInductionVars())
            return;
          operandNumber -= forOp.getNumInductionVars();
        }
        impl(yield->getOperand(operandNumber));
      }
    } else {
      auto definingOp = v.getDefiningOp();
      if (!definingOp)
        return;
      else if (visitedOps.contains(definingOp))
        return;
      visitedOps.insert(definingOp);
      if (auto matchOp = dyn_cast<T>(definingOp))
        matchingOps.insert(matchOp);
      else
        for (auto predValue : definingOp->getOperands())
          impl(predValue);
    }
  };
  impl(v);
}

// Populate the dotGlobalLoads, dotLocalLoads, and dotLocalStores set with
// any loads that are generated by the current dot product. This occurs in
// steps to:
// 1. Determine which loads are generated by the dot product via getA()
//    and getB().
// 2. Determine which local stores are used to populate the inputs to
//    the local loads.
// 3. Determine which global loads are used to populate the inputs to
//    the local stores.
// Note: This function currently depends on num_stages=2, which is a
// precondition for the pingpong scheduling.
void Pingponger::determineDotMemoryOps(
    tt::DotOp dotOp, DenseSet<tt::LoadOp> &dotGlobalLoads,
    DenseSet<ttg::LocalLoadOp> &dotLocalLoads,
    DenseSet<ttg::LocalStoreOp> &dotLocalStores) {
  // Find the locals loads used to compute the dot inputs. These
  // must come before the dot op.
  findClosestPredOps<ttg::LocalLoadOp>(dotOp.getA(), dotLocalLoads);
  findClosestPredOps<ttg::LocalLoadOp>(dotOp.getB(), dotLocalLoads);

  // Determine the local stores from the local loads.
  // With pipelining we expect this to be a single local
  // store within the loop based on a block argument after routing through
  // a ttg.MemDescSubviewOp.
  DenseSet<ttg::MemDescSubviewOp> subviews;
  for (auto &&localLoad : dotLocalLoads)
    findClosestPredOps<ttg::MemDescSubviewOp>(localLoad.getSrc(), subviews);

  for (auto &&subview : subviews)
    for (auto &&user : subview->getUsers())
      if (auto localStore = dyn_cast<ttg::LocalStoreOp>(user))
        dotLocalStores.insert(localStore);

  // Determine the global loads from the local stores.
  // We expect this to just be a global load
  // within the loop.
  for (auto &&localStore : dotLocalStores)
    findClosestPredOps<tt::LoadOp>(localStore.getSrc(), dotGlobalLoads);
}

// Transform a loop into one Dot - Memory (ping - pong) clusters
// Each cluster, especially the Dot cluster is guarded with setprio(1->0) so
// each warp can complete the execution of the cluster without being
// interrupted. This is also supposed to be used with the numWarps=4 case where
// each SIMD runs two warps from different blocks and those two warps don't need
// to be synchronized together.
// Splitting loading A/B and interleave global/local load in order to prevent
// the stalls.
// sched.barriers with 0 mask were used to enforce the boundary of the
// high-level operations, inserting `setPrio` also has a same effect of
// instruction scheduling boundary, too.
void Pingponger::transformOnePPClusters(OpBuilder &builder, Location loc) {
  auto dotLoc = dotOps[0]->getPrevNode();
  // sched barrier to prevent memory ops from cross but leave other ops to be
  // scheduled across the barrier.
  auto preDotBar = builder.create<ROCDL::SchedBarrier>(loc, 1);
  updateOpInsertion(dotLoc);
  appendOp(preDotBar);

  // Memory cluster #0
  updateOpInsertion(lLoadOps[0]);
  appendOp(builder.create<ROCDL::SetPrioOp>(loc, highPriority));
  moveOpAndPredecessorsUpSameBlock(gLoadOps[0]);
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  moveOpAndPredecessorsUpSameBlock(lLoadOps[1]);
  appendOp(builder.create<ROCDL::SetPrioOp>(loc, lowPriority));
  moveOpAndPredecessorsUpSameBlock(gLoadOps[1]);

  // Dot cluster #0
  updateOpInsertion(preDotBar);
  appendOpWithPrio(builder, dotOps[0], loc);
  // Add a remark for user feedback
  dotOps[0]->emitRemark() << "Performed one ping pong cluster transformation\n";
}

void Pingponger::genOffsetConstants(Location loc, OpBuilder &builder,
                                    unsigned numSlices, int64_t sliceWidth) {
  for (int i = 0; i < numSlices; i++) {
    int64_t offset = sliceWidth * i;
    constOffsets.push_back(
        builder.create<arith::ConstantIntOp>(loc, offset, 32));
  }
}

// Splits given local_loads for dot into multiple subviews and local_loads. This
// function tries to slice the local_load into the given number of the slices,
// generates ops when succeed, return fail() otherwise.
LogicalResult Pingponger::genLocalSlice(OpBuilder &builder, Value v,
                                        Attribute dotEncoding, unsigned opIdx,
                                        unsigned numSlices,
                                        int64_t sliceWidth) {
  SmallVector<Operation *> slices;
  SmallVector<Operation *> subviews;
  // TODO: support transformed input to dot
  auto localLoad = v.getDefiningOp<ttg::LocalLoadOp>();
  if (!localLoad)
    return failure();
  auto memDesc = localLoad.getSrc();
  auto type = cast<ttg::MemDescType>(memDesc.getType());
  SmallVector<int64_t> shape = llvm::to_vector(type.getShape());
  Type elementType = type.getElementType();
  int64_t kIdx = opIdx == 0 ? 1 : 0;
  shape[kIdx] = sliceWidth;
  // Each slice cannot be smaller than the smallest supported mfma width.
  if (sliceWidth < 16)
    return failure();
  auto dotOperandEnc = ttg::DotOperandEncodingAttr::get(
      builder.getContext(), opIdx, dotEncoding, kWidth);
  auto subviewDescType = ttg::MemDescType::get(
      shape, elementType, type.getEncoding(), type.getMemorySpace(),
      type.getMutableMemory(), type.getAllocShape());
  for (int i = 0; i < numSlices; i++) {
    SmallVector<Value> offsetsVal;
    SmallVector<int64_t> offsets = {0, 0};
    offsets[kIdx] = i;
    for (int64_t off : offsets) {
      offsetsVal.push_back(constOffsets[off]);
    }
    Value newSmem = builder.create<ttg::MemDescSubviewOp>(
        v.getLoc(), subviewDescType, memDesc, offsetsVal);
    Value prefetchSlice = builder.create<ttg::LocalLoadOp>(
        v.getLoc(), RankedTensorType::get(shape, elementType, dotOperandEnc),
        newSmem);
    subviews.push_back(newSmem.getDefiningOp());
    slices.push_back(prefetchSlice.getDefiningOp());
  }
  subViewOps.push_back(subviews);
  loadSliceOps.push_back(slices);
  return success();
}

// Split dot into 'numSlices' pieces. This is required by pingpong scheduling
// when it needs to schedule multiple dot clusters. Calls genLocalSlice to
// create corresponding local_load slices.
LogicalResult Pingponger::sliceDot(OpBuilder &builder, Location loc,
                                   tt::DotOp op, unsigned numSlices) {
  builder.setInsertionPointToStart(forOp.getBody());
  auto typeB = op.getB().getType();
  auto shapeB = typeB.getShape();
  int64_t sliceWidth = shapeB[0] / numSlices;
  if (shapeB[0] % numSlices != 0)
    return failure();
  genOffsetConstants(loc, builder, numSlices, sliceWidth);
  builder.setInsertionPointAfter(gLoadOps[0]);
  auto dotEncoding = op.getType().getEncoding();
  if (genLocalSlice(builder, op.getA(), dotEncoding, 0, numSlices, sliceWidth)
          .failed() ||
      genLocalSlice(builder, op.getB(), dotEncoding, 1, numSlices, sliceWidth)
          .failed())
    return failure();

  // Clone dots to consume all the slices
  Operation *prevDot = op;
  for (int i = 0; i < numSlices; i++) {
    IRMapping mapping;
    mapping.map(op.getA(), loadSliceOps[0][i]->getResult(0));
    mapping.map(op.getB(), loadSliceOps[1][i]->getResult(0));
    if (i > 0)
      mapping.map(op.getC(), prevDot->getResult(0));
    auto newOp = builder.clone(*op, mapping);
    prevDot = newOp;
    dotSliceOps.push_back(newOp);
  }
  op->replaceAllUsesWith(prevDot);
  op->erase();
  for (auto loads : lLoadOps)
    loads->erase();
  return success();
}

// Transform a loop into four Dot - Memory (ping - pong) clusters
// This transform is useful when the original dot tile is too large that there's
// not enough registers to hold data for a Dot cluster. This path slices the dot
// into four pieces and pair with four clusters of reordered memory operations.
// There are multiple guards at the boundary of each cluster.
// (1) sched.barrier : with mask0 to prevent compiler backed from reordering
//  instructions across the boundary
// (2) gpu.barrier : ensures asymmetric synchronization at each point
// (3) setprio (1->0) : in order to avoid incoming warp overtaking resource
//  while the other warp is actively using it.
//
// Here's overview of the instruction clusters
// mem0: global load A, local load A(1/4), local load B(1/4)
// dot0: dot A(1/4) * B(1/4)
// mem1: global load B, local load A(2/4), local load B(2/4)
// dot1: dot A(2/4) * B(2/4)
// mem2: local load A(3/4, 4/4), local load B(3/4, 4/4)
// dot2: dot A(3/4) * B(3/4)
// mem3: local store A and B
// dot3: dot A(4/4) * B(4/4)

LogicalResult Pingponger::transformFourPPClusters(OpBuilder &builder,
                                                  Location loc) {
  // First, slice local_loads and dot into 4 parts
  if (sliceDot(builder, loc, dotOps[0], 4).failed())
    return failure();
  builder.setInsertionPointAfter(gLoadOps[1]);
  // Reorder operations into four mem/dot clusters

  // mem0: global load A, local load A(1/4), local load B(1/4)
  // set insertion point at the last global_load where all the addresses are
  // ready to be used.
  updateOpInsertion(gLoadOps[1]);
  appendSlicedLoadAB(/*slice=*/0);
  appendClusterBarrier(builder, loc);

  // dot0 (1/4)
  appendOpWithPrio(builder, dotSliceOps[0], loc);
  appendClusterBarrier(builder, loc);

  // mem1: global load B, local load A(2/4), local load B(2/4)
  appendOp(gLoadOps[1]);
  appendSlicedLoadAB(/*slice=*/1);
  appendClusterBarrier(builder, loc);

  // dot1 (2/4)
  appendOpWithPrio(builder, dotSliceOps[1], loc);
  appendClusterBarrier(builder, loc);

  // mem2: local load A(3/4, 4/4), local load B(3/4, 4/4)
  appendSlicedLoadAB(/*slice=*/2);
  appendSlicedLoadAB(/*slice=*/3);
  appendClusterBarrier(builder, loc);

  // dot2 (3/4)
  appendOpWithPrio(builder, dotSliceOps[2], loc);
  appendClusterBarrier(builder, loc);

  // mem3: local store A and B
  // Matmul kernels may use the output of the dot product in another operation
  // before the local store (e.g. persistent matmul epilogue). To accommodate
  // such cases, we need to move the local store up in the loop.
  moveOpAndPredecessorsUpSameBlock(lStoreOps[0]);
  moveOpAndPredecessorsUpSameBlock(lStoreOps[1]);
  appendClusterBarrier(builder, loc);

  // dot3 (4/4)
  appendOpWithPrio(builder, dotSliceOps[3], loc);
  appendClusterBarrier(builder, loc);

  // Add a remark for user feedback
  dotSliceOps[0]->emitRemark()
      << "Performed four ping pong cluster transformation\n";
  return success();
}

// Transform a loop into two Dot - Memory (ping - pong) clusters
// This is useful for the medium sized tile which doesn't fit to either one/four
// cluster scheduling.
LogicalResult Pingponger::transformTwoPPClusters(OpBuilder &builder,
                                                 Location loc) {
  // First, slice local_loads and dot into 2 parts
  if (sliceDot(builder, loc, dotOps[0], 2).failed())
    return failure();
  // Reorder operations into two mem/dot clusters

  // Memory cluster #0
  // interleave local_loads and global_loads to minimize the stalling
  // cycles, sched.barrier prevents backend from canceling the interleaved order
  updateOpInsertion(gLoadOps[1]);
  appendSlicedLoadAB(/*slice=*/0);
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  appendOp(gLoadOps[0]);
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  appendSlicedLoadAB(/*slice=*/1);
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  appendOp(gLoadOps[1]);
  // The first cluster just fits into the two cluster pingpong and cannot
  // include wait of the local_load inserted by the gpu.barrier, using s.barrier
  // instead. backend will schedule the local memory fences later in the dot0
  // cluster.
  appendOp(builder.create<ROCDL::SBarrierOp>(loc));
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));

  // dot0 (1/2)
  appendOpWithPrio(builder, dotSliceOps[0], loc);
  appendClusterBarrier(builder, loc);

  // mem1: local store A and B
  // Matmul kernels may use the output of the dot product in another operation
  // before the local store (e.g. persistent matmul epilogue). To accommodate
  // such cases, we need to move the local store up in the loop.
  moveOpAndPredecessorsUpSameBlock(lStoreOps[0]);
  moveOpAndPredecessorsUpSameBlock(lStoreOps[1]);
  appendClusterBarrier(builder, loc);

  // dot1 (2/2)
  appendOpWithPrio(builder, dotSliceOps[1], loc);
  appendClusterBarrier(builder, loc);

  // Add a remark for user feedback
  dotSliceOps[0]->emitRemark()
      << "Performed two ping pong cluster transformation\n";
  return success();
}

// This function wraps forOp with cond_barrier. First, hold half of the warps
// (warpHigh) in a block before the loop so the barriers in the loop synchronize
// warps at the different point per the warp groups. After the loop, hold
// proceeding warps (warpLow) by calling cond_barrier on them.
void Pingponger::addAsymmetricSyncToLoop(OpBuilder &builder, Location loc) {
  builder.setInsertionPointAfter(forOp);
  // Set barrier before starting the loop. This resolves any remaining required
  // synchronization before beginning the specialized asymmetric
  // synchronization.
  auto preBarrier = builder.create<gpu::BarrierOp>(loc);
  preBarrier->moveBefore(forOp);
  builder.setInsertionPointAfter(preBarrier);

  // Insert condbarrier::second_half before starting the loop
  auto i32ty = builder.getIntegerType(32);
  auto workIDX = builder.create<ROCDL::ThreadIdXOp>(loc, i32ty);
  auto constZero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  auto constWarpSize = builder.create<arith::ConstantIntOp>(loc, 256, 32);
  auto warpIDX = builder.create<arith::DivSIOp>(loc, workIDX, constWarpSize);
  auto warpLow = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               warpIDX, constZero);
  auto warpHigh = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                warpIDX, constZero);
  auto condBarrierHigh =
      builder.create<tt::amdgpu::CondBarrierOp>(loc, warpHigh);

  // Insert condbarrier::first_half after the end of the loop
  builder.setInsertionPointAfter(forOp);
  auto condBarrierLow = builder.create<tt::amdgpu::CondBarrierOp>(loc, warpLow);
}

void Pingponger::getDotPingponged() {
  if (numStages != 2) {
    std::stringstream message;
    message << "All ping pong scheduling requires 2 stages. Found " << numStages
            << " stages";
    LDBG(message.str());
    return;
  }

  OpBuilder builder(forOp);
  MLIRContext *ctx = forOp.getContext();
  Location loc = forOp.getLoc();

  forOp->walk([&](Operation *op) {
    if (auto gLoad = dyn_cast<tt::LoadOp>(op))
      gLoadOps.push_back(gLoad);
    else if (auto lLoad = dyn_cast<ttg::LocalLoadOp>(op)) {
      // This scheduling doesn't help hiding intra-warp latency. So, we only
      // collect local_load ops that are software pipelined, which means their
      // source is from loop carried values
      auto src = lLoad.getSrc();
      if (auto arg = mlir::dyn_cast<BlockArgument>(src))
        if (auto tiedLoopInit = forOp.getTiedLoopInit(arg))
          if (tiedLoopInit->get())
            lLoadOps.push_back(lLoad);
    } else if (auto lStore = dyn_cast<ttg::LocalStoreOp>(op))
      lStoreOps.push_back(lStore);
    else if (auto pingpongDot = dyn_cast<tt::DotOp>(op))
      if (pingpongDot.getType().getRank() == 2)
        dotOps.push_back(pingpongDot);
  });

  // Currently, pingpong scheduling is known as helpful under limited condition.
  // Individual conditions are checked while collecting each operation such as
  // software pipelining and dot rank=2. Also only accept the for-loop with
  // supported combination of operations because this transformation is very
  // tightly scheduling the latencies.
  if (gLoadOps.size() < 2 || lLoadOps.size() < 2 || dotOps.size() != 1) {
    std::stringstream message;
    message << "Unable to match ping pong scheduling pattern. Details: "
            << gLoadOps.size() << " global loads, " << lLoadOps.size()
            << " local loads, " << dotOps.size() << " dot products";
    LDBG(message.str());
    return;
  }

  // The existing code depends on the loads being targeted being safe to move,
  // which will not hold if we do not properly have a GEMM. As a result, we
  // filter the associated load operations to only those that are associated
  // // with the GEMM.
  DenseSet<tt::LoadOp> dotGlobalLoads;
  DenseSet<ttg::LocalLoadOp> dotLocalLoads;
  DenseSet<ttg::LocalStoreOp> dotLocalStores;
  determineDotMemoryOps(dotOps[0], dotGlobalLoads, dotLocalLoads,
                        dotLocalStores);

  auto origGlobalLoadCount = gLoadOps.size();
  auto origLocalLoadCount = lLoadOps.size();
  // Prune Memory operations that may be moved to only those involved in dot
  // computation.
  auto gLoadIt = llvm::remove_if(gLoadOps, [&dotGlobalLoads](tt::LoadOp op) {
    return !dotGlobalLoads.contains(op);
  });
  gLoadOps.erase(gLoadIt, gLoadOps.end());
  auto lLoadIt =
      llvm::remove_if(lLoadOps, [&dotLocalLoads](ttg::LocalLoadOp op) {
        return !dotLocalLoads.contains(op);
      });
  lLoadOps.erase(lLoadIt, lLoadOps.end());
  auto lStoreIt =
      llvm::remove_if(lStoreOps, [&dotLocalStores](ttg::LocalStoreOp op) {
        return !dotLocalStores.contains(op);
      });
  lStoreOps.erase(lStoreIt, lStoreOps.end());
  // All PingPong Scheduler assumes there are 2 movable global loads and 2
  // movable local loads.
  if (gLoadOps.size() != 2 || lLoadOps.size() != 2) {
    std::stringstream message;
    message << "Unable to match ping pong slicing pattern. Details: "
            << gLoadOps.size() << " global loads in dot computation, "
            << lLoadOps.size() << " local loads in dot computation";
    LDBG(message.str());
    return;
  }

  // Pingpong scheduling tries to form two different types of the instruction
  // clusters, i.e., Dot clusters and Memory clusters. While each SIMD has
  // two concurrent warps, both warps can execute a different type of
  // instruction cluster in parallel. Here are currently available patterns,
  // more patterns could be added later.
  //
  // (1) One Dot-Memory (ping-pong) cluster
  //  :Ideal to support small tile size e.g., 128x128x64_FP16. Where amount
  //   of the data used per each iteration is small enough and not causing
  //   local_load waiting or register spilling. Currently used for numWarps=4
  //   case where SIMD can hold two warps from different blocks.
  //
  // (2) Four Dot-Memory (ping-pongx4) clusters
  //  :Useful for the larger tile size e.g., 256x256x64_FP16. Clustering
  //   the Dot instruction (mfma) all together without fetching data requires
  //   GPU to hold all the data for the calculation. Such large tile size
  //   exceeds the amount of register GPU has so, we need to split the dot
  //   into several pieces.
  //
  // (3) Two Dot-Memory (ping-pongx2) clusters
  //  :Covers medium sized tile e.g., 256x128x64_FP16. Different tile size may
  //  require different scheduling pattern because the loop consists of
  //  different amount of memory transfer and dot operation. This scheduling
  //  support the tile sizes not supported by above two methods.
  //
  // N.B., Tile size smaller than 128x128x64_FP16 is likely not compute-bound
  // that pingpong scheduling doesn't help much.

  auto dotType = dotOps[0].getType();
  auto dotShape = dotType.getShape();
  auto aType = dotOps[0].getA().getType();
  auto aShape = aType.getShape();
  auto elemWidth = aType.getElementTypeBitWidth();
  int64_t tileSize = dotShape[0] * dotShape[1] * aShape[1] * elemWidth;

  const int64_t minTile = 262144;      // e.g. 32x128x64x16bit
  const int64_t smallTile = 16777216;  // e.g. 128x128x64x16bit
  const int64_t mediumTile = 33554432; // smallTile x 2
  const int64_t largeTile = 67108864;  // e.g. 256x256x64x16bit

  auto encoding = cast<RankedTensorType>(aType).getEncoding();
  auto srcEncoding = cast<ttg::DotOperandEncodingAttr>(encoding);
  kWidth = srcEncoding.getKWidth();
  auto mfmaEncoding = cast<ttg::AMDMfmaEncodingAttr>(srcEncoding.getParent());
  SmallVector<int64_t> intShape;
  intShape.push_back(mfmaEncoding.getMDim());
  intShape.push_back(mfmaEncoding.getNDim());

  if (numWarps == 4) { // Pingpong between warps from different blocks
    // Transform a loop with small tile size.
    // We've observed that this small tile size spent almost equivalent cycle
    // times for issuing the memory operations and issuing dot operations,
    // smaller tile sizes are not likely to get any advantage from current dot
    // centric pingpong scheduling.
    if (tileSize <= smallTile && tileSize >= minTile)
      transformOnePPClusters(builder, loc);
    // numWarps=4 doesn't need asymmetric sync, return.
    return;
  } else if (numWarps == 8) { // Pingpong between warps from the same block
    if (origGlobalLoadCount != 2 || origLocalLoadCount != 2) {
      std::stringstream message;
      message << "Unable to match ping pong slicing pattern. Details: "
              << gLoadOps.size() << " global loads, " << lLoadOps.size()
              << " local loads";
      LDBG(message.str());
      return;
    }
    if (lStoreOps.size() != 2) {
      std::stringstream message;
      message << "Unable to match ping pong slicing pattern. Details: "
              << lStoreOps.size() << " local stores in dot computation ";
      LDBG(message.str());
      return;
    }
    // Transform a loop where the tile size requires dots to be sliced
    if (tileSize == mediumTile) {
      if (transformTwoPPClusters(builder, dotOps[0]->getLoc()).failed()) {
        LDBG("Encountered failure when trying to execute the two ping pong "
             "cluster transformation");
        return;
      }
    } else if (tileSize >= largeTile) {
      // Avoid known register spilling. i.e., mfma16x16x16 & largetile & kpack>1
      if (intShape[0] == 16 && intShape[1] == 16 && kWidth == 8) {
        LDBG("Reached known register spilling case, skip pingpong scheduling");
        return;
      }
      if (transformFourPPClusters(builder, dotOps[0]->getLoc()).failed()) {
        LDBG("Encountered failure when trying to execute the four ping pong "
             "cluster transformation");
        return;
      }
    } else
      return;

    // Let half of the warps start the loop first and the others follow later
    // but in the synchronized way. This can be accomplished by calling
    // cond_barrier for the second half before the beginning of the loop so they
    // can wait until the first half hit the first barrier in the loop. Also
    // need to call cond_barrier for the first_half after exiting the loop, so
    // all warps can converge again.
    addAsymmetricSyncToLoop(builder, loc);
  }
}

class TritonAMDGPUBlockPingpongPass
    : public TritonAMDGPUBlockPingpongBase<TritonAMDGPUBlockPingpongPass> {
public:
  TritonAMDGPUBlockPingpongPass() = default;
  TritonAMDGPUBlockPingpongPass(int32_t numStages) {
    this->numStages = numStages;
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();
    for (auto funcOp : m.getOps<tt::FuncOp>()) {
      funcOp.walk([&](scf::ForOp forOp) {
        Pingponger pingponger(forOp, ttg::lookupNumWarps(forOp), numStages);
        pingponger.getDotPingponged();
      });
    }
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::createTritonAMDGPUBlockPingpongPass(int32_t numStages) {
  return std::make_unique<TritonAMDGPUBlockPingpongPass>(numStages);
}
