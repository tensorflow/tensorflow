#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "triton-loop-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace gpu {

namespace {

/////////////////////////////
// UTILS
/////////////////////////////

class OpBuilderForStage : public OpBuilder {
  std::optional<int> _stage;
  std::optional<CoarseSchedule::Cluster> _cluster;
  CoarseSchedule &_schedule;

public:
  explicit OpBuilderForStage(Operation *op, CoarseSchedule &schedule, int stage,
                             CoarseSchedule::Cluster cluster)
      : OpBuilder(op, nullptr), _schedule(schedule), _stage(stage),
        _cluster(cluster) {}
  explicit OpBuilderForStage(Operation *op, CoarseSchedule &schedule)
      : OpBuilder(op, nullptr), _schedule(schedule) {
    if (_schedule.count(op)) {
      auto sc = _schedule[op];
      _stage = sc.first;
      _cluster = sc.second;
    }
  }
  void setStageCluster(std::pair<int, CoarseSchedule::Cluster> stageCluster) {
    _stage = stageCluster.first;
    _cluster = stageCluster.second;
  }

  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    OpTy op = OpBuilder::create<OpTy>(std::forward<Args>(args)...);
    if (_stage && _cluster) {
      _schedule.insert(op, *_stage, *_cluster);
    }
    return op;
  }
};

DenseSet<Operation *> getTopLevelUsersInLoop(Operation *op, scf::ForOp forOp) {
  DenseSet<Operation *> topLevelUsers;
  SmallVector<OpOperand *> q;
  for (auto &use : op->getUses())
    q.push_back(&use);
  while (!q.empty()) {
    auto use = q.pop_back_val();
    auto yieldOp = dyn_cast<scf::YieldOp>(use->getOwner());
    if (yieldOp && yieldOp->getParentOp() == forOp) {
      for (auto &use :
           forOp.getRegionIterArgs()[use->getOperandNumber()].getUses())
        q.push_back(&use);
      continue;
    }
    Operation *topLevelUser =
        forOp.getBody()->findAncestorOpInBlock(*use->getOwner());
    topLevelUsers.insert(topLevelUser);
  }
  return topLevelUsers;
}

Operation *getFirstUseOfPipelinedOp(SmallVector<Operation *> ops,
                                    scf::ForOp forOp,
                                    CoarseSchedule &schedule) {
  Operation *firstUser = nullptr;
  DenseSet<Operation *> topLevelUsers;
  for (Operation *op : ops) {
    auto users = getTopLevelUsersInLoop(op, forOp);
    topLevelUsers.insert(users.begin(), users.end());
  }
  for (Operation *topLevelUser : topLevelUsers) {
    assert(schedule.count(topLevelUser) && "op user not found in the schedule");
    auto [_useStage, _useCluster] = schedule[topLevelUser];
    if (!firstUser) {
      firstUser = topLevelUser;
    } else {
      auto [_firstUserStage, _firstUserCluster] = schedule[firstUser];
      if (_useStage < _firstUserStage ||
          (_useStage == _firstUserStage &&
           schedule.clusters.isBefore(_useCluster, _firstUserCluster))) {
        firstUser = topLevelUser;
      }
    }
  }
  return firstUser;
}

int getDefUseStageDiff(Operation *op, scf::ForOp forOp,
                       CoarseSchedule &schedule) {
  assert(schedule.count(op) && "LoadOp not found in the schedule");
  auto [defStage, _] = schedule[op];
  std::optional<int> useStage;
  DenseSet<Operation *> topLevelUsers = getTopLevelUsersInLoop(op, forOp);
  for (Operation *topLevelUser : topLevelUsers) {
    auto [_useStage, _] = schedule[topLevelUser];
    useStage = std::min(_useStage, useStage.value_or(_useStage));
  }
  if (!useStage)
    return 0;
  assert(useStage >= defStage && "LoadOp used before defined");
  return useStage.value() - defStage;
}

template <typename BuilderT>
Value createIncrementModulo(BuilderT &builder, Location loc, Value counter,
                            Value modulus, Value zero, Value one,
                            Value *outCond = nullptr) {
  Value addOne = builder.template create<arith::AddIOp>(loc, counter, one);
  Value inRangeCond = builder.template create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, addOne, modulus);
  if (outCond)
    *outCond = inRangeCond;
  return builder.template create<arith::SelectOp>(loc, inRangeCond, addOne,
                                                  zero);
}

/////////////////////////////
// LOWER LOADS
/////////////////////////////

// Create an allocation that can hold distance number of loadOp shapes.
static Value createAlloc(scf::ForOp &forOp, Operation *loadOp,
                         ttg::SharedEncodingTrait sharedEnc,
                         unsigned distance) {
  return triton::createAlloc(
      forOp, cast<RankedTensorType>(loadOp->getResultTypes().front()),
      loadOp->getLoc(), sharedEnc, distance);
}

template <typename BuilderT, typename... Args>
Operation *createWithStage(BuilderT &builder, Location loc, int stage,
                           CoarseSchedule::Cluster cluster, Args &&...args) {
  Operation *op = builder.template create<ttg::AsyncCopyGlobalToLocalOp>(
      loc, std::forward<Args>(args)...);

  return op;
}

// Check if the load can be pipelined entirely in shared memory, with user
// consuming directly the shared memory, without going through registers.
bool canBeShmemPipelined(Operation *op) {
  if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
    // AsyncCopyGlobalToLocalOp does not support the non-zero "other" value.
    // With consumer consuming directly the shared memory, there would be no way
    // to replace masked values with the "other" value.
    if (loadOp.getOther() && !isZeroConst(loadOp.getOther()))
      return false;
  }

  if (!op->hasOneUse())
    return false;
  Operation *user = *op->getUsers().begin();
  if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
    return isa<ttg::NVMMASharedEncodingAttr>(alloc.getType().getEncoding());
  }
  return false;
}

void createAsyncCopy(scf::ForOp forOp, tt::LoadOp loadOp, Value alloc,
                     Value insertIdx, Value extractIdx,
                     CoarseSchedule &schedule) {
  OpBuilderForStage builder(forOp, schedule);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);

  Operation *firstUse = getFirstUseOfPipelinedOp({loadOp}, forOp, schedule);
  assert(firstUse && "LoadOp has no users");
  // Replace the load with async copy, wait and loal_load.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(loadOp);
  builder.setStageCluster(schedule[loadOp]);
  Location loc = loadOp.getLoc();
  Value src = loadOp.getPtr();
  Value mask = loadOp.getMask();
  Value other = loadOp.getOther();
  ttg::MemDescType allocTy = cast<ttg::MemDescType>(alloc.getType());

  // Create async copy
  Value view = createSingleBufferView(builder, alloc, insertIdx);
  Operation *copy = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
      loc, src, view, mask, other, loadOp.getCache(), loadOp.getEvict(),
      loadOp.getIsVolatile());
  Operation *commit =
      builder.create<ttg::AsyncCommitGroupOp>(loc, copy->getResult(0));

  // Create wait and local load
  builder.setStageCluster(schedule[firstUse]);
  Operation *wait =
      builder.create<ttg::AsyncWaitOp>(loc, commit->getResult(0), 0);
  Value viewLoad = createSingleBufferView(builder, alloc, extractIdx);

  if (!loadOp.getOther() || isZeroConst(loadOp.getOther())) {
    // Remove redundant local_load -> local_alloc, but only if
    // we are not using the other value. AsyncCopyGlobalToLocalOp does not
    // support the masking.
    SmallVector<ttg::LocalAllocOp> allocsToErase;
    for (Operation *user : loadOp->getUsers()) {
      if (auto userAlloc = dyn_cast<ttg::LocalAllocOp>(user)) {
        if (allocTy.getEncoding() == userAlloc.getType().getEncoding()) {
          tt::replaceUsesAndPropagateType(builder, userAlloc, viewLoad);
          allocsToErase.push_back(userAlloc);
        }
      }
    }
    for (auto alloc : allocsToErase) {
      alloc.erase();
    }
  }

  // If there are some uses that were not local_allocs, we need to create a
  // local_load for them.
  if (loadOp->use_begin() != loadOp->use_end()) {
    auto sharedLoad = builder.create<ttg::LocalLoadOp>(
        loc, loadOp.getType(), viewLoad, wait->getResult(0));
    auto result = sharedLoad->getResults();

    // Create a select for non-zero other values as they are not handled by
    // AsyncCopyGlobalToLocalOp for now.
    if (other && !isZeroConst(other)) {
      auto select = builder.create<arith::SelectOp>(
          loc, loadOp.getType(),
          // Use the mask operand from the original load, not the one with a
          // potentially transformed layout.
          loadOp.getMask(), sharedLoad.getResult(), other);
      result = select->getResults();
    }
    loadOp->replaceAllUsesWith(result);
  }
  schedule.erase(loadOp);
  loadOp->erase();
}

void createTMAAsyncCopy(
    scf::ForOp forOp, Operation *loadOp, Value desc, Value alloc,
    Value insertIdx, Value extractIdx, Value barrier, Operation *waitOp,
    CoarseSchedule &schedule,
    function_ref<void(OpBuilderForStage &, Value, Value, Value, Value)>
        createCopy) {
  OpBuilderForStage builder(forOp, schedule);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);

  Operation *firstUse = getFirstUseOfPipelinedOp({loadOp}, forOp, schedule);
  assert(firstUse && "LoadOp has no users");
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  Location loc = loadOp->getLoc();

  builder.setInsertionPoint(loadOp);
  builder.setStageCluster(schedule[loadOp]);
  ttg::MemDescType allocTy = cast<ttg::MemDescType>(alloc.getType());

  // Create async copy
  Value view = createSingleBufferView(builder, alloc, insertIdx);

  Value pred = builder.create<arith::ConstantIntOp>(loc, 1, 1);
  Value tmaPtr =
      builder.create<triton::nvidia_gpu::TensorDescToTMAPtrOp>(loc, desc);
  createCopy(builder, tmaPtr, barrier, view, pred);

  // Create local load after the wait
  builder.setInsertionPointAfter(waitOp);
  builder.setStageCluster(schedule[firstUse]);
  Value viewLoad = createSingleBufferView(builder, alloc, extractIdx);
  //  Remove redundant local_load -> local_alloc
  SmallVector<ttg::LocalAllocOp> allocsToErase;
  for (Operation *user : loadOp->getUsers()) {
    if (auto userAlloc = dyn_cast<ttg::LocalAllocOp>(user)) {
      if (allocTy.getEncoding() == userAlloc.getType().getEncoding()) {
        tt::replaceUsesAndPropagateType(builder, userAlloc, viewLoad);
        allocsToErase.push_back(userAlloc);
      }
    }
  }
  for (auto alloc : allocsToErase) {
    alloc.erase();
  }

  // If there are some uses that were not local_allocs, we need to create a
  // local_load for them.
  if (loadOp->use_begin() != loadOp->use_end()) {
    auto sharedLoad = builder.create<ttg::LocalLoadOp>(
        loc, loadOp->getResultTypes().front(), viewLoad);
    auto result = sharedLoad->getResults();
    loadOp->replaceAllUsesWith(result);
  }
  schedule.erase(loadOp);
  loadOp->erase();
}

void createTMAAsyncLoad(scf::ForOp forOp, tt::DescriptorLoadOp loadOp,
                        Value alloc, Value insertIdx, Value extractIdx,
                        Value barrier, Operation *waitOp,
                        CoarseSchedule &schedule) {
  return createTMAAsyncCopy(
      forOp, loadOp, loadOp.getDesc(), alloc, insertIdx, extractIdx, barrier,
      waitOp, schedule,
      [&](OpBuilderForStage &builder, Value tmaPtr, Value barrier, Value view,
          Value pred) {
        auto loc = loadOp.getLoc();
        auto indices = ttng::translateTMAIndices(
            builder, loadOp.getLoc(),
            loadOp.getDesc().getType().getBlockType().getEncoding(),
            loadOp.getIndices());
        builder.create<ttng::AsyncTMACopyGlobalToLocalOp>(
            loadOp.getLoc(), tmaPtr, indices, barrier, view, pred);
      });
}

void createTMAAsyncGather(scf::ForOp forOp, tt::DescriptorGatherOp gatherOp,
                          Value alloc, Value insertIdx, Value extractIdx,
                          Value barrier, Operation *waitOp,
                          CoarseSchedule &schedule) {
  return createTMAAsyncCopy(forOp, gatherOp, gatherOp.getDesc(), alloc,
                            insertIdx, extractIdx, barrier, waitOp, schedule,
                            [&](OpBuilderForStage &builder, Value tmaPtr,
                                Value barrier, Value view, Value pred) {
                              builder.create<ttng::AsyncTMAGatherOp>(
                                  gatherOp.getLoc(), tmaPtr,
                                  gatherOp.getXOffsets(), gatherOp.getYOffset(),
                                  barrier, view, pred);
                            });
}

struct AsyncLoad {
  int stageDiff;
  Value alloc;
  Value barrier;
  Operation *waitOp;
  SharedEncodingTrait sharedEncoding;
};
struct LoadGroupInfo {
  Value insertIdx;
  Value extractIdx;
  Value phase;
  bool hasTMALoad = false;
};

void createTMABarrierAndWait(
    scf::ForOp forOp, llvm::MapVector<Operation *, AsyncLoad> &asyncLoads,
    const llvm::MapVector<int, LoadGroupInfo> &loadGroups,
    CoarseSchedule &schedule) {
  SmallVector<SmallVector<Operation *>> commonWaitGroups;
  llvm::SmallDenseSet<Operation *> visited;
  // Find groups of loads that can share the same barrier. We look consecutive
  // loads and check that there are uses in between.
  for (auto &[loadOp, asyncLoad] : asyncLoads) {
    if (!isTMALoad(loadOp) || visited.count(loadOp))
      continue;
    llvm::SmallDenseSet<Operation *> users;
    SmallVector<Operation *> group;
    Block *loadBlock = loadOp->getBlock();
    auto addToGroup = [&](Operation *loadOp) {
      group.push_back(loadOp);
      visited.insert(loadOp);
      for (Operation *user : loadOp->getUsers()) {
        // Special case for MMAv3 loads, we can ignore the alloc and only
        // consider uses of the alloc op since it will be removed.
        if (canBeShmemPipelined(loadOp)) {
          auto alloc = cast<ttg::LocalAllocOp>(*loadOp->getUsers().begin());
          if (alloc->getBlock() == loadBlock) {
            users.insert(alloc->getUsers().begin(), alloc->getUsers().end());
            continue;
          }
        }
        Operation *userInBlock = loadBlock->findAncestorOpInBlock(*user);
        if (userInBlock)
          users.insert(userInBlock);
      }
    };
    addToGroup(loadOp);
    Operation *nextOp = loadOp->getNextNode();
    int numBuffers = asyncLoad.stageDiff;
    while (nextOp) {
      if (users.count(nextOp) || visited.count(nextOp))
        break;
      if (isTMALoad(nextOp) && asyncLoads.count(nextOp)) {
        if (asyncLoads[nextOp].stageDiff != numBuffers)
          break;
        if (group.size() > 0 && schedule[group[0]] == schedule[nextOp]) {
          addToGroup(nextOp);
        }
      }
      nextOp = nextOp->getNextNode();
    }
    commonWaitGroups.push_back(group);
  }

  // For each group calculate the size and insert the barrier after the last
  // load.
  for (SmallVector<Operation *> &group : commonWaitGroups) {
    int sizeInBytes = 0;
    int numBuffers = asyncLoads[group[0]].stageDiff;
    const LoadGroupInfo loadGroup = loadGroups.find(numBuffers)->second;
    for (Operation *op : group) {
      auto tensorTy = cast<RankedTensorType>(op->getResultTypes()[0]);
      int loadSize = product(tensorTy.getShape());
      sizeInBytes += loadSize * tensorTy.getElementTypeBitWidth() / 8;
    }

    Value barrierAlloc = triton::createBarrierAlloc(forOp, numBuffers);
    Location loc = forOp.getLoc();
    OpBuilderForStage builder(group[0], schedule);
    Value barrier = triton::createSingleBufferView(builder, barrierAlloc,
                                                   loadGroup.insertIdx);
    Value pred = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    builder.create<ttng::BarrierExpectOp>(loc, barrier, sizeInBytes, pred);

    builder.setInsertionPointAfter(group.back());
    Operation *firstUse = getFirstUseOfPipelinedOp(group, forOp, schedule);
    builder.setStageCluster(schedule[firstUse]);
    Value barrierViewWait = triton::createSingleBufferView(
        builder, barrierAlloc, loadGroup.extractIdx);
    auto wait = builder.create<ttng::WaitBarrierOp>(loc, barrierViewWait,
                                                    loadGroup.phase);

    // Update the async loads info.
    for (Operation *op : group) {
      asyncLoads[op].barrier = barrier;
      asyncLoads[op].waitOp = wait;
    }
  }
}

// Check if load requires additional buffer for a mma pipelining
bool loadRequiresAdditionalBuffer(Operation *loadOp) {
  // TODO: Limit the cases to only the wgmma pipelining once mmav5
  // pipelining is integrated with the new pipeliner
  if (canBeShmemPipelined(loadOp)) {
    return true;
  }
  // Pattern match the op sequence used for mmav5 scales
  if (loadOp->hasOneUse()) {
    ttg::LocalAllocOp alloc =
        dyn_cast<ttg::LocalAllocOp>(*loadOp->getUsers().begin());
    if (alloc && alloc->hasOneUse()) {
      if (isa<ttng::TCGen5MMAScaledOp>(*alloc->getUsers().begin())) {
        return true;
      }
    }
  }
  return false;
}

scf::ForOp lowerLoads(scf::ForOp forOp, CoarseSchedule &schedule) {
  llvm::MapVector<Operation *, AsyncLoad> asyncLoads;
  llvm::MapVector<int, LoadGroupInfo> loadGroups;
  // Only visit the top level ops, we do not support pipelining conditional
  // loads for now
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op)) {
      int stageDiff = getDefUseStageDiff(&op, forOp, schedule);
      if (stageDiff == 0 || !isa<RankedTensorType>(op.getResultTypes()[0])) {
        // Don't care about non-pipelined loads. Don't use async loads for
        // scalar values.
        continue;
      }
      SharedEncodingTrait sharedEncoding = getSharedEncoding(&op);
      // Do not create async loads for small loads (cp.async requires at least 4
      // bytes)
      int copyVecBytes = getCopyVecBytes(
          cast<RankedTensorType>(op.getResultTypes()[0]), sharedEncoding);
      if (copyVecBytes >= 4 || isTMALoad(&op)) {
        if (loadRequiresAdditionalBuffer(&op)) {
          // Allocate additional buffer required by the wgmma pipelining.
          stageDiff += 1;
        }
        asyncLoads[&op] = {.stageDiff = stageDiff,
                           .sharedEncoding = sharedEncoding};
      } else if (stageDiff > 1) {
        // Distance-1 loads can in most cases be pipelined in registers without
        // any performance degradation, as the schedule will usually reorder the
        // user and the producer so there is no liverange overlap, and no copy
        // needed.
        op.emitRemark() << "Pipelining load that cannot use vectorized "
                           "copy. This will likely "
                           "lead to pipelining in registers and severe "
                           "performance degradation.";
      }
    }
  }

  if (asyncLoads.empty())
    return forOp;

  for (auto &[loadOp, asyncLoad] : asyncLoads) {
    Value alloc = createAlloc(forOp, loadOp, asyncLoad.sharedEncoding,
                              asyncLoad.stageDiff);
    asyncLoad.alloc = alloc;
    loadGroups.insert({asyncLoad.stageDiff, {}});
    if (isTMALoad(loadOp)) {
      loadGroups[asyncLoad.stageDiff].hasTMALoad = true;
    }
  }

  IRRewriter builder(forOp);
  builder.setInsertionPoint(forOp);
  Location loc = forOp.getLoc();
  // Create a counter to index into the allocations per loop iteration.
  // NOTE: We create two duplicates values, insertIdx and extractIdx so that the
  // pipeliner will re-materialize the value in later stages of the pipeline
  // instead of carrying it as a dependency across multiple iterations.
  Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  SmallVector<Value> newOperands;
  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  for (auto [_, loadGroup] : loadGroups) {
    newOperands.push_back(minusOne); // insertIdx
    newOperands.push_back(minusOne); // extractIdx
    if (loadGroup.hasTMALoad) {
      // A single barrier arrival sequence is a "phase" and two phases can
      // overlap, provided the phases are differentiated with an alternating
      // boolean value.
      newOperands.push_back(zero); // phase
    }
  }

  // Patch the loop to add the new loop carried dependencies.
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, newOperands);
  forOp.erase();
  forOp = newForOp;

  // Update yield op with temporary yield values
  auto forYield = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
  for (unsigned i = 0; i < newOperands.size(); ++i) {
    forYield.getResultsMutable().append(newOperands[i]);
  }

  builder.setInsertionPoint(forOp);
  loc = forOp.getLoc();
  int argIdx = newOperandIndex;
  for (auto &[numBuffers, loadGroup] : loadGroups) {
    Value insertIdx = newForOp.getBody()->getArgument(argIdx);
    argIdx++;
    Value extractIdx = newForOp.getBody()->getArgument(argIdx);
    argIdx++;
    Value phase = nullptr;
    if (loadGroup.hasTMALoad) {
      phase = newForOp.getBody()->getArgument(argIdx);
      argIdx++;
    }

    // Create two counters for the insert and extract indices to avoid creating
    // long liverange.
    builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());

    Value numBuffersVal =
        builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);
    loadGroup.insertIdx = createIncrementModulo(builder, loc, insertIdx,
                                                numBuffersVal, zero, one);
    Value cndExt = nullptr;
    loadGroup.extractIdx = createIncrementModulo(
        builder, loc, extractIdx, numBuffersVal, zero, one, &cndExt);
    if (phase) {
      Value nextPhase = builder.create<arith::XOrIOp>(loc, phase, one);
      phase = builder.create<arith::SelectOp>(loc, cndExt, phase, nextPhase);
      loadGroup.phase = phase;
    }
  }

  createTMABarrierAndWait(forOp, asyncLoads, loadGroups, schedule);

  for (auto [op, asyncLoad] : asyncLoads) {
    auto [insertIdx, extractIdx, phase, _] = loadGroups[asyncLoad.stageDiff];
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      createAsyncCopy(forOp, loadOp, asyncLoad.alloc, insertIdx, extractIdx,
                      schedule);
    } else if (auto loadOp = dyn_cast<tt::DescriptorLoadOp>(op)) {
      createTMAAsyncLoad(forOp, loadOp, asyncLoad.alloc, insertIdx, extractIdx,
                         asyncLoad.barrier, asyncLoad.waitOp, schedule);
    } else if (auto loadOp = dyn_cast<tt::DescriptorGatherOp>(op)) {
      createTMAAsyncGather(forOp, loadOp, asyncLoad.alloc, insertIdx,
                           extractIdx, asyncLoad.barrier, asyncLoad.waitOp,
                           schedule);
    }
  }
  // Patch the yield with the updated counters. Subtract to account for the loop
  // counter.
  argIdx = newOperandIndex - 1;
  for (auto &[numBuffers, loadGroup] : loadGroups) {
    forYield.setOperand(argIdx++, loadGroup.insertIdx);
    forYield.setOperand(argIdx++, loadGroup.extractIdx);
    if (loadGroup.phase)
      forYield.setOperand(argIdx++, loadGroup.phase);
  }

  // Automatically discover dependencies and schedule new insert/extract ops to
  // correct stages.
  scheduleDependencies(forOp, schedule);

  // Insert sync point for any possibly outstanding loads after the loop. This
  // can happen as we speculatively execute loads in the loop.
  builder.setInsertionPointAfter(forOp);
  builder.create<ttg::AsyncWaitOp>(loc, ValueRange({}), 0);

  // Make sure all ops have attributes.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    assert(schedule.count(&op) && "op not found in the schedule");
  }
  return forOp;
}

/////////////////////////////
// LOWER TMA DESCRIPTORS
/////////////////////////////

LogicalResult
allocTMABuffers(scf::ForOp forOp,
                llvm::MapVector<Operation *, Value> &tmaBufferMapping,
                int numStages) {
  IRRewriter rewriter(forOp);

  // Create a multi-buffered allocation for each MakeTensorDescOp call in the
  // loop
  forOp.walk([&](tt::MakeTensorDescOp op) {
    // TODO peter: walk to loop yield to find the init value if this is a
    // loop-carried value. That would save us from allocating another buffer
    // just for the init value
    auto loc = op.getLoc();
    Value alloc = rewriter.create<triton::gpu::GlobalScratchAllocOp>(
        loc, triton::getPointerType(rewriter.getI8Type()),
        numStages * ttng::TMA_SIZE_BYTES, ttng::TMA_ALIGN);
    tmaBufferMapping[op.getOperation()] = alloc;
  });
  return success();
}

template <typename BuilderT>
Value subviewTMADescriptor(BuilderT &builder, Location loc, Value alloc,
                           Value counter) {
  Value tmaSizeVal = builder.template create<arith::ConstantIntOp>(
      loc, ttng::TMA_SIZE_BYTES, 32);
  Value offset =
      builder.template create<arith::MulIOp>(loc, tmaSizeVal, counter);
  return builder.template create<triton::AddPtrOp>(loc, alloc.getType(), alloc,
                                                   offset);
}

LogicalResult rewriteTMABufferUpdates(
    scf::ForOp forOp,
    const llvm::MapVector<Operation *, Value> &tmaBufferMapping,
    ArrayRef<BlockArgument> tmaCounters, int numStages, Value one, Value zero,
    CoarseSchedule &schedule) {
  assert(tmaBufferMapping.size() == tmaCounters.size());

  Value numStagesVal = mlir::OpBuilder(forOp).create<arith::ConstantIntOp>(
      forOp.getLoc(), numStages, 32);

  for (auto [iOp, pair] : llvm::enumerate(tmaBufferMapping)) {
    auto &[op, alloc] = pair;

    // Rewriter MakeTensorDescOp as writing a TMA descriptor
    auto makeDescOp = cast<tt::MakeTensorDescOp>(op);

    OpBuilderForStage stageBuilder(makeDescOp, schedule);
    auto loc = makeDescOp.getLoc();

    BlockArgument counter = tmaCounters[iOp];
    Value nextBuf = subviewTMADescriptor(stageBuilder, loc, alloc, counter);
    if (failed(ttng::createTMADesc(nextBuf, makeDescOp, stageBuilder))) {
      return failure();
    }
    stageBuilder.create<triton::ExperimentalTensormapFenceproxyAcquireOp>(
        loc, nextBuf);
    Value nextDesc = stageBuilder.create<triton::ReinterpretTensorDescOp>(
        loc, makeDescOp.getType(), nextBuf);

    makeDescOp.getResult().replaceAllUsesWith(nextDesc);

    // Increment the buffer index counter
    Value nextCounter = createIncrementModulo(stageBuilder, loc, counter,
                                              numStagesVal, zero, one);

    // If we are in a (potentially nested) if region, propagate the counter
    // up to the main for op body scope
    Operation *curOp = op;
    Operation *parent = op->getParentOp();
    while (parent != forOp.getOperation()) {
      auto ifOp = dyn_cast<scf::IfOp>(parent);
      if (!ifOp) {
        std::string msg;
        llvm::raw_string_ostream ss(msg);
        ss << "Cannot pipeline MakeTensorDescOp inside:\n";
        parent->print(ss);
        ss << "\nOnly scf.if regions are supported";
        return makeDescOp->emitOpError(std::move(msg));
      }

      IRRewriter rewriter(parent);
      auto newIfOp =
          replaceIfOpWithNewSignature(rewriter, ifOp, {nextCounter.getType()});

      auto yieldNewBlock = newIfOp.thenBlock();
      auto yieldOldBlock = newIfOp.elseBlock();

      if (yieldNewBlock != curOp->getBlock()) {
        std::swap(yieldNewBlock, yieldOldBlock);
      }
      cast<scf::YieldOp>(yieldNewBlock->getTerminator())
          .getResultsMutable()
          .append(nextCounter);
      cast<scf::YieldOp>(yieldOldBlock->getTerminator())
          .getResultsMutable()
          .append(counter);

      ifOp.erase();
      nextCounter = newIfOp.getResults().back();
      curOp = newIfOp;
      parent = newIfOp->getParentOp();
    }

    // Finally, rewrite the loop level yield
    auto forYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    forYield.setOperand(counter.getArgNumber() - 1, nextCounter);
  }
  return success();
}

scf::ForOp lowerTMADescriptors(scf::ForOp forOp, CoarseSchedule &schedule) {
  llvm::MapVector<Operation *, Value> tmaBufferMapping;
  if (failed(
          allocTMABuffers(forOp, tmaBufferMapping, schedule.getNumStages()))) {
    llvm_unreachable("TMA pipelining failed");
  }

  IRRewriter builder(forOp);
  Location loc = forOp.getLoc();
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  SmallVector<Value> newOperands;
  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Create one counter per TMA buffer. This allows the descriptors to be
  // updated independently without needing to write duplicate of existing tma
  // descriptors.
  unsigned tmaCounterArgsStartIdx = newOperandIndex + newOperands.size();
  for (int i = 0; i < tmaBufferMapping.size(); ++i) {
    newOperands.push_back(zero);
  }

  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, newOperands);
  forOp.erase();
  forOp = newForOp;

  auto tmaCounters = ArrayRef<BlockArgument>(newForOp.getBody()->getArguments())
                         .slice(tmaCounterArgsStartIdx);

  // Update yield op with temporary yield values
  auto forYield = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
  for (unsigned i = 0; i < newOperands.size(); ++i) {
    forYield.getResultsMutable().append(newOperands[i]);
  }

  if (failed(rewriteTMABufferUpdates(newForOp, tmaBufferMapping, tmaCounters,
                                     schedule.getNumStages(), one, zero,
                                     schedule))) {
    llvm_unreachable("Failed to rewrite TMA ops");
  }
  return newForOp;
}

/////////////////////////////
// LOWER LOOP
/////////////////////////////

void lowerLoop(scf::ForOp forOp) {
  CoarseSchedule schedule;
  if (failed(schedule.deSerialize(forOp))) {
    return;
  }
  scf::ForOp newForOp = lowerLoads(forOp, schedule);
  newForOp = lowerTMADescriptors(newForOp, schedule);
  schedule.serialize(newForOp);
}

} // namespace

void lowerLoops(ModuleOp moduleOp) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  if (loops.empty())
    return;
  for (auto forOp : loops) {
    lowerLoop(forOp);
  }
}

} // namespace gpu
} // namespace triton
} // namespace mlir
