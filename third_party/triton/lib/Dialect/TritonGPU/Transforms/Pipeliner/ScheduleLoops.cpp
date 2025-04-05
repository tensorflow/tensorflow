#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-loop-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace gpu {

namespace {

bool hasGpuBarriers(scf::ForOp forOp) {
  WalkResult result = forOp.walk(
      [&](mlir::gpu::BarrierOp barrier) { return WalkResult::interrupt(); });
  return result.wasInterrupted();
}

// Return true if the preconditions for pipelining the loop are met.
bool isSafeToPipeline(scf::ForOp forOp) {
  // Skip loop with distance > 1.
  if (loopHasDistGreaterThanOne(forOp))
    return false;
  // Don't pipeline outer loops.
  if (isOuterLoop(forOp))
    return false;
  // Skip loops with barriers.
  if (hasGpuBarriers(forOp))
    return false;
  return true;
}

bool hasLatenciesAssigned(scf::ForOp forOp,
                          const DenseMap<Operation *, int> &opLatency) {
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (opLatency.count(&op))
      return true;
  }
  return false;
}

CoarseSchedule scheduleKeyOps(scf::ForOp forOp,
                              const DenseMap<Operation *, int> &opLatency) {
  llvm::MapVector<Operation *, int> opToStage;
  // Find terminator for later reference
  auto terminator = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  // Determine all operations that have a non-zero latency
  SmallVector<Operation *> latOps;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (opLatency.count(&op))
      latOps.push_back(&op);
  }
  // If no latency ops, nothing to schedule
  if (latOps.empty())
    return CoarseSchedule(0);

  // Compute the longest path to the yield for each operation reachable
  // from any latency operation.
  DenseMap<Operation *, int> distance;
  std::function<int(Operation *)> computeDistance = [&](Operation *op) -> int {
    auto it = distance.find(op);
    if (it != distance.end())
      return it->second;
    // Compute max distance among all users that are inside the loop body
    int maxDist = -1;
    for (Operation *user : op->getUsers()) {
      // Only consider users inside the same block and not the terminator
      Operation *inBlockUser = forOp.getBody()->findAncestorOpInBlock(*user);
      if (!inBlockUser || inBlockUser == terminator)
        continue;
      int distUser = computeDistance(inBlockUser);
      if (distUser > maxDist)
        maxDist = distUser;
    }
    int lat = 0;
    if (opLatency.count(op))
      lat = opLatency.lookup(op);
    // If an op has no users (maxDist == -1) but has latency, we include its
    // latency otherwise it contributes 0 to the distance.
    int d = lat + (maxDist < 0 ? 0 : maxDist);
    distance[op] = d;
    return d;
  };

  // Compute distances for all latency-starting ops
  int maxDistance = 0;
  for (Operation *latOp : latOps) {
    int d = computeDistance(latOp);
    if (d > maxDistance)
      maxDistance = d;
  }

  // Assign stage to each op reachable from a latency op
  for (auto [op, dist] : distance) {
    // We only schedule ops that are downstream of a latency op
    // (had a non-negative distance due to a latency op).
    if (dist >= 0)
      opToStage[op] = maxDistance - dist;
  }

  auto stages = llvm::make_second_range(opToStage);
  int maxStage = *llvm::max_element(stages);
  CoarseSchedule schedule(maxStage + 1);
  SmallVector<CoarseSchedule::Cluster> clusters(maxStage + 1);
  for (int i = 0; i <= maxStage; i++) {
    clusters[i] = schedule.clusters.newAtBack();
  }
  // Assign ops to the clusters in reverse-stage order;
  // ops with higher stage numbers are assigned first. This way we will
  // end up with roughly reverse program order in the clusters.
  for (auto [op, stage] : opToStage)
    schedule.insert(op, stage, clusters[maxStage - stage]);

  // Move `scf.if` ops in the current schedule (forward slice of the latency
  // ops) into a new epilogue cluster at the end of the schedule, pushing them
  // as close to the end of the loop body as possible.
  CoarseSchedule::Cluster epilogue = schedule.clusters.newAtBack();
  for (auto [op, stage] : opToStage) {
    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (!ifOp)
      continue;
    // If the `scf.if` op itself is a latency op, skip it.
    if (opLatency.contains(ifOp))
      continue;
    // Ensure this does not create scheduling conflicts by ensuring the forward
    // slice of the `scf.if` does not contain ops that are already scheduled, as
    // this will cause the `scf.if` to be scheduled after its dependents.
    SetVector<Operation *> slice;
    getForwardSlice(ifOp, &slice);
    if (llvm::any_of(slice, [&](Operation *op) { return opToStage.count(op); }))
      continue;
    schedule.insert(ifOp, stage, epilogue);
  }

  return schedule;
}

// Find dependencies with distance of 1. They will go to the next stage,
// but in the cluster before the current op.
void scheduleDistanceOneDependencies(scf::ForOp forOp,
                                     CoarseSchedule &schedule) {
  int numStages = schedule.getNumStages();

  // Mapping from the cluster to the cluster before it.
  DenseMap<CoarseSchedule::Cluster *, CoarseSchedule::Cluster> dist1Cluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0)
      continue;
    auto [stage, cluster] = schedule[&op];
    // Can't schedule past the last stage.
    if (stage == numStages - 1)
      continue;
    for (Value operand : getNestedOperands(&op)) {
      if (auto arg = dyn_cast<BlockArgument>(operand)) {
        if (arg.getArgNumber() > 0 && arg.getOwner() == op.getBlock()) {
          auto yieldOp = op.getBlock()->getTerminator();
          Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
          Operation *defOp = v.getDefiningOp();
          if (defOp && schedule.count(defOp) == 0) {
            if (isa<tt::LoadOp>(defOp)) {
              // Exception: Schedule loads with a distance of 1 together
              // with the current op.
              schedule.insertIfAbsent(defOp, stage, cluster);
              schedule.insertDepsOfOp(defOp, stage, cluster,
                                      /*includeArg=*/true,
                                      /*insertIfEarlier=*/true);
            } else {
              if (dist1Cluster.count(&cluster) == 0) {
                dist1Cluster[&cluster] = schedule.clusters.newBefore(cluster);
              }
              schedule.insertIfAbsent(defOp, stage + 1, dist1Cluster[&cluster]);
              schedule.insertDepsOfOp(defOp, stage + 1, dist1Cluster[&cluster],
                                      /*includeArg=*/true,
                                      /*includeIfEarlier=*/true);
            }
          }
        }
      }
    }
  }
}

// Schedule the prologue and epilogue `if` ops in the loop, pushing them as
// close to the loop boundaries as possible. Return the cluster after the
// prologue (or the beginning of the loop if there is no prologue).
CoarseSchedule::Cluster schedulePrologueAndEpilogue(scf::ForOp forOp,
                                                    CoarseSchedule &schedule) {
  int numStages = schedule.getNumStages();
  CoarseSchedule::Cluster afterPrologue = schedule.clusters.begin();

  // Look for the IfOp that is in the backward slice any of the currently
  // scheduled ops and put it at the beginning of the loop.
  DenseMap<scf::IfOp, int> ifsToStage;
  // Go stage by stage.
  for (int stage = 0; stage < numStages; stage++) {
    for (auto [op, stage_, cluster] : schedule.getOpsInOrder(forOp)) {
      if (stage_ != stage)
        continue;
      SetVector<Operation *> backwardSlice;
      BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      opt.omitUsesFromAbove = false;
      getBackwardSlice((Operation *)op, &backwardSlice, opt);

      for (auto op : backwardSlice) {
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          ifsToStage.insert({ifOp, stage});
        }
      }
    }
  }
  if (!ifsToStage.empty()) {
    CoarseSchedule::Cluster prologueCluster = schedule.clusters.newAtFront();
    for (auto [ifOp, stage] : ifsToStage) {
      schedule.insertIfAbsent(ifOp, stage, prologueCluster);
    }
  }

  // Other IfOps should be pushed to the end.
  CoarseSchedule::Cluster epilogueCluster = schedule.clusters.newAtBack();
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (ifsToStage.count(ifOp) == 0) {
        schedule.insertIfAbsent(ifOp, numStages - 1,
                                epilogueCluster); // after prefetch extracts
      }
    }
  }
  return afterPrologue;
}

void scheduleRemainingToLastStage(scf::ForOp forOp, CoarseSchedule &schedule,
                                  CoarseSchedule::Cluster afterPrologue) {
  int numStages = schedule.getNumStages();
  // Assign the rest of the ops to the last stage.
  // Take care of the ordering of the ops - uses cannot be scheduled to the
  // cluster before the definition.
  DenseMap<Operation *, CoarseSchedule::Cluster> opToCluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0) {
      opToCluster[&op] = afterPrologue;
    }
  }
  SmallVector<Operation *> queue;
  for (auto [op, stage, cluster] : schedule.getOpsInOrder(forOp)) {
    // We really only care about the producers from the last stage.
    // Others will be scheduled before these ops anyway.
    if (stage == numStages - 1) {
      queue.push_back(op);
    }
  }
  while (!queue.empty()) {
    Operation *op = queue.pop_back_val();
    for (auto user : op->getUsers()) {
      if (opToCluster.count(user)) {
        CoarseSchedule::Cluster userCluster = opToCluster[user];
        CoarseSchedule::Cluster opCluster;
        if (schedule.count(op))
          opCluster = schedule[op].second;
        else
          opCluster = opToCluster[op];
        if (*userCluster < *opCluster) {
          opToCluster[user] = opCluster;
          queue.push_back(user);
        }
      }
    }
  }
  for (auto [op, cluster] : opToCluster) {
    schedule.insert(op, numStages - 1, cluster);
  }
}

void scheduleLoop(scf::ForOp forOp,
                  const DenseMap<Operation *, int> &opLatency) {
  if (!hasLatenciesAssigned(forOp, opLatency) || !isSafeToPipeline(forOp))
    return;
  // Based on the latencies, schedule the key ops to the stages.
  CoarseSchedule schedule = scheduleKeyOps(forOp, opLatency);
  if (schedule.empty())
    return;
  LLVM_DEBUG({
    schedule.serialize(forOp);
    DBGS() << "Initial coarse schedule:\n" << forOp << "\n";
  });
  // Schedule the dependencies
  CoarseSchedule::Cluster afterPrologue =
      schedulePrologueAndEpilogue(forOp, schedule);
  LLVM_DEBUG({
    schedule.serialize(forOp);
    DBGS() << "Coarse schedule with prologue and epilogue:\n" << forOp << "\n";
  });
  scheduleDependencies(forOp, schedule);
  LLVM_DEBUG({
    schedule.serialize(forOp);
    DBGS() << "Coarse schedule with dependencies:\n" << forOp << "\n";
  });
  scheduleDistanceOneDependencies(forOp, schedule);
  LLVM_DEBUG({
    schedule.serialize(forOp);
    DBGS() << "Coarse schedule with dist 1:\n" << forOp << "\n";
  });
  scheduleRemainingToLastStage(forOp, schedule, afterPrologue);
  LLVM_DEBUG({
    schedule.serialize(forOp);
    DBGS() << "Final coarse schedule:\n" << forOp << "\n";
  });

  // Write the schedule to the IR
  schedule.serialize(forOp);
}

} // namespace

void scheduleLoops(ModuleOp moduleOp) {
  DenseMap<Operation *, int> opLatency = deserializeLatencies(moduleOp);
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  if (loops.empty())
    return;
  for (auto forOp : loops) {
    scheduleLoop(forOp, opLatency);
  }
}

} // namespace gpu
} // namespace triton
} // namespace mlir
