#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

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

// Return true if the preconditions for pipelining the loop are met.
bool preCondition(scf::ForOp forOp) {
  // Skip loop with distance > 1 for now.
  // TODO: relax the constraint in the expander.
  if (loopHasDistGreaterThanOne(forOp))
    return false;
  // Don't pipeline outer loops.
  if (isOuterLoop(forOp))
    return false;
  return true;
}

bool hasLatenciesAssigned(scf::ForOp forOp) {
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (op.hasAttr("tt_latency"))
      return true;
  }
  return false;
}

void assignUserProvidedLatencies(scf::ForOp forOp,
                                 DenseMap<Operation *, int> &opLatency) {
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (auto latencyAttr = op.getAttr("tt_latency")) {
      opLatency[&op] = mlir::cast<IntegerAttr>(latencyAttr).getInt();
    }
  }
}

class AssignLoadLatencies {
public:
  AssignLoadLatencies(scf::ForOp forOp, int numStages,
                      DenseMap<Operation *, int> &opLatency)
      : forOp(forOp), numStages(numStages), opLatency(opLatency) {};

  void run() {
    bool pipelineWithoutDot = forOp->hasAttr(mlir::triton::kNumStagesAttrName);
    ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
    tt::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    llvm::MapVector<Operation *, int> loadOpToIndLevel =
        loadOpsToIndirectionLevel(forOp, pipelineWithoutDot, axisInfoAnalysis);
    if (loadOpToIndLevel.empty())
      return;

    // We assume loads with different dist are assigned to different stages.
    // If numStages is 2, we will have no stage available for indirect loads
    // with dist >= 1. In general, when dist is equal to numStages - 1, we
    // should not pipeline it.
    for (auto iter = loadOpToIndLevel.begin();
         iter != loadOpToIndLevel.end();) {
      if (iter->second >= numStages - 1)
        iter = loadOpToIndLevel.erase(iter);
      else
        ++iter;
    }

    // Calculate the stage distance between applicable loads.
    auto vals = llvm::make_second_range(loadOpToIndLevel);
    int maxIndirectionLevel = vals.empty() ? 0 : *llvm::max_element(vals);
    unsigned loadLatency = (numStages - 1) / (maxIndirectionLevel + 1);

    for (auto [loadOp, dist] : loadOpToIndLevel) {
      opLatency[loadOp] = loadLatency;
    }
  }

private:
  scf::ForOp forOp;
  int numStages;
  DenseMap<Operation *, int> &opLatency;

  bool canHaveSharedEncoding(tt::LoadOp op) {
    // If used by an user with DotOp encoding, all the uses must be compatible.
    bool incompatible = false;
    getSharedEncIfAllUsersAreDotEnc(op.getResult(), incompatible);
    return !incompatible;
  }

  bool isSmallLoad(tt::LoadOp loadOp,
                   tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
    assert(!isLoadFromTensorPtr(loadOp) &&
           "Block ptr should have been lowered before this pass.");
    auto ptr = loadOp.getPtr();
    unsigned vec = axisInfoAnalysis.getContiguity(ptr);
    if (auto mask = loadOp.getMask())
      vec = std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return true;
    auto ty = cast<tt::PointerType>(tensorTy.getElementType()).getPointeeType();
    unsigned width = vec * ty.getIntOrFloatBitWidth();

    // We do not pipeline all loads for the following reasons:
    // 1. On nvidia GPUs, cp.async's cp-size can only be 4, 8, or 16.
    // 2. It's likely that pipling small loads won't offer much performance
    //    improvement and may even hurt performance by increasing register
    //    pressure.
    LDBG("Load " << *loadOp << " has width " << width);
    return width < 32;
  }

  bool isPipeliningBeneficial(Operation *op, Operation *finalUser,
                              tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      if (isSmallLoad(loadOp, axisInfoAnalysis)) {
        LDBG("Load " << *loadOp << " is too small for pipelining");
        return false;
      }
    }
    if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op))
      return true;
    if (!canHaveSharedEncoding(cast<tt::LoadOp>(op))) {
      LDBG("Load " << *op << " cannot have shared encoding");
      return false;
    }

    ttg::SharedEncodingTrait localAllocEnc;
    if (llvm::any_of(op->getUsers(), [&](Operation *user) {
          return isa<ttg::LocalAllocOp>(user);
        })) {
      for (auto user : op->getUsers()) {
        auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user);
        if (!localAlloc)
          continue;
        auto enc = mlir::cast<ttg::SharedEncodingTrait>(
            localAlloc.getType().getEncoding());
        if (!localAllocEnc) {
          localAllocEnc = enc;
        }
        if (enc != localAllocEnc) {
          // If the load is used by a LocalAllocOp, all the users need to have
          // the same encoding.
          return false;
        }
      }
    }

    if (localAllocEnc) {
      auto registerTy = cast<RankedTensorType>(op->getResultTypes()[0]);
      auto vecBytes = getCopyVecBytes(registerTy, localAllocEnc);
      if (vecBytes < 4) {
        // At least 4 bytes need to be consecutive for cp.async
        return false;
      }
    }

    return true;
  }

  // Create a map from load ops to their indirection level and the
  // final use of the load op (another load op, or a dot op).
  // Indirection level is "0" for the load op directly used by the dot op,
  // "1" for the load op used by the load op used by the dot op, and so on.
  llvm::MapVector<Operation *, int>
  loadOpsToIndirectionLevel(scf::ForOp forOp, bool pipelineWithoutDot,
                            tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
    llvm::MapVector<Operation *, int> loadOpToIndLevel;
    DenseSet<Operation *> seen;
    DenseSet<Operation *> excluded;

    std::function<void(Operation *, Operation *, int)> dfs =
        [&](Operation *op, Operation *finalUser, int distance) {
          if (!seen.insert(op).second || excluded.count(op))
            return;
          if (isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(
                  op)) {
            if (!isPipeliningBeneficial(op, finalUser, axisInfoAnalysis))
              return;
            if (loadOpToIndLevel.count(op)) {
              int level = loadOpToIndLevel[op];
              if (level != distance) {
                // If we have multiple uses at different distances, we don't
                // know which one to pick.
                LDBG("Load " << *op
                             << " has multiple uses at different distances:"
                             << level << " and " << distance);
                loadOpToIndLevel.erase(op);
                excluded.insert(op);
                return;
              }
            } else {
              LDBG("Load " << *op << " considered for pipelining with distance "
                           << distance);
              loadOpToIndLevel[op] = distance;
            }
            finalUser = op;
            distance++;
          }
          for (Value operand : getNestedOperands(op)) {
            if (isa<mlir::triton::DotOpInterface>(op)) {
              // Heuristic: only pipeline A and B operands of the dot op.
              if (operand == op->getOperand(2))
                continue;
            }
            Value v = operand;
            Operation *defOp = v.getDefiningOp();
            if (defOp && defOp->getBlock() == op->getBlock()) {
              dfs(defOp, finalUser, distance);
            }
          }
        };

    bool seenDot = false;
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!isa<mlir::triton::DotOpInterface>(op))
        continue;
      seenDot = true;
      seen.clear();
      dfs(&op, &op, 0);
    }

    // If the loop has numStages attribute, also consider pipelining other loads
    // that are not directly used by dot ops.
    if (pipelineWithoutDot && !seenDot) {
      for (Operation &op : forOp.getBody()->without_terminator()) {
        if (!isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op))
          dfs(&op, &op, 0);
      }
    }

    return loadOpToIndLevel;
  }
};

class AssignMMALatencies {
public:
  AssignMMALatencies(scf::ForOp forOp, DenseMap<Operation *, int> &opLatency)
      : forOp(forOp), opLatency(opLatency) {};

  void run() {
    for (auto &op : forOp.getBody()->without_terminator()) {
      if (isa<ttng::MMAv5OpInterface>(op) &&
          isPipeliningOfMMAOpPossible(&op, forOp)) {
        opLatency[&op] = 1;
      }
    }
  }

private:
  scf::ForOp forOp;
  DenseMap<Operation *, int> &opLatency;

  bool isPipeliningOfMMAOpPossible(Operation *op, scf::ForOp forOp) {
    assert((isa<ttng::MMAv5OpInterface>(op)) && "Only MMA ops are supported");
    // Operands of the MMA op must come from the load, or from outside the loop.
    auto comesFromLoadOrOutsideLoop = [&](Value v) {
      if (forOp.isDefinedOutsideOfLoop(v)) {
        return true;
      }
      // Do not walk through the Block Arguments.
      if (!v.getDefiningOp()) {
        return false;
      }
      if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(v.getDefiningOp())) {
        if (!localAlloc.getSrc()) {
          return false;
        }
        return (localAlloc.getSrc().getDefiningOp() &&
                isa<tt::LoadOp>(localAlloc.getSrc().getDefiningOp())) ||
               forOp.isDefinedOutsideOfLoop(localAlloc.getSrc());
      }
      return false;
    };
    if (auto dotOp = dyn_cast<tt::DotOpInterface>(op)) {
      if (!comesFromLoadOrOutsideLoop(dotOp.getA()) ||
          !comesFromLoadOrOutsideLoop(dotOp.getB())) {
        return false;
      }
    }

    // For scaled MMA check if the scales are passed through shared memory, and
    // also coming from load or outside the loop.
    if (auto scaledOp = dyn_cast<ttng::TCGen5MMAScaledOp>(op)) {
      if (!isa<ttg::SharedEncodingTrait>(
              scaledOp.getAScale().getType().getEncoding()) ||
          !isa<ttg::SharedEncodingTrait>(
              scaledOp.getBScale().getType().getEncoding()))
        return false;
      if (!comesFromLoadOrOutsideLoop(scaledOp.getAScale()) ||
          !comesFromLoadOrOutsideLoop(scaledOp.getBScale()))
        return false;
    }
    return true;
  }
};

} // namespace

// Look for load ops that directly or indirectly feed into dot ops. Based
// on the requested number of stages assign the latencies in a way that
// cover all the stages with the sum of latencies in the chain from the first
// load to the final dot op.
void assignLatencies(ModuleOp moduleOp, int defaultNumStages, bool assignMMA) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) {
    // Bail out for loops with num_stage <= 1.
    if (preCondition(forOp) &&
        getNumStagesOrDefault(forOp, defaultNumStages) > 1)
      loops.push_back(forOp);
  });
  if (loops.empty())
    return;

  DenseMap<Operation *, int> opLatency;
  for (auto forOp : loops) {
    if (hasLatenciesAssigned(forOp)) {
      assignUserProvidedLatencies(forOp, opLatency);
      continue;
    }
    int numStages = getNumStagesOrDefault(forOp, defaultNumStages);
    AssignLoadLatencies(forOp, numStages, opLatency).run();
    if (assignMMA) {
      AssignMMALatencies(forOp, opLatency).run();
    }
  }
  serializeLatencies(moduleOp, opLatency);
}
} // namespace gpu
} // namespace triton
} // namespace mlir
