#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// This file will create a schedule that will be handed over to the pipeline
// expander.
// Software pipeliners are usually separated into two pieces, one that create a
// modulo schedule and an expander that rewrites the loop and emits a prologue
// and epilogue. This pass first calls a helper that will pre-process the IR
// to create async operations and create a modulo schedule. Then we call the
// expander to generate the prologue and new loop.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPIPELINE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

static void pipelineWgmma(ModuleOp moduleOp) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

  for (scf::ForOp forOp : loops) {
    mlir::triton::asyncLaunchDots(forOp);
  }
}

static void expandLoops(ModuleOp moduleOp) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  for (scf::ForOp forOp : loops) {
    CoarseSchedule schedule;
    if (failed(schedule.deSerialize(forOp))) {
      continue;
    }

    std::vector<std::pair<Operation *, unsigned>> finalSchedule =
        schedule.createFinalSchedule(forOp);
    triton::PipeliningOption options;
    options.supportDynamicLoops = true;
    options.peelEpilogue = false;
    options.predicateFn = triton::predicateOp;
    options.getScheduleFn =
        [&](scf::ForOp forOp,
            std::vector<std::pair<Operation *, unsigned>> &schedule) {
          schedule = finalSchedule;
        };
    IRRewriter rewriter(forOp);
    FailureOr<scf::ForOp> newForOp =
        triton::pipelineForLoop(rewriter, forOp, options);
  }
}

static void removeAttributes(ModuleOp moduleOp) {
  moduleOp->walk([&](Operation *op) {
    op->removeAttr(mlir::triton::kLoopStageAttrName);
    op->removeAttr(mlir::triton::kLoopClusterAttrName);
    op->removeAttr(mlir::triton::kScheduledMaxStageAttrName);
  });
}

struct PipelinePass : public impl::TritonGPUPipelineBase<PipelinePass> {

  using impl::TritonGPUPipelineBase<PipelinePass>::TritonGPUPipelineBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    // Go over the interesting ops and assign latencies (based on the
    // numStages) to the them, trying to populate the allowed stages. This
    // step will be at some point extracted to separate pass that will be run
    // only for loops missing the latency information.
    assignLatencies(moduleOp, numStages);
    if (dumpIntermediateSteps) {
      llvm::dbgs() << "// -----// SoftwarePipeliner internal IR Dump After: "
                      "AssignLatencies\n"
                   << moduleOp << "\n\n\n";
    }
    // numStages should not be used below this point. We should know
    // everything based on the assigned stages

    // Schedule the loops
    scheduleLoops(moduleOp);
    if (dumpIntermediateSteps) {
      llvm::dbgs() << "// -----// SoftwarePipeliner internal IR Dump After: "
                      "ScheduleLoops\n"
                   << moduleOp << "\n\n\n";
    }

    // Transform the loop by introducing async operations to prepare it for
    // pipeline expansion.
    lowerLoops(moduleOp);
    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// SoftwarePipeliner internal IR Dump After: LowerLoops\n"
          << moduleOp << "\n\n\n";
    }

    // Apply the pipeline expansion.
    expandLoops(moduleOp);
    if (dumpIntermediateSteps) {
      llvm::dbgs() << "// -----// SoftwarePipeliner internal IR Dump After: "
                      "ExpandLoops\n"
                   << moduleOp << "\n\n\n";
    }

    // Cleanup the IR from the pipeline attributes.
    removeAttributes(moduleOp);

    pipelineWgmma(moduleOp);

    // There is a hard dependency between load pipelining and the TC05MMA
    // pipelining. We can pipeline the TC05MMA only after the loads are
    // pipelined and buffers are allocated.
    mlir::triton::pipelineTC05MMALoops(moduleOp, 2);

    // schedule the waits
    mlir::triton::updateWaits(getOperation());

    // Clean up arithmetic before applying the next level of pipelining to
    // simplify the IR.
    auto arithDialect =
        getOperation().getContext()->getLoadedDialect<arith::ArithDialect>();
    RewritePatternSet patterns(getOperation().getContext());
    arithDialect->getCanonicalizationPatterns(patterns);
    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
      return signalPassFailure();

    {
      SmallVector<scf::ForOp> loops;
      getOperation()->walk([&](scf::ForOp forOp) {
        // Bail out for loops with num_stage <= 1.
        if (getNumStagesOrDefault(forOp, numStages) > 1)
          loops.push_back(forOp);
      });

      for (scf::ForOp forOp : loops) {
        mlir::triton::pipelineTMAStores(forOp);
      }

      for (scf::ForOp forOp : loops) {
        mlir::triton::pipelineMMAWithScaledAcc(forOp);
      }
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
