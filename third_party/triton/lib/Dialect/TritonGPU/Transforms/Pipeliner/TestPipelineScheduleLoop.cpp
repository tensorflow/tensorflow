#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUTESTPIPELINESCHEDULELOOP
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

static const char *kLatencyAttrName = "tt.latency";

struct TestPipelineScheduleLoop
    : public impl::TritonGPUTestPipelineScheduleLoopBase<
          TestPipelineScheduleLoop> {
  using impl::TritonGPUTestPipelineScheduleLoopBase<
      TestPipelineScheduleLoop>::TritonGPUTestPipelineScheduleLoopBase;

  void runOnOperation() override { scheduleLoops(getOperation()); }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
