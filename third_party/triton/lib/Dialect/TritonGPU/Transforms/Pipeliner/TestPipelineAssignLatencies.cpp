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

#define GEN_PASS_DEF_TRITONGPUTESTPIPELINEASSIGNLATENCIES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct TestPipelineAssignLatencies
    : public impl::TritonGPUTestPipelineAssignLatenciesBase<
          TestPipelineAssignLatencies> {
  using impl::TritonGPUTestPipelineAssignLatenciesBase<
      TestPipelineAssignLatencies>::TritonGPUTestPipelineAssignLatenciesBase;

  void runOnOperation() override {
    assignLatencies(getOperation(), numStages, /*assignMMA=*/true);
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
