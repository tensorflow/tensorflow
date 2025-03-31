#ifndef TRITON_TRITONGPU_TRANSFORMS_MMAV5PIPELINEUTILITY_H_
#define TRITON_TRITONGPU_TRANSFORMS_MMAV5PIPELINEUTILITY_H_

#include <optional>
#include <tuple>

namespace mlir {
class OpBuilder;

namespace triton::nvidia_gpu {
class MMAv5OpInterface;
class TMEMAllocOp;
class TMEMLoadOp;

// Returns the TMEMAllocOp and TMEMLoadOp that are used to allocate and load the
// accumulator for the given MMA operation. The TMEMAllocOp and TMEMLoadOp must
// be in the same region as the MMA operation.
std::optional<std::pair<TMEMAllocOp, TMEMLoadOp>>
getTMemAllocAndLoad(MMAv5OpInterface mmaOp);
// Create a new TMEMAllocOp to use for the pipelined MMA operation. It is
// optionally multi-buffered based on the number of stages.
TMEMAllocOp createTMemAlloc(OpBuilder &builder, TMEMAllocOp oldTMemAllocOp,
                            bool multiBufferred, int numStages);
} // namespace triton::nvidia_gpu
} // namespace mlir

#endif // TRITON_TRITONGPU_TRANSFORMS_MMAV5PIPELINEUTILITY_H_
