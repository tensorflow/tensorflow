#ifndef TRITON_TRITONGPU_TRANSFORM_PIPELINE_WARPSPECIALIZATION_H_
#define TRITON_TRITONGPU_TRANSFORM_PIPELINE_WARPSPECIALIZATION_H_

#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace scf {
class ForOp;
} // namespace scf
namespace triton::gpu {
// Identify load-mma dependencies and specialize them to different partitions.
LogicalResult specializeLoadMMADependencies(scf::ForOp &loop,
                                            int defaultNumStages);
// This is the final step to prepare a loop for warp specialization. This takes
// a loop with a partition schedule and rewrites the loop such that all SSA
// dependencies between partitions are passed through shared memory and
// multibuffers them according to partition stages.
LogicalResult rewritePartitionDependencies(scf::ForOp &loop);
// Given a loop where the partitions' inputs and outputs have been fully
// rewritten to be reference semantic, partitiong the loop into a
// `ttg.warp_specialize` by duplicating the loop for each partition and
// rematerializing, as necessary, operations in the root partition.
LogicalResult partitionLoop(scf::ForOp loop);
} // namespace triton::gpu
} // namespace mlir

#endif // TRITON_TRITONGPU_TRANSFORM_PIPELINE_WARPSPECIALIZATION_H_
