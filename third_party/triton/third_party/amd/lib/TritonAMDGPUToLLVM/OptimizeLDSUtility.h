#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_OPTIMIZELDSUTILITY_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_OPTIMIZELDSUTILITY_H_

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {

int getCvtOpLDSUsage(RankedTensorType srcTy, RankedTensorType dstTy);

int getCvtOpLDSUsage(triton::gpu::ConvertLayoutOp op);

std::vector<SmallVector<unsigned>> factorizePowerOf2(int n, int rank);

/// Copy given layout with different warpsPerCTA parameter
///
/// \param layout original layout
/// \param warpsPerCTA new warpsPerCTA
/// \returns create layout
triton::gpu::DistributedEncodingTrait
createTmpLayout(triton::gpu::DistributedEncodingTrait layout,
                ArrayRef<unsigned> warpsPerCTA);

/// Creates two chained convert layout operations
///
/// %1 = cvtOp %0 (srcLayout -> dstLayout) // original operation
/// ->
/// %2 = cvtOp %0 (srcLayout -> tmpLayout) // <returned pair>.first
/// %3 = cvtOp %2 (tmpLayout -> dstLayout) // <returned pair>.second
///
/// \param builder
/// \param cvtOp original operation
/// \param tmpLayout
/// \returns pair of created operations
std::pair<triton::gpu::ConvertLayoutOp, triton::gpu::ConvertLayoutOp>
createNewConvertOps(OpBuilder &builder, triton::gpu::ConvertLayoutOp &cvtOp,
                    Attribute tmpLayout);

struct Resources {
  int LDS;
};

Resources
estimateResourcesForReplacement(OpBuilder builder,
                                mlir::triton::gpu::ConvertLayoutOp cvtOp,
                                Attribute tmpLayout);

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_OPTIMIZELDSUTILITY_H_
