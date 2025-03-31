#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_SCHEDINSTRUCTIONS_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_SCHEDINSTRUCTIONS_H_

#include "mlir/IR/Types.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

// The following functions are used to collect and set side-channel information
// during to LLVM conversion/lowering to facilitate instruction scheduling
// controls.
namespace mlir::triton {
template <typename DotOpType>
void setNumGeneratedMMAs(DotOpType op, size_t mmaCount, unsigned m, unsigned n,
                         unsigned k, Type elementType);

template <typename LoadOpType>
void setNumGeneratedGlobalLoads(LoadOpType op, size_t globalLoadsCount,
                                Type type);
void setNumGeneratedDsReads(gpu::LocalLoadOp op, size_t numDsReadsCount,
                            Type type);
void storeOpSchedAnnotations(triton::gpu::LocalStoreOp op, size_t llvmOpCount,
                             Type type);
triton::DotOp getSingleDotOpIfExists(scf::ForOp forOp);
} // namespace mlir::triton

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_SCHEDINSTRUCTIONS_H_
