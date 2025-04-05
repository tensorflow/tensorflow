#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PASSES_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PASSES_H_

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {

std::unique_ptr<Pass>
createTritonAMDGPUStreamPipelinePass(int numStages = 2, int globalPrefetch = 0,
                                     int localPrefetch = 0,
                                     bool useAsyncCopy = false);

std::unique_ptr<Pass>
createTritonAMDGPUAccelerateMatmulPass(std::string archGenName = std::string(),
                                       int matrixInstructionSize = 0,
                                       int kpack = 1);

std::unique_ptr<Pass> createTritonAMDGPUCanonicalizeLoopsPass();

std::unique_ptr<Pass> createTritonAMDGPUReorderInstructionsPass();

std::unique_ptr<Pass> createTritonAMDGPUVerifier();

std::unique_ptr<Pass> createTritonAMDGPUOptimizeEpiloguePass();

std::unique_ptr<Pass> createTritonAMDGPUHoistLayoutConversionsPass();

std::unique_ptr<Pass> createTritonAMDGPUCanonicalizePointersPass();

std::unique_ptr<Pass> createTritonAMDGPUConvertToBufferOpsPass(
    std::string archGenName = std::string());

std::unique_ptr<Pass>
createTritonAMDGPUBlockPingpongPass(int32_t numStages = 2);

std::unique_ptr<Pass> createTritonAMDGPUInThreadTransposePass();

std::unique_ptr<Pass>
createTritonAMDGPUCoalesceAsyncCopyPass(std::string archGenName = {});

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "TritonAMDGPUTransforms/Passes.h.inc"

} // namespace mlir
#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PASSES_H_
