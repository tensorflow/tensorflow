/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "absl/strings/str_format.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace xla {
namespace gpu {

namespace mt = ::mlir::triton;
namespace mt_xla = ::mlir::triton::xla;
namespace ttng = mlir::triton::nvidia_gpu;

// Based on make_ttir() in
// @triton//:third_party/nvidia/backend/compiler.py
static void MakeTTIR(mlir::OpPassManager* pm,
                     const stream_executor::CudaComputeCapability& cuda_cc) {
  pm->addPass(mt_xla::CreateRoundF32ToTF32ForTf32DotRewritePass());
  pm->addPass(mlir::createInlinerPass());
  pm->addPass(mt::createTritonRewriteTensorPointer());
  if (!cuda_cc.IsAtLeastHopper()) {
    pm->addPass(mt::createTritonRewriteTensorDescriptorToPointer());
  }
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mt::createTritonCombineOps());
  pm->addPass(mt::createTritonReorderBroadcast());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createSymbolDCEPass());
  pm->addPass(mt::createTritonLoopUnroll());
}

// Based on make_ttgir() in
// @triton//:third_party/nvidia/backend/compiler.py
static void MakeTTGIR(mlir::OpPassManager* pm,
                      const stream_executor::CudaComputeCapability& cuda_cc,
                      int num_warps, int num_ctas, int num_stages) {
  const int cuda_cc_as_int = cuda_cc.major * 10 + cuda_cc.minor;
  pm->addPass(mt::createConvertTritonToTritonGPU(
      {absl::StrFormat("cuda:%u", cuda_cc_as_int), num_warps,
       /*threads_per_warp=*/32, num_ctas}));
  pm->addPass(mt::gpu::createTritonGPUCoalesce());
  pm->addPass(mt::gpu::createTritonGPUF32DotTC({cuda_cc.IsAtLeastAmpere()}));
  pm->addPass(ttng::createTritonNvidiaGPUPlanCTAPass());
  pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(mt::gpu::createTritonGPUOptimizeThreadLocality());
  pm->addPass(mt::gpu::createTritonGPUAccelerateMatmul());
  pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(
      mt::gpu::createTritonGPUOptimizeDotOperands({cuda_cc.IsAtLeastAmpere()}));
  pm->addPass(ttng::createTritonNvidiaGPUOptimizeDescriptorEncodingPass());
  pm->addPass(mt::createTritonLoopAwareCSE());
  if (cuda_cc.IsAmpere() || cuda_cc.IsHopper()) {
    pm->addPass(mt::gpu::createTritonGPUFuseNestedLoops());
    pm->addPass(mlir::createCanonicalizerPass());
    pm->addPass(mlir::createLoopInvariantCodeMotionPass());
    pm->addPass(mlir::createCanonicalizerPass());
    pm->addPass(mt::gpu::createTritonGPUCombineTensorSelectAndIf());
    pm->addPass(mlir::createNVGPUWarpSpecialization({num_stages}));
    pm->addPass(mt::gpu::createTritonGPUAssignLatencies({num_stages}));
    pm->addPass(mt::gpu::createTritonGPUScheduleLoops());
    pm->addPass(mt::gpu::createTritonGPUPipeline({num_stages}));
  } else if (cuda_cc.IsAtLeastBlackwell()) {
    pm->addPass(mt::gpu::createTritonGPUFuseNestedLoops());
    pm->addPass(mlir::createCanonicalizerPass());
    pm->addPass(mlir::createLoopInvariantCodeMotionPass());
    pm->addPass(mt::gpu::createTritonGPUOptimizeAccumulatorInit());
    pm->addPass(mt::gpu::createTritonGPUHoistTMEMAlloc({false}));
    pm->addPass(ttng::createTritonNvidiaGPUPromoteLHSToTMemPass());
    pm->addPass(mt::gpu::createTritonGPUAssignLatencies({num_stages}));
    pm->addPass(mt::gpu::createTritonGPUScheduleLoops());
    pm->addPass(
        mt::gpu::createTritonGPUAutomaticWarpSpecialization({num_stages}));
    pm->addPass(mt::gpu::createTritonGPUPipeline({num_stages}));
    pm->addPass(mt::gpu::createTritonGPUOptimizePartitionWarps());
    pm->addPass(mt::gpu::createTritonGPUCombineTensorSelectAndIf());
    pm->addPass(mt::gpu::createTritonGPUHoistTMEMAlloc({true}));
    pm->addPass(ttng::createTritonNvidiaGPURemoveTMEMTokensPass());
  } else {
    pm->addPass(mlir::createLoopInvariantCodeMotionPass());
  }
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mt::createTritonLoopAwareCSE());
  pm->addPass(mt::gpu::createTritonGPUPrefetch());
  pm->addPass(
      mt::gpu::createTritonGPUOptimizeDotOperands({cuda_cc.IsAtLeastAmpere()}));
  pm->addPass(mt::gpu::createTritonGPUCoalesceAsyncCopy());
  pm->addPass(ttng::createTritonNvidiaGPUOptimizeTMemLayoutsPass());
  if (cuda_cc.IsAtLeastHopper()) {
    pm->addPass(ttng::createTritonNvidiaGPUTMALoweringPass());
  }
  pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(ttng::createTritonNvidiaGPUInterleaveTMemPass());
  pm->addPass(mt::gpu::createTritonGPUReduceDataDuplication());
  pm->addPass(mt::gpu::createTritonGPUReorderInstructions());
  pm->addPass(mt::createTritonLoopAwareCSE());
  pm->addPass(mlir::createSymbolDCEPass());
  pm->addPass(ttng::createTritonGPUFenceInsertion({cuda_cc_as_int}));
  pm->addPass(ttng::createTritonNvidiaGPUMMALoweringPass());
  pm->addPass(mlir::createSCCPPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createCanonicalizerPass());
  // Corresponds to "mod.get_tensordesc_metadata()"
  // in @triton//:third_party/nvidia/backend/compiler.py
  pm->addPass(mt_xla::CreateExtractTmaInfoPass());
}

int GetDefaultPtxVersion(
    const stream_executor::CudaComputeCapability& cuda_cc) {
  if (cuda_cc.IsAtLeastHopper()) {
    // Upstream defaults to 8.6
    // @triton//:third_party/nvidia/backend/compiler.py
    return 86;
  }
  // Fallback for older architectures.
  return 80;
}

static void MakeLLIR(mlir::OpPassManager* pm,
                     const stream_executor::CudaComputeCapability& cuda_cc) {
  const int cuda_cc_as_int = cuda_cc.major * 10 + cuda_cc.minor;
  const int final_ptx_version = GetDefaultPtxVersion(cuda_cc);

  pm->addPass(mt::gpu::createTritonGPUCombineTensorSelectAndIf());
  pm->addPass(mt::gpu::createTritonGPUAllocateWarpGroups());
  pm->addPass(mlir::createSCFToControlFlowPass());
  pm->addPass(mlir::triton::gluon::createGluonInline());
  pm->addPass(
      mt::createAllocateSharedMemoryNvPass(cuda_cc_as_int, final_ptx_version));
  pm->addPass(ttng::createTritonTensorMemoryAllocationPass());
  pm->addPass(ttng::createTritonNvidiaGPUCheckMatmulTwoCTAPass());
  // We could add a flag to XLA to optionally enable the following pass:
  // pm->addPass(mt::instrument::createTritonInstrumentConcurrencySanitizer());
  pm->addPass(mt::gpu::createTritonGPUGlobalScratchAllocationPass());
  pm->addPass(ttng::createTritonGPUProxyFenceInsertion({cuda_cc_as_int}));
  pm->addPass(
      mt::createConvertTritonGPUToLLVMPass(cuda_cc_as_int, final_ptx_version));
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mt::createConvertNVGPUToLLVM());
  pm->addPass(mt::createConvertWarpSpecializeToLLVM());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createSymbolDCEPass());
  pm->addPass(mlir::createConvertNVVMToLLVMPass());
  // Note: translateTritonGPUToLLVMIR adds line info with LLVMDIScopePass.
}

void CreateTritonCudaPipeline(
    mlir::OpPassManager* pm,
    const stream_executor::CudaComputeCapability& cuda_cc, int num_warps,
    int num_ctas, int num_stages) {
  MakeTTIR(pm, cuda_cc);
  MakeTTGIR(pm, cuda_cc, num_warps, num_ctas, num_stages);
  MakeLLIR(pm, cuda_cc);
}

}  // namespace gpu
}  // namespace xla
