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

#include <string>

#include "nvidia/include/NVGPUToLLVM/NVGPUToLLVMPass.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "xla/backends/gpu/codegen/triton/xla_triton_passes.h"
#include "xla/service/gpu/llvm_gpu_backend/nvptx_libdevice_path.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace xla {
namespace gpu {

namespace mt = ::mlir::triton;
namespace mt_xla = ::mlir::triton::xla;

absl::Status CreateTritonPipeline(mlir::OpPassManager* pm,
                                  std::string arch_name, int num_warps,
                                  int num_ctas, int num_stages,
                                  mt::nvidia_gpu::ClusterInfo& out_cluster_info,
                                  bool is_xla_fusion) {
  TF_ASSIGN_OR_RETURN(
      const stream_executor::CudaComputeCapability cc,
      stream_executor::CudaComputeCapability::FromString(arch_name));
  const int ccAsInt = cc.major * 10 + cc.minor;
  const int threadsPerWarp = 32;

  if (is_xla_fusion) {
    pm->addPass(mt_xla::CreateInt4ToPackedInt4RewritePass());
  }

  // Based on make_ttir() in
  // @triton//:third_party/nvidia/backend/compiler.py
  pm->addPass(mlir::createInlinerPass());
  pm->addPass(mt::createRewriteTensorPointerPass());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mt::createCombineOpsPass());
  pm->addPass(mt::createReorderBroadcastPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createLoopInvariantCodeMotionPass());
  pm->addPass(mlir::createSymbolDCEPass());
  pm->addPass(mt::createLoopUnrollPass());

  // Based on make_ttgir() in
  // @triton//:third_party/nvidia/backend/compiler.py
  pm->addPass(mt::createConvertTritonToTritonGPUPass(
      absl::StrFormat("cuda:%u", ccAsInt), num_warps, threadsPerWarp,
      num_ctas));
  pm->addPass(
      mt_xla::CreateSparseAddEncodingPass(num_warps, threadsPerWarp, num_ctas));
  pm->addPass(mt::gpu::createTritonGPUCoalesce());
  if (cc.IsAtLeastAmpere()) {
    pm->addPass(mt::gpu::createTritonGPUF32DotTC());
  }
  pm->addPass(mlir::createTritonNvidiaGPUPlanCTAPass(&out_cluster_info));
  pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(mt::gpu::createTritonGPUOptimizeThreadLocality());
  pm->addPass(mt_xla::CreateSparseBlockedToMMAPass());
  pm->addPass(mt::gpu::createTritonGPUAccelerateMatmul());
  pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(
      mt::gpu::createTritonGPUOptimizeDotOperands({cc.IsAtLeastAmpere()}));
  pm->addPass(mlir::createCSEPass());

  // Even though we don't run on pre-Ampere architectures anymore, we keep this
  // check for consistency with the upstream pipeline
  if (cc.IsAtLeastAmpere()) {
    pm->addPass(mt::gpu::createTritonGPUOptimizeAccumulatorInit());
    pm->addPass(mt::gpu::createTritonGPUCombineTensorSelectAndIf());
    pm->addPass(mt::gpu::createTritonGPULoopScheduling({num_stages}));
    pm->addPass(mt::gpu::createTritonGPUPipeline({num_stages}));
  }
  pm->addPass(mt::gpu::createTritonGPUPrefetch());
  pm->addPass(
      mt::gpu::createTritonGPUOptimizeDotOperands({cc.IsAtLeastAmpere()}));
  pm->addPass(mt::gpu::createTritonGPUCoalesceAsyncCopy());
  pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(mt_xla::CreateSparseRemoveLayoutConversionPass());
  pm->addPass(mt::gpu::createTritonGPUReduceDataDuplication());
  pm->addPass(mt::gpu::createTritonGPUReorderInstructions());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createSymbolDCEPass());
  if (cc.IsAtLeastHopper()) {
    pm->addPass(mlir::createTritonNvidiaGPUFenceInsertionPass(ccAsInt));
    pm->addPass(mlir::createTritonNvidiaGPUTMALoweringPass());
  }
  pm->addPass(mlir::createCanonicalizerPass());

  // Based on make_llir() in
  // @triton//:third_party/nvidia/backend/compiler.py
  // This pass reduces Hopper compile time extensively: b/344841434.
  if (cc.IsAtLeastHopper()) {
    pm->addPass(mt_xla::CreatePreventMmaV3LoopUnrollingPass());
  }
  pm->addPass(mt::gpu::createTritonGPUCombineTensorSelectAndIf());
  pm->addPass(mlir::createConvertSCFToCFPass());
  pm->addPass(mlir::createConvertIndexToLLVMPass());
  pm->addPass(mt::gpu::createAllocateSharedMemoryPass());
  pm->addPass(mt::gpu::createTritonGPUGlobalScratchAllocationPass());
  pm->addPass(mt_xla::CreateSparseLocalLoadToLLVMPass());
  pm->addPass(mt::createConvertTritonGPUToLLVMPass(ccAsInt));
  // The triton_xla.sparse_dot ops need to be rewritten after
  // ModuleAxisInfoAnalysis inside convert-triton-gpu-to-llvm.
  pm->addPass(mt_xla::CreateSparseDotOpToLLVMPass());
  pm->addPass(mt::createConvertNVGPUToLLVMPass());
  pm->addPass(mt_xla::CreateSparseWGMMAOpToLLVMPass());
  pm->addPass(mlir::createArithToLLVMConversionPass());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createSymbolDCEPass());
  // Note: translateTritonGPUToLLVMIR adds line info with LLVMDIScopePass.

  return absl::OkStatus();
}

std::string GetLibdevicePath(const HloModuleConfig& hlo_config,
                             const se::DeviceDescription& device_info) {
  return nvptx::LibDevicePath(
      hlo_config.debug_options().xla_gpu_cuda_data_dir());
}

}  // namespace gpu
}  // namespace xla
