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
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/triton_sparse_extensions.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace xla {
namespace gpu {

namespace mt = ::mlir::triton;

absl::Status CreateTritonPipeline(
    mlir::OpPassManager& pm, const se::GpuComputeCapability& cc,
    const BlockLevelParameters& block_level_parameters,
    mt::nvidia_gpu::ClusterInfo& out_cluster_info) {
  auto ccCuda = std::get<se::CudaComputeCapability>(cc);
  const int ccAsInt = ccCuda.major * 10 + ccCuda.minor;
  const int threadsPerWarp = 32;

  // Based on make_ttir() in
  // @triton//:third_party/nvidia/backend/compiler.py
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mt::createRewriteTensorPointerPass());
  pm.addPass(mt::createCombineOpsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mt::createReorderBroadcastPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSymbolDCEPass());

  // Based on make_ttgir() in
  // @triton//:third_party/nvidia/backend/compiler.py
  pm.addPass(mt::createConvertTritonToTritonGPUPass(
      absl::StrFormat("cuda:%u", ccAsInt), block_level_parameters.num_warps,
      threadsPerWarp, block_level_parameters.num_ctas));
  pm.addPass(createAddSparseDotEncodingPass(block_level_parameters.num_warps,
                                            threadsPerWarp,
                                            block_level_parameters.num_ctas));
  pm.addPass(mt::gpu::createTritonGPUCoalesce());
  if (ccCuda.IsAtLeastAmpere()) {
    pm.addPass(mt::gpu::createTritonGPUF32DotTC());
  }
  pm.addPass(mlir::createTritonNvidiaGPUPlanCTAPass(&out_cluster_info));
  pm.addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  pm.addPass(mt::gpu::createTritonGPUOptimizeThreadLocality());
  pm.addPass(mt::gpu::createTritonGPUAccelerateMatmul());
  pm.addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  pm.addPass(
      mt::gpu::createTritonGPUOptimizeDotOperands({ccCuda.IsAtLeastAmpere()}));
  pm.addPass(mlir::createCSEPass());

  // Even though we don't run on pre-Ampere architectures anymore, we keep this
  // check for consistency with the upstream pipeline
  if (ccCuda.IsAtLeastAmpere()) {
    pm.addPass(mt::gpu::createTritonGPUCombineTensorSelectAndIf());
    pm.addPass(
        mt::gpu::createTritonGPUPipeline({block_level_parameters.num_stages}));
  }
  pm.addPass(mt::gpu::createTritonGPUPrefetch());
  pm.addPass(
      mt::gpu::createTritonGPUOptimizeDotOperands({ccCuda.IsAtLeastAmpere()}));
  pm.addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  pm.addPass(mt::gpu::createTritonGPUReduceDataDuplication());
  pm.addPass(mt::gpu::createTritonGPUReorderInstructions());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  if (ccCuda.IsAtLeastHopper()) {
    pm.addPass(mlir::createTritonNvidiaGPUFenceInsertionPass(ccAsInt));
    pm.addPass(mlir::createTritonNvidiaGPUTMALoweringPass());
  }
  pm.addPass(mlir::createCanonicalizerPass());

  // Based on make_llir() in
  // @triton//:third_party/nvidia/backend/compiler.py
  pm.addPass(mt::NVIDIA::createDecomposeUnsupportedConversionsPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mt::gpu::createAllocateSharedMemoryPass());
  pm.addPass(mt::createConvertTritonGPUToLLVMPass(ccAsInt));
  pm.addPass(mt::createConvertNVGPUToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
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
