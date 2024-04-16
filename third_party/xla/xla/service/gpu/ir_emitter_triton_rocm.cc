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
// TODO(ROCm): Enable and include ROCm Triton passes when ROCm Triton is
// included in build.
// #include "third_party/amd/include/TritonAMDGPUToLLVM/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/hlo_module_config.h"
#include "tsl/platform/rocm_rocdl_path.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace xla {
namespace gpu {

namespace ma = ::mlir::arith;
namespace mm = ::mlir::math;
namespace ml = ::mlir::LLVM;
namespace mt = ::mlir::triton;

using ::llvm::SmallVector;
using mlir::ArrayRef;
using ::mlir::ShapedType;
using ::mlir::Type;
using ::mlir::Value;
using mlir::ValueRange;

absl::Status CreateTritonPipeline(
    mlir::OpPassManager& pm, const se::GpuComputeCapability& cc,
    const TritonGemmConfig& config,
    mt::nvidia_gpu::ClusterInfo& out_cluster_info) {
  // TODO(ROCm): Check whether value different than 0 can be used.
  const int ccAsInt = 0;
  // TODO(ROCm): Check why some test fail when threadsPerWarp is set to 64.
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
      config.num_warps, threadsPerWarp, config.num_ctas, ccAsInt));
  pm.addPass(mt::gpu::createCoalescePass());
  pm.addPass(mt::gpu::createRemoveLayoutConversionsPass());
  pm.addPass(mt::gpu::createOptimizeThreadLocalityPass());
  pm.addPass(mt::gpu::createAccelerateMatmulPass(ccAsInt));
  pm.addPass(mt::gpu::createRemoveLayoutConversionsPass());
  pm.addPass(mt::gpu::createOptimizeDotOperandsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mt::gpu::createPipelinePass(config.num_stages, config.num_warps,
                                         config.num_ctas, ccAsInt));
  pm.addPass(mt::gpu::createPrefetchPass());

  pm.addPass(mt::gpu::createOptimizeDotOperandsPass());
  pm.addPass(mt::gpu::createRemoveLayoutConversionsPass());
  pm.addPass(mt::gpu::createReduceDataDuplicationPass());
  pm.addPass(mt::gpu::createReorderInstructionsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Based on make_llir() in
  // @triton//:third_party/nvidia/backend/compiler.py
  // pm.addPass(mt::gpu::createDecomposeUnsupportedConversionsPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mt::gpu::createAllocateSharedMemoryPass());
  // pm.addPass(mt::createConvertTritonAMDGPUToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // Note: translateTritonGPUToLLVMIR adds line info with LLVMDIScopePass.
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());

  // There is no clusters in ROCm for now.
  out_cluster_info.clusterDimX = 1;
  out_cluster_info.clusterDimY = 1;
  out_cluster_info.clusterDimZ = 1;

  return absl::OkStatus();
}

std::string GetLibdevicePath(const HloModuleConfig& hlo_config,
                             const se::DeviceDescription& device_info) {
  std::string libdevice_dir = tsl::RocdlRoot();
  auto compute_capability = device_info.rocm_compute_capability();
  const std::string libdevice_path =
      amdgpu::LibDevicePath(compute_capability.gcn_arch_name(), libdevice_dir);
  return libdevice_path;
}

}  // namespace gpu
}  // namespace xla
