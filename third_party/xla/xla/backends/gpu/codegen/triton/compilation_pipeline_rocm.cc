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

#include <string>

#include "absl/strings/str_cat.h"
#include "third_party/amd/include/TritonAMDGPUToLLVM/Passes.h"
#include "third_party/amd/include/TritonAMDGPUTransforms/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "xla/stream_executor/device_description.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace xla {
namespace gpu {

namespace mt = ::mlir::triton;

// Based on make_ttir() in
// @triton//:third_party/amd/backend/compiler.py
static void MakeTTIR(mlir::OpPassManager* pm) {
  pm->addPass(mlir::createInlinerPass());
  pm->addPass(mt::createTritonRewriteTensorPointer());
  pm->addPass(mt::createTritonRewriteTensorDescriptorToPointer());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mt::createTritonCombineOps());
  pm->addPass(mt::createTritonReorderBroadcast());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createLoopInvariantCodeMotionPass());
  pm->addPass(mlir::createSymbolDCEPass());
  pm->addPass(mt::createTritonLoopUnroll());
}

static bool is_pingpong_schedule_enabled(
    const stream_executor::RocmComputeCapability& rocm_cc,
    bool use_async_copy) {
  return rocm_cc.gfx9_mi300() || (rocm_cc.gfx9_mi350() && use_async_copy);
}

static bool is_in_thread_transpose_enabled(
    const stream_executor::RocmComputeCapability& rocm_cc) {
  return rocm_cc.gfx9_mi300();
}

// Based on make_ttgir() in
// @triton//:third_party/amd/backend/compiler.py
static void MakeTTGIR(mlir::OpPassManager* pm,
                      const stream_executor::RocmComputeCapability& rocm_cc,
                      int num_warps, int num_ctas, int num_stages) {
  pm->addPass(mt::createConvertTritonToTritonGPU(
      {absl::StrCat("hip:", rocm_cc.gfx_version()), num_warps,
       rocm_cc.threads_per_warp(), num_ctas}));
  pm->addPass(mt::gpu::createTritonGPUCoalesce());
  pm->addPass(mt::gpu::createTritonGPUF32DotTC({false}));
  pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(mt::gpu::createTritonGPUOptimizeThreadLocality());
  pm->addPass(
      mlir::createTritonAMDGPUAccelerateMatmul({rocm_cc.gfx_version()}));
  pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  // TODO ROCm Check if we want to compare MI100 and greater
  pm->addPass(mlir::createTritonAMDGPUOptimizeEpilogue());
  pm->addPass(mt::amdgpu::createTritonAMDGPUOptimizeDotOperands(
      {rocm_cc.gfx_version()}));
  pm->addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUHoistLayoutConversions());

  pm->addPass(mt::gpu::createTritonGPUFuseNestedLoops());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createLoopInvariantCodeMotionPass());
  pm->addPass(mlir::createCanonicalizerPass());

  // TODO(ROCm) Modify when corresponding run time flags are introduced.
  std::string schedule_hint = "none";

  bool use_async_copy = false;  // Not enabled by default.
  bool use_block_pingpong =
      is_pingpong_schedule_enabled(rocm_cc, use_async_copy);

  pm->addPass(mlir::createTritonAMDGPUScheduleLoops({num_stages}));
  pm->addPass(
      mlir::createTritonAMDGPUPipeline({use_async_copy, use_block_pingpong}));
  if (use_async_copy) {
    pm->addPass(
        mlir::createTritonAMDGPUCoalesceAsyncCopy({rocm_cc.gfx_version()}));
  }
  pm->addPass(mlir::createCanonicalizerPass());
  if (schedule_hint != "none") {
    pm->addPass(
        mt::createTritonAMDGPUInsertInstructionSchedHintsPass({schedule_hint}));
  }
  pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(mt::gpu::createTritonGPUReduceDataDuplication());
  if (is_in_thread_transpose_enabled(rocm_cc)) {
    pm->addNestedPass<mlir::triton::FuncOp>(
        mlir::createTritonAMDGPUInThreadTranspose());
    pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  }
  pm->addPass(mlir::createTritonAMDGPUReorderInstructions());
  if (use_block_pingpong && num_stages > 1) {
    pm->addPass(mlir::createTritonAMDGPUBlockPingpong({num_stages}));
  }

  pm->addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUCanonicalizePointers());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createTritonAMDGPUConvertToBufferOps(
      {rocm_cc.gfx_version(), /*allowBufferAtomics*/ true,
       /*analyzeSmallTensorOfst*/ false}));

  pm->addPass(mlir::createTritonAMDFoldTrueCmpI());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createSymbolDCEPass());
}

// Based on make_llir() in
// @triton//:third_party/amd/backend/compiler.py
static void MakeLLIR(mlir::OpPassManager* pm,
                     const stream_executor::RocmComputeCapability& rocm_cc,
                     int num_stages) {
  pm->addPass(mlir::createTritonAMDGPUUpdateAsyncWaitCount());
  pm->addPass(mlir::triton::AMD::createConvertWarpPipelinePass());
  pm->addPass(mlir::createSCFToControlFlowPass());
  pm->addPass(mlir::createConvertIndexToLLVMPass());
  pm->addPass(mt::gpu::createAllocateSharedMemory());
  pm->addPass(
      mt::createConvertTritonAMDGPUToLLVMPass(rocm_cc.gfx_version(), true));
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  // Note: translateTritonGPUToLLVMIR adds line info with LLVMDIScopePass.
  pm->addPass(mlir::createConvertControlFlowToLLVMPass());
  pm->addPass(mlir::createArithToLLVMConversionPass());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createSymbolDCEPass());
  if (/*(instruction_sched_variant=="none") == */ /* DISABLES CODE */ (false)) {
    pm->addPass(mt::createTritonAMDGPULowerInstructionSchedHintsPass(
        rocm_cc.gfx_version(), num_stages));
  }
  pm->addPass(mt::createConvertBuiltinFuncToLLVMPass(/*ftz=*/true));
}

void CreateTritonRocmPipeline(
    mlir::OpPassManager* pm,
    const stream_executor::RocmComputeCapability& rocm_cc, int num_warps,
    int num_ctas, int num_stages) {
  MakeTTIR(pm);
  MakeTTGIR(pm, rocm_cc, num_warps, num_ctas, num_stages);
  MakeLLIR(pm, rocm_cc, num_stages);
}

}  // namespace gpu
}  // namespace xla
