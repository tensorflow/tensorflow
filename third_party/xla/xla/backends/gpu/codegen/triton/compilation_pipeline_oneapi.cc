/* Copyright 2026 The OpenXLA Authors.

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

#include "absl/strings/str_cat.h"
#include "third_party/intel/include/Dialect/Triton/Transforms/Passes.h"
#include "third_party/intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "third_party/intel/include/TritonAnnotateModule/Passes.h"
#include "third_party/intel/include/TritonGENToLLVM/Passes.h"
#include "third_party/intel/include/TritonIntelGPUToLLVM/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
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

static void MakeTTIR(mlir::OpPassManager* pm) {
  pm->addPass(mlir::createInlinerPass());
  pm->addPass(mt::intel::createTritonRewriteTensorDescriptorToPointer());
  pm->addPass(mlir::createCSEPass());

  mt::intel::TritonIntelReassociateDotScaleOptions options;
  // TODO(intel-tf): Make this configurable. Turned off by default.
  options.fastMath = false;
  pm->addPass(mt::intel::createTritonIntelReassociateDotScale(options));

  pm->addPass(mlir::createLoopInvariantCodeMotionPass());
  pm->addPass(mt::intel::createTritonIntelRemoveMasks());
  pm->addPass(mt::intel::createTritonIntelStrideVersioning());
  pm->addPass(mt::intel::createTritonIntelDescriptorVersioning());

  pm->addPass(mt::intel::createTritonIntelFuseReshape());
  pm->addPass(mt::intel::createTritonIntelGPUFoldTrueCmpI());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mt::createTritonCombineOps());
  pm->addPass(mt::intel::createTritonIntelSimplifySignedArithmetic());
  pm->addPass(mt::createTritonReorderBroadcast());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createSymbolDCEPass());
  pm->addPass(mt::createTritonLoopUnroll());
}

static void MakeTTGIR(mlir::OpPassManager* pm,
                      const stream_executor::OneAPIComputeCapability& oneapi_cc,
                      int num_warps, int num_ctas, int num_stages) {
  mt::gpu::intel::TritonAnnotateModuleOptions options;
  options.minSGSize = 16;
  options.support2DBlockIO = true;
  options.supportBF16Conversion = true;
  options.supportBfloat16Arithmetic = true;
  options.supportPredicatedIO = true;
  options.threadsPerWarp = 32;
  options.targetArch = "spirv64";

  pm->addPass(mt::gpu::intel::createTritonAnnotateModule(options));
  pm->addPass(
      mt::createConvertTritonToTritonGPU({"xpu", num_warps, 32, num_ctas}));
  pm->addPass(mt::gpu::createTritonGPUCoalesce());
  pm->addPass(mt::gpu::intel::createTritonIntelGPURemoveLayoutConversions());

  pm->addPass(mt::gpu::intel::createTritonIntelGPUAccelerateMatmul());
  pm->addPass(mt::gpu::intel::createTritonIntelGPUStageLargeFMADotsViaSLM());
  pm->addPass(mt::gpu::intel::createTritonIntelGPUMaterializeBlockPointer());
  pm->addPass(mt::gpu::intel::createTritonIntelGPURemoveLayoutConversions());
  pm->addPass(mt::gpu::intel::createTritonIntelGPUFoldFpToFp());
  pm->addPass(mt::gpu::intel::createTritonIntelGPUOptimizeDotOperands());
  pm->addPass(mt::gpu::intel::createTritonIntelGPUHoistLayoutConversions());
  mt::gpu::intel::TritonIntelGPUPipelineOptions pipeline_options;
  pipeline_options.numStages = num_stages;
  pipeline_options.useBarrier = true;
  pm->addPass(mt::gpu::intel::createTritonIntelGPUPipeline(pipeline_options));

  pm->addPass(mt::createTritonLoopAwareCSE());
  pm->addPass(mt::gpu::createTritonGPUFuseNestedLoops());

  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createLoopInvariantCodeMotionPass());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mt::gpu::createTritonGPUCombineTensorSelectAndIf());

  pm->addPass(mt::gpu::createTritonGPUOptimizeThreadLocality());
  mt::gpu::TritonGPUOptimizeDotOperandsOptions triton_dot_operands_options;
  triton_dot_operands_options.hoistLayoutConversion = true;
  pm->addPass(
      mt::gpu::createTritonGPUOptimizeDotOperands(triton_dot_operands_options));
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mt::gpu::createTritonGPUPrefetch());
  pm->addPass(
      mt::gpu::createTritonGPUOptimizeDotOperands(triton_dot_operands_options));
  pm->addPass(mt::gpu::intel::createTritonIntelGPURemoveLayoutConversions());
  pm->addPass(mt::gpu::intel::createTritonIntelGPUReduceDataDuplication());

  pm->addPass(mt::gpu::createTritonGPUReorderInstructions());
  pm->addPass(mt::createTritonLoopAwareCSE());
  pm->addPass(mlir::createSymbolDCEPass());
  pm->addPass(mlir::createSCCPPass());
  pm->addPass(mlir::createCanonicalizerPass());

  // TODO(intel-tf): Evaluate whether BF16 F32 emulation is still necessary with
  // current XLA float normalization.
  mlir::arith::ArithEmulateUnsupportedFloatsOptions emulate_options;
  emulate_options.sourceTypeStrs = {"bf16"};
  emulate_options.targetTypeStr = "f32";
  pm->addPass(
      mlir::arith::createArithEmulateUnsupportedFloats(emulate_options));
}

static void MakeLLIR(mlir::OpPassManager* pm,
                     const stream_executor::OneAPIComputeCapability& oneapi_cc,
                     int num_stages) {
  pm->addPass(mt::gpu::intel::createTritonIntelGPULowerTo2DBlockLoad());
  pm->addPass(mlir::createSCFToControlFlowPass());
  pm->addPass(mlir::createInlinerPass());
  pm->addPass(mlir::createConvertIndexToLLVMPass());

  pm->addPass(mt::gpu::intel::createIntelAllocateSharedMemory());
  pm->addPass(mt::gpu::createTritonGPUGlobalScratchAllocationPass());

  pm->addPass(mt::gpu::intel::createConvertTritonIntelGPUToLLVM());
  pm->addPass(mt::createConvertTritonGENToLLVM());

  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mt::gpu::intel::createTritonIntelGPURewriteStackPtr());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createArithToLLVMConversionPass());
  pm->addPass(mlir::createCanonicalizerPass());

  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createSymbolDCEPass());
}

void CreateTritonOneAPIPipeline(
    mlir::OpPassManager* pm,
    const stream_executor::OneAPIComputeCapability& oneapi_cc, int num_warps,
    int num_ctas, int num_stages) {
  MakeTTIR(pm);
  MakeTTGIR(pm, oneapi_cc, num_warps, num_ctas, num_stages);
  MakeLLIR(pm, oneapi_cc, num_stages);
}

}  // namespace gpu
}  // namespace xla
