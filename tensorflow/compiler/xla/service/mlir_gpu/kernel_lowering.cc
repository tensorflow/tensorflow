/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/mlir_gpu/kernel_lowering.h"

#include <memory>

#include "absl/memory/memory.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"  // TF:local_config_mlir
#include "mlir/Conversion/LoopsToGPU/LoopsToGPUPass.h"  // TF:local_config_mlir
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"  // TF:local_config_mlir
#include "mlir/Dialect/GPU/GPUDialect.h"  // TF:local_config_mlir
#include "mlir/Dialect/GPU/Passes.h"  // TF:local_config_mlir
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // TF:local_config_mlir
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // TF:local_config_mlir
#include "mlir/Dialect/Linalg/Passes.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/OperationSupport.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassManager.h"  // TF:local_config_mlir
#include "mlir/Transforms/DialectConversion.h"  // TF:local_config_mlir
#include "mlir/Transforms/Passes.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace mlir_gpu {

Status LowerLHLOToGPU(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());

  // Transform element-wise operations to LinAlg.
  pm.addPass(::mlir::xla_lhlo::createLegalizeToLinalgPass());
  // Go from affine to normal loops.
  pm.addPass(::mlir::linalg::createConvertLinalgToLoopsPass());
  // Lower affine to ordinary loops.
  pm.addPass(::mlir::createLowerAffinePass());
  // Move constants out of the loop.
  pm.addPass(::mlir::createLoopInvariantCodeMotionPass());
  // Coalesce generated loops to have 1d loops.
  pm.addPass(::mlir::createLoopCoalescingPass());
  // Transform the now 1d loops to gpu launches.
  pm.addPass(::mlir::createSimpleLoopsToGPUPass(/*numBlockDims=*/0,
                                                /*numThreadDims=*/1));
  // Some basic cleanup.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Take launches to launches with kernels.
  pm.addPass(::mlir::createGpuKernelOutliningPass());

  if (failed(pm.run(module))) {
    return InternalError("Lowering to GPU kernels failed.");
  }
  return Status::OK();
}

Status LowerKernelBodiesToNVVM(mlir::ModuleOp module) {
  // We cannot verify as the signature of the kernel is rewritten.
  ::mlir::PassManager pm(module.getContext(), /*verifyPasses=*/false);

  // Rewrite kernel functions to LLVM IR.
  auto &kernelPm = pm.nest<::mlir::ModuleOp>();
  kernelPm.addPass(::mlir::createLowerGpuOpsToNVVMOpsPass());
  // Some basic cleanup.
  kernelPm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  kernelPm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());

  if (failed(pm.run(module))) {
    return InternalError("Lowering to NVVM IR failed.");
  }
  return Status::OK();
}

StatusOr<mlir::ModuleOp> ExtractKernelModule(mlir::ModuleOp module) {
  auto kernelModule = ::mlir::ModuleOp::create(module.getLoc());
  // TODO(b/137624192): This also needs to resolve naming conflicts.
  module.walk([&kernelModule](mlir::ModuleOp nestedModule) {
    if (nestedModule.getAttrOfType<mlir::UnitAttr>(
            mlir::gpu::GPUDialect::getKernelModuleAttrName())) {
      for (auto& fn : nestedModule) {
        kernelModule.push_back(fn.clone());
      }
    }
  });
  return kernelModule;
}
}  // namespace mlir_gpu
}  // namespace xla
