/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

//===- kernel_creator.cc ----------------------------------------*- C++ -*-===//
//
// This file implements the function to compile a TF kernel function to gpu
// binary (hsaco for AMD, cubin for NVIDIA) or to a gpu binary with host side.
//
//===----------------------------------------------------------------------===//
#include "tensorflow/compiler/mlir/tools/kernel_gen/kernel_creator.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"  // from @llvm-project
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"  // from @llvm-project
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"  // from @llvm-project
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"  // from @llvm-project
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"  // from @llvm-project
#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/GPU/ParallelLoopMapper.h"  // from @llvm-project
#include "mlir/Dialect/GPU/Passes.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Passes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/Passes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/SCF/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/BufferPlacement.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/kernel_lowering.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/passes.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"

namespace tensorflow {
namespace kernel_gen {
namespace {

using tensorflow::Status;
using xla::InternalError;
using xla::StatusOr;

constexpr llvm::StringRef kGpuBinaryAttrName = "gpu.binary";

Status LowerTFtoGPU(mlir::ModuleOp module, bool gpu_binary_only,
                    llvm::ArrayRef<uint32_t> tile_sizes,
                    llvm::ArrayRef<uint32_t> unroll_factors) {
  mlir::PassManager pm(module.getContext());
  applyPassManagerCLOptions(pm);

  pm.addPass(mlir::mhlo::createLegalizeTFPass(false));
  if (gpu_binary_only) {
    pm.addNestedPass<mlir::FuncOp>(
        mlir::kernel_gen::transforms::CreateMaterializeBroadcastsPass());
    pm.addNestedPass<mlir::FuncOp>(
        mlir::kernel_gen::transforms::CreateUnfuseBatchNormPass());
    pm.addPass(mlir::mhlo::createLegalizeToLhloPass(
        /*results_escape_functions=*/true));
    // Moving `AllocOp`s and inserting missing `DeallocOp`s
    pm.addPass(::mlir::createBufferPlacementPass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createCopyRemovalPass());
    pm.addPass(mlir::kernel_gen::transforms::CreateShapeToDescriptorsPass());
  } else {
    pm.addPass(mlir::createTransformUnrankedHloPass());
    pm.addPass(mlir::kernel_gen::transforms::CreateShapeToDescriptorsPass());
    pm.addPass(mlir::kernel_gen::transforms::CreateBufferizePass());
  }

  // Clean up the IR for further processing.
  pm.addPass(mlir::createCanonicalizerPass());
  // We have to anticipate later unrolling in tiling to make sure that we get
  // the requested tiling after unrolling. Compute the new tiling here if
  // needed.
  llvm::SmallVector<unsigned, 4> tiling_for_unrolling;
  llvm::SmallVector<int64_t, 4> as_int64;
  if (!unroll_factors.empty()) {
    tiling_for_unrolling.reserve(tile_sizes.size());
    for (auto pair : llvm::zip(tile_sizes, unroll_factors)) {
      tiling_for_unrolling.push_back(std::get<0>(pair) * std::get<1>(pair));
      as_int64.push_back(std::get<1>(pair));
    }
  } else {
    tiling_for_unrolling.append(tile_sizes.begin(), tile_sizes.end());
  }
  // Transform LHLO operations to LinAlg.
  pm.addPass(::mlir::lmhlo::createLegalizeLhloToLinalgPass());
  // Fuse linalg operations.
  pm.addPass(::mlir::lmhlo::createLhloFuseLinalgPass(
      /*use_parallel_loops=*/true, tiling_for_unrolling));
  // Transform the Linalg operations inside of the loop nest into parallel
  // loops.
  pm.addPass(::mlir::createConvertLinalgToParallelLoopsPass());
  // Canonicalize the code to simplify index computations. This is needed so
  // that loop bounds have the same value.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Fuse the inner-most loops.
  pm.addPass(xla::mlir_gpu::createFuseInnerParallelLoopsPass());
  // Run CSE to ensure that loads and stores to the same subview get
  // recognized as such.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Forward stores to buffers to loads.
  pm.addPass(xla::mlir_gpu::createStoreForwardingPass());
  // Remove now unused temporary buffers.
  pm.addPass(xla::mlir_gpu::createDeadTempBufferRemovalPass());
  if (!unroll_factors.empty()) {
    pm.addPass(::mlir::createParallelLoopTilingPass(as_int64));
  }
  // Some basic cleanup.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Greedily map the remaining loop to GPU hardware dimensions.
  pm.addPass(xla::mlir_gpu::createMapParallelLoopsPass());
  // Apply the mapping.
  pm.addPass(mlir::createParallelLoopToGpuPass());

  // Embed TF Framework ops.
  if (!gpu_binary_only) {
    pm.addPass(mlir::kernel_gen::tf_framework::CreateEmbedTFFrameworkPass());
  }

  // Some basic cleanup.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Make loops with min bounds into a conditional plus static bounds.
  // Only do this if we unrolled in the first place.
  if (!unroll_factors.empty()) {
    pm.addNestedPass<::mlir::FuncOp>(mlir::createForLoopSpecializationPass());
  }
  // Approximate Tanh using standard operations.
  pm.addNestedPass<::mlir::FuncOp>(
      ::mlir::mhlo::createLegalizeTanhToApproximationPass());
  // Move scalar operations into the launch to ensure smaller signatures.
  pm.addPass(xla::mlir_gpu::createMoveScalarComputationsIntoGpuLaunchPass());
  // Take launches to launches with kernels.
  pm.addPass(::mlir::createGpuKernelOutliningPass());

  if (gpu_binary_only) {
    // Make kernel signature deterministic so that we can call it externally.
    pm.addPass(xla::mlir_gpu::createRewriteKernelSignaturePass());
  }
  pm.addPass(::mlir::createLowerAffinePass());
  pm.addPass(::mlir::createLowerToCFGPass());
  if (failed(pm.run(module))) {
    return InternalError("Lowering to GPU kernels failed.");
  }
  return Status::OK();
}

Status LowerGPUToLLVM(mlir::ModuleOp module, bool gpu_binary_only,
                      llvm::ArrayRef<uint32_t> same_shape,
                      llvm::StringRef gpu_binary_attr_name,
                      int32_t architecture) {
  mlir::PassManager pm(module.getContext());
  applyPassManagerCLOptions(pm);

  auto& kernel_pm = pm.nest<mlir::gpu::GPUModuleOp>();
  if (gpu_binary_only) {
    // Grab the original signature from the single function.
    auto func = *module.getBody()->op_begin<mlir::FuncOp>();
    kernel_pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(
        mlir::kernel_gen::transforms::CreatePropagateTensorFlowABIKnowledgePass(
            func.getType(), same_shape));
  }
  kernel_pm.addPass(mlir::createStripDebugInfoPass());
  kernel_pm.addPass(mlir::kernel_gen::transforms::CreateGpuKernelToBlobPass(
      gpu_binary_attr_name, architecture));

  if (!gpu_binary_only) {
    pm.addPass(mlir::kernel_gen::transforms::CreateTFKernelToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }
  return failed(pm.run(module)) ? InternalError("Lowering to LLVM IR failed.")
                                : Status::OK();
}

}  // namespace

StatusOr<mlir::OwningModuleRef> GenerateKernelForTfCode(
    mlir::MLIRContext& context, llvm::StringRef tf_code, bool gpu_binary_only,
    int32_t architecture, llvm::ArrayRef<uint32_t> tile_sizes,
    llvm::ArrayRef<uint32_t> same_shape,
    llvm::ArrayRef<uint32_t> unroll_factors) {
  mlir::RegisterAllTensorFlowDialects(context.getDialectRegistry());
  mlir::OwningModuleRef module = mlir::parseSourceString(tf_code, &context);
  TF_RETURN_IF_ERROR(
      LowerTFtoGPU(module.get(), gpu_binary_only, tile_sizes, unroll_factors));
#if !defined(TENSORFLOW_USE_ROCM) && !defined(GOOGLE_CUDA)
  return InternalError(
      "Neither TENSORFLOW_USE_ROCM nor GOOGLE_CUDA are defined."
      " Did you specify either --config=rocm or --config=cuda ?");
#endif

#if TENSORFLOW_USE_ROCM
  TF_RETURN_IF_ERROR(xla::mlir_gpu::LowerKernelBodiesToROCDL(module.get()));
#elif GOOGLE_CUDA
  TF_RETURN_IF_ERROR(xla::mlir_gpu::LowerKernelBodiesToNVVM(module.get()));
#endif
  TF_RETURN_IF_ERROR(LowerGPUToLLVM(module.get(), gpu_binary_only, same_shape,
                                    kGpuBinaryAttrName, architecture));
  return module;
}

StatusOr<std::string> ExtractGpuBinary(mlir::ModuleOp module) {
  auto gpu_modules = module.getOps<mlir::gpu::GPUModuleOp>();
  if (std::distance(gpu_modules.begin(), gpu_modules.end()) != 1) {
    return InternalError("There should be exactly one GPU Module");
  }
  mlir::gpu::GPUModuleOp gpu_mod = *gpu_modules.begin();
  auto blob = gpu_mod.getAttrOfType<mlir::StringAttr>(kGpuBinaryAttrName);
  if (blob == nullptr) {
    return InternalError("No binary blob found in the module");
  }
  return blob.getValue().str();
}

}  // namespace kernel_gen
}  // namespace tensorflow
