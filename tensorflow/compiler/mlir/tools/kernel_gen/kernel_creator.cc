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
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"  // from @llvm-project
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"  // from @llvm-project
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"  // from @llvm-project
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"  // from @llvm-project
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"  // from @llvm-project
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"  // from @llvm-project
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"  // from @llvm-project
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/GPU/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/Transforms/OptimizeForNVVM.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/SCF/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Transforms/gpu_passes.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
namespace kernel_gen {
namespace {

using mlir::Value;
using mlir::func::FuncOp;
using mlir::memref::RankOp;
using mlir::scf::ParallelOp;

constexpr llvm::StringRef kGpuBinaryAttrName = "gpu.binary";

/// Check if the size of the allocation is less than the given size. The
/// transformation is only applied to small buffers since large buffers could
/// exceed the stack space.
bool IsSmallAlloc(Value alloc) {
  constexpr unsigned kMaximumSizeInBytes = 64;
  constexpr unsigned kMaxRankOfAllocatedMemRef = 1;

  auto type = alloc.getType().dyn_cast<mlir::ShapedType>();
  if (!type || !alloc.getDefiningOp<mlir::memref::AllocOp>()) return false;
  if (!type.hasStaticShape()) {
    // Check if the dynamic shape dimension of the alloc is produced by RankOp
    // or SelectOp(_, RankOp, RankOp).
    // If this is the case, it is likely to be small. Furthermore, the dimension
    // is limited to the maximum rank of the allocated memref to avoid large
    // values by multiplying several small values.
    if (type.getRank() <= kMaxRankOfAllocatedMemRef) {
      for (Value alloc_arg : alloc.getDefiningOp()->getOperands()) {
        if (auto select = alloc_arg.getDefiningOp<mlir::arith::SelectOp>()) {
          if (!select.getTrueValue().getDefiningOp<RankOp>() ||
              !select.getFalseValue().getDefiningOp<RankOp>())
            return false;
        } else if (!alloc_arg.getDefiningOp<RankOp>()) {
          return false;
        }
      }
      return true;
    }
    return false;
  }
  unsigned bitwidth = mlir::DataLayout::closest(alloc.getDefiningOp())
                          .getTypeSizeInBits(type.getElementType());
  return type.getNumElements() * bitwidth <= kMaximumSizeInBytes * 8;
}

Status LowerTFToJITInvocation(mlir::ModuleOp module,
                              llvm::ArrayRef<int64_t> tile_sizes,
                              llvm::ArrayRef<int64_t> unroll_factors,
                              int64_t max_supported_rank, bool enable_ftz,
                              bool index_64bit, bool cpu_codegen,
                              bool jit_i64_indexed_for_large_tensors,
                              bool apply_cl_options) {
  mlir::PassManager pm(module.getContext());
  if (apply_cl_options) applyTensorflowAndCLOptions(pm);

  pm.addNestedPass<FuncOp>(
      mlir::kernel_gen::transforms::CreateTFToJITInvocationPass(
          tile_sizes, unroll_factors, max_supported_rank, enable_ftz,
          index_64bit, cpu_codegen, jit_i64_indexed_for_large_tensors));
  pm.addPass(mlir::kernel_gen::tf_framework::CreateEmbedTFFrameworkPass());
  pm.addNestedPass<FuncOp>(mlir::createLinalgInitTensorToAllocTensorPass());
  pm.addPass(mlir::createComputeOpAndFuncBufferizePass());

  pm.addPass(mlir::createFinalBufferizePass(
      /*alignment=*/64,
      mlir::kernel_gen::transforms::populateExtraBufferizeDialects,
      mlir::kernel_gen::transforms::populateExtraBufferizePatterns));

  if (failed(pm.run(module))) {
    return tensorflow::errors::Internal(
        "Lowering TF to JIT invocation failed.");
  }
  return OkStatus();
}

Status LowerTFtoLoops(mlir::ModuleOp module, llvm::ArrayRef<int64_t> tile_sizes,
                      llvm::ArrayRef<int64_t> unroll_factors,
                      int64_t max_supported_rank, bool enable_ftz,
                      bool index_64bit, bool cpu_codegen,
                      bool jit_i64_indexed_for_large_tensors,
                      bool apply_cl_options) {
  mlir::PassManager pm(module.getContext());
  if (apply_cl_options) applyTensorflowAndCLOptions(pm);
  if (jit_i64_indexed_for_large_tensors) {
    pm.addNestedPass<FuncOp>(
        mlir::kernel_gen::transforms::CreateTFToJITInvocationPass(
            tile_sizes, unroll_factors, max_supported_rank, enable_ftz,
            index_64bit, cpu_codegen,
            /*jit_i64_indexed_for_large_tensors=*/true));
  }
  pm.addNestedPass<FuncOp>(mlir::mhlo::createLegalizeTFNoFallbackPass(
      /*allow_partial_conversion=*/true));
  pm.addNestedPass<FuncOp>(mlir::mhlo::createRankSpecializationClusterPass());
  pm.addNestedPass<FuncOp>(
      mlir::mhlo::createRankSpecializationToSCFPass(max_supported_rank));
  pm.addNestedPass<FuncOp>(mlir::mhlo::createChloLegalizeToHloPass());

  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(mlir::createCSEPass());
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(mlir::createShapeSimplification());
  pm.addNestedPass<FuncOp>(mlir::mhlo::createMergeAssumingOpsPass());
  pm.addNestedPass<FuncOp>(mlir::mhlo::createBroadcastPropagationPass());
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(mlir::createCSEPass());

  // Transform HLO operations to LinAlg and standard.
  pm.addNestedPass<FuncOp>(::mlir::mhlo::createLegalizeHloToLinalgPass());
  pm.addPass(::mlir::mhlo::createLegalizeToArithmeticPass());
  pm.addNestedPass<FuncOp>(
      mlir::mhlo::createLegalizeHloShapeOpsToStandardPass());

  // Remove the remaining references to unsigned types after all HLO compute
  // operations were converted.
  pm.addPass(mlir::mhlo::createConvertToSignlessPass());

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(mlir::createCSEPass());

  // Convert operations from the Complex dialect to the Standard/Math dialects.
  pm.addNestedPass<FuncOp>(::mlir::createConvertComplexToStandardPass());

  // Fuse linalg operations.
  pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(mlir::createLinalgElementwiseOpFusionPass());

  // Partial bufferization: Transforms inparticular HLO and Linalg operations to
  // their corresponding LHLO operations and converts the function signature.
  // Leaves shape operations untouched.
  //
  // TODO(pifon): Rename the pass to CreateHloLinalgBufferizePass or bufferize
  // in 2 steps: first Linalg, then Hlo. That would need refactoring of
  // BufferizeTypeConverter.
  pm.addNestedPass<FuncOp>(mlir::createLinalgInitTensorToAllocTensorPass());
  pm.addPass(mlir::createComputeOpAndFuncBufferizePass());
  pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(::mlir::createCSEPass());
  // Remove copies which are introduced by canonicalizing
  // BufferCastOp(TensorLoadOp).
  pm.addNestedPass<FuncOp>(
      mlir::kernel_gen::transforms::CreateCopyCleanupPass());
  // Find candidates for buffer reuse. This is only successful if buffer size
  // equality can be determined based on `linalg.generic` operations.
  pm.addNestedPass<FuncOp>(
      mlir::kernel_gen::transforms::CreateBufferReusePass());
  // Approximate Tanh using standard operations.
  pm.addNestedPass<FuncOp>(
      ::mlir::mhlo::createLegalizeTrigonometricToApproximationPass());
  if (cpu_codegen) {
    pm.addNestedPass<FuncOp>(
        mlir::kernel_gen::transforms::CreateVectorizationPass());
    pm.addNestedPass<FuncOp>(
        mlir::bufferization::createBufferLoopHoistingPass());
    pm.addNestedPass<FuncOp>(mlir::createShapeSimplification());
    pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(::mlir::createCSEPass());
    pm.addNestedPass<FuncOp>(
        mlir::kernel_gen::transforms::CreateVectorizationCleanupPass());
    pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
  }
  // Transform the Linalg ops inside of the loop nest into parallel loops.
  pm.addNestedPass<FuncOp>(::mlir::createConvertLinalgToParallelLoopsPass());

  if (!cpu_codegen) {
    // Canonicalize the code to simplify index computations. This is needed so
    // that loop bounds have the same value.
    pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
    // Run CSE to ensure that loads and stores to the same subview get
    // recognized as such.
    pm.addNestedPass<FuncOp>(::mlir::createCSEPass());
    // Collapse and tile parallel loops for GPU only. Collapsing shouldn't
    // provide benefits to CPU and tiling is handled by vectorization.
    pm.addNestedPass<FuncOp>(mlir::createCollapseParallelLoopsTo1DPass());
    pm.addNestedPass<FuncOp>(
        mlir::createTileLoopsPass(tile_sizes, unroll_factors));
  }

  pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(::mlir::createCSEPass());
  if (failed(pm.run(module))) {
    return tensorflow::errors::Internal("Lowering TF to loops failed.");
  }
  return OkStatus();
}

Status LowerLoopsToGPUorCPU(mlir::ModuleOp module, bool embed_memref_prints,
                            bool cpu_codegen, bool index_64bit,
                            bool apply_cl_options) {
  mlir::PassManager pm(module.getContext());
  if (apply_cl_options) applyTensorflowAndCLOptions(pm);

  if (!cpu_codegen) {
    // Greedily map the remaining loop to GPU hardware dimensions.
    pm.addNestedPass<FuncOp>(mlir::createGpuMapParallelLoopsPass());
  }

  // Expand memref_reshape to its ranked form so that we can propagate
  // scalars and avoid allocation.
  pm.addNestedPass<FuncOp>(mlir::arith::createArithmeticExpandOpsPass());
  pm.addNestedPass<FuncOp>(mlir::memref::createExpandOpsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::kernel_gen::transforms::CreateShapeToDescriptorsPass());
  // Before bufferizing further, remove unused tensor_to_memref, so that we do
  // not create allocations for tensor computations that are not actually
  // needed.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(mlir::createCSEPass());
  // Before inserting more allocs, map the ones we already have to the
  // tf runtime. That ensures that all allocations for the actual computation
  // end up on the device, whereas allocations for shape computation and host
  // side things remain on the host.
  // Longer term, this should be handled by proper device placement.
  pm.addPass(mlir::kernel_gen::tf_framework::CreateEmbedTFFrameworkPass());
  // Now lower the shape computations, bufferize all remaining ops and insert
  // deallocs.
  pm.addPass(mlir::createFinalBufferizePass(
      /*alignment=*/64,
      mlir::kernel_gen::transforms::populateExtraBufferizeDialects,
      mlir::kernel_gen::transforms::populateExtraBufferizePatterns));
  // TODO(herhut): Enable once no-longer broken.
  pm.addNestedPass<FuncOp>(::mlir::bufferization::createBufferHoistingPass());
  pm.addNestedPass<FuncOp>(mlir::bufferization::createPromoteBuffersToStackPass(
      [](Value alloc) { return IsSmallAlloc(alloc); }));
  // Free all temporaries,
  pm.addNestedPass<FuncOp>(
      ::mlir::bufferization::createBufferDeallocationPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Apply the mapping and go to GPU. We cannot do this earlier due to missing
  // interfaces on the GPU dialect.
  // TODO(b/174830459): Move up once implemented.
  if (!cpu_codegen) {
    pm.addNestedPass<FuncOp>(mlir::createParallelLoopToGpuPass());
  }

  // Some basic cleanup.
  pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(::mlir::createCSEPass());
  // Make loops with min bounds into a conditional plus static bounds.
  pm.addNestedPass<FuncOp>(mlir::createForLoopSpecializationPass());
  // Take launches to launches with kernels.
  if (!cpu_codegen) {
    pm.addPass(mlir::createGpuLauchSinkIndexComputationsPass());
    const std::string gpuDataLayoutSpec =
        index_64bit ? "#dlti.dl_spec<#dlti.dl_entry<index,64:i64>>"
                    : "#dlti.dl_spec<#dlti.dl_entry<index,32:i32>>";
    pm.addPass(mlir::createGpuKernelOutliningPass(gpuDataLayoutSpec));
  }

  pm.addPass(::mlir::createLowerAffinePass());
  // Constraints are removed as late as possible and before lowering to CFG.
  pm.addNestedPass<FuncOp>(::mlir::createConvertShapeConstraintsPass());
  pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
  pm.addPass(::mlir::createConvertSCFToCFPass());
  if (cpu_codegen) pm.addPass(::mlir::createConvertVectorToLLVMPass());
  // Map asserts to the tensorflow framework.
  pm.addPass(mlir::kernel_gen::tf_framework::CreateRewriteTFFrameworkAssert());
  if (embed_memref_prints) {
    pm.addPass(mlir::kernel_gen::transforms::CreateEmbedMemRefPrintsPass());
  }
  if (failed(pm.run(module))) {
    return tensorflow::errors::Internal("Lowering to GPU kernels failed.");
  }
  return OkStatus();
}

Status LowerKernelBodiesToLowLevelIr(mlir::ModuleOp module,
                                     bool apply_cl_options) {
#if !defined(TENSORFLOW_USE_ROCM) && !defined(GOOGLE_CUDA)
  return tensorflow::errors::Internal(
      "Neither TENSORFLOW_USE_ROCM nor GOOGLE_CUDA are defined."
      " Did you specify either --config=rocm or --config=cuda ?");
#endif

#if TENSORFLOW_USE_ROCM
  auto gpu_modules = module.getOps<::mlir::gpu::GPUModuleOp>();
  for (::mlir::gpu::GPUModuleOp gpu_module : gpu_modules) {
    gpu_module.walk([&](mlir::gpu::GPUFuncOp gpu_kernel) {
      if (gpu_kernel.isKernel()) {
        gpu_kernel->setAttr(
            "rocdl.max_flat_work_group_size",
            mlir::IntegerAttr::get(
                mlir::IntegerType::get(module.getContext(), 32), 1024));
      }
    });
  }
#endif

  mlir::PassManager pm(module.getContext());
  // We cannot verify as the signature of the kernel is rewritten.
  // pm.enableVerifier(false);
  if (apply_cl_options) tensorflow::applyTensorflowAndCLOptions(pm);
  auto& kernelPm = pm.nest<::mlir::gpu::GPUModuleOp>();
  kernelPm.addPass(::mlir::createConvertSCFToCFPass());
#if TENSORFLOW_USE_ROCM
  kernelPm.addPass(mlir::createGpuKernelToRocdlPass());
#elif GOOGLE_CUDA
  kernelPm.addPass(mlir::createGpuKernelToNvvmPass());
  kernelPm.addPass(mlir::NVVM::createOptimizeForTargetPass());
#endif
  // Remove all location information to prevent a debug build.
  pm.addPass(::mlir::createStripDebugInfoPass());

  if (failed(pm.run(module))) {
    return tensorflow::errors::Internal(
        "Lowering to low-level device IR failed.");
  }

  return OkStatus();
}

Status AmendKernelLLVMIRWithStaticKnowledge(mlir::ModuleOp module,
                                            bool apply_cl_options) {
  mlir::PassManager pm(module.getContext());
  if (apply_cl_options) applyTensorflowAndCLOptions(pm);

  pm.addNestedPass<FuncOp>(
      mlir::kernel_gen::transforms::CreatePropagateShapeKnowledgeToKernels());
  pm.addNestedPass<FuncOp>(
      mlir::kernel_gen::transforms::CreatePropagateTfAbiKnowledgeToKernels());

  return failed(pm.run(module))
             ? tensorflow::errors::Internal(
                   "Amending LLVMIR with static knowledge failed.")
             : OkStatus();
}

Status GenerateDeviceCode(mlir::ModuleOp module,
                          llvm::StringRef gpu_binary_attr_name,
                          llvm::ArrayRef<std::string> architectures,
                          bool print_ptx, bool print_llvmir, bool enable_ftz,
                          bool apply_cl_options) {
  mlir::PassManager pm(module.getContext());
  if (apply_cl_options) applyTensorflowAndCLOptions(pm);
  mlir::registerLLVMDialectTranslation(*module->getContext());

  auto& kernel_pm = pm.nest<mlir::gpu::GPUModuleOp>();
  // Remove debug information to ensure we do not create debug PTX.
  kernel_pm.addPass(mlir::createStripDebugInfoPass());
  kernel_pm.addPass(mlir::kernel_gen::transforms::CreateGpuKernelToBlobPass(
      gpu_binary_attr_name, architectures, print_ptx, print_llvmir,
      enable_ftz));

  return failed(pm.run(module))
             ? tensorflow::errors::Internal("Generating device code failed.")
             : OkStatus();
}

Status LowerHostSideToFinalForm(mlir::ModuleOp module, bool apply_cl_options) {
  mlir::PassManager pm(module.getContext());
  if (apply_cl_options) applyTensorflowAndCLOptions(pm);

  pm.addPass(mlir::kernel_gen::transforms::CreateTFKernelToLLVMPass(
      kGpuBinaryAttrName));
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  return failed(pm.run(module)) ? tensorflow::errors::Internal(
                                      "Final lowering of host side failed.")
                                : OkStatus();
}

}  // namespace

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> SetupContextAndParseModule(
    mlir::MLIRContext& context, llvm::StringRef tf_code) {
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  registry.insert<mlir::chlo::ChloDialect, mlir::mhlo::MhloDialect>();
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);
  context.appendDialectRegistry(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(tf_code, &context);
  if (!module)
    return tensorflow::Status(tensorflow::error::Code::INVALID_ARGUMENT,
                              "invalid kernel IR");
  return module;
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GenerateKernelForTfCode(
    mlir::MLIRContext& context, llvm::StringRef tf_code,
    llvm::ArrayRef<std::string> architectures,
    llvm::ArrayRef<int64_t> tile_sizes, llvm::ArrayRef<int64_t> unroll_factors,
    int64_t max_supported_rank, bool embed_memref_prints, bool print_ptx,
    bool print_llvmir, bool enable_ftz, bool index_64bit, bool cpu_codegen,
    bool jit_compile, bool jit_i64_indexed_for_large_tensors,
    bool apply_cl_options) {
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      SetupContextAndParseModule(context, tf_code));

  if (jit_compile) {
    TF_RETURN_IF_ERROR(LowerTFToJITInvocation(
        module.get(), tile_sizes, unroll_factors, max_supported_rank,
        enable_ftz, index_64bit, cpu_codegen,
        /*jit_i64_indexed_for_large_tensors=*/false, apply_cl_options));
  } else {
    TF_RETURN_IF_ERROR(
        LowerTFtoLoops(module.get(), tile_sizes, unroll_factors,
                       max_supported_rank, enable_ftz, index_64bit, cpu_codegen,
                       jit_i64_indexed_for_large_tensors, apply_cl_options));
    TF_RETURN_IF_ERROR(LowerLoopsToGPUorCPU(module.get(), embed_memref_prints,
                                            cpu_codegen, index_64bit,
                                            apply_cl_options));
    if (!cpu_codegen) {
      TF_RETURN_IF_ERROR(
          LowerKernelBodiesToLowLevelIr(module.get(), apply_cl_options));
      TF_RETURN_IF_ERROR(
          AmendKernelLLVMIRWithStaticKnowledge(module.get(), apply_cl_options));
      TF_RETURN_IF_ERROR(GenerateDeviceCode(
          module.get(), kGpuBinaryAttrName, architectures, print_ptx,
          print_llvmir, enable_ftz, apply_cl_options));
    }
  }

  TF_RETURN_IF_ERROR(LowerHostSideToFinalForm(module.get(), apply_cl_options));

  return module;
}

}  // namespace kernel_gen
}  // namespace tensorflow
