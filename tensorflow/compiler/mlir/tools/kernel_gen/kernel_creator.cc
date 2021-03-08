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
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"  // from @llvm-project
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"  // from @llvm-project
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"  // from @llvm-project
#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/GPU/ParallelLoopMapper.h"  // from @llvm-project
#include "mlir/Dialect/GPU/Passes.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/SCF/Passes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/SCF/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/SCF/Utils.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Shape/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Transforms/Bufferize.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/LoopUtils.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"

namespace tensorflow {
namespace kernel_gen {
namespace {

using mlir::Value;
using mlir::scf::ParallelOp;
using tensorflow::Status;
using xla::InternalError;
using xla::StatusOr;

constexpr llvm::StringRef kGpuBinaryAttrName = "gpu.binary";

/// Check if the size of the allocation is less than the given size. The
/// transformation is only applied to small buffers since large buffers could
/// exceed the stack space.
bool IsSmallAlloc(Value alloc) {
  constexpr unsigned kMaximumSizeInBytes = 64;
  constexpr unsigned kBitwidthOfIndexType = 64;
  constexpr unsigned kMaxRankOfAllocatedMemRef = 1;

  auto type = alloc.getType().dyn_cast<mlir::ShapedType>();
  if (!type || !alloc.getDefiningOp<mlir::AllocOp>()) return false;
  if (!type.hasStaticShape()) {
    // Check if the dynamic shape dimension of the alloc is produced by RankOp
    // or SelectOp(_, RankOp, RankOp).
    // If this is the case, it is likely to be small. Furthermore, the dimension
    // is limited to the maximum rank of the allocated memref to avoid large
    // values by multiplying several small values.
    if (type.getRank() <= kMaxRankOfAllocatedMemRef) {
      for (Value alloc_arg : alloc.getDefiningOp()->getOperands()) {
        if (auto select = alloc_arg.getDefiningOp<mlir::SelectOp>()) {
          if (!select.true_value().getDefiningOp<mlir::RankOp>() ||
              !select.false_value().getDefiningOp<mlir::RankOp>())
            return false;
        } else if (!alloc_arg.getDefiningOp<mlir::RankOp>()) {
          return false;
        }
      }
      return true;
    }
    return false;
  }
  // For index types, use the provided size, as the type does not know.
  unsigned int bitwidth = type.getElementType().isIndex()
                              ? kBitwidthOfIndexType
                              : type.getElementTypeBitWidth();
  return type.getNumElements() * bitwidth <= kMaximumSizeInBytes * 8;
}

// TODO(herhut): Remove this once leftover tensor_to_memref are handled in core.
struct RemoveUnusedTensorToMemrefOperations
    : public mlir::PassWrapper<RemoveUnusedTensorToMemrefOperations,
                               mlir::FunctionPass> {
  void runOnFunction() override {
    getFunction().walk([](mlir::TensorToMemrefOp op) {
      // Drop all tensor_to_memref that have no more users. Currently this will
      // not happen, as tensor_to_memref has a side-effect. See
      // https://reviews.llvm.org/D91967 for a dicsussion.
      if (op.memref().getUsers().empty()) {
        op.erase();
      }
    });
  }
};

struct CollapseParallelLoopsTo1D
    : public mlir::PassWrapper<CollapseParallelLoopsTo1D, mlir::FunctionPass> {
  void runOnFunction() override {
    getFunction().walk([&](ParallelOp op) {
      unsigned num_loops = op.getNumLoops();
      if (num_loops == 1) return;
      std::vector<unsigned> combinedLoops;
      combinedLoops.reserve(num_loops);
      for (unsigned i = 0; i < num_loops; ++i) {
        combinedLoops.push_back(i);
      }
      mlir::collapseParallelLoops(op, {combinedLoops});
    });
  }
};

class TileLoops : public mlir::PassWrapper<TileLoops, mlir::FunctionPass> {
 public:
  explicit TileLoops(llvm::ArrayRef<int64_t> tile_sizes,
                     llvm::ArrayRef<int64_t> unroll_factors) {
    tile_sizes_ = llvm::to_vector<4>(tile_sizes);
    outer_tile_ = tile_sizes_;

    // We have to anticipate later unrolling in tiling to make sure that we get
    // the requested tiling after unrolling.
    if (unroll_factors.size() == tile_sizes.size()) {
      inner_tile_ = llvm::to_vector<4>(unroll_factors);
      for (auto en : llvm::enumerate(unroll_factors)) {
        outer_tile_[en.index()] *= en.value();
      }
    }
  }

  void runOnFunction() override {
    llvm::SmallVector<ParallelOp, 2> innermostPloops;
    mlir::getInnermostParallelLoops(this->getFunction().getOperation(),
                                    innermostPloops);
    for (ParallelOp ploop : innermostPloops) {
      // Support unrolling only for the simple shapes (same shapes or when one
      // of the arguments is a constant), i.e. it's not inside `shape.assuming`.
      if (ploop->getParentOfType<mlir::shape::AssumingOp>() != nullptr) {
        tileParallelLoop(ploop, tile_sizes_);
        continue;
      }
      auto tiled_loops = tileParallelLoop(ploop, outer_tile_);
      // Tile twice if the inner_tile is non-empty.
      if (!inner_tile_.empty()) {
        tileParallelLoop(tiled_loops.second, inner_tile_);
      }
    }
  }

 private:
  // Outer tile size = unroll_factor.empty() ? tile_sizes : tile_sizes *
  // unroll_factors.
  llvm::SmallVector<int64_t, 4> outer_tile_;
  // Inner tile size if the unrolling factors were specified.
  llvm::SmallVector<int64_t, 4> inner_tile_;
  // Original tile sizes.
  llvm::SmallVector<int64_t, 4> tile_sizes_;
};

Status LowerTFtoLoops(mlir::ModuleOp module, llvm::ArrayRef<int64_t> tile_sizes,
                      llvm::ArrayRef<int64_t> unroll_factors,
                      bool cpu_codegen) {
  mlir::PassManager pm(module.getContext());
  applyTensorflowAndCLOptions(pm);

  pm.addNestedPass<mlir::FuncOp>(mlir::mhlo::createLegalizeTFPass(
      /*allow_partial_conversion=*/false, /*legalize_chlo=*/false));
  pm.addNestedPass<mlir::FuncOp>(mlir::mhlo::createTransformUnrankedHloPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::mhlo::createChloLegalizeToHloPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::mhlo::createLowerComplexPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());

  // Transform HLO operations to LinAlg.
  pm.addNestedPass<mlir::FuncOp>(::mlir::mhlo::createLegalizeHloToLinalgPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
  // Fuse linalg operations.
  pm.addNestedPass<mlir::FuncOp>(mlir::createLinalgFusionOfTensorOpsPass());

  // Partial bufferization: Transforms inparticular HLO and Linalg operations to
  // their corresponding LHLO operations and converts the function signature.
  // Leaves shape operations untouched.
  //
  // TODO(pifon): Rename the pass to CreateHloLinalgBufferizePass or bufferize
  // in 2 steps: first Linalg, then Hlo. That would need refactoring of
  // BufferizeTypeConverter.
  pm.addPass(mlir::kernel_gen::transforms::CreateHloBufferizePass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Find candidates for buffer reuse. This is only successful if buffer size
  // equality can be determined based on `linalg.generic` operations.
  pm.addNestedPass<mlir::FuncOp>(
      mlir::kernel_gen::transforms::CreateBufferReusePass());
  // Transform the Linalg ops inside of the loop nest into parallel loops.
  pm.addNestedPass<mlir::FuncOp>(
      ::mlir::createConvertLinalgToParallelLoopsPass());
  // Canonicalize the code to simplify index computations. This is needed so
  // that loop bounds have the same value.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Run CSE to ensure that loads and stores to the same subview get
  // recognized as such.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());

  if (!cpu_codegen) {
    // Collapse and tile parallel loops. Collapsing shouldn't provide benefits
    // to CPU and tiling is handled by vectorization.
    pm.addNestedPass<mlir::FuncOp>(
        std::make_unique<CollapseParallelLoopsTo1D>());
    pm.addNestedPass<mlir::FuncOp>(
        std::make_unique<TileLoops>(tile_sizes, unroll_factors));
  }
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  if (failed(pm.run(module))) {
    return InternalError("Lowering TF to loops failed.");
  }
  return Status::OK();
}

Status LowerLoopsToGPUorCPU(mlir::ModuleOp module, bool embed_memref_prints,
                            bool cpu_codegen) {
  mlir::PassManager pm(module.getContext());
  applyTensorflowAndCLOptions(pm);

  if (!cpu_codegen) {
    // Greedily map the remaining loop to GPU hardware dimensions.
    pm.addNestedPass<::mlir::FuncOp>(
        mlir::kernel_gen::transforms::CreateMapParallelLoopsPass());
  }

  // Expand memref_reshape to its ranked form so that we can propagate
  // scalars and avoid allocation.
  pm.addNestedPass<mlir::FuncOp>(mlir::createStdExpandOpsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::kernel_gen::transforms::CreateShapeToDescriptorsPass());
  // Before bufferizing further, remove unused tensor_to_memref, so that we do
  // not create allocations for tensor computations that are not actually
  // needed.
  pm.addPass(mlir::createCanonicalizerPass());
  // TODO(herhut) Remove once handled in mlir core.
  pm.addNestedPass<mlir::FuncOp>(
      std::make_unique<RemoveUnusedTensorToMemrefOperations>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
  // Before inserting more allocs, map the ones we already have to the
  // tf runtime. That ensures that all allocations for the actual computation
  // end up on the device, whereas allocations for shape computation and host
  // side things remain on the host.
  // Longer term, this should be handled by proper device placement.
  pm.addPass(mlir::kernel_gen::tf_framework::
                 CreateEmbedTFFrameworkFunctionAndAllocPass());
  // Now lower the shape computations, bufferize all remaining ops and insert
  // deallocs.
  pm.addPass(mlir::kernel_gen::transforms::CreateFinalBufferizePass());
  // TODO(herhut): Enable once no-longer broken.
  // This depends on https://bugs.llvm.org/show_bug.cgi?id=49142 being fixed.
  // pm.addNestedPass<mlir::FuncOp>(::mlir::createBufferHoistingPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createPromoteBuffersToStackPass(
      [](Value alloc) { return IsSmallAlloc(alloc); }));
  // TODO(herhut): Depends on https://bugs.llvm.org/show_bug.cgi?id=48385.
  // We also cannot properly free temporaries until
  // https://llvm.discourse.group/t/remove-tight-coupling-of-the-bufferdeallocation-pass-to-std-and-linalg-operations/2162
  // is resolved.
  // pm.addNestedPass<mlir::FuncOp>(::mlir::createBufferDeallocationPass());
  // pm.addNestedPass<mlir::FuncOp>(mlir::createCopyRemovalPass());
  // Apply the mapping and go to GPU. We cannot do this earlier due to missing
  // interfaces on the GPU dialect.
  // TODO(b/174830459): Move up once implemented.
  if (!cpu_codegen) {
    pm.addNestedPass<::mlir::FuncOp>(mlir::createParallelLoopToGpuPass());
  }

  // Some basic cleanup.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Make loops with min bounds into a conditional plus static bounds.
  pm.addNestedPass<::mlir::FuncOp>(mlir::createForLoopSpecializationPass());
  // Approximate Tanh using standard operations.
  pm.addNestedPass<::mlir::FuncOp>(
      ::mlir::mhlo::createLegalizeTrigonometricToApproximationPass());
  // Take launches to launches with kernels.
  if (!cpu_codegen) {
    pm.addPass(::mlir::createGpuKernelOutliningPass());
  }

  pm.addPass(::mlir::createLowerAffinePass());
  // Constraints are removed as late as possible and before lowering to CFG.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createConvertShapeConstraintsPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addPass(::mlir::createLowerToCFGPass());
  // Map asserts to the tensorflow framework.
  pm.addPass(
      mlir::kernel_gen::tf_framework::CreateEmbedTFFrameworkAssertPass());
  if (embed_memref_prints) {
    pm.addNestedPass<::mlir::FuncOp>(
        mlir::kernel_gen::transforms::CreateEmbedMemRefPrintsPass());
  }
  if (failed(pm.run(module))) {
    return InternalError("Lowering to GPU kernels failed.");
  }
  return Status::OK();
}

Status LowerKernelBodiesToLowLevelIr(mlir::ModuleOp module) {
  auto gpu_modules = module.getOps<mlir::gpu::GPUModuleOp>();
  auto num_modules = std::distance(gpu_modules.begin(), gpu_modules.end());
  if (num_modules != 1) {
    LOG(WARNING) << "There should be exactly one GPU Module, but got "
                 << num_modules
                 << ". Currently we leak memory if there is more than one "
                    "module, see https://bugs.llvm.org/show_bug.cgi?id=48385";
  }
#if !defined(TENSORFLOW_USE_ROCM) && !defined(GOOGLE_CUDA)
  return InternalError(
      "Neither TENSORFLOW_USE_ROCM nor GOOGLE_CUDA are defined."
      " Did you specify either --config=rocm or --config=cuda ?");
#endif
  mlir::PassManager pm(module.getContext());
  // We cannot verify as the signature of the kernel is rewritten.
  // pm.enableVerifier(false);
  tensorflow::applyTensorflowAndCLOptions(pm);
  auto& kernelPm = pm.nest<::mlir::gpu::GPUModuleOp>();
  kernelPm.addPass(::mlir::createLowerToCFGPass());
#if TENSORFLOW_USE_ROCM
  kernelPm.addPass(mlir::kernel_gen::transforms::CreateGpuKernelToRocdlPass());
#elif GOOGLE_CUDA
  kernelPm.addPass(mlir::kernel_gen::transforms::CreateGpuKernelToNvvmPass());
#endif
  // Remove all location information to prevent a debug build.
  pm.addPass(::mlir::createStripDebugInfoPass());

  if (failed(pm.run(module))) {
    return InternalError("Lowering to low-level device IR failed.");
  }

  return Status::OK();
}

Status AmendKernelLLVMIRWithStaticKnowledge(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  applyTensorflowAndCLOptions(pm);

  pm.addNestedPass<mlir::FuncOp>(
      mlir::kernel_gen::transforms::CreatePropagateShapeKnowledgeToKernels());
  pm.addNestedPass<mlir::FuncOp>(
      mlir::kernel_gen::transforms::CreatePropagateTfAbiKnowledgeToKernels());

  return failed(pm.run(module))
             ? InternalError("Amending LLVMIR with static knowledge failed.")
             : Status::OK();
}

Status GenerateDeviceCode(mlir::ModuleOp module,
                          llvm::StringRef gpu_binary_attr_name,
                          llvm::ArrayRef<std::string> architectures,
                          bool generate_fatbin, bool print_ptx,
                          bool enable_ftz) {
  mlir::PassManager pm(module.getContext());
  applyTensorflowAndCLOptions(pm);
  mlir::registerLLVMDialectTranslation(*module->getContext());

  auto& kernel_pm = pm.nest<mlir::gpu::GPUModuleOp>();
  // Remove debug information to ensure we do not create debug PTX.
  kernel_pm.addPass(mlir::createStripDebugInfoPass());
  kernel_pm.addPass(mlir::kernel_gen::transforms::CreateGpuKernelToBlobPass(
      gpu_binary_attr_name, architectures, generate_fatbin, print_ptx,
      enable_ftz));

  return failed(pm.run(module))
             ? InternalError("Generating device code failed.")
             : Status::OK();
}

Status LowerHostSideToFinalForm(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  applyTensorflowAndCLOptions(pm);

  pm.addPass(mlir::kernel_gen::transforms::CreateTFKernelToLLVMPass(
      kGpuBinaryAttrName));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  return failed(pm.run(module))
             ? InternalError("Final lowering of host side failed.")
             : Status::OK();
}

}  // namespace

StatusOr<mlir::OwningModuleRef> GenerateKernelForTfCode(
    mlir::MLIRContext& context, llvm::StringRef tf_code,
    llvm::ArrayRef<std::string> architectures,
    llvm::ArrayRef<int64_t> tile_sizes, llvm::ArrayRef<int64_t> unroll_factors,
    bool embed_memref_prints, bool generate_fatbin, bool print_ptx,
    bool enable_ftz, bool cpu_codegen) {
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  registry.insert<mlir::chlo::HloClientDialect, mlir::mhlo::MhloDialect>();
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);
  context.appendDialectRegistry(registry);
  mlir::OwningModuleRef module = mlir::parseSourceString(tf_code, &context);

  TF_RETURN_IF_ERROR(
      LowerTFtoLoops(module.get(), tile_sizes, unroll_factors, cpu_codegen));
  TF_RETURN_IF_ERROR(
      LowerLoopsToGPUorCPU(module.get(), embed_memref_prints, cpu_codegen));
  if (!cpu_codegen) {
    TF_RETURN_IF_ERROR(LowerKernelBodiesToLowLevelIr(module.get()));
    TF_RETURN_IF_ERROR(AmendKernelLLVMIRWithStaticKnowledge(module.get()));
    TF_RETURN_IF_ERROR(GenerateDeviceCode(module.get(), kGpuBinaryAttrName,
                                          architectures, generate_fatbin,
                                          print_ptx, enable_ftz));
  }
  TF_RETURN_IF_ERROR(LowerHostSideToFinalForm(module.get()));
  return module;
}

}  // namespace kernel_gen
}  // namespace tensorflow
