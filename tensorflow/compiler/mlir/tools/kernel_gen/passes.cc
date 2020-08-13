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

#include "tensorflow/compiler/mlir/tools/kernel_gen/passes.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Target/NVVMIR.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/cuda_libdevice_path.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/gpu/asm_compiler.h"
#endif

namespace mlir {
namespace kernel_gen {
namespace {

xla::StatusOr<std::string> GetLibdeviceDir(
    const xla::HloModuleConfig& hlo_module_config) {
  for (const std::string& cuda_root : tensorflow::CandidateCudaRoots(
           hlo_module_config.debug_options().xla_gpu_cuda_data_dir())) {
    std::string libdevice_dir =
        tensorflow::io::JoinPath(cuda_root, "nvvm", "libdevice");
    VLOG(2) << "Looking for libdevice at " << libdevice_dir;
    if (tensorflow::Env::Default()->IsDirectory(libdevice_dir).ok()) {
      VLOG(2) << "Found libdevice dir " << libdevice_dir;
      return libdevice_dir;
    }
  }
  return xla::InternalError(
      "Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice");
}

struct MaterializeBroadcastsPass
    : public mlir::PassWrapper<MaterializeBroadcastsPass, mlir::FunctionPass> {
  void runOnFunction() override {
    mlir::ConversionTarget conversionTarget(getContext());
    mlir::OwningRewritePatternList conversionPatterns;

    // Consider the mhlo dialect legal for tests.
    conversionTarget.addLegalDialect<mlir::mhlo::MhloDialect>();
    // The conversion uses helpers from the Standard dialect.
    conversionTarget.addLegalDialect<mlir::StandardOpsDialect>();

    mlir::mhlo::SetupMaterializeBroadcastsLegality(&getContext(),
                                                   &conversionTarget);
    mlir::mhlo::PopulateMaterializeBroadcastsPatterns(&getContext(),
                                                      &conversionPatterns);

    if (failed(applyPartialConversion(getFunction(), conversionTarget,
                                      conversionPatterns))) {
      return signalPassFailure();
    }
  }
};

struct UnfuseBatchNormPass
    : public mlir::PassWrapper<UnfuseBatchNormPass, mlir::FunctionPass> {
  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns;
    mlir::mhlo::PopulateUnfuseBatchNormPatterns(&getContext(), &patterns);
    mlir::applyPatternsAndFoldGreedily(getOperation(), patterns);
  }
};

struct PropagateTensorFlowABIKnowledgePass
    : public mlir::PassWrapper<PropagateTensorFlowABIKnowledgePass,
                               mlir::OperationPass<mlir::LLVM::LLVMFuncOp>> {
  explicit PropagateTensorFlowABIKnowledgePass(
      mlir::FunctionType type, llvm::ArrayRef<uint32_t> same_shape_)
      : func_type(type), same_shape(same_shape_) {}

  void runOnOperation() override {
    // We know due to tensorflow ABI that the offset is always 0 and that the
    // innermost stride is always 1. To make this visible to the compiler,
    // we insert constants into the code and replace usages accordingly.
    // We do not change the signature so that we keep a somewhat stable ABI
    // that is easy to undertand by tools.
    // We also know that tensorflow aligns all allocated pointers by 16, so
    // we pass this on. Furthermore, we know that arguments never alias. More
    // precicely, they may only alias (due to reuse) if the kernel does not
    // read from a position it previously has written to. We express this with
    // the noalias attribute.
    mlir::LLVM::LLVMFuncOp func = getOperation();

    // This only works if the function is local and we can rewrite it.
    if (func.isExternal()) return;

    mlir::OpBuilder b(func.getBody());
    // Steal the LLVM representation of the index type from the third argument.
    auto index_type = func.getArgument(3).getType();
    mlir::Value one = b.create<mlir::LLVM::ConstantOp>(
        func.getLoc(), index_type, b.getIntegerAttr(b.getIndexType(), 1));
    mlir::Value zero = b.create<mlir::LLVM::ConstantOp>(
        func.getLoc(), index_type, b.getIntegerAttr(b.getIndexType(), 0));
    uint32_t arg_pos = 0;
    std::vector<uint32_t> positions;
    // Collect the agument and return types of the surrounding function.
    auto arg_types = llvm::to_vector<4>(llvm::concat<const mlir::Type>(
        func_type.getInputs(), func_type.getResults()));
    for (mlir::Type arg_type : arg_types) {
      if (!arg_type.isa<mlir::MemRefType>()) {
        func.emitError() << "argument of surrounding func is not ranked memref";
        return signalPassFailure();
      }
      positions.push_back(arg_pos);
      // Set alignment and aliasing on the pointers.
      func.setArgAttr(arg_pos + 1, "llvm.noalias", b.getBoolAttr(true));
      func.setArgAttr(arg_pos + 1, "llvm.align", b.getIndexAttr(16));
      // Replace the offset with zero. Offset is argument number 3.
      func.getArgument(arg_pos + 2).replaceAllUsesWith(zero);
      // Forward over base_ptr, aligned_ptr, offset, size and stride arguments.
      arg_pos += 3 + arg_type.cast<mlir::MemRefType>().getRank() * 2;
      // Replace the last stride with constant 1.
      func.getArgument(arg_pos - 1).replaceAllUsesWith(one);
    }

    // If we have knowledge that some arguments have the same shape, we
    // can use that here. Simply replace usages of the shape parameters within
    // the function body to a single shape parameter.
    if (!same_shape.empty()) {
      auto first = same_shape.front();
      auto first_offset = positions.at(first);
      auto first_type = arg_types[first].cast<mlir::ShapedType>();
      uint32_t rank = first_type.getRank();
      for (auto same : same_shape.drop_front(1)) {
        uint32_t same_offset = positions.at(same);
        auto same_type = arg_types[same].cast<mlir::ShapedType>();
        if (same_type.getRank() != rank) {
          func.emitOpError() << "same shape constraints on arguments with "
                                "non-matching shapes: #"
                             << first << " and #" << same;
          return signalPassFailure();
        }

        for (uint32_t i = 0; i < 2 * rank; ++i) {
          // Replace uses for second arg data with first arg.
          auto same_arg = func.getArgument(same_offset + 3 + i);
          auto first_arg = func.getArgument(first_offset + 3 + i);
          same_arg.replaceAllUsesWith(first_arg);
        }
      }
    }
  }

  mlir::FunctionType func_type;
  llvm::ArrayRef<uint32_t> same_shape;
};

class GpuKernelToBlobPass
    : public mlir::PassWrapper<GpuKernelToBlobPass,
                               mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
 public:
  GpuKernelToBlobPass(mlir::StringRef blob_annotation,
                      std::pair<int32_t, int32_t> compute_capability)
      : blob_annotation_(blob_annotation),
        compute_capability_(compute_capability) {}

  void runOnOperation() override {
    mlir::gpu::GPUModuleOp module = getOperation();

    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToNVVMIR(module, llvmContext);
    if (!llvmModule) {
      return signalPassFailure();
    }

    llvmModule->setModuleIdentifier("acme");
    llvmModule->setDataLayout(xla::gpu::nvptx::kDataLayout);
    xla::HloModuleConfig config;
    config.set_debug_options(xla::GetDebugOptionsFromFlags());

    auto enable_fusion = [](llvm::TargetMachine* target) {
      target->Options.AllowFPOpFusion = llvm::FPOpFusion::FPOpFusionMode::Fast;
    };

    auto libdevice_dir_or = GetLibdeviceDir(config);
    if (!libdevice_dir_or.ok()) {
      return signalPassFailure();
    }

    auto ptx_or = xla::gpu::nvptx::CompileToPtx(
        llvmModule.get(), compute_capability_, config,
        libdevice_dir_or.ValueOrDie(), enable_fusion);
    if (!ptx_or.ok()) {
      return signalPassFailure();
    }

    auto ptx = ptx_or.ValueOrDie();

#if GOOGLE_CUDA
    auto blob_or = tensorflow::se::CompileGpuAsm(
        std::get<0>(compute_capability_), std::get<1>(compute_capability_),
        ptx.c_str(), xla::gpu::PtxOptsFromConfig(config));
    if (blob_or.ok()) {
      const auto& blob = blob_or.ValueOrDie();
      std::string blob_string(blob.begin(), blob.end());
      module.setAttr(blob_annotation_,
                     mlir::StringAttr::get(blob_string, &getContext()));
      return;
    } else {
      return signalPassFailure();
    }
#endif
    return signalPassFailure();
  }

 private:
  mlir::StringRef blob_annotation_;
  std::pair<int32_t, int32_t> compute_capability_;
};

}  // namespace

std::unique_ptr<mlir::FunctionPass> createMaterializeBroadcastsPass() {
  return absl::make_unique<MaterializeBroadcastsPass>();
}

std::unique_ptr<mlir::FunctionPass> createUnfuseBatchNormPass() {
  return absl::make_unique<UnfuseBatchNormPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::LLVM::LLVMFuncOp>>
createPropagateTensorFlowABIKnowledgePass(mlir::FunctionType type,
                                          llvm::ArrayRef<uint32_t> same_shape) {
  return absl::make_unique<PropagateTensorFlowABIKnowledgePass>(type,
                                                                same_shape);
}

std::unique_ptr<mlir::OperationPass<mlir::gpu::GPUModuleOp>>
createGpuKernelToBlobPass(
    mlir::StringRef blob_annotation,
    const std::pair<int32_t, int32_t>& compute_capability) {
  return absl::make_unique<GpuKernelToBlobPass>(blob_annotation,
                                                compute_capability);
}

}  // namespace kernel_gen
}  // namespace mlir
