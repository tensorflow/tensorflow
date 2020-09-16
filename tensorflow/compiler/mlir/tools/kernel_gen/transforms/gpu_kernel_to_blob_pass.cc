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

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Target/NVVMIR.h"  // from @llvm-project
#include "mlir/Target/ROCDLIR.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
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
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/rocm_rocdl_path.h"
#endif

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

using xla::InternalError;

class GpuKernelToBlobPass
    : public GpuKernelToBlobPassBase<GpuKernelToBlobPass> {
 public:
  GpuKernelToBlobPass(mlir::StringRef blob_annotation, int32_t arch) {
    blob_annotation_ = blob_annotation;
    arch_ = arch;
  }

  void runOnOperation() override {
    mlir::gpu::GPUModuleOp gpu_module = getOperation();
    auto blob_or = GetGpuBinaryBlob(gpu_module);
    if (blob_or.ok()) {
      const auto& blob = blob_or.ValueOrDie();
      std::string blob_string(blob.begin(), blob.end());
      gpu_module.setAttr(blob_annotation_,
                         mlir::StringAttr::get(blob_string, &getContext()));
      return;
    }
    return signalPassFailure();
  }

  xla::StatusOr<std::vector<uint8_t>> GetGpuBinaryBlob(
      mlir::gpu::GPUModuleOp gpu_module) {
    llvm::LLVMContext llvmContext;
#if TENSORFLOW_USE_ROCM
    auto llvmModule = mlir::translateModuleToROCDLIR(gpu_module, llvmContext);
    if (!llvmModule) {
      return InternalError("Could not translate MLIR module to ROCDL IR");
    }

    llvmModule->setModuleIdentifier("acme");

    xla::HloModuleConfig config;
    config.set_debug_options(xla::GetDebugOptionsFromFlags());

    std::string libdevice_dir = tensorflow::RocdlRoot();

    return xla::gpu::amdgpu::CompileToHsaco(llvmModule.get(), arch_, config,
                                            libdevice_dir);

#elif GOOGLE_CUDA
    auto llvmModule = mlir::translateModuleToNVVMIR(gpu_module, llvmContext);
    if (!llvmModule) {
      return InternalError("Could not translate MLIR module to NVVM");
    }

    llvmModule->setModuleIdentifier("acme");
    llvmModule->setDataLayout(xla::gpu::nvptx::kDataLayout);

    xla::HloModuleConfig config;
    config.set_debug_options(xla::GetDebugOptionsFromFlags());

    auto enable_fusion = [](llvm::TargetMachine* target) {
      target->Options.AllowFPOpFusion = llvm::FPOpFusion::FPOpFusionMode::Fast;
    };

    int32_t cc_major = arch_ / 10;
    int32_t cc_minor = arch_ % 10;
    TF_ASSIGN_OR_RETURN(std::string libdevice_dir, GetLibdeviceDir(config));
    TF_ASSIGN_OR_RETURN(
        std::string ptx,
        xla::gpu::nvptx::CompileToPtx(llvmModule.get(),
                                      std::make_pair(cc_major, cc_minor),
                                      config, libdevice_dir, enable_fusion));
    VLOG(1) << ptx;

    return tensorflow::se::CompileGpuAsm(cc_major, cc_minor, ptx.c_str(),
                                         xla::gpu::PtxOptsFromConfig(config));
#endif
    return InternalError(
        "Neither TENSORFLOW_USE_ROCM nor GOOGLE_CUDA are defined."
        " Did you specify either --config=rocm or --config=cuda ?");
  }

 private:
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
    return InternalError(
        "Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice");
  }
};

}  // namespace

std::unique_ptr<OperationPass<gpu::GPUModuleOp>> CreateGpuKernelToBlobPass(
    mlir::StringRef blob_annotation, int32_t architecture) {
  return std::make_unique<GpuKernelToBlobPass>(blob_annotation, architecture);
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
