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

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Export.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "xla/debug_options_flags.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/gpu/gpu_asm_opts_util.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

#if GOOGLE_CUDA
#include "xla/service/gpu/llvm_gpu_backend/nvptx_backend.h"
#include "xla/stream_executor/cuda/cuda_asm_compiler.h"
#elif TENSORFLOW_USE_ROCM
#include "xla/stream_executor/gpu/asm_compiler.h"
#include "tensorflow/core/platform/rocm_rocdl_path.h"
#endif

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

#define GEN_PASS_DEF_GPUKERNELTOBLOBPASS
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

class GpuKernelToBlobPass
    : public impl::GpuKernelToBlobPassBase<GpuKernelToBlobPass> {
 public:
  GpuKernelToBlobPass(StringRef blob_annotation,
                      llvm::ArrayRef<std::string> architectures, bool print_ptx,
                      bool print_llvmir, bool enable_ftz) {
    if (!blob_annotation.empty()) blob_annotation_ = blob_annotation.str();
    architectures_ = architectures;
    print_ptx_ = print_ptx;
    print_llvmir_ = print_llvmir;
    enable_ftz_ = enable_ftz;
  }

  void runOnOperation() override {
    gpu::GPUModuleOp gpu_module = getOperation();
    auto blob_or = GetGpuBinaryBlob(gpu_module);
    if (blob_or.ok()) {
      const auto& blob = blob_or.value();
      std::string blob_string(blob.begin(), blob.end());
      gpu_module->setAttr(blob_annotation_,
                          StringAttr::get(&getContext(), blob_string));
      return;
    }
    // Forward the error by attaching the message to the gpu module.
    gpu_module.emitError(blob_or.status().message());
    return signalPassFailure();
  }

  absl::StatusOr<std::vector<uint8_t>> GetGpuBinaryBlob(
      gpu::GPUModuleOp gpu_module) {
    if (architectures_.empty()) {
      return tensorflow::errors::Internal(
          "Expected at least one GPU architecture.");
    }

    // Lower to LLVM module.
    llvm::LLVMContext llvmContext;
    auto llvmModule = translateModuleToLLVMIR(gpu_module, llvmContext);
    if (!llvmModule) {
      return tensorflow::errors::Internal(
          "Could not translate MLIR module to LLVM IR");
    }
    llvmModule->setModuleIdentifier(gpu_module.getName());

#if TENSORFLOW_USE_ROCM
    xla::DebugOptions options = xla::GetDebugOptionsFromFlags();
    options.set_xla_gpu_ftz(enable_ftz_);
    options.set_xla_gpu_dump_llvmir(print_llvmir_);

    using AmdGpuHsaco = std::vector<tensorflow::uint8>;
    std::vector<tensorflow::se::HsacoImage> images;
    images.reserve(architectures_.size());
    for (const std::string& arch_str : architectures_) {
      // Parse ROCm architecture.
      absl::string_view consumable_arch(arch_str);
      if (!absl::ConsumePrefix(&consumable_arch, "gfx")) {
        return tensorflow::errors::Internal(
            "Could not parse ROCm architecture prefix (expected gfx)");
      }
      auto llvm_module_copy = llvm::CloneModule(*llvmModule);
      auto hsaco_or = xla::gpu::amdgpu::CompileToHsaco(
          llvm_module_copy.get(),
          tensorflow::se::RocmComputeCapability{arch_str}, options,
          options.DebugString());
      if (!hsaco_or.ok()) {
        return tensorflow::errors::Internal("Failure when generating HSACO");
      }
      auto hsaco = hsaco_or.value();
      images.push_back({arch_str, std::move(hsaco)});
    }

    // TODO(b/169870789): Revisit the use of fatbins.
    // Bundle HSACO images into a single fatbin.
    if (images.size() == 1) return images.front().bytes;
    return tensorflow::se::BundleGpuAsm(images, tensorflow::RocmRoot());

#elif GOOGLE_CUDA
    xla::DebugOptions options = xla::GetDebugOptionsFromFlags();
    options.set_xla_gpu_ftz(enable_ftz_);
    options.set_xla_gpu_dump_llvmir(print_llvmir_);
    // Make sure we use full precision division operations.
    (*options.mutable_xla_backend_extra_options())["-nvptx-prec-divf32"] = "2";
    // Disable tail sinking as it interferes with load/store vectorization. If
    // we have common tails that is intentional.
    (*options.mutable_xla_backend_extra_options())["-simplifycfg-sink-common"] =
        "false";

    llvmModule->setDataLayout(xla::gpu::nvptx::DataLayout());
    llvmModule->setTargetTriple(llvm::Triple(xla::gpu::nvptx::TargetTriple()));

    // Compile and collect requested cubin and PTX images.
    std::vector<tensorflow::se::CubinOrPTXImage> images;
    auto gpu_asm_opts = xla::gpu::PtxOptsFromDebugOptions(options);
    for (const std::string& arch_str : architectures_) {
      TF_ASSIGN_OR_RETURN(auto arch_pair, ParseCudaArch(arch_str));
      bool is_compute_profile = arch_pair.first;
      int arch = arch_pair.second;
      int cc_major = arch / 10;
      int cc_minor = arch % 10;
      tensorflow::se::CudaComputeCapability cc{cc_major, cc_minor};

      // Generate PTX code.
      // Module may be changed by CompileToPtx.
      auto llvm_module_copy = llvm::CloneModule(*llvmModule);
      auto enable_fusion = [](llvm::TargetMachine* target) {
        target->Options.AllowFPOpFusion =
            llvm::FPOpFusion::FPOpFusionMode::Fast;
      };
      TF_ASSIGN_OR_RETURN(std::string ptx, xla::gpu::nvptx::CompileToPtx(
                                               llvm_module_copy.get(), cc,
                                               options, enable_fusion));
      if (print_ptx_) {
        llvm::dbgs() << "Generated PTX code for module '"
                     << gpu_module.getName() << "' on architecture sm_" << arch
                     << ":\n";
        llvm::dbgs() << ptx << "\n";
      }

      // Compile PTX code with ptxas if requested and possible and fall back to
      // a compute image, otherwise.
      if (!is_compute_profile) {
        auto gpu_asm = tensorflow::se::CompileGpuAsm(cc, ptx, gpu_asm_opts);
        if (gpu_asm.ok()) {
          images.push_back(
              {absl::StrCat("sm_", arch), std::move(gpu_asm.value())});
        } else {
#ifdef PLATFORM_GOOGLE
          // Require compilation with ptxas.
          return gpu_asm;
#else
          // Fall back to compilation by driver in OSS.
          LOG(WARNING) << "Failed to compile generated PTX with ptxas. Falling "
                          "back to compilation by driver.";
          is_compute_profile = true;
#endif
        }
      }
      if (is_compute_profile) {
        std::vector<uint8_t> ptx_bytes;
        ptx_bytes.reserve(ptx.size() + 1);
        std::copy(ptx.begin(), ptx.end(), std::back_inserter(ptx_bytes));
        ptx_bytes.push_back('\0');
        images.push_back(
            {absl::StrCat("compute_", arch), std::move(ptx_bytes)});
      }
    }

    // TODO(b/169870789): Revisit the use of fatbins.
    // Bundle cubin and PTX images into a single fatbin if needed.
    if (images.size() == 1) return images.front().bytes;
    return tensorflow::se::BundleGpuAsm(images, gpu_asm_opts);

#else
    return tensorflow::errors::Internal(
        "Neither TENSORFLOW_USE_ROCM nor GOOGLE_CUDA are defined."
        " Did you specify either --config=rocm or --config=cuda ?");
#endif
  }

 private:
  absl::StatusOr<std::pair<bool, int>> ParseCudaArch(
      const std::string& arch_str) {
    absl::string_view consumable_arch(arch_str);
    bool is_compute_profile;
    if (absl::ConsumePrefix(&consumable_arch, "compute_")) {
      is_compute_profile = true;
    } else if (absl::ConsumePrefix(&consumable_arch, "sm_")) {
      is_compute_profile = false;
    } else {
      return tensorflow::errors::Internal(
          "Could not parse cuda architecture prefix (expected sm_ or "
          "compute_)");
    }
    int arch;
    if (!absl::SimpleAtoi(consumable_arch, &arch)) {
      return tensorflow::errors::Internal(
          "Could not parse cuda architecture number");
    }
    return std::pair<bool, int>(is_compute_profile, arch);
  }

  bool enable_ftz_;
};

}  // namespace

std::unique_ptr<OperationPass<gpu::GPUModuleOp>> CreateGpuKernelToBlobPass(
    StringRef blob_annotation, ArrayRef<std::string> architectures,
    bool print_ptx, bool print_llvmir, bool enable_ftz) {
  return std::make_unique<GpuKernelToBlobPass>(
      blob_annotation, architectures, print_ptx, print_llvmir, enable_ftz);
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
