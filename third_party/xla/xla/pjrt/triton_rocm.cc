/* Copyright 2025 The OpenXLA Authors.

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

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"
#include "xla/debug_options_flags.h"
#include "xla/pjrt/triton.h"
#include "xla/service/gpu/llvm_gpu_backend/amdgpu_backend.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace xla::triton {

namespace {

absl::Status TritonToLLVM(mlir::ModuleOp module, absl::string_view arch_name,
                          int num_warps, int num_ctas, int num_stages) {
  mlir::PassManager pm(module.getContext());
  pm.enableVerifier();
  pm.addPass(mlir::createLowerAffinePass());
  stream_executor::RocmComputeCapability rocm_cc =
      stream_executor::RocmComputeCapability(std::string(arch_name));
  stream_executor::GpuComputeCapability gpu_version(rocm_cc);
  xla::gpu::CreateTritonPipeline(&pm, gpu_version, num_warps, num_ctas,
                                 num_stages);
  pm.addPass(mlir::createStripDebugInfoPass());
  return pm.run(module).succeeded()
             ? absl::OkStatus()
             : absl::InternalError("Failed to compile Triton to LLVM");
}

absl::StatusOr<std::string> LLVMToHSACO(mlir::ModuleOp module,
                                        absl::string_view arch_name,
                                        int num_warps) {
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);
  module.getContext()->appendDialectRegistry(registry);

  stream_executor::RocmComputeCapability rocm_cc =
      stream_executor::RocmComputeCapability(std::string(arch_name));
  stream_executor::GpuComputeCapability gpu_version(rocm_cc);
  int threads_per_block = num_warps * (rocm_cc.gfx9_mi100_or_later() ? 64 : 32);

  llvm::LLVMContext llvm_context;
  std::unique_ptr<llvm::Module> llvm_module =
      mlir::translateModuleToLLVMIR(module, llvm_context);
  if (!llvm_module) {
    return absl::InternalError("Failed to emit LLVM IR");
  }
  llvm_module->setTargetTriple(llvm::Triple(xla::gpu::amdgpu::TargetTriple()));
  for (llvm::Function& func : *llvm_module) {
    if (!func.isDeclaration() && func.hasExternalLinkage()) {
      func.setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
      func.addFnAttr("uniform-work-group-size", "true");
      func.addFnAttr(
          "amdgpu-flat-work-group-size",
          absl::StrJoin({threads_per_block, threads_per_block}, ","));
      func.addFnAttr("amdgpu-waves-per-eu", "2");
    }
  }

  xla::DebugOptions debug_opts = xla::DefaultDebugOptionsIgnoringFlags();
  return xla::gpu::amdgpu::CompileToHsacoAndReturnFilePath(
      llvm_module.get(), gpu_version, debug_opts, false);
}

}  // namespace

absl::StatusOr<CompilationResult> Compile(absl::string_view module,
                                          absl::string_view arch_name,
                                          int num_warps, int num_ctas,
                                          int num_stages) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::triton::TritonDialect,
                      mlir::triton::gpu::TritonGPUDialect,
                      mlir::arith::ArithDialect, mlir::affine::AffineDialect,
                      mlir::LLVM::LLVMDialect, mlir::func::FuncDialect,
                      mlir::tensor::TensorDialect>();
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  context.appendDialectRegistry(registry);

  mlir::OwningOpRef<mlir::ModuleOp> module_op =
      mlir::parseSourceString<mlir::ModuleOp>(module, &context);
  if (!module_op) {
    return absl::InvalidArgumentError("Failed to parse Triton module");
  }

  mlir::PassManager pm((*module_op)->getContext());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (!pm.run(*module_op).succeeded()) {
    return absl::InvalidArgumentError("Failed to canonicalize Triton module");
  }

  TF_RETURN_IF_ERROR(
      TritonToLLVM(*module_op, arch_name, num_warps, num_ctas, num_stages));

  int64_t shared_mem_bytes =
      (*module_op)->getAttrOfType<mlir::IntegerAttr>("ttg.shared").getInt();

  TF_ASSIGN_OR_RETURN(std::string hsaco_path,
                      LLVMToHSACO(*module_op, arch_name, num_warps));

  // There is no clusters in ROCm for now.
  return CompilationResult{
      HsacoPath{hsaco_path},
      shared_mem_bytes,
  };
}

}  // namespace xla::triton
