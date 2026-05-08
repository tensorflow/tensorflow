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

#include <cstdint>
#include <memory>
#include <string>

#include "nvidia/include/NVGPUToLLVM/NVGPUToLLVMPass.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"
#include "xla/pjrt/triton.h"
#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "xla/service/gpu/llvm_gpu_backend/nvptx_backend.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace xla::triton {

namespace {

namespace se = ::stream_executor;

absl::StatusOr<std::string> LLVMToPTX(mlir::ModuleOp module,
                                      absl::string_view arch_name) {
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  module.getContext()->appendDialectRegistry(registry);

  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    return absl::InternalError("Failed to emit LLVM IR");
  }

  TF_ASSIGN_OR_RETURN(auto cuda_cc,
                      se::CudaComputeCapability::FromString(arch_name));
  // Hopper and Blackwell require accelerated features ("a" suffix) for TMA and
  // other advanced instructions.
  if (cuda_cc.major >= 9) {
    cuda_cc.feature_extension =
        se::CudaComputeCapability::FeatureExtension::kAcceleratedFeatures;
  }

  static absl::once_flag init_target_once;
  absl::call_once(init_target_once, []() {
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  });

  llvm::Triple target_triple("nvptx64-nvidia-cuda");
  llvmModule->setTargetTriple(target_triple);

  auto features = absl::StrCat("+ptx", gpu::GetDefaultPtxVersion(cuda_cc));

  DebugOptions debug_options;

  std::unique_ptr<llvm::TargetMachine> machine = gpu::GetTargetMachine(
      target_triple, gpu::nvptx::GetSmName(cuda_cc), debug_options, features);

  llvmModule->setDataLayout(machine->createDataLayout());

  return gpu::nvptx::CompileToPtx(
      llvmModule.get(), se::GpuComputeCapability(cuda_cc), debug_options);
}

}  // namespace

absl::StatusOr<CompilationResult> Compile(absl::string_view module,
                                          absl::string_view arch_name,
                                          int num_warps, int num_ctas,
                                          int num_stages) {
  mlir::MLIRContext context;
  context.loadDialect<::mlir::triton::TritonDialect,
                      ::mlir::triton::gpu::TritonGPUDialect,
                      ::mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
                      ::mlir::arith::ArithDialect, ::mlir::LLVM::LLVMDialect>();
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  context.appendDialectRegistry(registry);

  mlir::OwningOpRef<mlir::ModuleOp> module_op =
      mlir::parseSourceString<mlir::ModuleOp>(module, &context);
  if (!module_op) {
    return absl::InvalidArgumentError("Failed to parse Triton module");
  }

  mlir::PassManager pm(&context);
  pm.enableVerifier();
  TF_ASSIGN_OR_RETURN(auto cuda_cc,
                      se::CudaComputeCapability::FromString(arch_name));

  gpu::CreateTritonPipeline(&pm, se::GpuComputeCapability(cuda_cc), num_warps,
                            num_ctas, num_stages);
  if (failed(pm.run(*module_op))) {
    return absl::InternalError("Failed to compile Triton IR to LLVM IR");
  }

  auto shared_mem_bytes =
      (*module_op)->getAttrOfType<::mlir::IntegerAttr>("ttg.shared").getInt();

  int32_t global_scratch_size = 0;
  if (auto attr = (*module_op)
                      ->getAttrOfType<::mlir::IntegerAttr>(
                          "ttg.global_scratch_memory_size")) {
    global_scratch_size = attr.getInt();
  }

  int cluster_dim_x = 1;
  int cluster_dim_y = 1;
  int cluster_dim_z = 1;
  if (auto attr =
          (*module_op)
              ->getAttrOfType<::mlir::DenseI32ArrayAttr>("ttg.num-ctas")) {
    auto vals = attr.asArrayRef();
    cluster_dim_x = vals[0];
    if (vals.size() > 1) {
      cluster_dim_y = vals[1];
    }
    if (vals.size() > 2) {
      cluster_dim_z = vals[2];
    }
  } else if (auto attr =
                 (*module_op)
                     ->getAttrOfType<::mlir::IntegerAttr>("ttg.num-ctas")) {
    cluster_dim_x = attr.getInt();
  }

  TF_ASSIGN_OR_RETURN(auto ptx, LLVMToPTX(*module_op, arch_name));

  return CompilationResult{
      AsmText{ptx},  shared_mem_bytes, global_scratch_size,
      cluster_dim_x, cluster_dim_y,    cluster_dim_z,
  };
}

}  // namespace xla::triton
