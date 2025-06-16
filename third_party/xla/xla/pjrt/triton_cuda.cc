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
#include <optional>
#include <string>
#include <utility>

#include "nvidia/include/NVGPUToLLVM/NVGPUToLLVMPass.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVM/NVVM/Utils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"
#include "xla/pjrt/triton.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/cuda_root_path.h"
#include "tsl/platform/path.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace xla::triton {

namespace {

absl::Status TritonToLLVM(
    mlir::ModuleOp module, absl::string_view arch_name, int num_warps,
    int num_ctas, int num_stages,
    mlir::triton::nvidia_gpu::ClusterInfo* out_cluster_info) {
  mlir::PassManager pm(module.getContext());
  pm.enableVerifier();
  TF_RETURN_IF_ERROR(
      xla::gpu::CreateTritonPipeline(&pm, std::string(arch_name), num_warps,
                                     num_ctas, num_stages, *out_cluster_info,
                                     /*is_xla_fusion=*/false));
  return pm.run(module).succeeded()
             ? absl::OkStatus()
             : absl::InternalError("Failed to compile Triton IR to LLVM IR");
}

absl::StatusOr<std::unique_ptr<llvm::TargetMachine>> CreateTargetMachine(
    llvm::Module* module, absl::string_view arch_name, bool enable_fp_fusion,
    absl::string_view features) {
  // Based on createTargetMachine() in triton/python/src/llvm.cc
  std::string error;
  const auto* target =
      llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
  if (target == nullptr) {
    return absl::InternalError(
        absl::StrFormat("Failed to lookup LLVM target based on triple %s: %s",
                        module->getTargetTriple().str(), error));
  }
  llvm::TargetOptions opt;
  if (enable_fp_fusion) {
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  }
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  opt.TrapUnreachable = true;
  opt.MCOptions.AsmVerbose = true;
  opt.MCOptions.PreserveAsmComments = true;
  return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
      module->getTargetTriple().str(), arch_name, features, opt,
      llvm::Reloc::PIC_, std::nullopt, llvm::CodeGenOptLevel::Aggressive));
}

absl::StatusOr<std::string> GetLibdeviceDir() {
  auto nvvm_cuda_root = mlir::NVVM::getCUDAToolkitPath().str();
  for (const std::string& cuda_root : tsl::CandidateCudaRoots(nvvm_cuda_root)) {
    auto libdevice_dir = tsl::io::JoinPath(cuda_root, "nvvm", "libdevice");
    if (tsl::Env::Default()->IsDirectory(libdevice_dir).ok()) {
      return libdevice_dir;
    }
  }
  return absl::InternalError(absl::StrCat(
      "Cannot find libdevice.10.bc in any of the CUDA roots. "
      "Searched for CUDA in the following directories:\n  ",
      absl::StrJoin(tsl::CandidateCudaRoots(nvvm_cuda_root), "\n  ")));
}

absl::Status LinkLibdevice(llvm::Module* module) {
  TF_ASSIGN_OR_RETURN(auto libdevice_dir, GetLibdeviceDir());
  auto libdevice_path = tsl::io::JoinPath(libdevice_dir, "libdevice.10.bc");

  llvm::LLVMContext& ctx = module->getContext();
  llvm::SMDiagnostic err;
  std::unique_ptr<llvm::Module> libdevice_module =
      llvm::parseIRFile(libdevice_path, err, ctx);
  if (!libdevice_module) {
    return absl::InternalError(
        absl::StrFormat("Failed to parse libdevice IR file at %s: %s",
                        libdevice_path, err.getMessage()));
  }

  llvm::Linker linker(*module);
  if (linker.linkInModule(std::move(libdevice_module),
                          llvm::Linker::Flags::LinkOnlyNeeded)) {
    return absl::InternalError("Failed to link libdevice");
  }

  return absl::OkStatus();
}

absl::StatusOr<std::string> LLVMToPTX(mlir::ModuleOp module,
                                      absl::string_view arch_name) {
  // Based on translateLLVMIRToASM() in triton/python/src/llvm.cc
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

  auto cc = absl::StrReplaceAll(arch_name, {{".", ""}});  // "8.0" -> "80"
  auto proc = absl::StrCat("sm_", cc, cc == "90" ? "a" : "");
  // We cap the ISA at 8.4 to align with Triton.
  // See get_features() in triton/third_party/nvidia/backend/compiler.py.
  auto features = cc >= "84" ? "+ptx84" : "+ptx" + cc;
  llvmModule->setTargetTriple(llvm::Triple("nvptx64-nvidia-cuda"));
  static absl::once_flag init_target_once;
  absl::call_once(init_target_once, []() {
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  });
  TF_ASSIGN_OR_RETURN(
      auto machine, CreateTargetMachine(llvmModule.get(), proc,
                                        /*enable_fp_fusion=*/false, features));

  llvmModule->setDataLayout(machine->createDataLayout());

  auto needsLibdevice =
      llvm::any_of(llvmModule->functions(), [](const auto& f) {
        return !f.isIntrinsic() && f.isDeclaration() &&
               f.getName().starts_with("__nv_");
      });
  if (needsLibdevice) {
    TF_RETURN_IF_ERROR(LinkLibdevice(llvmModule.get()));
  }

  auto transformer = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/machine.get());
  if (auto error = transformer(llvmModule.get()); error) {
    return absl::InternalError("Failed to optimize LLVM IR");
  }

  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream bstream(stream);
    llvm::legacy::PassManager pm;
    machine->addPassesToEmitFile(pm, bstream, nullptr,
                                 llvm::CodeGenFileType::AssemblyFile,
                                 /*DisableVerify=*/false);
    if (!pm.run(*llvmModule)) {
      return absl::InternalError("Failed to compile LLVM IR to PTX");
    }
  }
  return result;
}

}  // namespace

absl::StatusOr<CompilationResult> Compile(absl::string_view module,
                                          absl::string_view arch_name,
                                          int num_warps, int num_ctas,
                                          int num_stages) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::triton::TritonDialect,
                      mlir::triton::gpu::TritonGPUDialect,
                      mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect>();
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  context.appendDialectRegistry(registry);

  mlir::OwningOpRef<mlir::ModuleOp> module_op =
      mlir::parseSourceString<mlir::ModuleOp>(module, &context);
  if (!module_op) {
    return absl::InvalidArgumentError("Failed to parse Triton module");
  }
  mlir::triton::nvidia_gpu::ClusterInfo cluster_info;
  TF_RETURN_IF_ERROR(TritonToLLVM(*module_op, arch_name, num_warps, num_ctas,
                                  num_stages, &cluster_info));

  auto shared_mem_bytes =
      (*module_op)->getAttrOfType<mlir::IntegerAttr>("ttg.shared").getInt();

  TF_ASSIGN_OR_RETURN(auto ptx, LLVMToPTX(*module_op, arch_name));

  return CompilationResult{
      ptx,
      shared_mem_bytes,
      cluster_info.clusterDimX,
      cluster_info.clusterDimY,
      cluster_info.clusterDimZ,
  };
}

}  // namespace xla::triton
