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
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
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
#include "mlir/Transforms/Passes.h"
#include "xla/pjrt/triton.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace xla::triton {

namespace {

absl::Status TritonToLLVM(
    mlir::ModuleOp module, absl::string_view arch_name, int num_warps,
    int num_ctas, int num_stages,
    mlir::triton::nvidia_gpu::ClusterInfo* out_cluster_info) {
  std::pair<std::string, std::string> split = absl::StrSplit(arch_name, '.');
  int cc = std::stoi(split.first) * 10 + std::stoi(split.second);

  constexpr int threadsPerWarp = 32;

  mlir::PassManager pm(module.getContext());
  pm.enableVerifier();

  // Based on make_ttir() in triton/third_party/nvidia/backend/compiler.py
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::triton::createRewriteTensorPointerPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::triton::createCombineOpsPass());
  pm.addPass(mlir::triton::createReorderBroadcastPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::triton::createLoopUnrollPass());

  // Based on make_tgir() in triton/third_party/nvidia/backend/compiler.py
  pm.addPass(mlir::triton::createConvertTritonToTritonGPUPass(
      absl::StrFormat("cuda:%u", cc), num_warps, threadsPerWarp, num_ctas));
  pm.addPass(mlir::triton::gpu::createTritonGPUCoalesce());
  if (cc / 10 >= 8) {
    pm.addPass(mlir::triton::gpu::createTritonGPUF32DotTC());
  }
  pm.addPass(mlir::createTritonNvidiaGPUPlanCTAPass(out_cluster_info));
  pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
  pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeThreadLocality());
  pm.addPass(mlir::triton::gpu::createTritonGPUAccelerateMatmul());
  pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
  pm.addPass(
      mlir::triton::gpu::createTritonGPUOptimizeDotOperands({cc / 10 >= 8}));
  pm.addPass(mlir::createCSEPass());
  if (cc / 10 >= 8) {
    pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeAccumulatorInit());
    pm.addPass(mlir::triton::gpu::createTritonGPUCombineTensorSelectAndIf());
    pm.addPass(mlir::triton::gpu::createTritonGPULoopScheduling({num_stages}));
    pm.addPass(mlir::triton::gpu::createTritonGPUPipeline({num_stages}));
  }
  pm.addPass(mlir::triton::gpu::createTritonGPUPrefetch());
  pm.addPass(
      mlir::triton::gpu::createTritonGPUOptimizeDotOperands({cc / 10 >= 8}));
  pm.addPass(mlir::triton::gpu::createTritonGPUCoalesceAsyncCopy());
  pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
  pm.addPass(mlir::triton::gpu::createTritonGPUReduceDataDuplication());
  pm.addPass(mlir::triton::gpu::createTritonGPUReorderInstructions());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  if (cc / 10 >= 9) {
    pm.addPass(mlir::createTritonNvidiaGPUFenceInsertionPass(cc));
    pm.addPass(mlir::createTritonNvidiaGPUTMALoweringPass());
  }
  pm.addPass(mlir::createCanonicalizerPass());

  // Based on make_llir() in triton/third_party/nvidia/backend/compiler.py
  // TODO(slebedev): Uncomment once we upgrade Triton internally.
  // pm.addPass(mlir::triton::NVIDIA::createDecomposeUnsupportedConversionsPass());
  pm.addPass(mlir::triton::gpu::createTritonGPUCombineTensorSelectAndIf());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mlir::triton::gpu::createAllocateSharedMemoryPass());
  pm.addPass(mlir::triton::gpu::createTritonGPUGlobalScratchAllocationPass());
  pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(cc));
  pm.addPass(mlir::triton::createConvertNVGPUToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());

  // TODO(slebedev): Consider adding line info to align with Triton.
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
                        module->getTargetTriple(), error));
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
      module->getTargetTriple(), arch_name, features, opt, llvm::Reloc::PIC_,
      std::nullopt, llvm::CodeGenOptLevel::Aggressive));
}

absl::Status LinkLibdevice(llvm::Module* module) {
  // NOTE: We cannot use std::filesystem until XLA migrates to C++20.
  namespace fs = llvm::sys::fs;

  auto cuda_path = mlir::NVVM::getCUDAToolkitPath();
  if (cuda_path.empty() || !fs::is_directory(cuda_path)) {
    return absl::InternalError(absl::StrFormat(
        "CUDA path %s does not exist or is not a directory", cuda_path));
  }
  auto sep = llvm::sys::path::get_separator().str();
  std::string libdevice_path;
  absl::StrAppend(&libdevice_path, cuda_path.str(), sep, "nvvm", sep,
                  "libdevice", sep, "libdevice.10.bc");

  if (!fs::is_regular_file(libdevice_path)) {
    return absl::InternalError(
        absl::StrFormat("%s is not a regular file", libdevice_path));
  }

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
  llvmModule->setTargetTriple("nvptx64-nvidia-cuda");
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

  auto transformer = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/machine.get());
  if (auto error = transformer(llvmModule.get()); error) {
    return absl::InternalError("Failed to optimize LLVM IR");
  }

  if (auto status = LinkLibdevice(llvmModule.get()); !status.ok()) {
    // TODO(slebedev): Make this an error if the module requires libdevice.
    LOG(ERROR) << "Failed to link libdevice: " << status;
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
