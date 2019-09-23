//===- ConvertKernelFuncToCubin.cpp - MLIR GPU lowering passes ------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements a pass to convert gpu kernel functions into a
// corresponding binary blob that can be executed on a CUDA GPU. Currently
// only translates the function itself but no dependencies.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToCUDA/GPUToCUDAPass.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/NVVMIR.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace mlir;

namespace {
// TODO(herhut): Move to shared location.
static constexpr const char *kCubinAnnotation = "nvvm.cubin";

/// A pass converting tagged kernel modules to cubin blobs.
///
/// If tagged as a kernel module, each contained function is translated to NVVM
/// IR and further to PTX. A user provided CubinGenerator compiles the PTX to
/// GPU binary code, which is then attached as an attribute to the function. The
/// function body is erased.
class GpuKernelToCubinPass : public ModulePass<GpuKernelToCubinPass> {
public:
  GpuKernelToCubinPass(
      CubinGenerator cubinGenerator = compilePtxToCubinForTesting)
      : cubinGenerator(cubinGenerator) {}

  void runOnModule() override {
    if (!getModule().getAttrOfType<UnitAttr>(
            gpu::GPUDialect::getKernelModuleAttrName()))
      return;

    // Make sure the NVPTX target is initialized.
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();

    auto llvmModule = translateModuleToNVVMIR(getModule());
    if (!llvmModule)
      return signalPassFailure();

    for (auto function : getModule().getOps<FuncOp>()) {
      if (!gpu::GPUDialect::isKernel(function))
        continue;
      if (failed(translateGpuKernelToCubinAnnotation(*llvmModule, function)))
        signalPassFailure();
    }
  }

private:
  static OwnedCubin compilePtxToCubinForTesting(const std::string &ptx,
                                                FuncOp &function);

  std::string translateModuleToPtx(llvm::Module &module,
                                   llvm::TargetMachine &target_machine);

  /// Converts llvmModule to cubin using the user-provded generator.
  OwnedCubin convertModuleToCubin(llvm::Module &llvmModule, FuncOp &function);

  /// Translates llvmModule to cubin and assigns it to attribute of function.
  LogicalResult translateGpuKernelToCubinAnnotation(llvm::Module &llvmModule,
                                                    FuncOp &function);

  CubinGenerator cubinGenerator;
};

} // anonymous namespace

std::string GpuKernelToCubinPass::translateModuleToPtx(
    llvm::Module &module, llvm::TargetMachine &target_machine) {
  std::string ptx;
  {
    llvm::raw_string_ostream stream(ptx);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegen_passes;
    target_machine.addPassesToEmitFile(codegen_passes, pstream, nullptr,
                                       llvm::TargetMachine::CGFT_AssemblyFile);
    codegen_passes.run(module);
  }

  return ptx;
}

OwnedCubin
GpuKernelToCubinPass::compilePtxToCubinForTesting(const std::string &ptx,
                                                  FuncOp &function) {
  const char data[] = "CUBIN";
  return std::make_unique<std::vector<char>>(data, data + sizeof(data) - 1);
}

OwnedCubin GpuKernelToCubinPass::convertModuleToCubin(llvm::Module &llvmModule,
                                                      FuncOp &function) {
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  {
    std::string error;
    // TODO(herhut): Make triple configurable.
    constexpr const char *cudaTriple = "nvptx64-nvidia-cuda";
    llvm::Triple triple(cudaTriple);
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget("", triple, error);
    if (target == nullptr) {
      function.emitError("cannot initialize target triple");
      return {};
    }
    targetMachine.reset(
        target->createTargetMachine(triple.str(), "sm_35", "+ptx60", {}, {}));
  }

  // Set the data layout of the llvm module to match what the ptx target needs.
  llvmModule.setDataLayout(targetMachine->createDataLayout());

  auto ptx = translateModuleToPtx(llvmModule, *targetMachine);

  return cubinGenerator(ptx, function);
}

LogicalResult GpuKernelToCubinPass::translateGpuKernelToCubinAnnotation(
    llvm::Module &llvmModule, FuncOp &function) {
  auto cubin = convertModuleToCubin(llvmModule, function);
  if (!cubin)
    return function.emitError("translation to CUDA binary failed.");

  Builder builder(function.getContext());
  function.setAttr(kCubinAnnotation,
                   builder.getStringAttr({cubin->data(), cubin->size()}));

  // Remove the body of the kernel function now that it has been translated.
  // The main reason to do this is so that the resulting module no longer
  // contains the NVVM instructions (typically contained in the kernel bodies)
  // and hence can be compiled into host code by a separate pass.
  function.eraseBody();

  return success();
}

std::unique_ptr<OpPassBase<ModuleOp>>
mlir::createConvertGPUKernelToCubinPass(CubinGenerator cubinGenerator) {
  return std::make_unique<GpuKernelToCubinPass>(cubinGenerator);
}

static PassRegistration<GpuKernelToCubinPass>
    pass("test-kernel-to-cubin",
         "Convert all kernel functions to CUDA cubin blobs");
