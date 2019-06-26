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

#include "mlir/GPU/GPUDialect.h"
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

/// A pass converting tagged kernel functions to cubin blobs.
class GpuKernelToCubinPass : public ModulePass<GpuKernelToCubinPass> {
public:
  GpuKernelToCubinPass(
      CubinGenerator cubinGenerator = compilePtxToCubinForTesting)
      : cubinGenerator(cubinGenerator) {}

  // Run the dialect converter on the module.
  void runOnModule() override {
    // Make sure the NVPTX target is initialized.
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();

    for (auto &function : getModule()) {
      if (!gpu::GPUDialect::isKernel(&function) || function.isExternal()) {
        continue;
      }
      if (failed(translateGpuKernelToCubinAnnotation(function)))
        signalPassFailure();
    }
  }

private:
  static OwnedCubin compilePtxToCubinForTesting(const std::string &ptx,
                                                Function &function);

  std::string translateModuleToPtx(llvm::Module &module,
                                   llvm::TargetMachine &target_machine);
  OwnedCubin convertModuleToCubin(llvm::Module &llvmModule, Function &function);
  LogicalResult translateGpuKernelToCubinAnnotation(Function &function);

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
                                                  Function &function) {
  const char data[] = "CUBIN";
  return llvm::make_unique<std::vector<char>>(data, data + sizeof(data) - 1);
}

OwnedCubin GpuKernelToCubinPass::convertModuleToCubin(llvm::Module &llvmModule,
                                                      Function &function) {
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  {
    std::string error;
    // TODO(herhut): Make triple configurable.
    constexpr const char *cudaTriple = "nvptx64-nvidia-cuda";
    llvm::Triple triple(cudaTriple);
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget("", triple, error);
    if (target == nullptr) {
      function.emitError("Cannot initialize target triple");
      return {};
    }
    targetMachine.reset(
        target->createTargetMachine(triple.str(), "sm_75", "+ptx60", {}, {}));
  }

  // Set the data layout of the llvm module to match what the ptx target needs.
  llvmModule.setDataLayout(targetMachine->createDataLayout());

  auto ptx = translateModuleToPtx(llvmModule, *targetMachine);

  return cubinGenerator(ptx, function);
}

LogicalResult
GpuKernelToCubinPass::translateGpuKernelToCubinAnnotation(Function &function) {
  Builder builder(function.getContext());

  std::unique_ptr<Module> module(builder.createModule());

  // TODO(herhut): Also handle called functions.
  module->getFunctions().push_back(function.clone());

  auto llvmModule = translateModuleToNVVMIR(*module);
  auto cubin = convertModuleToCubin(*llvmModule, function);

  if (!cubin)
    return function.emitError("Translation to CUDA binary failed.");

  function.setAttr(kCubinAnnotation,
                   builder.getStringAttr({cubin->data(), cubin->size()}));

  // Remove the body of the kernel function now that it has been translated.
  // The main reason to do this is so that the resulting module no longer
  // contains the NVVM instructions (typically contained in the kernel bodies)
  // and hence can be compiled into host code by a separate pass.
  function.eraseBody();

  return success();
}

ModulePassBase *
mlir::createConvertGPUKernelToCubinPass(CubinGenerator cubinGenerator) {
  return new GpuKernelToCubinPass(cubinGenerator);
}

static PassRegistration<GpuKernelToCubinPass>
    pass("test-kernel-to-cubin",
         "Convert all kernel functions to CUDA cubin blobs");
