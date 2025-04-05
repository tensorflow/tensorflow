#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <vector>
using namespace llvm;
using namespace std;

namespace {

struct GpuHello : public PassInfoMixin<GpuHello> {
  PreservedAnalyses run(Module &module, ModuleAnalysisManager &) {
    bool modifiedCodeGen = runOnModule(module);

    return (modifiedCodeGen ? llvm::PreservedAnalyses::none()
                            : llvm::PreservedAnalyses::all());
  }
  bool runOnModule(llvm::Module &module);
  // isRequired being set to true keeps this pass from being skipped
  // if it has the optnone LLVM attribute
  static bool isRequired() { return true; }
};

} // end anonymous namespace

bool GpuHello::runOnModule(Module &module) {
  bool modifiedCodeGen = false;

  for (auto &function : module) {
    if (function.isIntrinsic())
      continue;
    StringRef functionName = function.getName();
    if (function.getCallingConv() == CallingConv::AMDGPU_KERNEL ||
        function.getCallingConv() == CallingConv::PTX_Kernel ||
        functionName.contains("kernel")) {
      for (Function::iterator basicBlock = function.begin();
           basicBlock != function.end(); basicBlock++) {
        for (BasicBlock::iterator inst = basicBlock->begin();
             inst != basicBlock->end(); inst++) {
          DILocation *debugLocation =
              dyn_cast<Instruction>(inst)->getDebugLoc();
          std::string sourceInfo =
              (function.getName() + "\t" + debugLocation->getFilename() + ":" +
               Twine(debugLocation->getLine()) + ":" +
               Twine(debugLocation->getColumn()))
                  .str();

          errs() << "Hello From First Instruction of GPU Kernel: " << sourceInfo
                 << "\n";
          return modifiedCodeGen;
        }
      }
    }
  }
  return modifiedCodeGen;
}

static PassPluginLibraryInfo getPassPluginInfo() {
  const auto callback = [](PassBuilder &pb) {
    pb.registerOptimizerLastEPCallback([&](ModulePassManager &mpm, auto, auto) {
      mpm.addPass(GpuHello());
      return true;
    });
  };

  return {LLVM_PLUGIN_API_VERSION, "gpu-hello", LLVM_VERSION_STRING, callback};
};

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getPassPluginInfo();
}
