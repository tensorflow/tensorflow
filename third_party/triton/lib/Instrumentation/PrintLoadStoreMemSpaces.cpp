#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include <map>

using namespace llvm;

namespace {

struct LoadStoreMemSpace : public PassInfoMixin<LoadStoreMemSpace> {
  PreservedAnalyses run(llvm::Module &module, ModuleAnalysisManager &) {
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

static std::map<int, std::string> AddrSpaceMap = {
    {0, "FLAT"}, {1, "GLOBAL"}, {3, "SHARED"}, {4, "CONSTANT"}, {5, "SCRATCH"}};

static std::map<std::string, uint32_t> LocationCounterSourceMap;

static std::string LoadOrStoreMap(const BasicBlock::iterator &I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return "LOAD";
  else if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return "STORE";
  else
    throw std::runtime_error("Error: unknown operation type");
}
template <typename LoadOrStoreInst>
static void InstrumentationFunction(const BasicBlock::iterator &I,
                                    const Function &F, const llvm::Module &M,
                                    uint32_t &LocationCounter) {
  auto LSI = dyn_cast<LoadOrStoreInst>(I);
  if (not LSI)
    return;
  Value *Op = LSI->getPointerOperand()->stripPointerCasts();
  uint32_t AddrSpace = cast<PointerType>(Op->getType())->getAddressSpace();
  DILocation *DL = dyn_cast<Instruction>(I)->getDebugLoc();

  std::string SourceAndAddrSpaceInfo =
      (F.getName() + "     " + DL->getFilename() + ":" + Twine(DL->getLine()) +
       ":" + Twine(DL->getColumn()))
          .str() +
      "     " + AddrSpaceMap[AddrSpace] + "     " + LoadOrStoreMap(I);

  if (LocationCounterSourceMap.find(SourceAndAddrSpaceInfo) ==
      LocationCounterSourceMap.end()) {
    errs() << LocationCounter << "     " << SourceAndAddrSpaceInfo << "\n";
    LocationCounterSourceMap[SourceAndAddrSpaceInfo] = LocationCounter;
    LocationCounter++;
  }
}

bool LoadStoreMemSpace::runOnModule(Module &M) {
  bool ModifiedCodeGen = false;
  uint32_t LocationCounter = 0;
  for (auto &F : M) {
    if (F.isIntrinsic())
      continue;
    StringRef functionName = F.getName();
    if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL ||
        F.getCallingConv() == CallingConv::PTX_Kernel ||
        functionName.contains("kernel")) {
      for (Function::iterator BB = F.begin(); BB != F.end(); BB++) {
        for (BasicBlock::iterator I = BB->begin(); I != BB->end(); I++) {
          if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
            InstrumentationFunction<LoadInst>(I, F, M, LocationCounter);
          } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
            InstrumentationFunction<StoreInst>(I, F, M, LocationCounter);
          }
        }
      }
    }
  }
  return ModifiedCodeGen;
}

static PassPluginLibraryInfo getPassPluginInfo() {
  const auto callback = [](PassBuilder &PB) {
    PB.registerOptimizerLastEPCallback([&](ModulePassManager &MPM, auto, auto) {
      MPM.addPass(LoadStoreMemSpace());
      return true;
    });
  };

  return {LLVM_PLUGIN_API_VERSION, "print-mem-space", LLVM_VERSION_STRING,
          callback};
};

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getPassPluginInfo();
}
