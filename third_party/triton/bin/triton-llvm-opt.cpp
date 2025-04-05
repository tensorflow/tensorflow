/// Trimmed down clone of llvm opt to be able to test triton custom llvm ir
/// passes.
#include "lib/Target/LLVMIR/LLVMPasses.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/TargetParser/Triple.h"
#include <optional>

using namespace llvm;

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input bitcode file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string> OutputFilename("o",
                                           cl::desc("Override output filename"),
                                           cl::value_desc("filename"));

static cl::opt<std::string> ClDataLayout("data-layout",
                                         cl::desc("data layout string to use"),
                                         cl::value_desc("layout-string"),
                                         cl::init(""));
static cl::opt<std::string>
    TargetTriple("mtriple", cl::desc("Override target triple for module"));

static cl::opt<bool>
    BreakStructPhiNodes("break-struct-phi-nodes",
                        llvm::cl::desc("run pass to break phi struct"),
                        cl::init(false));

namespace {
static std::function<Error(Module *)> makeOptimizingPipeline() {
  return [](Module *m) -> Error {
    PipelineTuningOptions tuningOptions;
    PassBuilder pb(nullptr, tuningOptions);

    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm;
    llvm::FunctionPassManager fpm;
    if (BreakStructPhiNodes)
      fpm.addPass(BreakStructPhiNodesPass());
    mpm.addPass(createModuleToFunctionPassAdaptor(std::move(fpm)));
    mpm.run(*m, mam);
    return Error::success();
  };
}
} // namespace

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(
      argc, argv, "llvm .bc -> .bc modular optimizer and analysis printer\n");

  LLVMContext Context;
  SMDiagnostic Err;

  // Load the input module...
  auto SetDataLayout = [](StringRef, StringRef) -> std::optional<std::string> {
    if (ClDataLayout.empty())
      return std::nullopt;
    return ClDataLayout;
  };
  std::unique_ptr<Module> M;
  M = parseIRFile(InputFilename, Err, Context, ParserCallbacks(SetDataLayout));
  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }
  // If we are supposed to override the target triple or data layout, do so now.
  if (!TargetTriple.empty())
    M->setTargetTriple(Triple(Triple::normalize(TargetTriple)));
  auto optPipeline = makeOptimizingPipeline();
  if (auto err = optPipeline(M.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
  }

  if (verifyModule(*M, &errs())) {
    errs() << argv[0] << ": " << InputFilename
           << ": error: input module is broken!\n";
    return 1;
  }

  // Write to standard output.
  std::unique_ptr<ToolOutputFile> Out;
  // Default to standard output.
  if (OutputFilename.empty())
    OutputFilename = "-";
  std::error_code EC;
  sys::fs::OpenFlags Flags = sys::fs::OF_TextWithCRLF;
  Out.reset(new ToolOutputFile(OutputFilename, EC, Flags));
  if (EC) {
    errs() << EC.message() << '\n';
    return 1;
  }
  Out->os() << *M << "\n";
  Out->keep();
  return 0;
}
