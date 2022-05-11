// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- tf_to_kernel.cc ------------------------------------------*- C++ -*-===//
//
// This file implements the entry point to compile a tf op to a kernel.
//
//===----------------------------------------------------------------------===//
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/ExecutionEngine/OptUtils.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Export.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/kernel_creator.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
namespace kernel_gen {
namespace {

static llvm::codegen::RegisterCodeGenFlags CGF;

std::unique_ptr<llvm::TargetMachine> GetTargetMachine(llvm::Module* module) {
  llvm::Triple triple(module->getTargetTriple());
  if (triple.getTriple().empty()) {
    triple = llvm::Triple(llvm::sys::getDefaultTargetTriple());
    module->setTargetTriple(triple.getTriple());
  }

  std::string error;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget("", triple, error);
  if (!target) {
    return nullptr;
  }

  llvm::TargetOptions target_options =
      llvm::codegen::InitTargetOptionsFromCodeGenFlags(llvm::Triple());
  return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
      triple.str(), "generic", "", target_options, llvm::Reloc::Model::PIC_));
}

// Compiles the given MLIR module via LLVM into an executable binary format.
StatusOr<std::string> EmitToBinary(mlir::ModuleOp module) {
  // Translate the module.
  llvm::LLVMContext llvm_context;
  mlir::registerLLVMDialectTranslation(*module->getContext());
  std::unique_ptr<llvm::Module> llvm_module =
      mlir::translateModuleToLLVMIR(module, llvm_context);

  auto target_machine = GetTargetMachine(llvm_module.get());
  llvm_module->setDataLayout(target_machine->createDataLayout());

  // Run LLVM's mid-level optimizer to clean up the IR.
  if (mlir::makeOptimizingTransformer(
          /*optLevel=*/2, /*sizeLevel=*/0,
          target_machine.get())(llvm_module.get())) {
    return tensorflow::errors::Internal("Failed to run LLVM optimizer passess");
  }

  // Set up the output stream.
  llvm::SmallString<8> outstr;
  llvm::raw_svector_ostream ostream(outstr);
  ostream.SetUnbuffered();

  llvm::legacy::PassManager codegen_passes;
  codegen_passes.add(new llvm::TargetLibraryInfoWrapperPass(
      llvm::Triple(llvm_module->getTargetTriple())));

  if (target_machine->addPassesToEmitFile(codegen_passes, ostream, nullptr,
                                          llvm::CGFT_ObjectFile, false)) {
    return tensorflow::errors::Internal("Failed add passes to emit file");
  }
  codegen_passes.run(*llvm_module);
  return ostream.str().str();
}

Status Run(llvm::StringRef input_file, llvm::StringRef output_file,
           llvm::ArrayRef<std::string> architectures,
           llvm::ArrayRef<int64_t> tile_sizes,
           llvm::ArrayRef<int64_t> unroll_factors, int64_t max_supported_rank,
           bool embed_memref_prints, bool print_ptx, bool print_llvmir,
           bool enable_ftz, bool index_64bit, bool cpu_codegen,
           bool jit_compile) {
  // 64 bit indexing is not incorporated yet
  if (index_64bit) {
    return tensorflow::errors::Unimplemented(
        "64 bit indexing is not supported yet");
  }
  // Read TF code.
  std::string tf_code;
  TF_RETURN_IF_ERROR(
      ReadFileToString(Env::Default(), input_file.str(), &tf_code));
  // Compile.
  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      GenerateKernelForTfCode(context, tf_code, architectures, tile_sizes,
                              unroll_factors, max_supported_rank,
                              embed_memref_prints, print_ptx, print_llvmir,
                              enable_ftz, index_64bit, cpu_codegen, jit_compile,
                              /*jit_i64_indexed_for_large_tensors=*/false,
                              /*apply_cl_options=*/true));
  // Get binary.
  TF_ASSIGN_OR_RETURN(std::string binary, EmitToBinary(*module));

  // Write .a file.
  TF_RETURN_IF_ERROR(
      WriteStringToFile(Env::Default(), output_file.str(), binary));
  return Status::OK();
}

}  // namespace
}  // namespace kernel_gen
}  // namespace tensorflow

int main(int argc, char** argv) {
  llvm::cl::opt<std::string> input_file("input", llvm::cl::desc("input file"),
                                        llvm::cl::value_desc("filename"),
                                        llvm::cl::init("foo.mlir"));
  llvm::cl::opt<std::string> output_file(
      "output", llvm::cl::desc("output file"), llvm::cl::value_desc("filename"),
      llvm::cl::init("foo.bin"));
  llvm::cl::opt<bool> cpu_codegen("cpu_codegen",
                                  llvm::cl::desc("enable CPU code generation"),
                                  llvm::cl::init(false));
  llvm::cl::opt<bool> index_64bit("index_64bit",
                                  llvm::cl::desc("enable 64 bit indexing"),
                                  llvm::cl::init(false));
  llvm::cl::opt<bool> embed_memref_prints(
      "embed_memref_prints",
      llvm::cl::desc("embed memref prints at the end of their lifetime"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> print_ptx(
      "print-ptx",
      llvm::cl::desc("print generated PTX code per target architecture."),
      llvm::cl::init(false));
  llvm::cl::opt<bool> print_llvmir(
      "print-llvmir", llvm::cl::desc("print llvm ir during lowering to ptx."),
      llvm::cl::init(false));
  llvm::cl::opt<bool> enable_ftz(
      "enable_ftz",
      llvm::cl::desc(
          "enable the denormal flush to zero mode when generating code."),
      llvm::cl::init(false));
  llvm::cl::opt<bool> jit_compile(
      "jit", llvm::cl::desc("Generate only a JIT compiler invocation."),
      llvm::cl::init(false));
  llvm::cl::list<std::string> architectures(
      "arch", llvm::cl::desc("target architectures (e.g. sm_70 or compute_75)"),
      llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated);
  llvm::cl::opt<int64_t> max_supported_rank(
      "max-supported-rank",
      llvm::cl::desc("maximum supported rank to be guaranteed by rank "
                     "specialization lowering"),
      llvm::cl::init(5));
  llvm::cl::list<int64_t> tile_sizes(
      "tile_sizes", llvm::cl::desc("tile sizes to use"), llvm::cl::ZeroOrMore,
      llvm::cl::CommaSeparated);
  llvm::cl::list<int64_t> unroll_factors(
      "unroll_factors",
      llvm::cl::desc("factors to unroll by, separated by commas"),
      llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated);

  tensorflow::InitMlir y(&argc, &argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::registerPassManagerCLOptions();
  mlir::registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "TF op kernel generator\n");

  auto status = tensorflow::kernel_gen::Run(
      input_file, output_file, architectures, tile_sizes, unroll_factors,
      max_supported_rank, embed_memref_prints, print_ptx, print_llvmir,
      enable_ftz, index_64bit, cpu_codegen, jit_compile);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return 1;
  }
  return 0;
}
