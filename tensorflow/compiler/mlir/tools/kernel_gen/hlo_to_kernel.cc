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

//===- hlo_to_kernel.cc -----------------------------------------*- C++ -*-===//
//
// This file implements the entry point to compile a hlo op to a kernel.
//
//===----------------------------------------------------------------------===//
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/ExecutionEngine/OptUtils.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Export.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/kernel_creator.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace kernel_gen {
namespace {

static llvm::codegen::RegisterCodeGenFlags CGF;

std::unique_ptr<llvm::TargetMachine> GetTargetMachine(
    llvm::StringRef host_triple, llvm::Module* module) {
  llvm::Triple triple(module->getTargetTriple());
  if (triple.getTriple().empty()) {
    if (!host_triple.empty()) {
      triple = llvm::Triple(host_triple);
    } else {
      triple = llvm::Triple(llvm::sys::getDefaultTargetTriple());
    }
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
absl::StatusOr<std::string> EmitToBinary(llvm::StringRef host_triple,
                                         mlir::ModuleOp module) {
  // Translate the module.
  llvm::LLVMContext llvm_context;
  mlir::registerLLVMDialectTranslation(*module->getContext());
  std::unique_ptr<llvm::Module> llvm_module =
      mlir::translateModuleToLLVMIR(module, llvm_context);

  auto target_machine = GetTargetMachine(host_triple, llvm_module.get());
  llvm_module->setDataLayout(target_machine->createDataLayout());

  // Run LLVM's mid-level optimizer to clean up the IR.
  if (mlir::makeOptimizingTransformer(
          /*optLevel=*/2, /*sizeLevel=*/0,
          target_machine.get())(llvm_module.get())) {
    return absl::InternalError("Failed to run LLVM optimizer passess");
  }

  // Set up the output stream.
  llvm::SmallString<8> outstr;
  llvm::raw_svector_ostream ostream(outstr);
  ostream.SetUnbuffered();

  llvm::legacy::PassManager codegen_passes;
  codegen_passes.add(new llvm::TargetLibraryInfoWrapperPass(
      llvm::Triple(llvm_module->getTargetTriple())));

  if (target_machine->addPassesToEmitFile(codegen_passes, ostream, nullptr,
                                          llvm::CodeGenFileType::ObjectFile,
                                          false)) {
    return absl::InternalError("Failed add passes to emit file");
  }
  codegen_passes.run(*llvm_module);
  return ostream.str().str();
}

absl::Status Run(llvm::StringRef input_file, llvm::StringRef output_file,
                 llvm::StringRef host_triple,
                 llvm::ArrayRef<std::string> architectures,
                 llvm::ArrayRef<int64_t> tile_sizes,
                 llvm::ArrayRef<int64_t> unroll_factors, bool print_ptx,
                 bool print_llvmir, bool enable_ftz, bool index_64bit,
                 bool jit_compile, bool jit_i64_indexed_for_large_tensors) {
  // Read TF code.
  std::string hlo_code;
  TF_RETURN_IF_ERROR(
      ReadFileToString(Env::Default(), input_file.str(), &hlo_code));

  // Compile.
  mlir::DialectRegistry registry;
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::MLIRContext context(registry);

  llvm::SourceMgr source_mgr;
  mlir::SourceMgrDiagnosticHandler source_mgr_handler(source_mgr, &context);

  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      GenerateKernelForHloCode(context, hlo_code, architectures, tile_sizes,
                               unroll_factors, print_ptx, print_llvmir,
                               enable_ftz, index_64bit, jit_compile,
                               jit_i64_indexed_for_large_tensors,
                               /*apply_cl_options=*/true));

  // Get binary.
  TF_ASSIGN_OR_RETURN(std::string binary, EmitToBinary(host_triple, *module));

  // Write .a file.
  TF_RETURN_IF_ERROR(
      WriteStringToFile(Env::Default(), output_file.str(), binary));
  return absl::OkStatus();
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
  llvm::cl::opt<bool> index_64bit("index_64bit",
                                  llvm::cl::desc("enable 64 bit indexing"),
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
  llvm::cl::opt<std::string> host_triple(
      "host-triple", llvm::cl::desc("Override host triple for module"));
  llvm::cl::list<std::string> architectures(
      "arch", llvm::cl::desc("target architectures (e.g. sm_70 or compute_75)"),
      llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated);
  llvm::cl::list<int64_t> tile_sizes(
      "tile_sizes", llvm::cl::desc("tile sizes to use"), llvm::cl::ZeroOrMore,
      llvm::cl::CommaSeparated);
  llvm::cl::list<int64_t> unroll_factors(
      "unroll_factors",
      llvm::cl::desc("factors to unroll by, separated by commas"),
      llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated);
  llvm::cl::opt<bool> jit_i64_indexed_for_large_tensors(
      "jit_i64_indexed_for_large_tensors",
      llvm::cl::desc(
          "Enable JIT compilation of i64-indexed kernels for large inputs."),
      llvm::cl::init(false));

  tensorflow::InitMlir y(&argc, &argv);

#ifdef TF_LLVM_X86_AVAILABLE
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmPrinter();
#endif

#ifdef TF_LLVM_AARCH64_AVAILABLE
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64TargetMC();
  LLVMInitializeAArch64AsmPrinter();
#endif

  mlir::registerPassManagerCLOptions();
  mlir::registerMLIRContextCLOptions();

  // Forward cli options to XLA, as it will reset llvm options internally
  // during the first invocation.
  auto& xla_llvm_global_options =
      xla::llvm_ir::LLVMCommandLineOptionsLock::GetGlobalOptions();
  xla_llvm_global_options.insert(xla_llvm_global_options.end(), argv + 1,
                                 argv + argc);
  llvm::cl::ParseCommandLineOptions(argc, argv, "TF op kernel generator\n");

  auto status = tensorflow::kernel_gen::Run(
      input_file, output_file, host_triple, architectures, tile_sizes,
      unroll_factors, print_ptx, print_llvmir, enable_ftz, index_64bit,
      jit_compile, jit_i64_indexed_for_large_tensors);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return 1;
  }
  return 0;
}
