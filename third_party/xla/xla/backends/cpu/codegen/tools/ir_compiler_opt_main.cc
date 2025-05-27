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

// A tool for reading LLVM IR and running XLA CPU codegen IR compiler passes.
// This tool is designed for FileCheck testing of IR transformations.

#include <cstdio>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"

namespace {
const char* const kUsage = R"(
This tool runs XLA CPU codegen IR compiler passes on LLVM IR input and outputs
the transformed LLVM IR for use in FileCheck tests.

Usage:
  bazel run ir_compiler_opt -- [options] input.ll

Options:
  --output_file=<file>    Output file (default: stdout)
  --help                  Show this help message
)";

struct IrCompilerOptConfig {
  bool help{false};
  std::string input_file{""};
  std::string output_file{"-"};
  int opt_level = static_cast<int>(llvm::CodeGenOptLevel::Aggressive);
};

}  // namespace

namespace xla::cpu {

namespace {

std::string GetInputPath(const IrCompilerOptConfig& opts, int argc,
                         char** argv) {
  if (!opts.input_file.empty()) {
    return opts.input_file;
  }
  QCHECK(argc == 2) << "Must specify a single input file";
  return argv[1];
}

absl::StatusOr<std::string> GetInputContents(const IrCompilerOptConfig& opts,
                                             int argc, char** argv) {
  std::string input_path = GetInputPath(opts, argc, argv);
  if (input_path == "-") {
    std::string input;
    std::getline(std::cin, input, static_cast<char>(EOF));
    return input;
  }

  std::string data;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), input_path, &data));
  return data;
}

absl::StatusOr<std::unique_ptr<llvm::Module>> ParseLlvmIr(
    const std::string& ir_content, llvm::LLVMContext& context) {
  llvm::SMDiagnostic error;
  std::unique_ptr<llvm::Module> module =
      llvm::parseIR(llvm::MemoryBufferRef(ir_content, "input"), error, context);

  if (!module) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse LLVM IR: ", error.getMessage().str()));
  }

  return module;
}

absl::StatusOr<std::string> RunIrCompilerPasses(const IrCompilerOptConfig& opts,
                                                int argc, char** argv) {
  llvm::TargetOptions target_options;
  IrCompiler::Options ir_compiler_options;
  CHECK(opts.opt_level >= 0 && opts.opt_level <= 3)
      << "Optimization level must be between 0 and 3";
  ir_compiler_options.opt_level =
      static_cast<llvm::CodeGenOptLevel>(opts.opt_level);
  auto ir_compiler = IrCompiler::Create(target_options, ir_compiler_options,
                                        IrCompiler::CompilationHooks());

  TF_ASSIGN_OR_RETURN(std::string ir_content,
                      GetInputContents(opts, argc, argv));

  llvm::LLVMContext context;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<llvm::Module> module,
                      ParseLlvmIr(ir_content, context));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<llvm::TargetMachine> target_machine,
      ir_compiler->InferTargetMachine(
          target_options, static_cast<llvm::CodeGenOptLevel>(opts.opt_level),
          std::nullopt));

  llvm::Error error = ir_compiler->RunIrPasses(*module, target_machine.get());
  if (error) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to run IR compiler passes: ",
                     llvm::toString(std::move(error))));
  }
  return llvm::to_string(*module);
}

absl::Status RunIrCompilerOptMain(int argc, char** argv,
                                  const IrCompilerOptConfig& opts) {
  TF_ASSIGN_OR_RETURN(std::string output,
                      RunIrCompilerPasses(opts, argc, argv));

  if (opts.output_file == "-") {
    std::cout << output << std::endl;
  } else {
    TF_RETURN_IF_ERROR(
        tsl::WriteStringToFile(tsl::Env::Default(), opts.output_file, output));
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace xla::cpu

int main(int argc, char** argv) {
  IrCompilerOptConfig opts;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("help", &opts.help, "Show help message and exit"),
      tsl::Flag("input_file", &opts.input_file, "Input LLVM IR file"),
      tsl::Flag("output_file", &opts.output_file,
                "Output filename, or '-' for stdout (default)."),
      tsl::Flag("opt_level", &opts.opt_level, "Optimization level"),
  };

  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));

  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);

  if (!parse_ok || opts.help) {
    std::cout << kUsageString << std::endl;
    return opts.help ? 0 : 1;
  }

  absl::Status s = xla::cpu::RunIrCompilerOptMain(argc, argv, opts);
  if (!s.ok()) {
    std::cerr << s << std::endl;
    return 1;
  }
  return 0;
}
