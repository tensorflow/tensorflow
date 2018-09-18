//===- mlir-translate.cpp - MLIR Translate Driver -------------------------===//
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
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "mlir-translate.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

static llvm::cl::opt<std::string>
    translationRequested(llvm::cl::Positional,
                         llvm::cl::desc("<translation-requested>"));

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

// Static map between translations registered and the TranslateFunctions that
// perform those translations.
llvm::ManagedStatic<llvm::StringMap<TranslateFunction>> translations;

TranslateRegistration::TranslateRegistration(
    llvm::StringRef name, const TranslateFunction &function) {
  if (translations->find(name) != translations->end())
    llvm::report_fatal_error("Attempting to overwrite an existing function");
  assert(function && "Attempting to register an empty translate function");
  (*translations)[name] = function;
}

TranslateFunction getTranslation(llvm::StringRef name) {
  auto it = translations->find(name);
  if (it == translations->end())
    return nullptr;
  return it->second;
}

extern void initializeMLIRContext(MLIRContext *ctx);

Module *mlir::parseMLIRInput(StringRef inputFilename, MLIRContext *context) {
  // Set up the input file.
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file '" << inputFilename
                 << "': " << error.message();
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  return parseSourceFile(sourceMgr, context);
}

bool mlir::printMLIROutput(const Module &module,
                           llvm::StringRef outputFilename) {
  std::error_code error;
  auto result = llvm::make_unique<llvm::ToolOutputFile>(outputFilename, error,
                                                        llvm::sys::fs::F_None);
  if (error) {
    llvm::errs() << error.message();
    return true;
  }
  module.print(result->os());
  result->keep();
  return false;
}

// Example translation registration. This performs a MLIR to MLIR "translation"
// which simply parses and prints the MLIR input file.
static TranslateRegistration MLIRToMLIRTranslate(
    "mlir-to-mlir", [](StringRef inputFilename, StringRef outputFilename,
                       MLIRContext *context) {
      std::unique_ptr<Module> module(parseMLIRInput(inputFilename, context));
      if (!module)
        return true;

      std::error_code error;
      auto result = llvm::make_unique<llvm::ToolOutputFile>(
          outputFilename, error, llvm::sys::fs::F_None);
      return printMLIROutput(*module, outputFilename);
    });

// Returns a comma-separated sorted list of the registered translations.
static std::string registeredTranslationNames() {
  std::vector<StringRef> keys(translations->keys().begin(),
                              translations->keys().end());
  llvm::sort(keys);
  return llvm::join(keys, ", ");
}

int main(int argc, char **argv) {
  llvm::PrettyStackTraceProgram x(argc, argv);
  llvm::InitLLVM y(argc, argv);

  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "MLIR translation driver\n\nRegistered translations:\n\t" +
          registeredTranslationNames() + "\n");

  auto translate = getTranslation(translationRequested);
  if (!translate) {
    llvm::errs() << "Translation requested '" << translationRequested
                 << "' not registered\n";
    return 1;
  }

  MLIRContext context;
  initializeMLIRContext(&context);
  return translate(inputFilename, outputFilename, &context);
}
