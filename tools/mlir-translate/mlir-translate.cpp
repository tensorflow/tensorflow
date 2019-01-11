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

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Translation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static Module *parseMLIRInput(StringRef inputFilename, MLIRContext *context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  return parseSourceFile(sourceMgr, context);
}

static bool printMLIROutput(const Module &module,
                            llvm::StringRef outputFilename) {
  auto file = openOutputFile(outputFilename);
  if (!file)
    return true;
  module.print(file->os());
  file->keep();
  return false;
}

// Common interface for source-to-source translation functions.
using TranslateFunction =
    std::function<bool(StringRef, StringRef, MLIRContext *)>;

// Storage for the translation function wrappers that survive the parser.
static llvm::SmallVector<TranslateFunction, 8> wrapperStorage;

// Custom parser for TranslateFunction.
// Wraps TranslateToMLIRFunctions and TranslateFromMLIRFunctions into
// TranslateFunctions before registering them as options.
struct TranslationParser : public llvm::cl::parser<const TranslateFunction *> {
  TranslationParser(llvm::cl::Option &opt)
      : llvm::cl::parser<const TranslateFunction *>(opt) {
    for (const auto &kv : getTranslationToMLIRRegistry()) {
      TranslateToMLIRFunction function = kv.second;
      TranslateFunction wrapper = [function](StringRef inputFilename,
                                             StringRef outputFilename,
                                             MLIRContext *context) {
        std::unique_ptr<Module> module = function(inputFilename, context);
        if (!module)
          return true;
        printMLIROutput(*module, outputFilename);
        return false;
      };
      wrapperStorage.emplace_back(std::move(wrapper));

      addLiteralOption(kv.first(), &wrapperStorage.back(), kv.first());
    }
    for (const auto &kv : getTranslationFromMLIRRegistry()) {
      TranslateFromMLIRFunction function = kv.second;
      TranslateFunction wrapper = [function](StringRef inputFilename,
                                             StringRef outputFilename,
                                             MLIRContext *context) {
        auto module =
            std::unique_ptr<Module>(parseMLIRInput(inputFilename, context));
        if (!module)
          return true;
        return function(module.get(), outputFilename);
      };
      wrapperStorage.emplace_back(std::move(wrapper));

      addLiteralOption(kv.first(), &wrapperStorage.back(), kv.first());
    }
  }

  void printOptionInfo(const llvm::cl::Option &O,
                       size_t GlobalWidth) const override {
    TranslationParser *TP = const_cast<TranslationParser *>(this);
    llvm::array_pod_sort(TP->Values.begin(), TP->Values.end(),
                         [](const TranslationParser::OptionInfo *VT1,
                            const TranslationParser::OptionInfo *VT2) {
                           return VT1->Name.compare(VT2->Name);
                         });
    using llvm::cl::parser;
    parser<const TranslateFunction *>::printOptionInfo(O, GlobalWidth);
  }
};

int main(int argc, char **argv) {
  llvm::PrettyStackTraceProgram x(argc, argv);
  llvm::InitLLVM y(argc, argv);

  // Add flags for all the registered translations.
  llvm::cl::opt<const TranslateFunction *, false, TranslationParser>
      translationRequested("", llvm::cl::desc("Translation to perform"),
                           llvm::cl::Required);
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR translation driver\n");

  MLIRContext context;
  return (*translationRequested)(inputFilename, outputFilename, &context);
}
