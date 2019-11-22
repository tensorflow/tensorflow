//===- TranslateClParser.h - Translations command line parser -------------===//
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
// This file contains custom command line parser for translations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/TranslateClParser.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

// Storage for the translation function wrappers that survive the parser.
static llvm::SmallVector<TranslateFunction, 16> wrapperStorage;

static LogicalResult printMLIROutput(ModuleOp module, llvm::raw_ostream &os) {
  if (failed(verify(module)))
    return failure();
  module.print(os);
  return success();
}

TranslationParser::TranslationParser(llvm::cl::Option &opt)
    : llvm::cl::parser<const TranslateFunction *>(opt) {
  const auto &toMLIRRegistry = getTranslationToMLIRRegistry();
  const auto &fromMLIRRegistry = getTranslationFromMLIRRegistry();
  const auto &fileToFileRegistry = getTranslationRegistry();

  // Reserve the required capacity upfront so that pointers are not
  // invalidated on reallocation.
  wrapperStorage.reserve(toMLIRRegistry.size() + fromMLIRRegistry.size() +
                         fileToFileRegistry.size());
  for (const auto &kv : toMLIRRegistry) {
    TranslateSourceMgrToMLIRFunction function = kv.second;
    TranslateFunction wrapper = [function](llvm::SourceMgr &sourceMgr,
                                           llvm::raw_ostream &output,
                                           MLIRContext *context) {
      OwningModuleRef module = function(sourceMgr, context);
      if (!module)
        return failure();
      return printMLIROutput(*module, output);
    };
    wrapperStorage.emplace_back(std::move(wrapper));

    addLiteralOption(kv.first(), &wrapperStorage.back(), kv.first());
  }

  for (const auto &kv : fromMLIRRegistry) {
    TranslateFromMLIRFunction function = kv.second;
    TranslateFunction wrapper = [function](llvm::SourceMgr &sourceMgr,
                                           llvm::raw_ostream &output,
                                           MLIRContext *context) {
      auto module = OwningModuleRef(parseSourceFile(sourceMgr, context));
      if (!module)
        return failure();
      return function(module.get(), output);
    };
    wrapperStorage.emplace_back(std::move(wrapper));

    addLiteralOption(kv.first(), &wrapperStorage.back(), kv.first());
  }
  for (const auto &kv : fileToFileRegistry) {
    wrapperStorage.emplace_back(kv.second);
    addLiteralOption(kv.first(), &wrapperStorage.back(), kv.first());
  }
}

void TranslationParser::printOptionInfo(const llvm::cl::Option &O,
                                        size_t GlobalWidth) const {
  TranslationParser *TP = const_cast<TranslationParser *>(this);
  llvm::array_pod_sort(TP->Values.begin(), TP->Values.end(),
                       [](const TranslationParser::OptionInfo *VT1,
                          const TranslationParser::OptionInfo *VT2) {
                         return VT1->Name.compare(VT2->Name);
                       });
  using llvm::cl::parser;
  parser<const TranslateFunction *>::printOptionInfo(O, GlobalWidth);
}
