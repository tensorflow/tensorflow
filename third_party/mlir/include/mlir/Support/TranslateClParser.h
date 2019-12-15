//===- TranslateClParser.h - Translations command line parser ---*- C++ -*-===//
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

#ifndef MLIR_SUPPORT_TRANSLATE_CL_PARSER_H_
#define MLIR_SUPPORT_TRANSLATE_CL_PARSER_H_

#include "mlir/Support/LLVM.h"
#include "mlir/Translation.h"
#include "llvm/Support/CommandLine.h"
#include <functional>

namespace mlir {

struct LogicalResult;
class MLIRContext;

/// Custom parser for TranslateFunction.
/// Wraps TranslateToMLIRFunctions and TranslateFromMLIRFunctions into
/// TranslateFunctions before registering them as options.
struct TranslationParser : public llvm::cl::parser<const TranslateFunction *> {
  TranslationParser(llvm::cl::Option &opt);

  void printOptionInfo(const llvm::cl::Option &O,
                       size_t GlobalWidth) const override;
};

} // namespace mlir

#endif // MLIR_SUPPORT_TRANSLATE_CL_PARSER_H_
