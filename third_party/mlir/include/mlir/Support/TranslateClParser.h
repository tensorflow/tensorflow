//===- TranslateClParser.h - Translations command line parser ---*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
