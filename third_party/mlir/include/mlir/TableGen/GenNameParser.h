//===- GenNameParser.h - Command line parser for generators -----*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The GenNameParser class adds all passes linked in to the system that are
// creatable to the tool.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENNAMEPARSER_H_
#define MLIR_TABLEGEN_GENNAMEPARSER_H_

#include "llvm/Support/CommandLine.h"

namespace mlir {
class GenInfo;

/// Adds command line option for each registered generator.
struct GenNameParser : public llvm::cl::parser<const GenInfo *> {
  GenNameParser(llvm::cl::Option &opt);

  void printOptionInfo(const llvm::cl::Option &O,
                       size_t GlobalWidth) const override;
};
} // end namespace mlir

#endif // MLIR_TABLEGEN_GENNAMEPARSER_H_
