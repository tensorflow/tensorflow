//===- GenNameParser.h - Command line parser for generators -----*- C++ -*-===//
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
// The GenNameParser class adds all passes linked in to the system that are
// creatable to the tool.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_GENNAMEPARSER_H_
#define MLIR_SUPPORT_GENNAMEPARSER_H_

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

#endif // MLIR_SUPPORT_GENNAMEPARSER_H_
