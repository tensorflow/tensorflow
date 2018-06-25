//===- Parser.h - MLIR Parser Library Interface -----------------*- C++ -*-===//
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
// This file is contains the interface to the MLIR parser library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PARSER_H
#define MLIR_PARSER_H

#include <functional>

namespace llvm {
  class SourceMgr;
  class SMDiagnostic;
} // end namespace llvm

namespace mlir {
class Module;
class MLIRContext;

using SMDiagnosticHandlerTy = std::function<void(llvm::SMDiagnostic error)>;

/// This parses the file specified by the indicated SourceMgr and returns an
/// MLIR module if it was valid.  If not, the errorReporter is used to report
/// the error diagnostics and this function returns null.
Module *parseSourceFile(llvm::SourceMgr &sourceMgr, MLIRContext *context,
                        const SMDiagnosticHandlerTy &errorReporter);

} // end namespace mlir

#endif // MLIR_PARSER_H
