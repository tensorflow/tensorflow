//===- ViewOpGraph.h - View/write op graphviz graphs ------------*- C++ -*-===//
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
// Defines interface to produce Graphviz outputs of MLIR op within block.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_VIEWOPGRAPH_H_
#define MLIR_TRANSFORMS_VIEWOPGRAPH_H_

#include "mlir/Support/LLVM.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
class Block;
class ModuleOp;
template <typename T> class OpPassBase;

/// Displays the graph in a window. This is for use from the debugger and
/// depends on Graphviz to generate the graph.
void viewGraph(Block &block, const Twine &name, bool shortNames = false,
               const Twine &title = "",
               llvm::GraphProgram::Name program = llvm::GraphProgram::DOT);

raw_ostream &writeGraph(raw_ostream &os, Block &block, bool shortNames = false,
                        const Twine &title = "");

/// Creates a pass to print op graphs.
std::unique_ptr<OpPassBase<ModuleOp>>
createPrintOpGraphPass(raw_ostream &os = llvm::errs(), bool shortNames = false,
                       const Twine &title = "");

} // end namespace mlir

#endif // MLIR_TRANSFORMS_VIEWOPGRAPH_H_
