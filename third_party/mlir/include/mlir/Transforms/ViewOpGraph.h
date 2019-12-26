//===- ViewOpGraph.h - View/write op graphviz graphs ------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
