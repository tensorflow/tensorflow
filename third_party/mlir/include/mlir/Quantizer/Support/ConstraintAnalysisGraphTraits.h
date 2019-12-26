//===- ConstraintAnalysisGraphTraits.h - Traits for CAGs --------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides graph traits for constraint analysis graphs.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_QUANTIZER_SUPPORT_CONSTRAINTANALYSISGRAPHTRAITS_H
#define MLIR_QUANTIZER_SUPPORT_CONSTRAINTANALYSISGRAPHTRAITS_H

#include "mlir/Quantizer/Support/ConstraintAnalysisGraph.h"
#include "llvm/ADT/GraphTraits.h"

namespace llvm {

template <>
struct GraphTraits<const mlir::quantizer::CAGNode *> {
  using NodeRef = const mlir::quantizer::CAGNode *;

  static NodeRef getEntryNode(NodeRef node) { return node; }

  // Successors.
  using ChildIteratorType = mlir::quantizer::CAGNode::const_iterator;
  static ChildIteratorType child_begin(NodeRef node) { return node->begin(); }
  static ChildIteratorType child_end(NodeRef node) { return node->end(); }
};

template <>
struct GraphTraits<const mlir::quantizer::CAGSlice *>
    : public llvm::GraphTraits<const mlir::quantizer::CAGNode *> {
  using nodes_iterator = mlir::quantizer::CAGSlice::const_iterator;
  static mlir::quantizer::CAGSlice::const_iterator
  nodes_begin(const mlir::quantizer::CAGSlice *G) {
    return G->begin();
  }
  static mlir::quantizer::CAGSlice::const_iterator
  nodes_end(const mlir::quantizer::CAGSlice *G) {
    return G->end();
  }
};

} // end namespace llvm

#endif // MLIR_QUANTIZER_SUPPORT_CONSTRAINTANALYSISGRAPHTRAITS_H
