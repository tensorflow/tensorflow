//===- ConstraintAnalysisGraphTraits.h - Traits for CAGs --------*- C++ -*-===//
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
