//===- RegionGraphTraits.h - llvm::GraphTraits for CFGs ---------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements specializations of llvm::GraphTraits for various MLIR
// CFG data types.  This allows the generic LLVM graph algorithms to be applied
// to CFGs.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_REGIONGRAPHTRAITS_H
#define MLIR_IR_REGIONGRAPHTRAITS_H

#include "mlir/IR/Region.h"
#include "llvm/ADT/GraphTraits.h"

namespace llvm {
template <> struct GraphTraits<mlir::Block *> {
  using ChildIteratorType = mlir::Block::succ_iterator;
  using Node = mlir::Block;
  using NodeRef = Node *;

  static NodeRef getEntryNode(NodeRef bb) { return bb; }

  static ChildIteratorType child_begin(NodeRef node) {
    return node->succ_begin();
  }
  static ChildIteratorType child_end(NodeRef node) { return node->succ_end(); }
};

template <> struct GraphTraits<Inverse<mlir::Block *>> {
  using ChildIteratorType = mlir::Block::pred_iterator;
  using Node = mlir::Block;
  using NodeRef = Node *;
  static NodeRef getEntryNode(Inverse<NodeRef> inverseGraph) {
    return inverseGraph.Graph;
  }
  static inline ChildIteratorType child_begin(NodeRef node) {
    return node->pred_begin();
  }
  static inline ChildIteratorType child_end(NodeRef node) {
    return node->pred_end();
  }
};

template <>
struct GraphTraits<mlir::Region *> : public GraphTraits<mlir::Block *> {
  using GraphType = mlir::Region *;
  using NodeRef = mlir::Block *;

  static NodeRef getEntryNode(GraphType fn) { return &fn->front(); }

  using nodes_iterator = pointer_iterator<mlir::Region::iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn->end());
  }
};

template <>
struct GraphTraits<Inverse<mlir::Region *>>
    : public GraphTraits<Inverse<mlir::Block *>> {
  using GraphType = Inverse<mlir::Region *>;
  using NodeRef = NodeRef;

  static NodeRef getEntryNode(GraphType fn) { return &fn.Graph->front(); }

  using nodes_iterator = pointer_iterator<mlir::Region::iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn.Graph->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn.Graph->end());
  }
};

} // namespace llvm

#endif
