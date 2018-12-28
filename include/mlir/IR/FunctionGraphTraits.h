//===- CFGFunctionGraphTraits.h - llvm::GraphTraits for CFGs ----*- C++ -*-===//
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
// This file implements specializations of llvm::GraphTraits for various MLIR
// CFG data types.  This allows the generic LLVM graph algorithms to be applied
// to CFGs.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_CFGFUNCTIONGRAPHTRAITS_H
#define MLIR_IR_CFGFUNCTIONGRAPHTRAITS_H

#include "mlir/IR/Function.h"
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

template <> struct GraphTraits<const mlir::Block *> {
  using ChildIteratorType = mlir::Block::const_succ_iterator;
  using Node = const mlir::Block;
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

template <> struct GraphTraits<Inverse<const mlir::Block *>> {
  using ChildIteratorType = mlir::Block::const_pred_iterator;
  using Node = const mlir::Block;
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
struct GraphTraits<mlir::Function *> : public GraphTraits<mlir::Block *> {
  using GraphType = mlir::Function *;
  using NodeRef = mlir::Block *;

  static NodeRef getEntryNode(GraphType fn) { return &fn->front(); }

  using nodes_iterator = pointer_iterator<mlir::Function::iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn->end());
  }
};

template <>
struct GraphTraits<const mlir::Function *>
    : public GraphTraits<const mlir::Block *> {
  using GraphType = const mlir::Function *;
  using NodeRef = const mlir::Block *;

  static NodeRef getEntryNode(GraphType fn) { return &fn->front(); }

  using nodes_iterator = pointer_iterator<mlir::Function::const_iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn->end());
  }
};

template <>
struct GraphTraits<Inverse<mlir::Function *>>
    : public GraphTraits<Inverse<mlir::Block *>> {
  using GraphType = Inverse<mlir::Function *>;
  using NodeRef = NodeRef;

  static NodeRef getEntryNode(GraphType fn) { return &fn.Graph->front(); }

  using nodes_iterator = pointer_iterator<mlir::Function::iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn.Graph->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn.Graph->end());
  }
};

template <>
struct GraphTraits<Inverse<const mlir::Function *>>
    : public GraphTraits<Inverse<const mlir::Block *>> {
  using GraphType = Inverse<const mlir::Function *>;
  using NodeRef = NodeRef;

  static NodeRef getEntryNode(GraphType fn) { return &fn.Graph->front(); }

  using nodes_iterator = pointer_iterator<mlir::Function::const_iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn.Graph->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn.Graph->end());
  }
};

template <>
struct GraphTraits<mlir::BlockList *> : public GraphTraits<mlir::Block *> {
  using GraphType = mlir::BlockList *;
  using NodeRef = mlir::Block *;

  static NodeRef getEntryNode(GraphType fn) { return &fn->front(); }

  using nodes_iterator = pointer_iterator<mlir::Function::iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn->end());
  }
};

template <>
struct GraphTraits<const mlir::BlockList *>
    : public GraphTraits<const mlir::Block *> {
  using GraphType = const mlir::BlockList *;
  using NodeRef = const mlir::Block *;

  static NodeRef getEntryNode(GraphType fn) { return &fn->front(); }

  using nodes_iterator = pointer_iterator<mlir::Function::const_iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn->end());
  }
};

template <>
struct GraphTraits<Inverse<mlir::BlockList *>>
    : public GraphTraits<Inverse<mlir::Block *>> {
  using GraphType = Inverse<mlir::BlockList *>;
  using NodeRef = NodeRef;

  static NodeRef getEntryNode(GraphType fn) { return &fn.Graph->front(); }

  using nodes_iterator = pointer_iterator<mlir::Function::iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn.Graph->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn.Graph->end());
  }
};

template <>
struct GraphTraits<Inverse<const mlir::BlockList *>>
    : public GraphTraits<Inverse<const mlir::Block *>> {
  using GraphType = Inverse<const mlir::BlockList *>;
  using NodeRef = NodeRef;

  static NodeRef getEntryNode(GraphType fn) { return &fn.Graph->front(); }

  using nodes_iterator = pointer_iterator<mlir::Function::const_iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn.Graph->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn.Graph->end());
  }
};

} // namespace llvm

#endif
