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
template <> struct GraphTraits<mlir::BasicBlock *> {
  using ChildIteratorType = mlir::BasicBlock::succ_iterator;
  using Node = mlir::BasicBlock;
  using NodeRef = Node *;

  static NodeRef getEntryNode(NodeRef bb) { return bb; }

  static ChildIteratorType child_begin(NodeRef node) {
    return node->succ_begin();
  }
  static ChildIteratorType child_end(NodeRef node) { return node->succ_end(); }
};

template <> struct GraphTraits<const mlir::BasicBlock *> {
  using ChildIteratorType = mlir::BasicBlock::const_succ_iterator;
  using Node = const mlir::BasicBlock;
  using NodeRef = Node *;

  static NodeRef getEntryNode(NodeRef bb) { return bb; }

  static ChildIteratorType child_begin(NodeRef node) {
    return node->succ_begin();
  }
  static ChildIteratorType child_end(NodeRef node) { return node->succ_end(); }
};

template <> struct GraphTraits<Inverse<mlir::BasicBlock *>> {
  using ChildIteratorType = mlir::BasicBlock::pred_iterator;
  using Node = mlir::BasicBlock;
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

template <> struct GraphTraits<Inverse<const mlir::BasicBlock *>> {
  using ChildIteratorType = mlir::BasicBlock::const_pred_iterator;
  using Node = const mlir::BasicBlock;
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
struct GraphTraits<mlir::CFGFunction *>
    : public GraphTraits<mlir::BasicBlock *> {
  using GraphType = mlir::CFGFunction *;
  using NodeRef = mlir::BasicBlock *;

  static NodeRef getEntryNode(GraphType fn) { return &fn->front(); }

  using nodes_iterator = pointer_iterator<mlir::CFGFunction::iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn->end());
  }
};

template <>
struct GraphTraits<const mlir::CFGFunction *>
    : public GraphTraits<const mlir::BasicBlock *> {
  using GraphType = const mlir::CFGFunction *;
  using NodeRef = const mlir::BasicBlock *;

  static NodeRef getEntryNode(GraphType fn) { return &fn->front(); }

  using nodes_iterator = pointer_iterator<mlir::CFGFunction::const_iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn->end());
  }
};

template <>
struct GraphTraits<Inverse<mlir::CFGFunction *>>
    : public GraphTraits<Inverse<mlir::BasicBlock *>> {
  using GraphType = Inverse<mlir::CFGFunction *>;
  using NodeRef = NodeRef;

  static NodeRef getEntryNode(GraphType fn) { return &fn.Graph->front(); }

  using nodes_iterator = pointer_iterator<mlir::CFGFunction::iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn.Graph->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn.Graph->end());
  }
};

template <>
struct GraphTraits<Inverse<const mlir::CFGFunction *>>
    : public GraphTraits<Inverse<const mlir::BasicBlock *>> {
  using GraphType = Inverse<const mlir::CFGFunction *>;
  using NodeRef = NodeRef;

  static NodeRef getEntryNode(GraphType fn) { return &fn.Graph->front(); }

  using nodes_iterator = pointer_iterator<mlir::CFGFunction::const_iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn.Graph->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn.Graph->end());
  }
};

template <>
struct GraphTraits<mlir::StmtBlockList *>
    : public GraphTraits<mlir::BasicBlock *> {
  using GraphType = mlir::StmtBlockList *;
  using NodeRef = mlir::BasicBlock *;

  static NodeRef getEntryNode(GraphType fn) { return &fn->front(); }

  using nodes_iterator = pointer_iterator<mlir::CFGFunction::iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn->end());
  }
};

template <>
struct GraphTraits<const mlir::StmtBlockList *>
    : public GraphTraits<const mlir::BasicBlock *> {
  using GraphType = const mlir::StmtBlockList *;
  using NodeRef = const mlir::BasicBlock *;

  static NodeRef getEntryNode(GraphType fn) { return &fn->front(); }

  using nodes_iterator = pointer_iterator<mlir::CFGFunction::const_iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn->end());
  }
};

template <>
struct GraphTraits<Inverse<mlir::StmtBlockList *>>
    : public GraphTraits<Inverse<mlir::BasicBlock *>> {
  using GraphType = Inverse<mlir::StmtBlockList *>;
  using NodeRef = NodeRef;

  static NodeRef getEntryNode(GraphType fn) { return &fn.Graph->front(); }

  using nodes_iterator = pointer_iterator<mlir::CFGFunction::iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn.Graph->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn.Graph->end());
  }
};

template <>
struct GraphTraits<Inverse<const mlir::StmtBlockList *>>
    : public GraphTraits<Inverse<const mlir::BasicBlock *>> {
  using GraphType = Inverse<const mlir::StmtBlockList *>;
  using NodeRef = NodeRef;

  static NodeRef getEntryNode(GraphType fn) { return &fn.Graph->front(); }

  using nodes_iterator = pointer_iterator<mlir::CFGFunction::const_iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn.Graph->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn.Graph->end());
  }
};

} // namespace llvm

#endif
