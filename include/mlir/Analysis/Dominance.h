//===- Dominance.h - Dominator analysis for CFG Functions -------*- C++ -*-===//
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

#ifndef MLIR_ANALYSIS_DOMINANCE_H
#define MLIR_ANALYSIS_DOMINANCE_H

#include "mlir/IR/CFGFunction.h"
#include "llvm/Support/GenericDomTree.h"

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
  static NodeRef getEntryNode(Inverse<mlir::BasicBlock *> inverseGraph) {
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

  static NodeRef getEntryNode(Inverse<const mlir::BasicBlock *> inverseGraph) {
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
} // namespace llvm

extern template class llvm::DominatorTreeBase<mlir::BasicBlock, false>;
extern template class llvm::DominatorTreeBase<mlir::BasicBlock, true>;
extern template class llvm::DomTreeNodeBase<mlir::BasicBlock>;

namespace llvm {
namespace DomTreeBuilder {

using MLIRDomTree = llvm::DomTreeBase<mlir::BasicBlock>;
using MLIRPostDomTree = llvm::PostDomTreeBase<mlir::BasicBlock>;

// extern template void Calculate<MLIRDomTree>(MLIRDomTree &DT);
// extern template void Calculate<MLIRPostDomTree>(MLIRPostDomTree &DT);

} // namespace DomTreeBuilder
} // namespace llvm

namespace mlir {
using DominatorTreeBase = llvm::DominatorTreeBase<BasicBlock, false>;
using PostDominatorTreeBase = llvm::DominatorTreeBase<BasicBlock, true>;
using DominanceInfoNode = llvm::DomTreeNodeBase<BasicBlock>;

/// A class for computing basic dominance information.
class DominanceInfo : public DominatorTreeBase {
  using super = DominatorTreeBase;

public:
  DominanceInfo(CFGFunction *F);

  /// Return true if instruction A properly dominates instruction B.
  bool properlyDominates(const Instruction *a, const Instruction *b);

  /// Return true if instruction A dominates instruction B.
  bool dominates(const Instruction *a, const Instruction *b) {
    return a == b || properlyDominates(a, b);
  }

  /// Return true if value A properly dominates instruction B.
  bool properlyDominates(const SSAValue *a, const Instruction *b);

  /// Return true if instruction A dominates instruction B.
  bool dominates(const SSAValue *a, const Instruction *b) {
    return a->getDefiningInst() == b || properlyDominates(a, b);
  }

  // dominates/properlyDominates for basic blocks.
  using DominatorTreeBase::dominates;
  using DominatorTreeBase::properlyDominates;
};

} //  end namespace mlir

namespace llvm {

/// DominatorTree GraphTraits specialization so the DominatorTree can be
/// iterated by generic graph iterators.
template <> struct GraphTraits<mlir::DominanceInfoNode *> {
  using ChildIteratorType = mlir::DominanceInfoNode::iterator;
  using NodeRef = mlir::DominanceInfoNode *;

  static NodeRef getEntryNode(NodeRef N) { return N; }
  static inline ChildIteratorType child_begin(NodeRef N) { return N->begin(); }
  static inline ChildIteratorType child_end(NodeRef N) { return N->end(); }
};

template <> struct GraphTraits<const mlir::DominanceInfoNode *> {
  using ChildIteratorType = mlir::DominanceInfoNode::const_iterator;
  using NodeRef = const mlir::DominanceInfoNode *;

  static NodeRef getEntryNode(NodeRef N) { return N; }
  static inline ChildIteratorType child_begin(NodeRef N) { return N->begin(); }
  static inline ChildIteratorType child_end(NodeRef N) { return N->end(); }
};

} // end namespace llvm
#endif
