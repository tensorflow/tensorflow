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

#include "mlir/IR/FunctionGraphTraits.h"
#include "llvm/Support/GenericDomTree.h"

extern template class llvm::DominatorTreeBase<mlir::Block, false>;
extern template class llvm::DominatorTreeBase<mlir::Block, true>;
extern template class llvm::DomTreeNodeBase<mlir::Block>;

namespace llvm {
namespace DomTreeBuilder {

using MLIRDomTree = llvm::DomTreeBase<mlir::Block>;
using MLIRPostDomTree = llvm::PostDomTreeBase<mlir::Block>;

// extern template void Calculate<MLIRDomTree>(MLIRDomTree &DT);
// extern template void Calculate<MLIRPostDomTree>(MLIRPostDomTree &DT);

} // namespace DomTreeBuilder
} // namespace llvm

namespace mlir {
using DominatorTreeBase = llvm::DominatorTreeBase<Block, false>;
using PostDominatorTreeBase = llvm::DominatorTreeBase<Block, true>;
using DominanceInfoNode = llvm::DomTreeNodeBase<Block>;

/// A class for computing basic dominance information.
class DominanceInfo : public DominatorTreeBase {
  using super = DominatorTreeBase;

public:
  DominanceInfo(Function *F);

  /// Return true if instruction A properly dominates instruction B.
  bool properlyDominates(const Instruction *a, const Instruction *b);

  /// Return true if instruction A dominates instruction B.
  bool dominates(const Instruction *a, const Instruction *b) {
    return a == b || properlyDominates(a, b);
  }

  /// Return true if value A properly dominates instruction B.
  bool properlyDominates(const Value *a, const Instruction *b);

  /// Return true if instruction A dominates instruction B.
  bool dominates(const Value *a, const Instruction *b) {
    return (Instruction *)a->getDefiningInst() == b || properlyDominates(a, b);
  }

  /// Return true if the specified block A properly dominates block B.
  bool properlyDominates(const Block *a, const Block *b);

  /// Return true if the specified block A dominates block B.
  bool dominates(const Block *a, const Block *b) {
    return a == b || properlyDominates(a, b);
  }
};

/// A class for computing basic postdominance information.
class PostDominanceInfo : public PostDominatorTreeBase {
  using super = PostDominatorTreeBase;

public:
  PostDominanceInfo(Function *F);

  /// Return true if instruction A properly postdominates instruction B.
  bool properlyPostDominates(const Instruction *a, const Instruction *b);

  /// Return true if instruction A postdominates instruction B.
  bool postDominates(const Instruction *a, const Instruction *b) {
    return a == b || properlyPostDominates(a, b);
  }
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
