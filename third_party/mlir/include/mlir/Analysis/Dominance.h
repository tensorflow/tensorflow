//===- Dominance.h - Dominator analysis for CFGs ----------------*- C++ -*-===//
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

#include "mlir/IR/RegionGraphTraits.h"
#include "llvm/Support/GenericDomTree.h"

extern template class llvm::DominatorTreeBase<mlir::Block, false>;
extern template class llvm::DominatorTreeBase<mlir::Block, true>;

namespace mlir {
using DominanceInfoNode = llvm::DomTreeNodeBase<Block>;
class Operation;

namespace detail {
template <bool IsPostDom> class DominanceInfoBase {
  using base = llvm::DominatorTreeBase<Block, IsPostDom>;

public:
  DominanceInfoBase(Operation *op) { recalculate(op); }
  DominanceInfoBase(DominanceInfoBase &&) = default;
  DominanceInfoBase &operator=(DominanceInfoBase &&) = default;

  DominanceInfoBase(const DominanceInfoBase &) = delete;
  DominanceInfoBase &operator=(const DominanceInfoBase &) = delete;

  /// Recalculate the dominance info.
  void recalculate(Operation *op);

  /// Get the root dominance node of the given region.
  DominanceInfoNode *getRootNode(Region *region) {
    assert(dominanceInfos.count(region) != 0);
    return dominanceInfos[region]->getRootNode();
  }

protected:
  using super = DominanceInfoBase<IsPostDom>;

  /// Return true if the specified block A properly dominates block B.
  bool properlyDominates(Block *a, Block *b);

  /// A mapping of regions to their base dominator tree.
  DenseMap<Region *, std::unique_ptr<base>> dominanceInfos;
};
} // end namespace detail

/// A class for computing basic dominance information.
class DominanceInfo : public detail::DominanceInfoBase</*IsPostDom=*/false> {
public:
  using super::super;

  /// Return true if operation A properly dominates operation B.
  bool properlyDominates(Operation *a, Operation *b);

  /// Return true if operation A dominates operation B.
  bool dominates(Operation *a, Operation *b) {
    return a == b || properlyDominates(a, b);
  }

  /// Return true if value A properly dominates operation B.
  bool properlyDominates(Value a, Operation *b);

  /// Return true if operation A dominates operation B.
  bool dominates(Value a, Operation *b) {
    return (Operation *)a->getDefiningOp() == b || properlyDominates(a, b);
  }

  /// Return true if the specified block A dominates block B.
  bool dominates(Block *a, Block *b) {
    return a == b || properlyDominates(a, b);
  }

  /// Return true if the specified block A properly dominates block B.
  bool properlyDominates(Block *a, Block *b) {
    return super::properlyDominates(a, b);
  }

  /// Return the dominance node from the Region containing block A.
  DominanceInfoNode *getNode(Block *a);

  /// Update the internal DFS numbers for the dominance nodes.
  void updateDFSNumbers();
};

/// A class for computing basic postdominance information.
class PostDominanceInfo : public detail::DominanceInfoBase</*IsPostDom=*/true> {
public:
  using super::super;

  /// Return true if operation A properly postdominates operation B.
  bool properlyPostDominates(Operation *a, Operation *b);

  /// Return true if operation A postdominates operation B.
  bool postDominates(Operation *a, Operation *b) {
    return a == b || properlyPostDominates(a, b);
  }

  /// Return true if the specified block A properly postdominates block B.
  bool properlyPostDominates(Block *a, Block *b) {
    return super::properlyDominates(a, b);
  }

  /// Return true if the specified block A postdominates block B.
  bool postDominates(Block *a, Block *b) {
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
