// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_TEXT_CORE_KERNELS_DISJOINT_SET_FOREST_H_
#define TENSORFLOW_TEXT_CORE_KERNELS_DISJOINT_SET_FOREST_H_

#include <stddef.h>

#include <type_traits>
#include <vector>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace text {

// An implementation of the disjoint-set forest data structure.  The universe of
// elements is the dense range of indices [0,n).  Thread-compatible.
//
// By default, this uses the path compression and union by rank optimizations,
// achieving near-constant runtime on all operations.  However, the user may
// disable the union by rank optimization, which allows the user to control how
// roots are selected when a union occurs.  When union by rank is disabled, the
// runtime of all operations increases to O(log n) amortized.
//
// Template args:
//   Index: An unsigned integral type wide enough to hold n.
//   kUseUnionByRank: Whether to use the union by rank optimization.
template <class Index, bool kUseUnionByRank = true>
class DisjointSetForest {
 public:
  static_assert(std::is_integral<Index>::value, "Index must be integral");
  static_assert(!std::is_signed<Index>::value, "Index must be unsigned");
  using IndexType = Index;

  // Creates an empty forest.
  DisjointSetForest() = default;

  // Initializes this to hold the elements [0,|size|), each initially in its own
  // singleton set.  Replaces existing state, if any.
  void Init(Index size);

  // Returns the root of the set containing |element|, which uniquely identifies
  // the set.  Note that the root of a set may change as the set is merged with
  // other sets; do not cache the return value of FindRoot(e) across calls to
  // Union() or UnionOfRoots() that could merge the set containing e.
  Index FindRoot(Index element);

  // For convenience, returns true if |element1| and |element2| are in the same
  // set.  When performing a large batch of queries it may be more efficient to
  // cache the value of FindRoot(), modulo caveats regarding caching above.
  bool SameSet(Index element1, Index element2);

  // Merges the sets rooted at |root1| and |root2|, which must be the roots of
  // their respective sets.  Either |root1| or |root2| will be the root of the
  // merged set.  If |kUseUnionByRank| is true, then it is unspecified whether
  // |root1| or |root2| will be the root; otherwise, |root2| will be the root.
  void UnionOfRoots(Index root1, Index root2);

  // As above, but for convenience finds the root of |element1| and |element2|.
  void Union(Index element1, Index element2);

  // The number of elements in this.
  Index size() const { return size_; }

 private:
  // The number of elements in the universe underlying the sets.
  Index size_ = 0;

  // The parent of each element, where self-loops are roots.
  std::vector<Index> parents_;

  // The rank of each element, for the union by rank optimization.  Only used if
  // |kUseUnionByRank| is true.
  std::vector<Index> ranks_;
};

// Implementation details below.

template <class Index, bool kUseUnionByRank>
void DisjointSetForest<Index, kUseUnionByRank>::Init(Index size) {
  size_ = size;
  parents_.resize(size_);
  if (kUseUnionByRank) ranks_.resize(size_);

  // Create singleton sets.
  for (Index i = 0; i < size_; ++i) {
    parents_[i] = i;
    if (kUseUnionByRank) ranks_[i] = 0;
  }
}

template <class Index, bool kUseUnionByRank>
Index DisjointSetForest<Index, kUseUnionByRank>::FindRoot(Index element) {
  DCHECK_LT(element, size());
  Index *const __restrict parents = parents_.data();

  // Walk up to the root of the |element|.  Unroll the first two comparisons
  // because path compression ensures most FindRoot() calls end there.  In
  // addition, if a root is found within the first two comparisons, then the
  // path compression updates can be skipped.
  Index current = element;
  Index parent = parents[current];
  if (current == parent) return current;  // |element| is a root
  current = parent;
  parent = parents[current];
  if (current == parent) return current;  // |element| is the child of a root
  do {  // otherwise, continue upwards until root
    current = parent;
    parent = parents[current];
  } while (current != parent);
  const Index root = current;

  // Apply path compression on the traversed nodes.
  current = element;
  parent = parents[current];  // not root, thanks to unrolling above
  do {
    parents[current] = root;
    current = parent;
    parent = parents[current];
  } while (parent != root);

  return root;
}

template <class Index, bool kUseUnionByRank>
bool DisjointSetForest<Index, kUseUnionByRank>::SameSet(Index element1,
                                                        Index element2) {
  return FindRoot(element1) == FindRoot(element2);
}

template <class Index, bool kUseUnionByRank>
void DisjointSetForest<Index, kUseUnionByRank>::UnionOfRoots(Index root1,
                                                             Index root2) {
  DCHECK_LT(root1, size());
  DCHECK_LT(root2, size());
  DCHECK_EQ(root1, parents_[root1]);
  DCHECK_EQ(root2, parents_[root2]);
  if (root1 == root2) return;  // already merged
  Index *const __restrict parents = parents_.data();

  if (kUseUnionByRank) {
    // Attach the lesser-rank root to the higher-rank root.
    Index *const __restrict ranks = ranks_.data();
    const Index rank1 = ranks[root1];
    const Index rank2 = ranks[root2];
    if (rank2 < rank1) {
      parents[root2] = root1;
    } else if (rank1 < rank2) {
      parents[root1] = root2;
    } else {
      // Equal ranks; choose one arbitrarily and promote its rank.
      parents[root1] = root2;
      ranks[root2] = rank2 + 1;
    }
  } else {
    // Always make |root2| the root of the merged set.
    parents[root1] = root2;
  }
}

template <class Index, bool kUseUnionByRank>
void DisjointSetForest<Index, kUseUnionByRank>::Union(Index element1,
                                                      Index element2) {
  UnionOfRoots(FindRoot(element1), FindRoot(element2));
}

}  // namespace text
}  // namespace tensorflow

#endif  // TENSORFLOW_TEXT_CORE_KERNELS_DISJOINT_SET_FOREST_H_
