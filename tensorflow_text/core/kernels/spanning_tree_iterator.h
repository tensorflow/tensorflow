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

#ifndef TENSORFLOW_TEXT_CORE_KERNELS_SPANNING_TREE_ITERATOR_H_
#define TENSORFLOW_TEXT_CORE_KERNELS_SPANNING_TREE_ITERATOR_H_

#include <vector>

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace text {

// A class that iterates over all possible spanning trees of a complete digraph.
// Thread-compatible.  Useful for brute-force comparison tests.
//
// TODO(terrykoo): Try using Prufer sequences, which are more efficient to
// enumerate as there are no non-trees to filter out.
class SpanningTreeIterator {
 public:
  // An array that provides the source of the inbound arc for each node.  Roots
  // are represented as self-loops.
  using SourceList = std::vector<uint32>;

  // Creates a spanning tree iterator.  If |forest| is true, then this iterates
  // over forests instead of trees (i.e., multiple roots are allowed).
  explicit SpanningTreeIterator(bool forest);

  // Applies the |functor| to all spanning trees (or forests, if |forest_| is
  // true) of a complete digraph containing |num_nodes| nodes.  Each tree is
  // passed to the |functor| as a SourceList.
  template <class Functor>
  void ForEachTree(uint32 num_nodes, Functor functor) {
    // Conveniently, the all-zero vector represents a valid tree.
    SourceList sources(num_nodes, 0);
    do {
      functor(sources);
    } while (NextTree(&sources));
  }

 private:
  // Returns true if the |sources| contains a cycle.
  bool HasCycle(const SourceList &sources);

  // Returns the number of roots in the |sources|.
  static uint32 NumRoots(const SourceList &sources);

  // Advances |sources| to the next source list, or returns false if there are
  // no more source lists.
  static bool NextSourceList(SourceList *sources);

  // Advances |sources| to the next tree (or forest, if |forest_| is true), or
  // returns false if there are no more trees.
  bool NextTree(SourceList *sources);

  // If true, iterate over spanning forests instead of spanning trees.
  const bool forest_;

  // Workspaces used by the search in HasCycle().
  std::vector<bool> searched_;
  std::vector<bool> visiting_;
};

}  // namespace text
}  // namespace tensorflow

#endif  // TENSORFLOW_TEXT_CORE_KERNELS_SPANNING_TREE_ITERATOR_H_
