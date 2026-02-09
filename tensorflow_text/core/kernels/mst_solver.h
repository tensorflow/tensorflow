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

#ifndef TENSORFLOW_TEXT_CORE_KERNELS_MST_SOLVER_H_
#define TENSORFLOW_TEXT_CORE_KERNELS_MST_SOLVER_H_

#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_text/core/kernels/disjoint_set_forest.h"

namespace tensorflow {
namespace text {

// Maximum spanning tree solver for directed graphs.  Thread-compatible.
//
// The solver operates on a digraph of n nodes and m arcs and outputs a maximum
// spanning tree rooted at any node.  Scores can be associated with arcs and
// root selections, and the score of a tree is the sum of the relevant arc and
// root-selection scores.
//
// The implementation is based on:
//
//   go/tarjan-1977  google-only
//   R.E. Tarjan.  1977.  Finding Optimum Branchings.  Networks 7(1), pp. 25-35.
//   [In particular, see Section 4 "a modification for dense graphs"]
//
// which itself is an improvement of the Chu-Liu-Edmonds algorithm.  Note also
// the correction in:
//
//   go/camerini-1979  google-only
//   P.M. Camerini, L. Fratta, F. Maffioli.  1979.  A Note on Finding Optimum
//     Branchings.  Networks 9(4), pp. 309-312.
//
// The solver runs in O(n^2) time, which is optimal for dense digraphs but slow
// for sparse digraphs where O(m + n log n) can be achieved.  The solver uses
// O(n^2) space to store the digraph, which is also optimal for dense digraphs.
//
// Although this algorithm has an inferior asymptotic runtime on sparse graphs,
// it avoids high-constant-overhead data structures like Fibonacci heaps, which
// are required in the asymptotically faster algorithms.  Therefore, this solver
// may still be competitive on small sparse graphs.
//
// TODO(terrykoo): If we start running on large sparse graphs, implement the
// following, which runs in O(m + n log n):
//
//   go/tarjan-1986  google-only
//   H.N. Gabow, Z. Galil, T. Spencer, and R.E. Tarjan.  1986.  Efficient
//     algorithms for finding minimum spanning trees in undirected and directed
//     graphs.  Combinatorica, 6(2), pp. 109-122.
//
// Template args:
//   Index: An unsigned integral type wide enough to hold 2n.
//   Score: A signed arithmetic (integral or floating-point) type.
template <class Index, class Score>
class MstSolver {
 public:
  static_assert(std::is_integral<Index>::value, "Index must be integral");
  static_assert(!std::is_signed<Index>::value, "Index must be unsigned");
  static_assert(std::is_arithmetic<Score>::value, "Score must be arithmetic");
  static_assert(std::is_signed<Score>::value, "Score must be signed");
  using IndexType = Index;
  using ScoreType = Score;

  // Creates an empty solver.  Call Init() before use.
  MstSolver() = default;

  // Initializes this for a digraph with |num_nodes| nodes, or returns non-OK on
  // error.  Discards existing state; call AddArc() and AddRoot() to add arcs
  // and root selections.  If |forest| is true, then this solves for a maximum
  // spanning forest (i.e., a set of disjoint trees that span the digraph).
  absl::Status Init(bool forest, Index num_nodes);

  // Adds an arc from the |source| node to the |target| node with the |score|.
  // The |source| and |target| must be distinct node indices in [0,n), and the
  // |score| must be finite.  Calling this multiple times on the same |source|
  // and |target| overwrites the score instead of adding parallel arcs.
  void AddArc(Index source, Index target, Score score);

  // As above, but adds a root selection for the |root| node with the |score|.
  void AddRoot(Index root, Score score);

  // Returns the score of the arc from |source| to |target|, which must have
  // been added by a previous call to AddArc().
  Score ArcScore(Index source, Index target) const;

  // Returns the score of selecting the |root|, which must have been added by a
  // previous call to AddRoot().
  Score RootScore(Index root) const;

  // Populates |argmax| with the maximum directed spanning tree of the current
  // digraph, or returns non-OK on error.  The |argmax| array must contain at
  // least n elements.  On success, argmax[t] is the source of the arc directed
  // into t, or t itself if t is a root.
  //
  // NB: If multiple spanning trees achieve the maximum score, |argmax| will be
  // set to one of the maximal trees, but it is unspecified which one.
  absl::Status Solve(absl::Span<Index> argmax);

  // Convience method
  absl::Status Solve(std::vector<Index> *argmax) {
    return Solve(absl::MakeSpan(argmax->data(), argmax->size()));
  }

 private:
  // Implementation notes:
  //
  // The solver does not operate on the "original" digraph as specified by the
  // user, but a "transformed" digraph that differs as follows:
  //
  // * The transformed digraph adds an "artificial root" node at index 0 and
  //   offsets all original node indices by +1 to make room.  For each root
  //   selection, the artificial root has one outbound arc directed into the
  //   candidate root that carries the root-selection score.  The artificial
  //   root has no inbound arcs.
  //
  // * When solving for a spanning tree (i.e., when |forest_| is false), the
  //   outbound arcs of the artificial root are penalized to ensure that the
  //   artificial root has exactly one child.
  //
  // In the remainder of this file, all mentions of nodes, arcs, etc., refer to
  // the transformed digraph unless otherwise specified.
  //
  // The algorithm is divided into two phases, the "contraction phase" and the
  // "expansion phase".  The contraction phase finds the arcs that make up the
  // maximum spanning tree by applying a series of "contractions" which further
  // modify the digraph.  The expansion phase "expands" these modifications and
  // recovers the maximum spanning tree in the original digraph.
  //
  // During the contraction phase, the algorithm selects the best inbound arc
  // for each node.  These arcs can form cycles, which are "contracted" by
  // removing the cycle nodes and replacing them with a new contracted node.
  // Since each contraction removes 2 or more cycle nodes and adds 1 contracted
  // node, at most n-1 contractions will occur.  (The digraph initially contains
  // n+1 nodes, but one is the artificial root, which cannot form a cycle).
  //
  // When contracting a cycle, nodes are not explicitly removed and replaced.
  // Instead, a contracted node is appended to the digraph and the cycle nodes
  // are remapped to the contracted node, which implicitly removes and replaces
  // the cycle.  As a result, each contraction actually increases the size of
  // the digraph, up to a maximum of 2n nodes.  One advantage of adding and
  // remapping nodes is that it is convenient to recover the argmax spanning
  // tree during the expansion phase.
  //
  // Note that contractions can be nested, because the best inbound arc for a
  // contracted node may itelf form a cycle.  During the expansion phase, the
  // algorithm picks a root of the hierarchy of contracted nodes, breaks the
  // cycle it represents, and repeats until all cycles are broken.

  // Constants, as enums to avoid the need for static variable definitions.
  enum Constants : Index {
    // An index reserved for "null" values.
    kNullIndex = std::numeric_limits<Index>::max(),
  };

  // A possibly-nonexistent arc in the digraph.
  struct Arc {
    // Creates a nonexistent arc.
    Arc() = default;

    // Returns true if this arc exists.
    bool Exists() const { return target != 0; }

    // Returns true if this is a root-selection arc.
    bool IsRoot() const { return source == 0; }

    // Returns a string representation of this arc.
    std::string DebugString() const {
      if (!Exists()) return "[null]";
      if (IsRoot()) {
        return absl::StrCat("[*->", target, "=", score, "]");
      }
      return absl::StrCat("[", source, "->", target, "=", score, "]");
    }

    // Score of this arc.
    Score score;

    // Source of this arc in the initial digraph.
    Index source;

    // Target of this arc in the initial digraph, or 0 if this is nonexistent.
    Index target = 0;
  };

  // Returns the index, in |arcs_|, of the arc from |source| to |target|.  The
  // |source| must be one of the initial n+1 nodes.
  size_t ArcIndex(size_t source, size_t target) const;

  // Penalizes the root arc scores to ensure that this finds a tree, or does
  // nothing if |forest_| is true.  Must be called before ContractionPhase().
  void MaybePenalizeRootScoresForTree();

  // Returns the maximum inbound arc of the |node|, or null if there is none.
  const Arc *MaximumInboundArc(Index node) const;

  // Merges the inbound arcs of the |cycle_node| into the inbound arcs of the
  // |contracted_node|.  Arcs are merged as follows:
  // * If the source and target of the arc belong to the same strongly-connected
  //   component, it is ignored.
  // * If exactly one of the nodes had an arc from some source, then on exit the
  //   |contracted_node| has that arc.
  // * If both of the nodes had an arc from the same source, then on exit the
  //   |contracted_node| has the better-scoring arc.
  // The |score_offset| is added to the arc scores of the |cycle_node| before
  // they are merged into the |contracted_node|.
  void MergeInboundArcs(Index cycle_node, Score score_offset,
                        Index contracted_node);

  // Contracts the cycle in |argmax_arcs_| that contains the |node|.
  void ContractCycle(Index node);

  // Runs the contraction phase of the solver, or returns non-OK on error.  This
  // phase finds the best inbound arc for each node, contracting cycles as they
  // are formed.  Stops when every node has selected an inbound arc and there
  // are no cycles.
  absl::Status ContractionPhase();

  // Runs the expansion phase of the solver, or returns non-OK on error.  This
  // phase expands each contracted node, breaks cycles, and populates |argmax|
  // with the maximum spanning tree.
  absl::Status ExpansionPhase(absl::Span<Index> argmax);

  // If true, solve for a spanning forest instead of a spanning tree.
  bool forest_ = false;

  // The number of nodes in the original digraph; i.e., n.
  Index num_original_nodes_ = 0;

  // The number of nodes in the initial digraph; i.e., n+1.
  Index num_initial_nodes_ = 0;

  // The maximum number of possible nodes in the digraph; i.e., 2n.
  Index num_possible_nodes_ = 0;

  // The number of nodes in the current digraph, which grows from n+1 to 2n.
  Index num_current_nodes_ = 0;

  // Column-major |num_initial_nodes_| x |num_current_nodes_| matrix of arcs,
  // where rows and columns correspond to source and target nodes.  Columns are
  // added as cycles are contracted into new nodes.
  //
  // TODO(terrykoo): It is possible to squeeze the nonexistent arcs out of each
  // column and run the algorithm with each column being a sorted list (sorted
  // by source node).  This is in fact the suggested representation in Tarjan
  // (1977).  This won't improve the asymptotic runtime but still might improve
  // speed in practice.  I haven't done this because it adds complexity versus
  // checking Arc::Exists() in a few loops.  Try this out when we can benchmark
  // this on real data.
  std::vector<Arc> arcs_;

  // Disjoint-set forests tracking the weakly-connected and strongly-connected
  // components of the initial digraph, based on the arcs in |argmax_arcs_|.
  // Weakly-connected components are used to detect cycles; strongly-connected
  // components are used to detect self-loops.
  DisjointSetForest<Index> weak_components_;
  DisjointSetForest<Index> strong_components_;

  // A disjoint-set forest that maps each node to the top-most contracted node
  // that contains it.  Nodes that have not been contracted map to themselves.
  // NB: This disjoint-set forest does not use union by rank so we can control
  // the outcome of a set union.  There will only be O(n) operations on this
  // instance, so the increased O(log n) cost of each operation is acceptable.
  DisjointSetForest<Index, false> contracted_nodes_;

  // An array that represents the history of cycle contractions, as follows:
  // * If contracted_into_[t] is |kNullIndex|, then t is deleted.
  // * If contracted_into_[t] is 0, then t is a "root" contracted node; i.e., t
  //   has not been contracted into another node.
  // * Otherwise, contracted_into_[t] is the node into which t was contracted.
  std::vector<Index> contracted_into_;

  // The maximum inbound arc for each node.  The first element is null because
  // the artificial root has no inbound arcs.
  std::vector<const Arc *> argmax_arcs_;

  // Workspace for ContractCycle(), which records the nodes and arcs in the
  // cycle being contracted.
  std::vector<std::pair<Index, const Arc *>> cycle_;
};

// Implementation details below.

template <class Index, class Score>
absl::Status MstSolver<Index, Score>::Init(bool forest, Index num_nodes) {
  if (num_nodes <= 0) {
    return tensorflow::errors::InvalidArgument("Non-positive number of nodes: ",
                                               num_nodes);
  }

  // Upcast to size_t to avoid overflow.
  if (2 * static_cast<size_t>(num_nodes) >= static_cast<size_t>(kNullIndex)) {
    return tensorflow::errors::InvalidArgument("Too many nodes: ", num_nodes);
  }

  forest_ = forest;
  num_original_nodes_ = num_nodes;
  num_initial_nodes_ = num_original_nodes_ + 1;
  num_possible_nodes_ = 2 * num_original_nodes_;
  num_current_nodes_ = num_initial_nodes_;

  // Allocate the full n+1 x 2n matrix, but start with a n+1 x n+1 prefix.
  const size_t num_initial_arcs = static_cast<size_t>(num_initial_nodes_) *
                                  static_cast<size_t>(num_initial_nodes_);
  const size_t num_possible_arcs = static_cast<size_t>(num_initial_nodes_) *
                                   static_cast<size_t>(num_possible_nodes_);
  arcs_.reserve(num_possible_arcs);
  arcs_.assign(num_initial_arcs, {});

  weak_components_.Init(num_initial_nodes_);
  strong_components_.Init(num_initial_nodes_);
  contracted_nodes_.Init(num_possible_nodes_);
  contracted_into_.assign(num_possible_nodes_, 0);
  argmax_arcs_.assign(num_possible_nodes_, nullptr);

  // This doesn't need to be cleared now; it will be cleared before use.
  cycle_.reserve(num_original_nodes_);

  return absl::OkStatus();
}

template <class Index, class Score>
void MstSolver<Index, Score>::AddArc(Index source, Index target, Score score) {
  DCHECK_NE(source, target);
  DCHECK(std::isfinite(score));
  Arc &arc = arcs_[ArcIndex(source + 1, target + 1)];
  arc.score = score;
  arc.source = source + 1;
  arc.target = target + 1;
}

template <class Index, class Score>
void MstSolver<Index, Score>::AddRoot(Index root, Score score) {
  DCHECK(std::isfinite(score));
  Arc &arc = arcs_[ArcIndex(0, root + 1)];
  arc.score = score;
  arc.source = 0;
  arc.target = root + 1;
}

template <class Index, class Score>
Score MstSolver<Index, Score>::ArcScore(Index source, Index target) const {
  const Arc &arc = arcs_[ArcIndex(source + 1, target + 1)];
  DCHECK(arc.Exists());
  return arc.score;
}

template <class Index, class Score>
Score MstSolver<Index, Score>::RootScore(Index root) const {
  const Arc &arc = arcs_[ArcIndex(0, root + 1)];
  DCHECK(arc.Exists());
  return arc.score;
}

template <class Index, class Score>
absl::Status MstSolver<Index, Score>::Solve(absl::Span<Index> argmax) {
  MaybePenalizeRootScoresForTree();
  TF_RETURN_IF_ERROR(ContractionPhase());
  TF_RETURN_IF_ERROR(ExpansionPhase(argmax));
  return absl::OkStatus();
}

template <class Index, class Score>
inline size_t MstSolver<Index, Score>::ArcIndex(size_t source,
                                                size_t target) const {
  DCHECK_LT(source, num_initial_nodes_);
  DCHECK_LT(target, num_current_nodes_);
  return source + target * static_cast<size_t>(num_initial_nodes_);
}

template <class Index, class Score>
void MstSolver<Index, Score>::MaybePenalizeRootScoresForTree() {
  if (forest_) return;
  DCHECK_EQ(num_current_nodes_, num_initial_nodes_)
      << "Root penalties must be applied before starting the algorithm.";

  // Find the minimum and maximum arc scores.  These allow us to bound the range
  // of possible tree scores.
  Score max_score = std::numeric_limits<Score>::lowest();
  Score min_score = std::numeric_limits<Score>::max();
  for (const Arc &arc : arcs_) {
    if (!arc.Exists()) continue;
    max_score = std::max(max_score, arc.score);
    min_score = std::min(min_score, arc.score);
  }

  // Nothing to do, no existing arcs.
  if (max_score < min_score) return;

  // A spanning tree or forest contains n arcs.  The penalty below ensures that
  // every structure with one root has a higher score than every structure with
  // two roots, and so on.
  const Score root_penalty = 1 + num_initial_nodes_ * (max_score - min_score);
  for (Index root = 1; root < num_initial_nodes_; ++root) {
    Arc &arc = arcs_[ArcIndex(0, root)];
    if (!arc.Exists()) continue;
    arc.score -= root_penalty;
  }
}

template <class Index, class Score>
const typename MstSolver<Index, Score>::Arc *
MstSolver<Index, Score>::MaximumInboundArc(Index node) const {
  const Arc *__restrict arc = &arcs_[ArcIndex(0, node)];
  const Arc *arc_end = arc + num_initial_nodes_;

  Score max_score = std::numeric_limits<Score>::lowest();
  const Arc *argmax_arc = nullptr;
  for (; arc < arc_end; ++arc) {
    if (!arc->Exists()) continue;
    const Score score = arc->score;
    if (max_score <= score) {
      max_score = score;
      argmax_arc = arc;
    }
  }
  return argmax_arc;
}

template <class Index, class Score>
void MstSolver<Index, Score>::MergeInboundArcs(Index cycle_node,
                                               Score score_offset,
                                               Index contracted_node) {
  const Arc *__restrict cycle_arc = &arcs_[ArcIndex(0, cycle_node)];
  const Arc *cycle_arc_end = cycle_arc + num_initial_nodes_;
  Arc *__restrict contracted_arc = &arcs_[ArcIndex(0, contracted_node)];

  for (; cycle_arc < cycle_arc_end; ++cycle_arc, ++contracted_arc) {
    if (!cycle_arc->Exists()) continue;  // nothing to merge

    // Skip self-loops; they are useless because they cannot be used to break
    // the cycle represented by the |contracted_node|.
    if (strong_components_.SameSet(cycle_arc->source, cycle_arc->target)) {
      continue;
    }

    // Merge the |cycle_arc| into the |contracted_arc|.
    const Score cycle_score = cycle_arc->score + score_offset;
    if (!contracted_arc->Exists() || contracted_arc->score < cycle_score) {
      contracted_arc->score = cycle_score;
      contracted_arc->source = cycle_arc->source;
      contracted_arc->target = cycle_arc->target;
    }
  }
}

template <class Index, class Score>
void MstSolver<Index, Score>::ContractCycle(Index node) {
  // Append a new node for the contracted cycle.
  const Index contracted_node = num_current_nodes_++;
  DCHECK_LE(num_current_nodes_, num_possible_nodes_);
  arcs_.resize(arcs_.size() + num_initial_nodes_);

  // We make two passes through the cycle.  The first pass updates everything
  // except the |arcs_|, and the second pass updates the |arcs_|.  The |arcs_|
  // must be updated in a second pass because MergeInboundArcs() requires that
  // the |strong_components_| are updated with the newly-contracted cycle.
  cycle_.clear();
  Index cycle_node = node;
  do {
    // Gather the nodes and arcs in |cycle_| for the second pass.
    const Arc *cycle_arc = argmax_arcs_[cycle_node];
    DCHECK(!cycle_arc->IsRoot()) << cycle_arc->DebugString();
    cycle_.emplace_back(cycle_node, cycle_arc);

    // Mark the cycle nodes as members of a strongly-connected component.
    strong_components_.Union(cycle_arc->source, cycle_arc->target);

    // Mark the cycle nodes as members of the new contracted node.  Juggling is
    // required because |contracted_nodes_| also determines the next cycle node.
    const Index next_node = contracted_nodes_.FindRoot(cycle_arc->source);
    contracted_nodes_.UnionOfRoots(cycle_node, contracted_node);
    contracted_into_[cycle_node] = contracted_node;
    cycle_node = next_node;

    // When the cycle repeats, |cycle_node| will be equal to |contracted_node|,
    // not |node|, because the first iteration of this loop mapped |node| to
    // |contracted_node| in |contracted_nodes_|.
  } while (cycle_node != contracted_node);

  // Merge the inbound arcs of each cycle node into the |contracted_node|.
  for (const auto &node_and_arc : cycle_) {
    // Set the |score_offset| to the cost of breaking the cycle by replacing the
    // arc currently directed into the |cycle_node|.
    const Index cycle_node = node_and_arc.first;
    const Score score_offset = -node_and_arc.second->score;
    MergeInboundArcs(cycle_node, score_offset, contracted_node);
  }
}

template <class Index, class Score>
absl::Status MstSolver<Index, Score>::ContractionPhase() {
  // Skip the artificial root since it has no inbound arcs.
  for (Index target = 1; target < num_current_nodes_; ++target) {
    // Find the maximum inbound arc for the current |target|, if any.
    const Arc *arc = MaximumInboundArc(target);
    if (arc == nullptr) {
      return tensorflow::errors::FailedPrecondition("Infeasible digraph");
    }
    argmax_arcs_[target] = arc;

    // The articifial root cannot be part of a cycle, so we do not need to check
    // for cycles or even update its membership in the connected components.
    if (arc->IsRoot()) continue;

    // Since every node has at most one selected inbound arc, cycles can be
    // detected using weakly-connected components.
    const Index source_component = weak_components_.FindRoot(arc->source);
    const Index target_component = weak_components_.FindRoot(arc->target);
    if (source_component == target_component) {
      // Cycle detected; contract it into a new node.
      ContractCycle(target);
    } else {
      // No cycles, just update the weakly-connected components.
      weak_components_.UnionOfRoots(source_component, target_component);
    }
  }

  return absl::OkStatus();
}

template <class Index, class Score>
absl::Status MstSolver<Index, Score>::ExpansionPhase(absl::Span<Index> argmax) {
  if (argmax.size() < num_original_nodes_) {
    return tensorflow::errors::InvalidArgument(
        "Argmax array too small: ", num_original_nodes_,
        " elements required, but got ", argmax.size());
  }

  // Select and expand a root contracted node until no contracted nodes remain.
  // Thanks to the (topological) order in which contracted nodes are appended,
  // root contracted nodes are easily enumerated using a backward scan.  After
  // this loop, entries [1,n] of |argmax_arcs_| provide the arcs of the maximum
  // spanning tree.
  for (Index i = num_current_nodes_ - 1; i >= num_initial_nodes_; --i) {
    if (contracted_into_[i] == kNullIndex) continue;  // already deleted
    const Index root = i;  // if not deleted, must be a root due to toposorting

    // Copy the cycle-breaking arc to its specified target.
    const Arc *arc = argmax_arcs_[root];
    argmax_arcs_[arc->target] = arc;

    // The |arc| not only breaks the cycle associated with the |root|, but also
    // breaks every nested cycle between the |root| and the target of the |arc|.
    // Delete the contracted nodes corresponding to all broken cycles.
    Index node = contracted_into_[arc->target];
    while (node != kNullIndex && node != root) {
      const Index parent = contracted_into_[node];
      contracted_into_[node] = kNullIndex;
      node = parent;
    }
  }

  // Copy the spanning tree from |argmax_arcs_| to |argmax|.  Also count roots
  // for validation below.
  Index num_roots = 0;
  for (Index target = 0; target < num_original_nodes_; ++target) {
    const Arc &arc = *argmax_arcs_[target + 1];
    DCHECK_EQ(arc.target, target + 1) << arc.DebugString();
    if (arc.IsRoot()) {
      ++num_roots;
      argmax[target] = target;
    } else {
      argmax[target] = arc.source - 1;
    }
  }
  DCHECK_GE(num_roots, 1);

  // Even when |forest_| is false, |num_roots| can still be more than 1.  While
  // the root score penalty discourages structures with multiple root arcs, it
  // is not a hard constraint.  For example, if the original digraph contained
  // one root selection per node and no other arcs, the solver would incorrectly
  // produce an all-root structure in spite of the root score penalty.  As this
  // example illustrates, however, |num_roots| will be more than 1 if and only
  // if the original digraph is infeasible for trees.
  if (!forest_ && num_roots != 1) {
    return tensorflow::errors::FailedPrecondition("Infeasible digraph");
  }

  return absl::OkStatus();
}

}  // namespace text
}  // namespace tensorflow

#endif  // TENSORFLOW_TEXT_CORE_KERNELS_MST_SOLVER_H_
