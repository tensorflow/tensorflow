/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This module implements an online topological sort using the two-way-search
// algorithm for sparse graphs of Bender et al., Section 2. The algorithm
// incorporates the extension from section 4 to maintain the topological order
// explicitly in a doubly-linked list.
//
// All topological order metadata and links are stored in a contiguous array
// (std::vector) managed by TopologicalSort, indexed directly by the node's
// dense integer ID (IndexInParent).
//
// Per Bender et al, inserting m edges into a graph of n nodes takes
// O(m*min(m**(1/2), n**(2/3))). For the use case of our compiler IR, we assume
// that the number of edges is at most a small multiple of the number of nodes,
// and so the graph is quite sparse, and the dominant bound is O(m**(3/2)).
//
// We implement several extensions to the algorithm:
// - we allow adding and removing nodes. This does not require any significant
//   changes to the algorithm. The original algorithm uses the values of m and n
//   as part of a scheme for numbering nodes, but the purpose of that scheme is
//   to combine (level, index) tuples into a single total order. We don't need
//   explicit position numbers, only the topological order, so we can just use
//   a lexicographic order of (level, index) tuples directly.
// - we number indices decreasing from std::numeric_limits<int>::max(). The
//   careful numbering of indices in the original paper is only to avoid
//   collisions in the ID space with the level numbers, but since we don't try
//   to combine these into a single number, we don't need to be quite as
//   careful.
// - we allow removing edges. This is a trivial extension; removing an edge
//   preserves topological ordering. Removing edges may affect the algorithmic
//   complexity guarantees, but we probably don't care that much.
//
// This implementation is not thread-safe.
//
// Type parameters:
// - T is the type of the nodes in the graph.
// - Index is the type of the index_in_parent field in the nodes. We only care
//   that the index values form a reasonably dense range starting at 0, since
//   we use them to index into vectors. If we didn't have a dense range, we
//   could use an associative map data structure instead, but that would be
//   slower to lookup.
// - IndexInParent is a pointer to the index_in_parent field in T.
//   These indices must remain fixed only during a call to AddEdge(), which
//   is obviously true because we don't allow threads and the topological sort
//   will not change them, but they are allowed to change between calls (if
//   Reindex() is called to update the internal node table).
// - PredecessorIterator, PredecessorsBegin, PredecessorsEnd iterate over the
//   predecessors of the node. Duplicates are allowed.
// - SuccessorIterator, SuccessorsBegin, SuccessorsEnd iterate over the
//   successors of the node. Duplicates are allowed.
//
// References:
// * Bender, M.A., Fineman, J.T., Gilbert, S. and Tarjan, R.E., 2015. A new
//   approach to incremental cycle detection and related problems.
//   ACM Transactions on Algorithms (TALG), 12(2), pp.1-22.
//   https://dl.acm.org/doi/abs/10.1145/2756553

#ifndef XLA_ONLINE_TOPSORT_H_
#define XLA_ONLINE_TOPSORT_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/logging.h"

namespace xla {

template <typename T, typename Index, Index T::* IndexInParent,
          typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
class TopologicalSort;

template <typename Index>
class TopologicalSortForwardIterator;

template <typename Index>
class TopologicalSortReverseIterator;

// Internal node stored in the contiguous vector of TopologicalSort.
// The nodes form a circular doubly-linked list through `next` and `prev`
// indices into the vector (with kSentinel = 0 acting as the list head).
template <typename Index>
struct TopologicalSortInternalNode {
  Index next = 0;
  Index prev = 0;
  int level = -1;
  int index = -1;
};

// Iterator that traverses through the topological sort in order.
template <typename Index>
class TopologicalSortForwardIterator {
 public:
  using Node = TopologicalSortInternalNode<Index>;

  TopologicalSortForwardIterator() : nodes_(nullptr), current_(0) {}
  TopologicalSortForwardIterator(const std::vector<Node>* nodes, Index current)
      : nodes_(nodes), current_(current) {}

  TopologicalSortForwardIterator(const TopologicalSortForwardIterator&) =
      default;
  TopologicalSortForwardIterator(TopologicalSortForwardIterator&&) = default;
  TopologicalSortForwardIterator& operator=(
      const TopologicalSortForwardIterator&) = default;
  TopologicalSortForwardIterator& operator=(TopologicalSortForwardIterator&&) =
      default;

  Index operator*() const { return static_cast<Index>(current_ - 1); }

  bool operator==(const TopologicalSortForwardIterator& other) const {
    return current_ == other.current_;
  }
  bool operator!=(const TopologicalSortForwardIterator& other) const {
    return current_ != other.current_;
  }

  TopologicalSortForwardIterator& operator++() {
    current_ = (*nodes_)[current_].next;
    return *this;
  }

  TopologicalSortForwardIterator& operator--() {
    current_ = (*nodes_)[current_].prev;
    return *this;
  }

 private:
  const std::vector<Node>* nodes_;
  // Index into `nodes_` of the current node in the traversal (kSentinel = 0 at
  // end).
  Index current_;
};

// Iterator that traverses through the topological sort in reverse order.
template <typename Index>
class TopologicalSortReverseIterator {
 public:
  using Node = TopologicalSortInternalNode<Index>;

  TopologicalSortReverseIterator() : nodes_(nullptr), current_(0) {}
  TopologicalSortReverseIterator(const std::vector<Node>* nodes, Index current)
      : nodes_(nodes), current_(current) {}

  TopologicalSortReverseIterator(const TopologicalSortReverseIterator&) =
      default;
  TopologicalSortReverseIterator(TopologicalSortReverseIterator&&) = default;
  TopologicalSortReverseIterator& operator=(
      const TopologicalSortReverseIterator&) = default;
  TopologicalSortReverseIterator& operator=(TopologicalSortReverseIterator&&) =
      default;

  Index operator*() const { return static_cast<Index>(current_ - 1); }

  bool operator==(const TopologicalSortReverseIterator& other) const {
    return current_ == other.current_;
  }
  bool operator!=(const TopologicalSortReverseIterator& other) const {
    return current_ != other.current_;
  }

  TopologicalSortReverseIterator& operator++() {
    current_ = (*nodes_)[current_].prev;
    return *this;
  }

  TopologicalSortReverseIterator& operator--() {
    current_ = (*nodes_)[current_].next;
    return *this;
  }

 private:
  const std::vector<Node>* nodes_;
  // Index into `nodes_` of the current node in the traversal (kSentinel = 0 at
  // rend).
  Index current_;
};

template <typename T, typename Index, Index T::* IndexInParent,
          typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
class TopologicalSort {
 public:
  using Node = TopologicalSortInternalNode<Index>;
  static constexpr Index kSentinel = 0;

  TopologicalSort() {
    nodes_.emplace_back();  // nodes_[0] is kSentinel
    nodes_[kSentinel].next = kSentinel;
    nodes_[kSentinel].prev = kSentinel;
    nodes_[kSentinel].level = 0;
    first_in_level_.push_back(kSentinel);
  }

  ~TopologicalSort() = default;

  // Invalidates iterators.
  void AddNode(T* v);

  // Invalidates iterators.
  void RemoveNode(T* v);

  // Caution: this data structure assumes that there are no parallel edges.
  // Invalidates any iterators. We assume the user has added the edge to their
  // own data structure before calling this method.
  void AddEdge(T* v, T* w);

  // You might wonder why we don't have the following method:
  // void RemoveEdge(T* v, T* w);
  // The reason is that we don't need it. Removing an edge preserves topological
  // ordering, and there's nothing for us to do here. The user still needs to
  // remove the edge from their own data structure, of course.

  // Rebuilds the internal node table to reflect updated IndexInParent values
  // (e.g., after ID compaction) by remapping node IDs according to old_to_new
  // (where old_to_new[old_idx] is the new IndexInParent, or -1 if removed),
  // while preserving every node's exact level, index, and topological order.
  void Reindex(absl::Span<const Index> old_to_new);

  // Returns an iterator over the nodes in topological order.
  TopologicalSortForwardIterator<Index> begin() const {
    return TopologicalSortForwardIterator<Index>(&nodes_,
                                                 nodes_[kSentinel].next);
  }
  TopologicalSortForwardIterator<Index> end() const {
    return TopologicalSortForwardIterator<Index>(&nodes_, kSentinel);
  }

  // Returns an iterator over the nodes in reverse topological order.
  TopologicalSortReverseIterator<Index> rbegin() const {
    return TopologicalSortReverseIterator<Index>(&nodes_,
                                                 nodes_[kSentinel].prev);
  }
  TopologicalSortReverseIterator<Index> rend() const {
    return TopologicalSortReverseIterator<Index>(&nodes_, kSentinel);
  }

  template <typename Fn>
  void ForEachPostOrder(Fn&& fn) const {
    const Node* const data = nodes_.data();
    for (Index curr = data[kSentinel].prev; curr != kSentinel;
         curr = data[curr].prev) {
      fn(static_cast<Index>(curr - 1));
    }
  }

  // This is a helper for debugging. It logs the current order and checks a
  // number of invariants.
  void LogOrder() {
    std::vector<Index> order;
    int level = -1;
    for (Index idx : *this) {
      Index id = idx + 1;
      const auto& link = nodes_[id];
      CHECK_GE(link.level, level);
      level = link.level;
      CHECK(nodes_[link.next].prev == id);
      CHECK(nodes_[link.prev].next == id);
      order.push_back(idx);
    }
    auto node_formatter = [this](std::string* out, Index idx) {
      Index id = idx + 1;
      absl::StrAppend(out, idx, "[", nodes_[id].level, ":", nodes_[id].index,
                      "]");
    };
    DVLOG(2) << this << " order=" << absl::StrJoin(order, ", ", node_formatter);
    auto first_in_level_formatter = [this](std::string* out, Index v) {
      if (nodes_[v].next != kSentinel) {
        absl::StrAppend(out, nodes_[v].next - 1, ":",
                        nodes_[nodes_[v].next].level);
      } else {
        absl::StrAppend(out, "-:-");
      }
    };
    DVLOG(2) << this << " first_in_level_="
             << absl::StrJoin(first_in_level_, ", ", first_in_level_formatter);

    CHECK(first_in_level_[0] == kSentinel);
    auto it = order.begin();
    for (Index v : first_in_level_) {
      if (nodes_[v].next != kSentinel) {
        it = std::find(it, order.end(), static_cast<Index>(nodes_[v].next - 1));
        CHECK(it != order.end());
      }
    }
  }

  void clear() {
    nodes_.clear();
    nodes_.emplace_back();
    nodes_[kSentinel].next = kSentinel;
    nodes_[kSentinel].prev = kSentinel;
    nodes_[kSentinel].level = 0;
    num_edges_ = 0;
    num_nodes_ = 0;
    delta_ = 0;
    next_index_ = std::numeric_limits<int>::max();
    first_in_level_.clear();
    first_in_level_.push_back(kSentinel);
    visited_backwards_.clear();
    visited_backwards_nodes_.clear();
    visited_forwards_.clear();
    increased_.clear();
  }

 private:
  Index NodeId(const T* v) const {
    return static_cast<Index>(v->*IndexInParent) + 1;
  }

  // Returns true if this node has been added to a topological order.
  // It may have temporarily been removed from a specific location in that
  // order if we are in the middle of an AddEdge() operation.
  bool in_topological_order(const T* v) const {
    Index id = NodeId(v);
    return id > 0 && static_cast<size_t>(id) < nodes_.size() &&
           nodes_[id].level >= 0;
  }

  // Updates delta_ after we have increased num_edges_ and num_nodes_.
  // We don't bother decreasing delta_ after removals, since we assume that our
  // graphs will not significantly shrink.
  void UpdateDelta();

  // Performs a DFS backwards from v of at most delta_ nodes on the same level,
  // populating b with nodes in postorder with respect to the search (i.e., a
  // node appears later in b than its predecessors). Returns true if we should
  // run a forwards search.
  bool SearchBackwards(T* v, T* w, std::vector<T*>& b);

  // Performs a DFS forwards from v populating f with nodes in postorder with
  // respect to the search (i.e., a node appears later in f than all its
  // predecessors).
  // (Note "f" is reversed from the paper, which just because we can save time
  // and reverse it when updating the indices, rather than explicitly reversing
  // it here.)
  void SearchForwards(T* v, T* w, std::vector<T*>& f);

  // Removes the node with ID `id` from the topological order.
  void RemoveFromOrder(Index id);
  void UpdateIndex(T* v);

  // Helper that makes sure that the AddEdge() data structures are large enough
  // to hold nodes with ID `id`.
  void UpdateMaxNodeId(Index id) {
    if (static_cast<size_t>(id) >= visited_backwards_.size()) {
      visited_backwards_.resize(id + 1, false);
      visited_forwards_.resize(id + 1, false);
      increased_.resize(id + 1, false);
    }
  }

  std::vector<Node> nodes_;

  int num_edges_ = 0;  // aka "m" in the paper.
  int num_nodes_ = 0;  // aka "n" in the paper.

  // How many nodes to search backwards when adding an edge. This should be
  // ceil(min(m**(1/2), n**(2/3))), but we compute that bound online as we add
  // nodes and edges via UpdateDelta().
  int64_t delta_ = 0;

  // The next value of index to assign, aka "a" in the paper. Monotonically
  // decreasing as indices are assigned.
  // You might also wonder where 'b' from the paper is, but we simply don't
  // need it, since we're trying to maintain a doubly-linked list in topological
  // order, and we don't care about computing a topological numbering.
  int next_index_ = std::numeric_limits<int>::max();

  // The first node in each level or a higher level.
  // As is the usual convention for this data structure, this is actually the
  // index of the node in nodes_ whose next pointer points to that node, if any.
  // Invariant: There is always at least one level. Further, these indices are
  // always valid: there's always a preceding node (kSentinel = 0, if nothing
  // else).
  std::vector<Index> first_in_level_;

  // Visited state for forwards and backwards searches which are used during
  // AddEdge(). We keep this state in the class to save repeatedly allocating
  // it. This would not be thread-safe, but neither is AddEdge().
  std::vector<bool> visited_backwards_;
  std::vector<Index> visited_backwards_nodes_;
  std::vector<bool> visited_forwards_;
  std::vector<bool> increased_;
};

template <typename T, typename Index, Index T::* IndexInParent,
          typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
void TopologicalSort<T, Index, IndexInParent, PredecessorIterator,
                     PredecessorsBegin, PredecessorsEnd, SuccessorIterator,
                     SuccessorsBegin, SuccessorsEnd>::AddNode(T* v) {
  if (VLOG_IS_ON(1)) {
    DVLOG(1) << this << " AddNode(" << v->*IndexInParent << ")";
    LogOrder();
  }

  CHECK(!in_topological_order(v));
  Index id = NodeId(v);
  if (static_cast<size_t>(id) >= nodes_.size()) {
    nodes_.resize(id + 1);
  }
  Node& node = nodes_[id];
  node.level = 0;
  node.index = next_index_--;
  node.prev = -1;
  node.next = -1;
  ++num_nodes_;
  UpdateDelta();

  // Add the node to the front of the topological ordering.
  Index first_0 = first_in_level_[0];
  node.next = nodes_[first_0].next;
  node.prev = first_0;
  nodes_[node.next].prev = id;
  nodes_[first_0].next = id;
  for (int level = 1;
       level < first_in_level_.size() && first_in_level_[level] == kSentinel;
       ++level) {
    first_in_level_[level] = id;
  }
  if (VLOG_IS_ON(1)) {
    LogOrder();
  }
}

template <typename T, typename Index, Index T::* IndexInParent,
          typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
void TopologicalSort<T, Index, IndexInParent, PredecessorIterator,
                     PredecessorsBegin, PredecessorsEnd, SuccessorIterator,
                     SuccessorsBegin, SuccessorsEnd>::RemoveNode(T* v) {
  DVLOG(1) << this << " RemoveNode(" << v->*IndexInParent << ")";
  CHECK(in_topological_order(v));
  Index id = NodeId(v);
  --num_nodes_;
  if (VLOG_IS_ON(1)) {
    LogOrder();
  }
  RemoveFromOrder(id);
  nodes_[id].level = -1;
  nodes_[id].index = -1;
  if (VLOG_IS_ON(1)) {
    LogOrder();
  }
}

template <typename T, typename Index, Index T::* IndexInParent,
          typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
void TopologicalSort<T, Index, IndexInParent, PredecessorIterator,
                     PredecessorsBegin, PredecessorsEnd, SuccessorIterator,
                     SuccessorsBegin, SuccessorsEnd>::AddEdge(T* v, T* w) {
  Index v_id = NodeId(v);
  Index w_id = NodeId(w);
  Node& v_node = nodes_[v_id];
  Node& w_node = nodes_[w_id];

  ++num_edges_;
  UpdateDelta();

  DVLOG(1) << this << " AddEdge(" << v->*IndexInParent << ", "
           << w->*IndexInParent << ") v={level=" << v_node.level << " "
           << "index=" << v_node.index << "} "
           << " w={level=" << w_node.level << " "
           << "index=" << w_node.index << "} "
           << "delta_=" << delta_;

  // Verify that both nodes are in the topological order.
  DCHECK(in_topological_order(v));
  DCHECK(in_topological_order(w));

  // Step 1: test order: if w is already higher than v in the lexicographical
  // order then the current ordering is fine.
  if (std::tie(v_node.level, v_node.index) <
      std::tie(w_node.level, w_node.index)) {
    if (VLOG_IS_ON(1)) {
      LogOrder();
    }
    return;
  }

  // Step 2: search backwards from v, until we either find `w`, which means we
  // have a cycle, visit delta_ edges, or run out of edges to visit.
  std::vector<T*> b;
  bool should_search_forwards;
  bool visited_delta_edges = SearchBackwards(v, w, b);
  if (visited_delta_edges) {
    b.resize(1);
    b.front() = v;
    RemoveFromOrder(w_id);
    w_node.level = v_node.level + 1;

    should_search_forwards = true;
  } else if (w_node.level == v_node.level) {
    // l = b;
    should_search_forwards = false;
  } else {
    // We know that w_node.level < v_node.level, by the case above and by the
    // test in step 1.
    DCHECK_LT(w_node.level, v_node.level);
    RemoveFromOrder(w_id);
    w_node.level = v_node.level;
    should_search_forwards = true;
  }

  // Step 3: search forwards from w, following outgoing edges only from nodes
  // whose level increases.
  std::vector<T*> f;
  if (should_search_forwards) {
    SearchForwards(v, w, f);
    if (v_node.level < w_node.level) {
      b.clear();  // l = reverse(f)
    } else {
      CHECK_EQ(v_node.level, w_node.level);
      // l = b + reverse(f)
    }
  }

  for (Index id : visited_backwards_nodes_) {
    visited_backwards_[id] = false;
  }
  visited_backwards_nodes_.clear();

  // Step 4: update indices.
  auto node_formatter = [](std::string* out, T* v) {
    absl::StrAppend(out, v->*IndexInParent);
  };
  DVLOG(2) << "b=" << absl::StrJoin(b, ", ", node_formatter)
           << " f=" << absl::StrJoin(f, ", ", node_formatter);
  for (auto it = f.begin(); it != f.end(); ++it) {
    Index id = NodeId(*it);
    visited_forwards_[id] = false;
    increased_[id] = false;
    UpdateIndex(*it);
  }
  for (auto it = b.rbegin(); it != b.rend(); ++it) {
    UpdateIndex(*it);
  }

  // Step 5: add the edge.
  // There's actually nothing to do here, because it's up to the user to add
  // the edge to their own data structures. It doesn't matter whether the user
  // does that before or after they call our AddEdge(), since we only search
  // backwards from v and forwards from w.
  if (VLOG_IS_ON(1)) {
    LogOrder();

    DVLOG(1) << "end AddEdge(" << v->*IndexInParent << ", " << w->*IndexInParent
             << ") v={level=" << v_node.level << " "
             << "index=" << v_node.index << "} "
             << " w={level=" << w_node.level << " "
             << "index=" << w_node.index << "} "
             << "delta_=" << delta_;
  }
}

template <typename T, typename Index, Index T::* IndexInParent,
          typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
void TopologicalSort<
    T, Index, IndexInParent, PredecessorIterator, PredecessorsBegin,
    PredecessorsEnd, SuccessorIterator, SuccessorsBegin,
    SuccessorsEnd>::Reindex(absl::Span<const Index> old_to_new) {
  auto map_id = [&](Index old_id) -> Index {
    if (old_id <= kSentinel) {
      return old_id;
    }
    Index old_idx = old_id - 1;
    CHECK_LT(static_cast<size_t>(old_idx), old_to_new.size());
    Index new_idx = old_to_new[old_idx];
    CHECK_GE(new_idx, 0);
    return new_idx + 1;
  };

  Index max_new_id = kSentinel;
  for (Index new_idx : old_to_new) {
    if (new_idx >= 0) {
      max_new_id = std::max<Index>(max_new_id, new_idx + 1);
    }
  }

  std::vector<Node> new_nodes(max_new_id + 1);
  new_nodes[kSentinel].next = map_id(nodes_[kSentinel].next);
  new_nodes[kSentinel].prev = map_id(nodes_[kSentinel].prev);
  new_nodes[kSentinel].level = nodes_[kSentinel].level;
  new_nodes[kSentinel].index = nodes_[kSentinel].index;

  for (size_t old_idx = 0; old_idx < old_to_new.size(); ++old_idx) {
    Index new_idx = old_to_new[old_idx];
    if (new_idx < 0) {
      continue;
    }
    Index old_id = static_cast<Index>(old_idx) + 1;
    Index new_id = new_idx + 1;
    if (static_cast<size_t>(old_id) < nodes_.size() &&
        nodes_[old_id].level >= 0) {
      new_nodes[new_id].next = map_id(nodes_[old_id].next);
      new_nodes[new_id].prev = map_id(nodes_[old_id].prev);
      new_nodes[new_id].level = nodes_[old_id].level;
      new_nodes[new_id].index = nodes_[old_id].index;
    }
  }

  for (Index& v : first_in_level_) {
    v = map_id(v);
  }

  nodes_ = std::move(new_nodes);
  visited_backwards_.assign(nodes_.size(), false);
  visited_forwards_.assign(nodes_.size(), false);
  increased_.assign(nodes_.size(), false);
  if (VLOG_IS_ON(1)) {
    LogOrder();
  }
}

template <typename T, typename Index, Index T::* IndexInParent,
          typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
bool TopologicalSort<T, Index, IndexInParent, PredecessorIterator,
                     PredecessorsBegin, PredecessorsEnd, SuccessorIterator,
                     SuccessorsBegin,
                     SuccessorsEnd>::SearchBackwards(T* v, T* w,
                                                     std::vector<T*>& b) {
  std::vector<std::pair<T*, bool>> agenda;
  int num_edges_visited = 0;
  agenda.emplace_back(v, false);
  while (!agenda.empty()) {
    auto [y, post] = agenda.back();
    agenda.pop_back();
    Index y_id = NodeId(y);
    DVLOG(3) << "SearchBackwards visiting " << y->*IndexInParent
             << " post=" << post;
    CHECK(y != w) << "Cycle detected";
    int level = nodes_[y_id].level;
    if (post) {
      b.push_back(y);
      continue;
    }

    UpdateMaxNodeId(y_id);
    if (visited_backwards_[y_id]) {
      continue;
    }
    visited_backwards_[y_id] = true;
    visited_backwards_nodes_.push_back(y_id);

    agenda.emplace_back(y, true);
    for (auto it = std::invoke(PredecessorsBegin, y);
         num_edges_visited < delta_ && it != std::invoke(PredecessorsEnd, y);
         ++it) {
      T* x = *it;
      if (!in_topological_order(x)) {
        continue;
      }
      Index x_id = NodeId(x);
      int x_level = nodes_[x_id].level;
      CHECK_LE(x_level, level);
      VLOG(2) << "visiting edge " << x->*IndexInParent;
      if (x_level == level) {
        ++num_edges_visited;
        if (num_edges_visited >= delta_) {
          return true;
        }
        agenda.emplace_back(x, false);
      }
    }
  }
  return false;
}

template <typename T, typename Index, Index T::* IndexInParent,
          typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
void TopologicalSort<T, Index, IndexInParent, PredecessorIterator,
                     PredecessorsBegin, PredecessorsEnd, SuccessorIterator,
                     SuccessorsBegin,
                     SuccessorsEnd>::SearchForwards(T* v, T* w,
                                                    std::vector<T*>& f) {
  std::vector<std::pair<T*, bool>> agenda;
  agenda.emplace_back(w, false);
  Index w_id = NodeId(w);
  UpdateMaxNodeId(w_id);
  increased_[w_id] = true;

  // f list of vertices whose level increases, in reverse postorder, i.e.,
  // a vertex appears in f before its successors.
  while (!agenda.empty()) {
    auto [x, post] = agenda.back();
    agenda.pop_back();
    Index x_id = NodeId(x);
    DVLOG(3) << "SearchForwards visiting " << x->*IndexInParent
             << " post=" << post;
    if (post) {
      f.push_back(x);
      continue;
    }
    if (visited_forwards_[x_id] || !increased_[x_id]) {
      continue;
    }
    visited_forwards_[x_id] = true;

    agenda.emplace_back(x, true);

    int x_level = nodes_[x_id].level;
    for (auto it = std::invoke(SuccessorsBegin, x);
         it != std::invoke(SuccessorsEnd, x); ++it) {
      T* y = *it;
      if (!in_topological_order(y)) {
        continue;
      }
      Index y_id = NodeId(y);
      VLOG(3) << "fwd edge to " << y->*IndexInParent;
      DCHECK(y != v) << "Cycle detected " << y->*IndexInParent;
      UpdateMaxNodeId(y_id);
      DCHECK(!visited_backwards_[y_id])
          << "Cycle detected " << y->*IndexInParent;
      agenda.emplace_back(y, false);
      if (x_level > nodes_[y_id].level) {
        RemoveFromOrder(y_id);
        nodes_[y_id].level = x_level;
        increased_[y_id] = true;
      }
    }
  }
}

template <typename T, typename Index, Index T::* IndexInParent,
          typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
void TopologicalSort<T, Index, IndexInParent, PredecessorIterator,
                     PredecessorsBegin, PredecessorsEnd, SuccessorIterator,
                     SuccessorsBegin, SuccessorsEnd>::RemoveFromOrder(Index
                                                                          id) {
  Node& v_node = nodes_[id];
  Index prev_id = v_node.prev;
  Index next_id = v_node.next;
  // If this node is the last node in any level, it may appear in the
  // first_in_level_ vector for subsequent levels.
  for (int level = v_node.level + 1;
       level < first_in_level_.size() && first_in_level_[level] == id;
       ++level) {
    first_in_level_[level] = prev_id;
  }
  nodes_[prev_id].next = next_id;
  nodes_[next_id].prev = prev_id;
  v_node.next = -1;
  v_node.prev = -1;
}

template <typename T, typename Index, Index T::* IndexInParent,
          typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
void TopologicalSort<T, Index, IndexInParent, PredecessorIterator,
                     PredecessorsBegin, PredecessorsEnd, SuccessorIterator,
                     SuccessorsBegin, SuccessorsEnd>::UpdateIndex(T* v) {
  Index id = NodeId(v);
  Node& v_node = nodes_[id];

  if (v_node.prev != -1) {
    // TODO(phawkins): could we just do this above?
    RemoveFromOrder(id);
  }

  // Since this node just decreased in index, it now becomes the first node on
  // its level.
  v_node.index = next_index_--;
  if (v_node.level >= first_in_level_.size()) {
    Index t = first_in_level_.back();
    while (nodes_[t].next != kSentinel) {
      t = nodes_[t].next;
    }
    first_in_level_.resize(v_node.level + 1, t);
  }

  Index old_first = first_in_level_[v_node.level];
  v_node.next = nodes_[old_first].next;
  v_node.prev = old_first;
  nodes_[v_node.next].prev = id;
  nodes_[old_first].next = id;
  for (int level = v_node.level + 1;
       level < first_in_level_.size() && first_in_level_[level] == old_first;
       ++level) {
    first_in_level_[level] = id;
  }
}

template <typename T, typename Index, Index T::* IndexInParent,
          typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
void TopologicalSort<T, Index, IndexInParent, PredecessorIterator,
                     PredecessorsBegin, PredecessorsEnd, SuccessorIterator,
                     SuccessorsBegin, SuccessorsEnd>::UpdateDelta() {
  int64_t m = num_edges_;
  int64_t n = num_nodes_;
  // delta should be ceil(min(m**(1/2), n**(2/3)))
  while (delta_ * delta_ < m && delta_ * delta_ * delta_ < n * n) {
    ++delta_;
  }
}

}  // namespace xla

#endif  // XLA_ONLINE_TOPSORT_H_
