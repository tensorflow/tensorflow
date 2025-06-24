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
// - Link is a pointer to the embedded TopologicalSortNode<T> field in T.
// - IndexInParent is a pointer to the index_in_parent field in T.
//   These indices must remain fixed only during a call to AddEdge(), which
//   is obviously true because we don't allow threads and the topological sort
//   will not change them, but they are allowed to change between calls.
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
#include "xla/tsl/platform/logging.h"

// The topological sort is an intrusive data structure. Nodes of type T that
// participate in the topological sort must have a TopologicalSortNode<T>
// embedded within them.
template <typename T>
class TopologicalSortNode {
 public:
  TopologicalSortNode() = default;
  ~TopologicalSortNode() { DCHECK(!in_topological_order()) << level_; }

  TopologicalSortNode(const TopologicalSortNode&) = delete;
  TopologicalSortNode(TopologicalSortNode&&) = delete;
  TopologicalSortNode& operator=(const TopologicalSortNode&) = delete;
  TopologicalSortNode& operator=(TopologicalSortNode&&) = delete;

  void clear() {
    next_ = nullptr;
    prev_ = nullptr;
    level_ = -1;
    index_ = -1;
  }

  // Returns true if this node has been added to a topological order.
  // It may have temporarily been removed from a specific location in that
  // order if we are in the middle of an AddEdge() operation.
  bool in_topological_order() const { return level_ >= 0; }

 private:
  template <typename S, typename Index, TopologicalSortNode<S> S::* Link,
            Index S::* IndexInParent, typename PredecessorIterator,
            PredecessorIterator (S::*PredecessorsBegin)() const,
            PredecessorIterator (S::*PredecessorsEnd)() const,
            typename SuccessorIterator,
            SuccessorIterator (S::*SuccessorsBegin)() const,
            SuccessorIterator (S::*SuccessorsEnd)() const>
  friend class TopologicalSort;

  template <typename S, TopologicalSortNode<S> S::* Link>
  friend class TopologicalSortForwardIterator;
  template <typename S, TopologicalSortNode<S> S::* Link>
  friend class TopologicalSortReverseIterator;

  int index_ = -1;
  int level_ = -1;

  // The nodes form a doubly-linked list, where the `next_` pointers are not
  // circular, but the `prev_` pointers are circular.
  // There is also an asymmetry in the types of `next_` and `prev_`: the former
  // is a pointer to a node, while the latter is a pointer to a
  // TopologicalSortNode embedded within a node. This trick helps us define
  // an intrusive templated list in C++.
  T* next_ = nullptr;
  TopologicalSortNode<T>* prev_ = nullptr;
};

// Iterator that traverses through the topological sort in order.
template <typename T, TopologicalSortNode<T> T::* Link>
class TopologicalSortForwardIterator {
 public:
  TopologicalSortForwardIterator() : current_(nullptr) {}
  explicit TopologicalSortForwardIterator(const TopologicalSortNode<T>* current)
      : current_(current) {}

  TopologicalSortForwardIterator(const TopologicalSortForwardIterator&) =
      default;
  TopologicalSortForwardIterator(TopologicalSortForwardIterator&&) = default;
  TopologicalSortForwardIterator& operator=(
      const TopologicalSortForwardIterator&) = default;
  TopologicalSortForwardIterator& operator=(TopologicalSortForwardIterator&&) =
      default;

  T& operator*() const { return *current_->next_; }
  T* operator->() const { return current_->next_; }

  bool operator==(const TopologicalSortForwardIterator& other) const {
    return current_ == other.current_;
  }
  bool operator!=(const TopologicalSortForwardIterator& other) const {
    return current_ != other.current_;
  }

  TopologicalSortForwardIterator& operator++() {
    current_ = &(current_->next_->*Link);
    return *this;
  }

  TopologicalSortForwardIterator& operator--() {
    current_ = &current_->prev_;
    return *this;
  }

 private:
  // Note: the iterator is a pointer to a node whose *next* pointer points to
  // the current node.
  TopologicalSortNode<T> const* current_;
};

// Iterator that traverses through the topological sort in reverse order.
template <typename T, TopologicalSortNode<T> T::* Link>
class TopologicalSortReverseIterator {
 public:
  TopologicalSortReverseIterator() : current_(nullptr) {}
  explicit TopologicalSortReverseIterator(const TopologicalSortNode<T>* current)
      : current_(current) {}

  TopologicalSortReverseIterator(const TopologicalSortReverseIterator&) =
      default;
  TopologicalSortReverseIterator(TopologicalSortReverseIterator&&) = default;
  TopologicalSortReverseIterator& operator=(
      const TopologicalSortReverseIterator&) = default;
  TopologicalSortReverseIterator& operator=(TopologicalSortReverseIterator&&) =
      default;

  T& operator*() const { return *current_->next_; }
  T* operator->() const { return current_->next_; }

  bool operator==(const TopologicalSortReverseIterator& other) const {
    return current_ == other.current_;
  }
  bool operator!=(const TopologicalSortReverseIterator& other) const {
    return current_ != other.current_;
  }

  TopologicalSortReverseIterator& operator++() {
    current_ = current_->prev_;
    return *this;
  }

 private:
  // Note: the iterator is a pointer to a node whose *next* pointer points to
  // the current node.
  TopologicalSortNode<T> const* current_;
};

template <typename T, typename Index, TopologicalSortNode<T> T::* Link,
          Index T::* IndexInParent, typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
class TopologicalSort {
 public:
  TopologicalSort() {
    node_.next_ = nullptr;
    node_.prev_ = &node_;
    first_in_level_.push_back(&node_);
  }

  ~TopologicalSort();

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

  // Returns an iterator over the nodes in topological order.
  TopologicalSortForwardIterator<T, Link> begin() const {
    return TopologicalSortForwardIterator<T, Link>(&node_);
  }
  TopologicalSortForwardIterator<T, Link> end() const {
    return TopologicalSortForwardIterator<T, Link>(node_.prev_);
  }

  // Returns an iterator over the nodes in reverse topological order.
  TopologicalSortReverseIterator<T, Link> rbegin() const {
    return TopologicalSortReverseIterator<T, Link>(node_.prev_->prev_);
  }
  TopologicalSortReverseIterator<T, Link> rend() const {
    return TopologicalSortReverseIterator<T, Link>(node_.prev_);
  }

  // This is a helper for debugging. It logs the current order and checks a
  // number of invariants.
  void LogOrder() {
    std::vector<T*> order;
    int level = -1;
    for (T& node : *this) {
      const auto& link = node.*Link;
      CHECK_GE(link.level_, level);
      level = link.level_;
      if (link.next_) {
        CHECK((link.next_->*Link).prev_ == &link);
      } else {
        CHECK(node_.prev_ == &link);
      }
      CHECK(link.prev_->next_ == &node);
      order.push_back(&node);
    }
    auto node_formatter = [](std::string* out, T* v) {
      absl::StrAppend(out, v->*IndexInParent, "[", (v->*Link).level_, ":",
                      (v->*Link).index_, "]");
    };
    DVLOG(2) << this << " order=" << absl::StrJoin(order, ", ", node_formatter);
    auto first_in_level_formatter = [](std::string* out,
                                       TopologicalSortNode<T>* v) {
      if (v->next_) {
        absl::StrAppend(out, v->next_->*IndexInParent, ":",
                        (v->next_->*Link).level_);
      } else {
        absl::StrAppend(out, "-:-");
      }
    };
    DVLOG(2) << this << " first_in_level_="
             << absl::StrJoin(first_in_level_, ", ", first_in_level_formatter);

    CHECK(first_in_level_[0] == &node_);
    auto it = order.begin();
    for (TopologicalSortNode<T>* v : first_in_level_) {
      it = std::find(it, order.end(), v->next_);
      CHECK(v->next_ == nullptr || it != order.end());
    }
  }

  void clear() { node_.clear(); }

 private:
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

  // Removes v from the topological order.
  void RemoveFromOrder(T* v);

  void UpdateIndex(T* v);

  // Helper that makes sure that the AddEdge() data structures are large enough
  // to hold nodes with index max_index_in_parent.
  void UpdateMaxIndexInParent(Index max_index_in_parent) {
    if (max_index_in_parent >= visited_backwards_.size()) {
      visited_backwards_.resize(max_index_in_parent + 1);
      visited_forwards_.resize(max_index_in_parent + 1);
      increased_.resize(max_index_in_parent + 1);
    }
  }

  TopologicalSortNode<T> node_;

  int num_edges_ = 0;  // aka "m" in the paper.
  int num_nodes_ = 0;  // aka "n" in the paper.

  // How many nodes to search backwards when adding an edge. This should be
  // ceil(min(m**(1/2), n**(2/3))), but we compute that bound online as we add
  // nodes and edges via UpdateDelta().
  int64_t delta_ = 0;

  // The next value of index_ to assign, aka "a" in the paper. Monotonically
  // decreasing as indices are assigned.
  // You might also wonder where 'b' from the paper is, but we simply don't
  // need it, since we're trying to maintain a doubly-linked list in topological
  // order, and we don't care about computing a topological numbering.
  int next_index_ = std::numeric_limits<int>::max();

  // The first node in each level or a higher level.
  // As is the usual convention for this data structure, this is actually the
  // TopologicalSortNode whose next_ pointer points to that node, if any.
  // Invariant: There is always at least one level. Futher, these pointers are
  // never nullptr: there's always a preceding node (node_, if nothing else).
  std::vector<TopologicalSortNode<T>*> first_in_level_;

  // Visited state for forwards and backwards searches which are used during
  // AddEdge(). We keep this state in the class to save repeatedly allocating
  // it. This would not be thread-safe, but neither is AddEdge().
  std::vector<bool> visited_backwards_;
  std::vector<bool> visited_forwards_;
  std::vector<bool> increased_;
};

template <typename T, typename Index, TopologicalSortNode<T> T::* Link,
          Index T::* IndexInParent, typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
TopologicalSort<T, Index, Link, IndexInParent, PredecessorIterator,
                PredecessorsBegin, PredecessorsEnd, SuccessorIterator,
                SuccessorsBegin, SuccessorsEnd>::~TopologicalSort() {
  TopologicalSortNode<T>* next;
  for (TopologicalSortNode<T>* node = &node_; node != nullptr; node = next) {
    if (node->next_) {
      next = &(node->next_->*Link);
    } else {
      next = nullptr;
    }
    node->clear();
  }
}

template <typename T, typename Index, TopologicalSortNode<T> T::* Link,
          Index T::* IndexInParent, typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
void TopologicalSort<T, Index, Link, IndexInParent, PredecessorIterator,
                     PredecessorsBegin, PredecessorsEnd, SuccessorIterator,
                     SuccessorsBegin, SuccessorsEnd>::AddNode(T* v) {
  TopologicalSortNode<T>* node = &(v->*Link);
  if (VLOG_IS_ON(1)) {
    DVLOG(1) << this << " AddNode(" << v->*IndexInParent << ")";
    LogOrder();
  }

  // next_ and prev_ should be nullptr for a new node.
  CHECK(node->next_ == nullptr);
  CHECK(node->prev_ == nullptr);
  node->level_ = 0;
  node->index_ = next_index_--;
  ++num_nodes_;
  UpdateDelta();

  // Add the node to the front of the topological ordering.
  node->next_ = first_in_level_[0]->next_;
  node->prev_ = first_in_level_[0];
  if (node->next_) {
    (node->next_->*Link).prev_ = node;
  } else {
    node_.prev_ = node;
  }
  first_in_level_[0]->next_ = v;
  for (int level = 1;
       level < first_in_level_.size() && first_in_level_[level] == &node_;
       ++level) {
    first_in_level_[level] = node;
  }
  if (VLOG_IS_ON(1)) {
    LogOrder();
  }
}

template <typename T, typename Index, TopologicalSortNode<T> T::* Link,
          Index T::* IndexInParent, typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
void TopologicalSort<T, Index, Link, IndexInParent, PredecessorIterator,
                     PredecessorsBegin, PredecessorsEnd, SuccessorIterator,
                     SuccessorsBegin, SuccessorsEnd>::RemoveNode(T* v) {
  TopologicalSortNode<T>* node = &(v->*Link);
  DVLOG(1) << this << " RemoveNode(" << v->*IndexInParent << ")";
  CHECK(node->prev_ == &node_ || node->prev_->in_topological_order());
  --num_nodes_;
  if (VLOG_IS_ON(1)) {
    LogOrder();
  }
  RemoveFromOrder(v);
  node->level_ = -1;
  node->index_ = -1;
  if (VLOG_IS_ON(1)) {
    LogOrder();
  }
}

template <typename T, typename Index, TopologicalSortNode<T> T::* Link,
          Index T::* IndexInParent, typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
void TopologicalSort<T, Index, Link, IndexInParent, PredecessorIterator,
                     PredecessorsBegin, PredecessorsEnd, SuccessorIterator,
                     SuccessorsBegin, SuccessorsEnd>::AddEdge(T* v, T* w) {
  TopologicalSortNode<T>* v_node = &(v->*Link);
  TopologicalSortNode<T>* w_node = &(w->*Link);

  ++num_edges_;
  UpdateDelta();

  DVLOG(1) << this << " AddEdge(" << v->*IndexInParent << ", "
           << w->*IndexInParent << ") v={level=" << v_node->level_ << " "
           << "index=" << v_node->index_ << "} "
           << " w={level=" << w_node->level_ << " "
           << "index=" << w_node->index_ << "} "
           << "delta_=" << delta_;

  // Verify that both nodes are in the topological order.
  DCHECK(v_node->in_topological_order());
  DCHECK(w_node->in_topological_order());

  // Step 1: test order: if w is already higher than v in the lexicographical
  // order then the current ordering is fine.
  if (std::tie(v_node->level_, v_node->index_) <
      std::tie(w_node->level_, w_node->index_)) {
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
    RemoveFromOrder(w);
    w_node->level_ = v_node->level_ + 1;

    should_search_forwards = true;
  } else if (w_node->level_ == v_node->level_) {
    // l = b;
    should_search_forwards = false;
  } else {
    // We know that w_node->level < v_node->level, by the case above and by the
    // test in step 1.
    DCHECK_LT(w_node->level_, v_node->level_);
    RemoveFromOrder(w);
    w_node->level_ = v_node->level_;
    should_search_forwards = true;
  }

  // Step 3: search forwards from w, following outgoing edges only from nodes
  // whose level increases.
  std::vector<T*> f;
  if (should_search_forwards) {
    SearchForwards(v, w, f);
    if (v_node->level_ < w_node->level_) {
      b.clear();  // l = reverse(f)
    } else {
      CHECK_EQ(v_node->level_, w_node->level_);
      // l = b + reverse(f)
    }
  }

  // Step 4: update indices.
  auto node_formatter = [](std::string* out, T* v) {
    absl::StrAppend(out, v->*IndexInParent);
  };
  DVLOG(2) << "b=" << absl::StrJoin(b, ", ", node_formatter)
           << " f=" << absl::StrJoin(f, ", ", node_formatter);
  for (auto it = f.begin(); it != f.end(); ++it) {
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
             << ") v={level=" << v_node->level_ << " "
             << "index=" << v_node->index_ << "} "
             << " w={level=" << w_node->level_ << " "
             << "index=" << w_node->index_ << "} "
             << "delta_=" << delta_;
  }
}

template <typename T, typename Index, TopologicalSortNode<T> T::* Link,
          Index T::* IndexInParent, typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
bool TopologicalSort<T, Index, Link, IndexInParent, PredecessorIterator,
                     PredecessorsBegin, PredecessorsEnd, SuccessorIterator,
                     SuccessorsBegin,
                     SuccessorsEnd>::SearchBackwards(T* v, T* w,
                                                     std::vector<T*>& b) {
  std::vector<std::pair<T*, bool>> agenda;
  std::fill(visited_backwards_.begin(), visited_backwards_.end(), false);
  int num_edges_visited = 0;
  agenda.emplace_back(v, false);
  while (!agenda.empty()) {
    auto [y, post] = agenda.back();
    agenda.pop_back();
    DVLOG(3) << "SearchBackwards visiting " << y->*IndexInParent
             << " post=" << post;
    CHECK(y != w) << "Cycle detected";
    TopologicalSortNode<T>* y_node = &(y->*Link);
    int level = y_node->level_;
    if (post) {
      b.push_back(y);
      continue;
    }

    Index y_index_in_parent = y->*IndexInParent;
    UpdateMaxIndexInParent(y_index_in_parent);
    if (visited_backwards_[y_index_in_parent]) {
      continue;
    }
    visited_backwards_[y_index_in_parent] = true;

    agenda.emplace_back(y, true);
    for (auto it = std::invoke(PredecessorsBegin, y);
         num_edges_visited < delta_ && it != std::invoke(PredecessorsEnd, y);
         ++it) {
      T* x = *it;
      TopologicalSortNode<T>* x_node = &(x->*Link);
      if (!x_node->in_topological_order()) {
        continue;
      }
      CHECK_LE(x_node->level_, level);
      VLOG(2) << "visiting edge " << x->*IndexInParent;
      if (x_node->level_ == level) {
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

template <typename T, typename Index, TopologicalSortNode<T> T::* Link,
          Index T::* IndexInParent, typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
void TopologicalSort<T, Index, Link, IndexInParent, PredecessorIterator,
                     PredecessorsBegin, PredecessorsEnd, SuccessorIterator,
                     SuccessorsBegin,
                     SuccessorsEnd>::SearchForwards(T* v, T* w,
                                                    std::vector<T*>& f) {
  std::fill(visited_forwards_.begin(), visited_forwards_.end(), false);
  std::fill(increased_.begin(), increased_.end(), false);
  std::vector<std::pair<T*, bool>> agenda;
  agenda.emplace_back(w, false);
  UpdateMaxIndexInParent(w->*IndexInParent);
  increased_[w->*IndexInParent] = true;

  // f list of vertices whose level increases, in reverse postorder, i.e.,
  // a vertex appears in f before its successors.
  while (!agenda.empty()) {
    auto [x, post] = agenda.back();
    agenda.pop_back();
    DVLOG(3) << "SearchForwards visiting " << x->*IndexInParent
             << " post=" << post;
    if (post) {
      f.push_back(x);
      continue;
    }
    Index x_index_in_parent = x->*IndexInParent;
    UpdateMaxIndexInParent(x_index_in_parent);
    if (visited_forwards_[x_index_in_parent] ||
        !increased_[x_index_in_parent]) {
      continue;
    }
    visited_forwards_[x_index_in_parent] = true;

    agenda.emplace_back(x, true);

    TopologicalSortNode<T>* x_node = &(x->*Link);
    for (auto it = std::invoke(SuccessorsBegin, x);
         it != std::invoke(SuccessorsEnd, x); ++it) {
      T* y = *it;
      VLOG(3) << "fwd edge to " << y->*IndexInParent;
      TopologicalSortNode<T>* y_node = &(y->*Link);
      if (!y_node->in_topological_order()) {
        continue;
      }
      Index y_index_in_parent = y->*IndexInParent;
      UpdateMaxIndexInParent(y_index_in_parent);
      DCHECK(y != v) << "Cycle detected " << y->*IndexInParent;
      DCHECK(!visited_backwards_[y_index_in_parent])
          << "Cycle detected " << y->*IndexInParent;
      agenda.emplace_back(y, false);
      if (x_node->level_ > y_node->level_) {
        RemoveFromOrder(y);
        y_node->level_ = x_node->level_;
        increased_[y_index_in_parent] = true;
      }
    }
  }
}

template <typename T, typename Index, TopologicalSortNode<T> T::* Link,
          Index T::* IndexInParent, typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
void TopologicalSort<T, Index, Link, IndexInParent, PredecessorIterator,
                     PredecessorsBegin, PredecessorsEnd, SuccessorIterator,
                     SuccessorsBegin, SuccessorsEnd>::RemoveFromOrder(T* v) {
  TopologicalSortNode<T>* v_node = &(v->*Link);
  // If this node is the last node in any level, it may appear in the
  // first_in_level_ vector for subsequent levels.
  for (int level = v_node->level_ + 1;
       level < first_in_level_.size() && first_in_level_[level] == v_node;
       ++level) {
    first_in_level_[level] = v_node->prev_;
  }
  v_node->prev_->next_ = v_node->next_;
  if (v_node->next_) {
    (v_node->next_->*Link).prev_ = v_node->prev_;
  } else {
    node_.prev_ = v_node->prev_;
  }
  v_node->next_ = nullptr;
  v_node->prev_ = nullptr;
}

template <typename T, typename Index, TopologicalSortNode<T> T::* Link,
          Index T::* IndexInParent, typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
void TopologicalSort<T, Index, Link, IndexInParent, PredecessorIterator,
                     PredecessorsBegin, PredecessorsEnd, SuccessorIterator,
                     SuccessorsBegin, SuccessorsEnd>::UpdateIndex(T* v) {
  TopologicalSortNode<T>* v_node = &(v->*Link);

  if (v_node->prev_) {
    // TODO(phawkins): could we just do this above?
    RemoveFromOrder(v);
  }

  // Since this node just decreased in index, it now becomes the first node on
  // its level.
  v_node->index_ = next_index_--;
  if (v_node->level_ >= first_in_level_.size()) {
    TopologicalSortNode<T>* t = first_in_level_.back();
    while (t->next_ != nullptr) {
      t = &(t->next_->*Link);
    }
    first_in_level_.resize(v_node->level_ + 1, t);
  }

  TopologicalSortNode<T>* old_first = first_in_level_[v_node->level_];
  v_node->next_ = old_first->next_;
  v_node->prev_ = old_first;
  if (v_node->next_) {
    (v_node->next_->*Link).prev_ = v_node;
  } else {
    node_.prev_ = v_node;
  }
  old_first->next_ = v;
  for (int level = v_node->level_ + 1;
       level < first_in_level_.size() && first_in_level_[level] == old_first;
       ++level) {
    first_in_level_[level] = v_node;
  }
}

template <typename T, typename Index, TopologicalSortNode<T> T::* Link,
          Index T::* IndexInParent, typename PredecessorIterator,
          PredecessorIterator (T::*PredecessorsBegin)() const,
          PredecessorIterator (T::*PredecessorsEnd)() const,
          typename SuccessorIterator,
          SuccessorIterator (T::*SuccessorsBegin)() const,
          SuccessorIterator (T::*SuccessorsEnd)() const>
void TopologicalSort<T, Index, Link, IndexInParent, PredecessorIterator,
                     PredecessorsBegin, PredecessorsEnd, SuccessorIterator,
                     SuccessorsBegin, SuccessorsEnd>::UpdateDelta() {
  int64_t m = num_edges_;
  int64_t n = num_nodes_;
  // delta should be ceil(min(m**(1/2), n**(2/3)))
  while (delta_ * delta_ < m && delta_ * delta_ * delta_ < n * n) {
    ++delta_;
  }
}

#endif  // XLA_ONLINE_TOPSORT_H_
