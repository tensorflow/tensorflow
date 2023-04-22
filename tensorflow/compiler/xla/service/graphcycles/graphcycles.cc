/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// GraphCycles provides incremental cycle detection on a dynamic
// graph using the following algorithm:
//
// A dynamic topological sort algorithm for directed acyclic graphs
// David J. Pearce, Paul H. J. Kelly
// Journal of Experimental Algorithmics (JEA) JEA Homepage archive
// Volume 11, 2006, Article No. 1.7
//
// Brief summary of the algorithm:
//
// (1) Maintain a rank for each node that is consistent
//     with the topological sort of the graph. I.e., path from x to y
//     implies rank[x] < rank[y].
// (2) When a new edge (x->y) is inserted, do nothing if rank[x] < rank[y].
// (3) Otherwise: adjust ranks in the neighborhood of x and y.

#include "tensorflow/compiler/xla/service/graphcycles/graphcycles.h"

#include <algorithm>
#include <unordered_set>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/graphcycles/ordered_set.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace {

using NodeSet = absl::flat_hash_set<int32>;
using OrderedNodeSet = OrderedSet<int32>;

template <typename T>
struct VecStruct {
  typedef absl::InlinedVector<T, 4> type;
};
template <typename T>
using Vec = typename VecStruct<T>::type;

struct Node {
  int32 rank;    // rank number assigned by Pearce-Kelly algorithm
  bool visited;  // Temporary marker used by depth-first-search
  void* data;    // User-supplied data
  OrderedNodeSet in;   // List of immediate predecessor nodes in graph
  OrderedNodeSet out;  // List of immediate successor nodes in graph
};

}  // namespace

struct GraphCycles::Rep {
  Vec<Node*> nodes_;
  Vec<int32> free_nodes_;  // Indices for unused entries in nodes_

  // Temporary state.
  Vec<int32> deltaf_;  // Results of forward DFS
  Vec<int32> deltab_;  // Results of backward DFS
  Vec<int32> list_;    // All nodes to reprocess
  Vec<int32> merged_;  // Rank values to assign to list_ entries
  Vec<int32> stack_;   // Emulates recursion stack when doing depth first search
};

GraphCycles::GraphCycles() : rep_(new Rep) {}

GraphCycles::~GraphCycles() {
  for (Vec<Node*>::size_type i = 0; i < rep_->nodes_.size(); i++) {
    delete rep_->nodes_[i];
  }
  delete rep_;
}

bool GraphCycles::CheckInvariants() const {
  Rep* r = rep_;
  NodeSet ranks;  // Set of ranks seen so far.
  for (Vec<Node*>::size_type x = 0; x < r->nodes_.size(); x++) {
    Node* nx = r->nodes_[x];
    if (nx->visited) {
      LOG(FATAL) << "Did not clear visited marker on node " << x;
    }
    if (!ranks.insert(nx->rank).second) {
      LOG(FATAL) << "Duplicate occurrence of rank " << nx->rank;
    }
    for (int32 y : nx->out.GetSequence()) {
      Node* ny = r->nodes_[y];
      if (nx->rank >= ny->rank) {
        LOG(FATAL) << "Edge " << x << "->" << y << " has bad rank assignment "
                   << nx->rank << "->" << ny->rank;
      }
    }
  }
  return true;
}

int32 GraphCycles::NewNode() {
  if (rep_->free_nodes_.empty()) {
    Node* n = new Node;
    n->visited = false;
    n->data = nullptr;
    n->rank = rep_->nodes_.size();
    rep_->nodes_.push_back(n);
    return n->rank;
  } else {
    // Preserve preceding rank since the set of ranks in use must be
    // a permutation of [0,rep_->nodes_.size()-1].
    int32 r = rep_->free_nodes_.back();
    rep_->nodes_[r]->data = nullptr;
    rep_->free_nodes_.pop_back();
    return r;
  }
}

void GraphCycles::RemoveNode(int32 node) {
  Node* x = rep_->nodes_[node];
  for (int32 y : x->out.GetSequence()) {
    rep_->nodes_[y]->in.Erase(node);
  }
  for (int32 y : x->in.GetSequence()) {
    rep_->nodes_[y]->out.Erase(node);
  }
  x->in.Clear();
  x->out.Clear();
  rep_->free_nodes_.push_back(node);
}

void* GraphCycles::GetNodeData(int32 node) const {
  return rep_->nodes_[node]->data;
}

void GraphCycles::SetNodeData(int32 node, void* data) {
  rep_->nodes_[node]->data = data;
}

bool GraphCycles::HasEdge(int32 x, int32 y) const {
  return rep_->nodes_[x]->out.Contains(y);
}

void GraphCycles::RemoveEdge(int32 x, int32 y) {
  rep_->nodes_[x]->out.Erase(y);
  rep_->nodes_[y]->in.Erase(x);
  // No need to update the rank assignment since a previous valid
  // rank assignment remains valid after an edge deletion.
}

static bool ForwardDFS(GraphCycles::Rep* r, int32 n, int32 upper_bound);
static void BackwardDFS(GraphCycles::Rep* r, int32 n, int32 lower_bound);
static void Reorder(GraphCycles::Rep* r);
static void Sort(const Vec<Node*>&, Vec<int32>* delta);
static void MoveToList(GraphCycles::Rep* r, Vec<int32>* src, Vec<int32>* dst);
static void ClearVisitedBits(GraphCycles::Rep* r, const Vec<int32>& nodes);

bool GraphCycles::InsertEdge(int32 x, int32 y) {
  if (x == y) return false;
  Rep* r = rep_;
  Node* nx = r->nodes_[x];
  if (!nx->out.Insert(y)) {
    // Edge already exists.
    return true;
  }

  Node* ny = r->nodes_[y];
  ny->in.Insert(x);

  if (nx->rank <= ny->rank) {
    // New edge is consistent with existing rank assignment.
    return true;
  }

  // Current rank assignments are incompatible with the new edge.  Recompute.
  // We only need to consider nodes that fall in the range [ny->rank,nx->rank].
  if (!ForwardDFS(r, y, nx->rank)) {
    // Found a cycle.  Undo the insertion and tell caller.
    nx->out.Erase(y);
    ny->in.Erase(x);
    // Since we do not call Reorder() on this path, clear any visited
    // markers left by ForwardDFS.
    ClearVisitedBits(r, r->deltaf_);
    return false;
  }
  BackwardDFS(r, x, ny->rank);
  Reorder(r);
  return true;
}

static bool ForwardDFS(GraphCycles::Rep* r, int32 n, int32 upper_bound) {
  // Avoid recursion since stack space might be limited.
  // We instead keep a stack of nodes to visit.
  r->deltaf_.clear();
  r->stack_.clear();
  r->stack_.push_back(n);
  while (!r->stack_.empty()) {
    n = r->stack_.back();
    r->stack_.pop_back();
    Node* nn = r->nodes_[n];
    if (nn->visited) continue;

    nn->visited = true;
    r->deltaf_.push_back(n);

    for (auto w : nn->out.GetSequence()) {
      Node* nw = r->nodes_[w];
      if (nw->rank == upper_bound) {
        return false;  // Cycle
      }
      if (!nw->visited && nw->rank < upper_bound) {
        r->stack_.push_back(w);
      }
    }
  }
  return true;
}

static void BackwardDFS(GraphCycles::Rep* r, int32 n, int32 lower_bound) {
  r->deltab_.clear();
  r->stack_.clear();
  r->stack_.push_back(n);
  while (!r->stack_.empty()) {
    n = r->stack_.back();
    r->stack_.pop_back();
    Node* nn = r->nodes_[n];
    if (nn->visited) continue;

    nn->visited = true;
    r->deltab_.push_back(n);

    for (auto w : nn->in.GetSequence()) {
      Node* nw = r->nodes_[w];
      if (!nw->visited && lower_bound < nw->rank) {
        r->stack_.push_back(w);
      }
    }
  }
}

static void Reorder(GraphCycles::Rep* r) {
  Sort(r->nodes_, &r->deltab_);
  Sort(r->nodes_, &r->deltaf_);

  // Adds contents of delta lists to list_ (backwards deltas first).
  r->list_.clear();
  MoveToList(r, &r->deltab_, &r->list_);
  MoveToList(r, &r->deltaf_, &r->list_);

  // Produce sorted list of all ranks that will be reassigned.
  r->merged_.resize(r->deltab_.size() + r->deltaf_.size());
  std::merge(r->deltab_.begin(), r->deltab_.end(), r->deltaf_.begin(),
             r->deltaf_.end(), r->merged_.begin());

  // Assign the ranks in order to the collected list.
  for (Vec<int32>::size_type i = 0; i < r->list_.size(); i++) {
    r->nodes_[r->list_[i]]->rank = r->merged_[i];
  }
}

static void Sort(const Vec<Node*>& nodes, Vec<int32>* delta) {
  struct ByRank {
    const Vec<Node*>* nodes;
    bool operator()(int32 a, int32 b) const {
      return (*nodes)[a]->rank < (*nodes)[b]->rank;
    }
  };
  ByRank cmp;
  cmp.nodes = &nodes;
  std::sort(delta->begin(), delta->end(), cmp);
}

static void MoveToList(GraphCycles::Rep* r, Vec<int32>* src, Vec<int32>* dst) {
  for (Vec<int32>::size_type i = 0; i < src->size(); i++) {
    int32 w = (*src)[i];
    (*src)[i] = r->nodes_[w]->rank;  // Replace src entry with its rank
    r->nodes_[w]->visited = false;   // Prepare for future DFS calls
    dst->push_back(w);
  }
}

static void ClearVisitedBits(GraphCycles::Rep* r, const Vec<int32>& nodes) {
  for (Vec<int32>::size_type i = 0; i < nodes.size(); i++) {
    r->nodes_[nodes[i]]->visited = false;
  }
}

int GraphCycles::FindPath(int32 x, int32 y, int max_path_len,
                          int32 path[]) const {
  // Forward depth first search starting at x until we hit y.
  // As we descend into a node, we push it onto the path.
  // As we leave a node, we remove it from the path.
  int path_len = 0;

  Rep* r = rep_;
  NodeSet seen;
  r->stack_.clear();
  r->stack_.push_back(x);
  while (!r->stack_.empty()) {
    int32 n = r->stack_.back();
    r->stack_.pop_back();
    if (n < 0) {
      // Marker to indicate that we are leaving a node
      path_len--;
      continue;
    }

    if (path_len < max_path_len) {
      path[path_len] = n;
    }
    path_len++;
    r->stack_.push_back(-1);  // Will remove tentative path entry

    if (n == y) {
      return path_len;
    }

    for (auto w : r->nodes_[n]->out.GetSequence()) {
      if (seen.insert(w).second) {
        r->stack_.push_back(w);
      }
    }
  }

  return 0;
}

bool GraphCycles::IsReachable(int32 x, int32 y) const {
  return FindPath(x, y, 0, nullptr) > 0;
}

bool GraphCycles::IsReachableNonConst(int32 x, int32 y) {
  if (x == y) return true;
  Rep* r = rep_;
  Node* nx = r->nodes_[x];
  Node* ny = r->nodes_[y];

  if (nx->rank >= ny->rank) {
    // x cannot reach y since it is after it in the topological ordering
    return false;
  }

  // See if x can reach y using a DFS search that is limited to y's rank
  bool reachable = !ForwardDFS(r, x, ny->rank);

  // Clear any visited markers left by ForwardDFS.
  ClearVisitedBits(r, r->deltaf_);
  return reachable;
}

bool GraphCycles::CanContractEdge(int32 a, int32 b) {
  CHECK(HasEdge(a, b)) << "No edge exists from " << a << " to " << b;
  RemoveEdge(a, b);
  bool reachable = IsReachableNonConst(a, b);
  // Restore the graph to its original state.
  InsertEdge(a, b);
  // If reachable, then contracting edge will cause cycle.
  return !reachable;
}

absl::optional<int32> GraphCycles::ContractEdge(int32 a, int32 b) {
  CHECK(HasEdge(a, b));
  RemoveEdge(a, b);

  if (IsReachableNonConst(a, b)) {
    // Restore the graph to its original state.
    InsertEdge(a, b);
    return absl::nullopt;
  }

  if (rep_->nodes_[b]->in.Size() + rep_->nodes_[b]->out.Size() >
      rep_->nodes_[a]->in.Size() + rep_->nodes_[a]->out.Size()) {
    // Swap "a" and "b" to minimize copying.
    std::swap(a, b);
  }

  Node* nb = rep_->nodes_[b];
  OrderedNodeSet out = std::move(nb->out);
  OrderedNodeSet in = std::move(nb->in);
  for (int32 y : out.GetSequence()) {
    rep_->nodes_[y]->in.Erase(b);
  }
  for (int32 y : in.GetSequence()) {
    rep_->nodes_[y]->out.Erase(b);
  }
  rep_->free_nodes_.push_back(b);

  rep_->nodes_[a]->out.Reserve(rep_->nodes_[a]->out.Size() + out.Size());
  for (int32 y : out.GetSequence()) {
    InsertEdge(a, y);
  }

  rep_->nodes_[a]->in.Reserve(rep_->nodes_[a]->in.Size() + in.Size());
  for (int32 y : in.GetSequence()) {
    InsertEdge(y, a);
  }

  // Note, if the swap happened it might be what originally was called "b".
  return a;
}

absl::Span<const int32> GraphCycles::Successors(int32 node) const {
  return rep_->nodes_[node]->out.GetSequence();
}

absl::Span<const int32> GraphCycles::Predecessors(int32 node) const {
  return rep_->nodes_[node]->in.GetSequence();
}

std::vector<int32> GraphCycles::SuccessorsCopy(int32 node) const {
  absl::Span<const int32> successors = Successors(node);
  return std::vector<int32>(successors.begin(), successors.end());
}

std::vector<int32> GraphCycles::PredecessorsCopy(int32 node) const {
  absl::Span<const int32> predecessors = Predecessors(node);
  return std::vector<int32>(predecessors.begin(), predecessors.end());
}

namespace {
void SortInPostOrder(absl::Span<Node* const> nodes,
                     std::vector<int32>* to_sort) {
  absl::c_sort(*to_sort, [&](int32 a, int32 b) {
    DCHECK(a == b || nodes[a]->rank != nodes[b]->rank);
    return nodes[a]->rank > nodes[b]->rank;
  });
}
}  // namespace

std::vector<int32> GraphCycles::AllNodesInPostOrder() const {
  absl::flat_hash_set<int32> free_nodes_set;
  absl::c_copy(rep_->free_nodes_,
               std::inserter(free_nodes_set, free_nodes_set.begin()));

  std::vector<int32> all_nodes;
  all_nodes.reserve(rep_->nodes_.size() - free_nodes_set.size());
  for (int64 i = 0, e = rep_->nodes_.size(); i < e; i++) {
    if (!free_nodes_set.contains(i)) {
      all_nodes.push_back(i);
    }
  }

  SortInPostOrder(rep_->nodes_, &all_nodes);
  return all_nodes;
}

string GraphCycles::DebugString() const {
  absl::flat_hash_set<int32> free_nodes_set;
  for (int32 free_node : rep_->free_nodes_) {
    free_nodes_set.insert(free_node);
  }

  string result = "digraph {\n";
  for (int i = 0, end = rep_->nodes_.size(); i < end; i++) {
    if (free_nodes_set.contains(i)) {
      continue;
    }

    for (int32 succ : rep_->nodes_[i]->out.GetSequence()) {
      absl::StrAppend(&result, "  \"", i, "\" -> \"", succ, "\"\n");
    }
  }

  absl::StrAppend(&result, "}\n");

  return result;
}

}  // namespace tensorflow
