/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "utils/cycle_detector.h"

#include <algorithm>
#include <optional>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

namespace {

using NodeSet = llvm::DenseSet<int32_t>;
using OrderedNodeSet = OrderedSet<int32_t>;

template <typename T>
struct VecStruct {
  using type = llvm::SmallVector<T, 4>;
};
template <typename T>
using Vec = typename VecStruct<T>::type;

struct Node {
  // rank number assigned by Pearce-Kelly algorithm
  int32_t rank;
  // Temporary marker used by depth-first-search
  bool visited;
  // User-supplied data
  void* data;
  // List of immediate predecessor nodes in graph
  OrderedNodeSet in;
  // List of immediate successor nodes in graph
  OrderedNodeSet out;
};

}  // namespace

struct GraphCycles::Rep {
  Vec<Node*> nodes;
  // Indices for unused entries in nodes
  Vec<int32_t> freeNodes;

  // Temporary state.
  // Results of forward DFS
  Vec<int32_t> deltaf;
  // Results of backward DFS
  Vec<int32_t> deltab;
  // All nodes to reprocess
  Vec<int32_t> list;
  // Rank values to assign to list entries
  Vec<int32_t> merged;
  // Emulates recursion stack when doing depth first search
  Vec<int32_t> stack;
};

GraphCycles::GraphCycles(int32_t numNodes) : rep_(new Rep) {
  rep_->nodes.reserve(numNodes);
  for (int32_t i = 0; i < numNodes; ++i) {
    Node* n = new Node;
    n->visited = false;
    n->data = nullptr;
    n->rank = rep_->nodes.size();
    rep_->nodes.push_back(n);
  }
}

GraphCycles::~GraphCycles() {
  for (Vec<Node*>::size_type i = 0, e = rep_->nodes.size(); i < e; ++i) {
    delete rep_->nodes[i];
  }
  delete rep_;
}

bool GraphCycles::HasEdge(int32_t x, int32_t y) const {
  return rep_->nodes[x]->out.Contains(y);
}

void GraphCycles::RemoveEdge(int32_t x, int32_t y) {
  rep_->nodes[x]->out.Erase(y);
  rep_->nodes[y]->in.Erase(x);
  // No need to update the rank assignment since a previous valid
  // rank assignment remains valid after an edge deletion.
}

static bool forwardDfs(GraphCycles::Rep* r, int32_t n, int32_t upperBound);
static void backwardDfs(GraphCycles::Rep* r, int32_t n, int32_t lowerBound);
static void reorder(GraphCycles::Rep* r);
static void sort(const Vec<Node*>&, Vec<int32_t>* delta);
static void moveToList(GraphCycles::Rep* r, Vec<int32_t>* src,
                       Vec<int32_t>* dst);
static void clearVisitedBits(GraphCycles::Rep* r, const Vec<int32_t>& nodes);

bool GraphCycles::InsertEdge(int32_t x, int32_t y) {
  if (x == y) return false;
  Rep* r = rep_;
  Node* nx = r->nodes[x];
  if (!nx->out.Insert(y)) {
    // Edge already exists.
    return true;
  }

  Node* ny = r->nodes[y];
  ny->in.Insert(x);

  if (nx->rank <= ny->rank) {
    // New edge is consistent with existing rank assignment.
    return true;
  }

  // Current rank assignments are incompatible with the new edge.  Recompute.
  // We only need to consider nodes that fall in the range [ny->rank,nx->rank].
  if (forwardDfs(r, y, nx->rank)) {
    // Found a cycle.  Undo the insertion and tell caller.
    nx->out.Erase(y);
    ny->in.Erase(x);
    // Since we do not call Reorder() on this path, clear any visited
    // markers left by ForwardDFS.
    clearVisitedBits(r, r->deltaf);
    return false;
  }
  backwardDfs(r, x, ny->rank);
  reorder(r);
  return true;
}

// Follows the edges from producer to consumer and searchs if the node having
// rank `n` can reach the node having rank `upper_bound` using a DFS search.
// When doing DFS search, We only consider the pathes that satisfy the ranks
// of the nodes of the path are all smaller than `upper_bound`.
//
// Returns true if such path exists.
static bool forwardDfs(GraphCycles::Rep* r, int32_t n, int32_t upperBound) {
  // Avoid recursion since stack space might be limited.
  // We instead keep a stack of nodes to visit.
  r->deltaf.clear();
  r->stack.clear();
  r->stack.push_back(n);
  while (!r->stack.empty()) {
    n = r->stack.back();
    r->stack.pop_back();
    Node* nn = r->nodes[n];
    if (nn->visited) continue;

    nn->visited = true;
    r->deltaf.push_back(n);

    for (auto w : nn->out.GetSequence()) {
      Node* nw = r->nodes[w];
      if (nw->rank == upperBound) {
        return true;
      }
      if (!nw->visited && nw->rank < upperBound) {
        r->stack.push_back(w);
      }
    }
  }
  return false;
}

// Follows the edges from consumer to producer and visit all the nodes that
// is reachable from node `n` and have rank larger than `lower_bound`.
static void backwardDfs(GraphCycles::Rep* r, int32_t n, int32_t lowerBound) {
  r->deltab.clear();
  r->stack.clear();
  r->stack.push_back(n);
  while (!r->stack.empty()) {
    n = r->stack.back();
    r->stack.pop_back();
    Node* nn = r->nodes[n];
    if (nn->visited) continue;

    nn->visited = true;
    r->deltab.push_back(n);

    for (auto w : nn->in.GetSequence()) {
      Node* nw = r->nodes[w];
      if (!nw->visited && lowerBound < nw->rank) {
        r->stack.push_back(w);
      }
    }
  }
}

// Recomputes rank assignments to make them compatible with the edges (producer
// has smaller rank than its consumer)
static void reorder(GraphCycles::Rep* r) {
  sort(r->nodes, &r->deltab);
  sort(r->nodes, &r->deltaf);

  // Adds contents of delta lists to list (backwards deltas first).
  r->list.clear();
  moveToList(r, &r->deltab, &r->list);
  moveToList(r, &r->deltaf, &r->list);

  // Produce sorted list of all ranks that will be reassigned.
  r->merged.resize(r->deltab.size() + r->deltaf.size());
  std::merge(r->deltab.begin(), r->deltab.end(), r->deltaf.begin(),
             r->deltaf.end(), r->merged.begin());

  // Assign the ranks in order to the collected list.
  for (Vec<int32_t>::size_type i = 0, e = r->list.size(); i < e; ++i) {
    r->nodes[r->list[i]]->rank = r->merged[i];
  }
}

// Sorts nodes in the vector according to their ranks. Small rank first.
static void sort(const Vec<Node*>& nodes, Vec<int32_t>* delta) {
  struct ByRank {
    const Vec<Node*>* nodes;
    bool operator()(int32_t a, int32_t b) const {
      return (*nodes)[a]->rank < (*nodes)[b]->rank;
    }
  };
  ByRank cmp;
  cmp.nodes = &nodes;
  std::sort(delta->begin(), delta->end(), cmp);
}

// Collects ranks of nodes in vector `src` to vector `dst`
static void moveToList(GraphCycles::Rep* r, Vec<int32_t>* src,
                       Vec<int32_t>* dst) {
  for (Vec<int32_t>::size_type i = 0, e = src->size(); i < e; i++) {
    int32_t w = (*src)[i];
    // Replace src entry with its rank
    (*src)[i] = r->nodes[w]->rank;
    // Prepare for future DFS calls
    r->nodes[w]->visited = false;
    dst->push_back(w);
  }
}

// Clears bookkeeping fileds used during the last DFS process.
static void clearVisitedBits(GraphCycles::Rep* r, const Vec<int32_t>& nodes) {
  for (Vec<int32_t>::size_type i = 0, e = nodes.size(); i < e; i++) {
    r->nodes[nodes[i]]->visited = false;
  }
}

bool GraphCycles::IsReachable(int32_t x, int32_t y) {
  if (x == y) return true;
  Rep* r = rep_;
  Node* nx = r->nodes[x];
  Node* ny = r->nodes[y];

  if (nx->rank >= ny->rank) {
    // x cannot reach y since it is after it in the topological ordering
    return false;
  }

  // See if x can reach y using a DFS search that is limited to y's rank
  bool reachable = forwardDfs(r, x, ny->rank);

  // Clear any visited markers left by ForwardDFS.
  clearVisitedBits(r, r->deltaf);
  return reachable;
}

std::optional<int32_t> GraphCycles::ContractEdge(int32_t a, int32_t b) {
  assert(HasEdge(a, b));
  RemoveEdge(a, b);

  if (IsReachable(a, b)) {
    // Restore the graph to its original state.
    InsertEdge(a, b);
    return {};
  }

  if (rep_->nodes[b]->in.Size() + rep_->nodes[b]->out.Size() >
      rep_->nodes[a]->in.Size() + rep_->nodes[a]->out.Size()) {
    // Swap "a" and "b" to minimize copying.
    std::swap(a, b);
  }

  Node* nb = rep_->nodes[b];
  OrderedNodeSet out = std::move(nb->out);
  OrderedNodeSet in = std::move(nb->in);
  for (int32_t y : out.GetSequence()) {
    rep_->nodes[y]->in.Erase(b);
  }
  for (int32_t y : in.GetSequence()) {
    rep_->nodes[y]->out.Erase(b);
  }
  rep_->freeNodes.push_back(b);

  rep_->nodes[a]->out.Reserve(rep_->nodes[a]->out.Size() + out.Size());
  for (int32_t y : out.GetSequence()) {
    InsertEdge(a, y);
  }

  rep_->nodes[a]->in.Reserve(rep_->nodes[a]->in.Size() + in.Size());
  for (int32_t y : in.GetSequence()) {
    InsertEdge(y, a);
  }

  // Note, if the swap happened it might be what originally was called "b".
  return a;
}

std::vector<int32_t> GraphCycles::SuccessorsCopy(int32_t node) const {
  return rep_->nodes[node]->out.GetSequence();
}

namespace {
void sortInPostOrder(const Vec<Node*>& nodes, std::vector<int32_t>* toSort) {
  std::sort(toSort->begin(), toSort->end(), [&](int32_t a, int32_t b) {
    return nodes[a]->rank > nodes[b]->rank;
  });
}
}  // namespace

std::vector<int32_t> GraphCycles::AllNodesInPostOrder() const {
  llvm::DenseSet<int32_t> freeNodesSet;
  for (int32_t n : rep_->freeNodes) freeNodesSet.insert(n);

  std::vector<int32_t> allNodes;
  allNodes.reserve(rep_->nodes.size() - freeNodesSet.size());
  for (size_t i = 0, e = rep_->nodes.size(); i < e; i++) {
    if (!freeNodesSet.count(i)) {
      allNodes.push_back(i);
    }
  }

  sortInPostOrder(rep_->nodes, &allNodes);
  return allNodes;
}

}  // namespace mlir
