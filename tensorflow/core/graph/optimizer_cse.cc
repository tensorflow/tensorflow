/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// This module implements a common subexpression elimination pass.  We
// process the nodes in the graph in reverse postorder
// (i.e. inputs before their downstream dependencies).  The rough algorithm is
// as follows:
//
// std::unordered_map<size_t, Node*> available
// for each node n in forward topological order:
//   h = NodeHash(n)
//   if available[h] exists and Equivalent(available(h), h)
//     redirect downstream uses of outputs of n to available[h]
//     remove n from graph
//   else
//     if available[h] does not exist
//       available[h] = n
//
// This is similar to the global value number algorithm describe in this
// paper:
//
// "Global code motion/global value numbering", Cliff Click, PLDI '95
// Proceedings of the ACM SIGPLAN 1995 conference on Programming
// language design and implementation, Pages 246-257
//      http://dl.acm.org/citation.cfm?id=207154

#include "tensorflow/core/graph/optimizer_cse.h"

#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

class OptimizerCSE {
 public:
  explicit OptimizerCSE(Graph* g) : g_(g) {}

  bool Optimize(const std::function<bool(const Node*)>& consider_fn);

 private:
  static size_t NodeHash(const Node* n);
  static bool Equivalent(const Node* a, const Node* b,
                         AttrSlice::Scratch* scratch);

  Graph* g_;
};

static void FillInputs(const Node* n,
                       gtl::InlinedVector<Node*, 4>* control_edges,
                       gtl::InlinedVector<std::pair<Node*, int>, 4>* in) {
  DCHECK_EQ(in->size(), n->num_inputs());
  control_edges->clear();
  for (const Edge* e : n->in_edges()) {
    if (e->IsControlEdge()) {
      control_edges->push_back(e->src());
    } else {
      (*in)[e->dst_input()] = std::make_pair(e->src(), e->src_output());
    }
  }
  std::sort(control_edges->begin(), control_edges->end());
  if (n->op_def().is_commutative()) {
    // For commutative inputs, we sort the input by the input Node*
    // to get a canonical ordering (so that add(a,b) and add(b, a) will
    // hash to the same value if is_commutative is true for 'add').
    std::sort(in->begin(), in->end());
  }
}

static size_t kIllegalNodeHash = 0;

size_t OptimizerCSE::NodeHash(const Node* n) {
  const DataTypeVector& out = n->output_types();
  string str_to_hash = strings::StrCat(n->type_string(), out.size());
  for (DataType dt : out) {
    strings::StrAppend(&str_to_hash, dt);
  }

  const int N_in = n->num_inputs();
  strings::StrAppend(&str_to_hash, N_in);
  gtl::InlinedVector<Node*, 4> control_edges;
  gtl::InlinedVector<std::pair<Node*, int>, 4> in(N_in);
  FillInputs(n, &control_edges, &in);
  for (const auto& edge : in) {
    strings::StrAppend(&str_to_hash, edge.first->id(), edge.second);
  }

  size_t h = Hash64(str_to_hash);

#if !defined(__ANDROID__)
  // Hash the attrs.  For example, this makes sure different constants
  // end up in different hash buckets.
  string tmp;
  for (const auto& attr : n->attrs()) {
    tmp = attr.first;
    attr.second.AppendToString(&tmp);
    // Add hashes of attrs, so the order of attrs doesn't matter.
    h += Hash32(tmp.data(), tmp.size(), 0x87341245);
  }
#endif

  if (h == kIllegalNodeHash) h = kIllegalNodeHash + 1;
  return h;
}

static bool HasRefInput(const Node* n) {
  for (auto dt : n->input_types()) {
    if (IsRefType(dt)) return true;
  }
  return false;
}

bool OptimizerCSE::Equivalent(const Node* a, const Node* b,
                              AttrSlice::Scratch* scratch) {
  // Different op names are different
  if (a->type_string() != b->type_string()) return false;

  // Never consider stateful nodes (such as non-const inputs) equivalent.
  if (a->op_def().is_stateful()) return false;

  // For now, we consider any node that takes a ref input to not be
  // equivalent to any other node.
  if (HasRefInput(a) || HasRefInput(b)) return false;

  // Compare attrs.  Note that equal attrs implies equal input and
  // output types.
  if (!a->attrs().EqualAttrs(b->attrs(), scratch)) return false;

  // Compare input sources
  if (a->num_inputs() != b->num_inputs()) return false;
  const int N_in = a->num_inputs();
  gtl::InlinedVector<Node*, 4> a_control_edges;
  gtl::InlinedVector<Node*, 4> b_control_edges;
  gtl::InlinedVector<std::pair<Node*, int>, 4> a_in(N_in);
  gtl::InlinedVector<std::pair<Node*, int>, 4> b_in(N_in);
  FillInputs(a, &a_control_edges, &a_in);
  FillInputs(b, &b_control_edges, &b_in);
  if (a_in != b_in) return false;
  if (a_control_edges != b_control_edges) return false;

  return true;
}

bool OptimizerCSE::Optimize(
    const std::function<bool(const Node*)>& consider_fn) {
  // This very simple implementation works if the whole graph is one
  // giant basic block (because we just traverse nodes in a
  // topological order). This simple implementation works well
  // with control flow/loops/etc. But we need to be careful about
  // control flow if we want to add more sophisticated CSE optimizations.

  // TODO(jeff): We need to handle Update nodes specially, but dealing
  // with more general control flow will also solve this issue, and for
  // now, our updates are almost always the most downstream nodes in
  // the graph.
  std::vector<Node*> order;
  GetReversePostOrder(*g_, &order);

  // Our value is just a single Node*, meaning we keep just a single
  // candidate for a given node hash value.  This may cause us to
  // (rarely) lose some optimization opportunities if there are
  // hash collisions, but it allows us to avoid having the value
  // be a set<Node*> (or equivalent).
  std::unordered_map<size_t, Node*> available;

  // Scratch space for Equivalent calls.  Allocated here and passed in to
  // Equivalent to avoid allocation inside the loop below.
  bool changed = false;
  AttrSlice::Scratch scratch;
  for (Node* n : order) {
    if (!n->IsOp()) continue;

    // See if we should consider this node at all
    if (consider_fn != nullptr && !consider_fn(n)) continue;

    size_t h = NodeHash(n);
    Node** candidate = &available[h];
    if (*candidate == nullptr) {
      // No existing match: insert "n" into the hash table under "h"
      *candidate = n;
    } else if (Equivalent(*candidate, n, &scratch)) {
      VLOG(1) << "CSE: equivalent: " << (*candidate)->name() << " and "
              << n->name();
      // *candidate and n are equivalent.  Therefore, we can replace
      // n with *candidate by fixing up outgoing edges from "n" to instead
      // come from "*candidate", and then delete n from the graph
      for (const Edge* e : n->out_edges()) {
        g_->AddEdge(*candidate, e->src_output(), e->dst(), e->dst_input());
      }
      g_->RemoveNode(n);
      changed = true;
    }
  }
  return changed;
}

bool OptimizeCSE(Graph* g,
                 const std::function<bool(const Node*)>& consider_fn) {
  OptimizerCSE opt(g);
  return opt.Optimize(consider_fn);
}

}  // namespace tensorflow
