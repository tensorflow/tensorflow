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

#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

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
                       gtl::InlinedVector<const Node*, 4>* control_edges,
                       gtl::InlinedVector<std::pair<const Node*, int>, 4>* in) {
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

class Hasher {
 public:
  uint64 hash() { return h_ == kIllegalNodeHash ? kIllegalNodeHash + 1 : h_; }

  void MixString(const string& s) { h_ = Hash64(s.data(), s.size(), h_); }

  void MixInteger(size_t z) { h_ = Hash64Combine(h_, z); }

  void MixProto(const protobuf::MessageLite& msg) {
    msg.ByteSizeLong();  // Ensure sizes are cached accurately.
    HashingOutputStream hasher;
    {
      // CodedOutputStream doesn't call BackUp until it's destroyed, so we need
      // it to be destroyed before we call hasher.hash().
      protobuf::io::CodedOutputStream stream(&hasher);
      stream.EnableAliasing(true);
      stream.SetSerializationDeterministic(true);
      msg.SerializeWithCachedSizes(&stream);
    }
    h_ = Hash64Combine(h_, hasher.hash());
  }

 private:
  // HashingOutputStream produces the same exact hash as if you serialized the
  // proto and hashed it sequentially in kBufSize chunks, except it doesn't
  // manifest the entire proto into memory at any point.
  class HashingOutputStream : public protobuf::io::ZeroCopyOutputStream {
   public:
    // This kBufSize makes sizeof(HashingOutputStream) == 256.  It's not chosen
    // for any particular reason except it's a nice even number of cache lines.
    static constexpr size_t kBufSize = 228;
    static constexpr uint64 kDefaultSeed = 2570847921467975139ULL;
    bool Next(void** data, int* size) override {
      if (i_ == kBufSize) {
        // Mix the chunk in.
        Mix(buf_, kBufSize);
        *data = buf_;
        *size = kBufSize;
      } else {
        *data = buf_ + i_;
        *size = kBufSize - i_;
      }
      // We always set i_ to be past the end, since we've given the rest of buf_
      // out.
      i_ = kBufSize;
      return true;
    }

    void BackUp(int count) override { i_ -= count; }

    int64_t ByteCount() const override { return byte_count_; }

    bool WriteAliasedRaw(const void* void_data, int size) override {
      // We can't do math on void*.
      const char* data = static_cast<const char*>(void_data);
      const auto remaining = kBufSize - i_;
      if (remaining > 0) {
        if (size < remaining) {
          memcpy(buf_ + i_, data, size);
          i_ += size;
          return true;
        }
        memcpy(buf_ + i_, data, remaining);
        i_ = kBufSize;
        data += remaining;
        size -= remaining;
      }
      if (i_ == kBufSize) {
        Mix(buf_, kBufSize);
        i_ = 0;
      }
      while (size >= kBufSize) {
        Mix(data, kBufSize);
        data += kBufSize;
        size -= kBufSize;
      }
      memcpy(buf_, data, size);
      i_ = size;
      return true;
    }

    bool AllowsAliasing() const override { return true; }

    uint64 hash() {
      if (i_ != 0) {
        Mix(buf_, i_);
        i_ = 0;
      }
      return h_;
    }

   private:
    void Mix(const char* p, size_t n) {
      byte_count_ += n;
      h_ = Hash64(p, n, h_);
    }
    char buf_[kBufSize];
    int i_ = 0;
    int64_t byte_count_ = 0;
    uint64 h_ = kDefaultSeed;
  };

  uint64 h_ = HashingOutputStream::kDefaultSeed;
};

size_t OptimizerCSE::NodeHash(const Node* n) {
  Hasher hasher;
  hasher.MixString(n->type_string());
  hasher.MixInteger(n->output_types().size());
  for (DataType dt : n->output_types()) {
    hasher.MixInteger(dt);
  }

  hasher.MixInteger(n->num_inputs());
  gtl::InlinedVector<const Node*, 4> control_edges;
  gtl::InlinedVector<std::pair<const Node*, int>, 4> in(n->num_inputs());
  FillInputs(n, &control_edges, &in);
  for (const auto& edge : in) {
    hasher.MixInteger(edge.first->id());
    hasher.MixInteger(edge.second);
  }

#if !defined(__ANDROID__)
  // Hash the attrs.  For example, this makes sure different constants
  // end up in different hash buckets.
  size_t attr_hashes = 0;
  for (const auto& attr : n->attrs()) {
    Hasher h;
    h.MixString(attr.first);
    h.MixProto(attr.second);
    attr_hashes = Hash64CombineUnordered(attr_hashes, h.hash());
  }
  hasher.MixInteger(attr_hashes);
#endif

  return hasher.hash();
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
  gtl::InlinedVector<const Node*, 4> a_control_edges;
  gtl::InlinedVector<const Node*, 4> b_control_edges;
  gtl::InlinedVector<std::pair<const Node*, int>, 4> a_in(N_in);
  gtl::InlinedVector<std::pair<const Node*, int>, 4> b_in(N_in);
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

    // Don't prune placeholder nodes.
    if (n->type_string() == "Placeholder" ||
        n->type_string() == "PlaceholderV2" ||
        n->type_string() == "PlaceholderWithDefault") {
      continue;
    }

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

      MergeDebugInfo(NodeDebugInfo(*n), *candidate);
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
