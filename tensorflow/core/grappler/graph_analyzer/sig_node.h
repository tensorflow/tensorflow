/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_SIG_NODE_H_
#define TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_SIG_NODE_H_

#include <map>
#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/graph_analyzer/gen_node.h"
#include "tensorflow/core/grappler/graph_analyzer/hash_tools.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {

namespace test {
class SigBaseTest;
}  // end namespace test

class SigNode;

// To find nodes by name. Having the map ordered makes the tests easier,
// and it isn't used in production code often enough to get any win from
// using an unordered map.
using SigNodeMap = std::map<string, std::unique_ptr<SigNode>>;

// One node in the graph, in the form convenient for generation of the signature
// of the graph, and comparison of two (sub)graphs for equivalence. It refers to
// the original NodeDef protobuf for most information and adds the extra
// enrichment.
//
// The graph building is 2-stage: first match a SigNode with each NodeDef and
// collect them into a map that finds them by name, then process the map,
// deep-parse the underlying NodeDefs and connect the SigNodes together.
class SigNode {
 public:
  friend struct Signature;

  // Will keep the pointer to the underlying NodeDef, so that
  // underlying object must not be deleted while SigNode is alive.
  explicit SigNode(const NodeDef* node);

  // Access wrappers.
  const string& name() const { return node_->name(); }
  const string& opcode() const { return node_->op(); }
  const NodeDef* node_def() const { return node_; }

  // For extraction of subgraphs into a separate SigNodeMap, copies the links
  // that point inside the subgraph from a full-graph SigNode to a subgraph
  // SigNode. The translation map defines the subgraph and gives the mapping
  // from the nodes in the full graph to the matching nodes in subgraph.
  using TranslationMap =
      std::unordered_map<const GenNode* /*full_graph*/, SigNode* /*subgraph*/>;
  void CopyLinks(const GenNode& from, const TranslationMap& map);

  // A link is an edge of the graph that connects 2 nodes. Each of the connected
  // nodes has its own perspective on the link, seeing its local port, remote
  // port and the remote node. The direction of the link is encoded in the
  // ports, one port is always incoming and another one outgoing.
  //
  // The link tag here contains both ports of the link viewed from the
  // perspective of this node; consisting of both the local port (i.e. at this
  // node) and remote port (i.e. on the other node), the local one going first.
  struct LinkTag {
    struct Hasher {
      size_t operator()(const LinkTag& tag) const noexcept {
        size_t hval = port_hasher(tag.local);
        CombineHash(port_hasher(tag.remote), &hval);
        return hval;
      }
      GenNode::Port::Hasher port_hasher;
    };

    LinkTag(GenNode::Port a_local, GenNode::Port a_remote)
        : local(a_local), remote(a_remote) {}

    // The default constructor is used for the default values in maps.
    // (false, 99) is an arbitrary value that makes the uninitialized
    // links easy to tell when debugging (they should never happen).
    LinkTag() : local(false, 99), remote(false, 99) {}

    // Port of the link on the local node.
    GenNode::Port local;
    // Port of the link on the remote node.
    GenNode::Port remote;

    bool operator==(const LinkTag& other) const {
      return local == other.local && remote == other.remote;
    }
    bool operator<(const LinkTag& other) const {
      return local < other.local ||
             (local == other.local && remote < other.remote);
    }
  };

  // Since the signature logic doesn't differentiate between the links
  // with the same tag (other than by the "peer" nodes on their other ends),
  // all the links with the same tag are grouped into a single structure.
  struct Link {
    LinkTag tag;
    size_t unique_hash;  // Hash of the tag after conflict resolution.
    // The remote node(s) on the other side on the link(s).
    using PeerVector = std::vector<SigNode*>;
    PeerVector peers;
  };

  // A way to look up the link description by its hash.
  using LinkHashMap = std::map<size_t, Link>;
  const LinkHashMap& hash_to_link() const { return hash_to_link_; }

  // The enumeration of all the peer nodes in a predictable order.
  // Before the signature generation, only the link values determine the
  // order, after the signature generation the entries at the same
  // links get further sorted by their peer node ranks.
  struct HashedPeer {
    HashedPeer(size_t l, SigNode* p) : link_hash(l), peer(p) {}

    struct LessByRank {
      bool operator()(const SigNode::HashedPeer& left,
                      const SigNode::HashedPeer& right) {
        return left.peer->unique_rank_ < right.peer->unique_rank_;
      }
    };

    size_t link_hash;
    SigNode* peer;
  };
  using HashedPeerVector = std::vector<HashedPeer>;
  const HashedPeerVector& hashed_peers() const { return hashed_peers_; }

  // Compares two nodes in two different graphs for equivalence (two nodes in
  // the same graph would never be equivalent). Expects that the signatures of
  // the graphs have already been computed, so unique_rank_ is filled in and
  // the hashed_peers_ properly ordered.
  bool operator==(const SigNode& other) const;

  bool operator!=(const SigNode& other) const { return !(*this == other); }

 private:
  friend class test::SigBaseTest;

  // The CopyLinks code is split into 2 parts for testability.
  // The first pass builds a map ordered by LinkTag for predictability.
  void CopyLinksPass1(const GenNode& from, const TranslationMap& map,
                      std::map<LinkTag, Link>* link_map);
  // The second pass converts to the map by hash value,
  // resolves any hash conflicts, and builds the hashed peer vector.
  void CopyLinksPass2(std::map<LinkTag, Link>* link_map);

  // Computes the topological hash at distance 0. Resets the topo_hash_ vector
  // and hashed_nodes_;
  void ComputeTopoHash0();

  // Compute the topological has at the given distance. The hashes for all the
  // lower distances must be already computed for all the nodes in the graph.
  // Also computes next_hashed_nodes_ from last_hashed_nodes_.
  void ComputeTopoHash(int distance);

  // Get the hash value for a particular distance. It must be previously
  // computed.
  size_t GetTopoHash(int distance) const;

  // The hash value for the highest computed distance. It must be previously
  // computed.
  size_t GetHighTopoHash() const {
    CHECK(!topo_hash_.empty());
    return topo_hash_.back();
  }

  // Rehash the topmost hash, to avoid conflicts.
  void ReHighTopoHash() {
    CHECK(!topo_hash_.empty());
    CombineHash(1, &topo_hash_.back());
  }

  // Ordering by node order and highest available hash (it must be
  // previously computed).
  struct NodeOrderLess {
    bool operator()(const SigNode* left, const SigNode* right) {
      return left->topo_hash_.back() < right->topo_hash_.back();
    }
  };

 private:
  const NodeDef* node_;

  // The bitmap mask with 1 bit set that represents this node in the set
  // during the computation of the signature.
  uint64_t node_mask_ = 0;

  // The code that populates this map makes sure that there are no hash
  // conflicts, rehashing if necessary.
  LinkHashMap hash_to_link_;

  // The enumeration of all the direct peers in the predictable order (which
  // happens to be the order ot their link tags, but the order of the hashes
  // would do too). It is used for the quick enumeration during the signature
  // computation. After the signature building is completed, the entries that
  // have the same link tag get further sorted in the order of the ranks of
  // their nodes.
  HashedPeerVector hashed_peers_;

  // The unique rank represents the order in which the node will be included
  // into the signature. It gets assigned in order either when the topo_hash_ of
  // this node becomes unique in the graph, or when the nodes are completely
  // equivalent, one of them is picked at random to assign the next rank, and
  // then the rest of the nodes attempt to disambiguate based on that
  // information.
  size_t unique_rank_ = ~0;
  // When hash_is_final_ is set, the topo_has_ vector stops growing, and the
  // last value from it is used for all the further hashes.
  bool hash_is_final_ = false;
  // The hashes that include the topology of the nodes up to the distance N. The
  // hash for distance 0 is produced from the attributes of this node itself and
  // its general connectivity properties but no information about the
  // neighboring nodes. The hash for distance D+1 is build from hashes at level
  // D of this node and of all its immediate neighbors. The neighbors that are
  // connected by equivalent links are included in a commutative way.
  std::vector<size_t> topo_hash_;
  // The set of nodes that got included into the computation of the
  // last topo_hash_ entry.
  uint64_t last_hashed_nodes_ = 0;
  // The next set of nodes that gets used for the current topo_hash entry.
  uint64_t next_hashed_nodes_ = 0;
};

// Signature of a graph. The computation is intertwined with the private methods
// of SigNode, so keeping both in the same file looks more convenient.
struct Signature {
  friend class test::SigBaseTest;

  // Maximal size of the graphs for which the signature can be computed.
  // Changing this constant won't magically add the support for a larger size,
  // the rest of implementation would have to be extended. The value of 64 is
  // driven by the size of a bitset in an uint64_t, and should be enough for our
  // purposes, while having a high efficiency of implementation.
  static constexpr int kMaxGraphSize = 64;

  // Using the map, computes the rest of the fields of a signature.
  // Returns an error is the graph is too big.
  absl::Status Compute();

  // Convert the computed signature to a string representation.
  string ToString() const;

  SigNodeMap map;        // The nodes in the graph, accessible by name.
  size_t sig_short = 0;  // Hash of the signature, for the quick equality check.
  // The full signature: hashes of the nodes in a predictable order.
  std::vector<size_t> sig_full;
  // The nodes in the same order as they go in the signature.
  std::vector<SigNode*> nodes;

  // For building the unordered maps.
  size_t Hash() const { return sig_short; }

  // Returns true if the graphs are equivalent. The signature must be already
  // computed.
  bool operator==(const Signature& other) const;

 private:
  // Populates the nodes vector from the map and initializes the state of the
  // nodes for the signature computation.
  void PrepareNodes();

  // Finds the nodes with the hashes that are unique and assigns the unique ids
  // to them. If there are nodes with non-unique hashes, exactly one node from
  // the first such sequence (in the order of hash values) will be picked and
  // assigned a unique id. Assumes that the nodes[0...(next_node_id-1)] have
  // been already assigned the unique ids. Advances next_node_id by at least 1.
  void FindUniqueHashes(size_t* next_node_id_p);

  // One round of the signature computation. Assumes that the
  // nodes[0...(next_node_id-1)] have been already assigned the fixed
  // positions, and thus computes the hashes only for the remaining nodes.
  void ComputeOneRound(size_t next_node_id);

  // Additional ordering of the hashed_peers_ links in the nodes, so that they
  // can be compared and printed in a predictable order.
  void OrderLinks();
};

}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_SIG_NODE_H_
