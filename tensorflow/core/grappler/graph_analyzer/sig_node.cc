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

#include "tensorflow/core/grappler/graph_analyzer/sig_node.h"

#include <algorithm>

#include "absl/strings/str_format.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {

static constexpr bool debug = false;

//=== SigNode

SigNode::SigNode(const NodeDef* node) : node_(node) {}

void SigNode::CopyLinks(const GenNode& from, const TranslationMap& map) {
  hash_to_link_.clear();
  hashed_peers_.clear();

  std::map<LinkTag, Link> link_map;
  CopyLinksPass1(from, map, &link_map);
  CopyLinksPass2(&link_map);
}

void SigNode::CopyLinksPass1(const GenNode& from, const TranslationMap& map,
                             std::map<LinkTag, Link>* link_map) {
  LinkTag::Hasher link_hasher;

  for (const auto& entry : from.links()) {
    for (const auto& target : entry.second) {
      auto nodeit = map.find(target.node);
      if (nodeit == map.end()) {
        // Node is not in the subgraph, ignore.
        continue;
      }

      LinkTag tag(entry.first, target.port);
      size_t hval = link_hasher(tag);

      // This instantiates the entry if it was not present.
      Link& map_entry = (*link_map)[tag];
      if (map_entry.peers.empty()) {
        map_entry.tag = tag;
        map_entry.unique_hash = hval;
      }
      map_entry.peers.push_back(nodeit->second);
    }
  }
}

void SigNode::CopyLinksPass2(std::map<LinkTag, Link>* link_map) {
  for (auto& entry : *link_map) {
    Link* hl_entry_ptr = &hash_to_link_[entry.second.unique_hash];
    // In case of a conflict, rehash. This should almost never happen.
    // Because the order of iteration is predictable, the rehashed values
    // will also be predictable.
    while (!hl_entry_ptr->peers.empty()) {
      CombineHash(1, &entry.second.unique_hash);
      hl_entry_ptr = &hash_to_link_[entry.second.unique_hash];
    }

    for (const auto& peer : entry.second.peers) {
      hashed_peers_.emplace_back(HashedPeer(entry.second.unique_hash, peer));
    }

    hl_entry_ptr->tag = entry.second.tag;
    hl_entry_ptr->unique_hash = entry.second.unique_hash;
    hl_entry_ptr->peers.swap(entry.second.peers);
  }
}

void SigNode::ComputeTopoHash0() {
  topo_hash_.clear();
  last_hashed_nodes_ = next_hashed_nodes_ = node_mask_;

  // TODO(babkin): include the attributes too, as an option.
  size_t hval = std::hash<string>()(opcode());

  // Getting the topology of the links in to the hash early should get more
  // conflicts resolved early.
  for (const auto& entry : hashed_peers_) {
    CombineHash(entry.link_hash, &hval);
  }

  topo_hash_.push_back(hval);
}

void SigNode::ComputeTopoHash(int distance) {
  // The new starting point.
  next_hashed_nodes_ = last_hashed_nodes_;
  if (debug) {
    LOG(INFO) << "DEBUG    node " << name() << " mask=" << std::hex
              << next_hashed_nodes_;
  }

  if (hash_is_final_) {
    return;
  }

  const int64_t topo_hash_size = topo_hash_.size();
  CHECK(topo_hash_size == distance);

  int prev = distance - 1;

  // Start with own's local topology hash. This value is stable, so
  // if the hashes of the surrounding nodes don't change on the following
  // distances, the hash of this node won't change either.
  size_t hval = topo_hash_[0];

  if (!hashed_peers_.empty()) {
    size_t last_link_hash = hashed_peers_[0].link_hash;
    size_t comm_hash = 0;

    for (const auto& entry : hashed_peers_) {
      if (entry.link_hash != last_link_hash) {
        CombineHash(last_link_hash, &hval);
        CombineHash(comm_hash, &hval);
        comm_hash = 0;
        last_link_hash = entry.link_hash;
      }

      // The links in the same vector are commutative, so combine their
      // hashes in a commutative way.
      CombineHashCommutative(entry.peer->GetTopoHash(prev), &comm_hash);
      next_hashed_nodes_ |= entry.peer->last_hashed_nodes_;
      if (debug) {
        LOG(INFO) << "DEBUG    node " << name() << " += " << entry.peer->name()
                  << " mask=" << std::hex << next_hashed_nodes_;
      }
    }

    // The last commutative group.
    CombineHash(last_link_hash, &hval);
    CombineHash(comm_hash, &hval);
  }

  topo_hash_.push_back(hval);
}

size_t SigNode::GetTopoHash(int distance) const {
  CHECK(!topo_hash_.empty());
  const int64_t topo_hash_size = topo_hash_.size();
  if (distance >= topo_hash_size) {
    CHECK(hash_is_final_);
    return topo_hash_.back();
  } else {
    return topo_hash_[distance];
  }
}

bool SigNode::operator==(const SigNode& other) const {
  // TODO(babkin): add attributes too.
  if (opcode() != other.opcode()) {
    return false;
  }

  // Normally the caller is expected to compare the nodes
  // at the same rank in different graphs, but just in case...
  if (unique_rank_ != other.unique_rank_) {
    return false;
  }

  if (hashed_peers_.size() != other.hashed_peers_.size()) {
    return false;
  }

  for (auto it1 = hashed_peers_.begin(), it2 = other.hashed_peers_.begin();
       it1 != hashed_peers_.end(); ++it1, ++it2) {
    // TODO(babkin): might compare the actual values too
    // but the hash is probably just as good.
    if (it1->link_hash != it2->link_hash) {
      return false;
    }
    if (it1->peer->unique_rank_ != it2->peer->unique_rank_) {
      return false;
    }
  }

  return true;
}

//=== Signature

constexpr int Signature::kMaxGraphSize;

string Signature::ToString() const {
  string result;
  for (size_t n = 0; n < nodes.size(); ++n) {
    // TODO(babkin): add attributes too.
    result += absl::StrFormat("%d:%s", n, nodes[n]->opcode());
    for (const auto& entry : nodes[n]->hashed_peers_) {
      const auto& link = nodes[n]->hash_to_link_[entry.link_hash];

      // The link entries are already sorted, by tags and then by the
      // node ranks.
      if (link.tag.local.IsInbound()) {
        result +=
            absl::StrFormat("[%s:%s:%d]", string(link.tag.local),
                            string(link.tag.remote), entry.peer->unique_rank_);
      }
    }
    result.push_back(',');
  }
  return result;
}

Status Signature::Compute() {
  if (map.size() > kMaxGraphSize) {
    return Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "A graph of %d nodes is too big for signature computation, "
            "the maximal supported node count is %d.",
            map.size(), kMaxGraphSize));
  }

  // The value that will be assigned next as the unique node id.
  // This also means that all the entries in nodes at indexes less than this
  // have been finalized and don't need to be touched any more.
  size_t next_node_id = 0;

  sig_short = 0;
  sig_full.resize(0);  // Keep the storage.

  // The main signature generation.
  PrepareNodes();
  FindUniqueHashes(&next_node_id);
  while (next_node_id < map.size()) {
    ComputeOneRound(next_node_id);
    FindUniqueHashes(&next_node_id);
  }

  OrderLinks();

  return OkStatus();
}

void Signature::PrepareNodes() {
  nodes.resize(0);  // Keep the storage.

  // Initialize the nodes.
  int64_t mask = 1;
  for (const auto& entry : map) {
    SigNode* node = entry.second.get();
    node->last_hashed_nodes_ = node->node_mask_ = mask;
    mask <<= 1;
    node->unique_rank_ = ~0;
    node->hash_is_final_ = false;
    node->ComputeTopoHash0();
    if (node->GetHighTopoHash() <= map.size()) {
      // Would conflict with one of the reserved values.
      node->ReHighTopoHash();
    }

    // The initial order is random.
    nodes.emplace_back(node);
  }
}

void Signature::FindUniqueHashes(size_t* next_node_id_p) {
  // Start by sorting by the hash value.
  std::stable_sort(nodes.begin() + *next_node_id_p, nodes.end(),
                   SigNode::NodeOrderLess());

  // At each call, if no nodes have unique hashes, one node that has a
  // non-unique (shared) hash can be made unique by assigning a unique id.
  // This node gets picked predictably by taking the last node.
  // TODO(babkin): Technically, more than one node can be unshared,
  // as long as their last_hashed_nodes_ overlap only by the nodes that
  // already had the assigned ids before the current round. But it's not clear
  // yet, how often would this beneficial, because it looks like for many
  // subgraphs unsharing one node should be enough to untangle them. This
  // would need more measurement before implementing.
  bool found_unique = false;
  for (size_t n = *next_node_id_p; n < nodes.size(); ++n) {
    size_t cur_hash = nodes[n]->GetHighTopoHash();
    if (n + 1 < nodes.size() && nodes[n + 1]->GetHighTopoHash() == cur_hash) {
      // A sequence of nodes sharing the same hash. Skip over it.
      // TODO(babkin): check here for the arbitrary hash conflicts and resolve
      // them.
      for (++n;
           n + 1 < nodes.size() && nodes[n + 1]->GetHighTopoHash() == cur_hash;
           ++n) {
      }
      if (found_unique || n != nodes.size() - 1) {
        // Either some unique nodes have already been found, or this is
        // not the last chance, keep trying to find the unique nodes.
        continue;
      }
      // Here we're at the last node and haven't found any unique ones.
      // So fall through and make this last node unique.
    }

    found_unique = true;
    size_t id = (*next_node_id_p)++;
    nodes[n]->unique_rank_ = id;

    size_t last_hash = nodes[n]->GetHighTopoHash();
    CombineHash(last_hash, &sig_short);
    sig_full.push_back(last_hash);

    // Take the hash at 0 and mix the unique rank into it. After that it will
    // stay fixed.
    nodes[n]->topo_hash_.resize(1);
    nodes[n]->topo_hash_[0] = id + 1;  // Avoid the value of 0.

    nodes[n]->hash_is_final_ = true;
    nodes[n]->last_hashed_nodes_ = nodes[n]->node_mask_;
    if (n != id) {
      std::swap(nodes[id], nodes[n]);
    }
  }
}

void Signature::ComputeOneRound(size_t next_node_id) {
  // Reset the state of the nodes.
  int debug_i = 0;
  for (auto it = nodes.begin() + next_node_id; it != nodes.end(); ++it) {
    auto node = *it;
    // The hash at distance 0 never changes, so preserve it.
    node->topo_hash_.resize(1);
    node->last_hashed_nodes_ = node->node_mask_;
    node->hash_is_final_ = false;
    if (debug) {
      LOG(INFO) << "DEBUG distance=" << 0 << " node " << debug_i++ << " "
                << node->name() << " mask=" << std::hex
                << node->last_hashed_nodes_;
    }
  }

  bool stop = false;
  // The distance can reach up to nodes.size()+1, to include not only all the
  // nodes but also all the redundant paths.
  for (int distance = 1; !stop; ++distance) {
    for (auto it = nodes.begin() + next_node_id; it != nodes.end(); ++it) {
      auto node = *it;
      if (node->hash_is_final_) {
        continue;
      }
      node->ComputeTopoHash(distance);
      if (node->GetHighTopoHash() <= nodes.size()) {
        // Would conflict with one of the reserved values.
        node->ReHighTopoHash();
      }
    }

    // Will be looking for the indications to not stop.
    stop = true;

    debug_i = 0;
    // The bitmasks get moved after all the hash computations are done.
    for (auto it = nodes.begin() + next_node_id; it != nodes.end(); ++it) {
      auto node = *it;
      if (debug) {
        LOG(INFO) << "DEBUG distance=" << distance << " node " << debug_i++
                  << " " << node->name() << " oldmask=" << std::hex
                  << node->last_hashed_nodes_ << " mask=" << std::hex
                  << node->next_hashed_nodes_;
      }
      if (node->last_hashed_nodes_ == node->next_hashed_nodes_) {
        // Stopped growing, this part of the graph must be fully
        // surrounded by nodes that already have the unique ids.
        node->hash_is_final_ = true;
      } else {
        node->last_hashed_nodes_ = node->next_hashed_nodes_;
        stop = false;
      }
    }
  }
}

void Signature::OrderLinks() {
  for (const auto& node : nodes) {
    if (node->hashed_peers_.empty()) {
      continue;
    }

    size_t cur_link_hash = node->hashed_peers_[0].link_hash + 1;
    int first_idx = -1;

    int idx;
    for (idx = 0; idx < static_cast<int64_t>(node->hashed_peers_.size());
         ++idx) {
      auto& entry = node->hashed_peers_[idx];
      if (entry.link_hash == cur_link_hash) {
        continue;
      }
      if (idx - first_idx > 1) {
        // Need to sort.
        std::sort(node->hashed_peers_.begin() + first_idx,
                  node->hashed_peers_.begin() + idx,
                  SigNode::HashedPeer::LessByRank());
      }

      cur_link_hash = entry.link_hash;
      first_idx = idx;
    }
    if (idx - first_idx > 1) {
      // Sort the last bunch.
      std::sort(node->hashed_peers_.begin() + first_idx,
                node->hashed_peers_.begin() + idx,
                SigNode::HashedPeer::LessByRank());
    }
  }
}

bool Signature::operator==(const Signature& other) const {
  // Tries to find the differences as early as possible by
  // comparing the hashes first.

  if (sig_short != other.sig_short) {
    return false;
  }
  if (sig_full.size() != other.sig_full.size()) {
    return false;
  }

  for (auto it1 = sig_full.begin(), it2 = other.sig_full.begin();
       it1 != sig_full.end(); ++it1, ++it2) {
    if (*it1 != *it2) {
      return false;
    }
  }

  if (nodes.size() != other.nodes.size()) {
    return false;
  }
  for (auto it1 = nodes.begin(), it2 = other.nodes.begin(); it1 != nodes.end();
       ++it1, ++it2) {
    if (**it1 != **it2) {
      return false;
    }
  }

  return true;
}

}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow
