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

#include "tensorflow/core/grappler/graph_analyzer/subgraph.h"

#include <functional>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/grappler/graph_analyzer/hash_tools.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {

//=== Subgraph::Identity

Subgraph::Identity::Identity(InitializerList init) {
  for (auto element : init) {
    insert(element);
  }
}

bool Subgraph::Identity::operator<(const Identity& other) const {
  // Shorter sets go first.
  if (this->size() < other.size()) {
    return true;
  }
  if (this->size() > other.size()) {
    return false;
  }
  for (auto lit = this->begin(), rit = other.begin(); lit != this->end();
       ++lit, ++rit) {
    if (*lit < *rit) {
      return true;
    }
    if (*lit > *rit) {
      return false;
    }
  }
  return false;  // Equal.
}

bool Subgraph::Identity::operator==(const Identity& other) const {
  if (this->size() != other.size()) {
    return false;
  }
  for (auto lit = this->begin(), rit = other.begin(); lit != this->end();
       ++lit, ++rit) {
    if (*lit != *rit) {
      return false;
    }
  }
  return true;  // Equal.
}

size_t Subgraph::Identity::Hash() const {
  std::hash<const GenNode*> hasher;
  size_t result = 0;
  for (auto ptr : *this) {
    CombineHash(hasher(ptr), &result);
  }
  return result;
}

string Subgraph::Dump() {
  // TODO(babkin): this is simplified for now.
  std::vector<string> nodes;
  for (const auto& n : id_) {
    if (specific_) {
      nodes.emplace_back(absl::StrFormat("%s(%s)", n->opcode(), n->name()));
    } else {
      nodes.emplace_back(n->opcode());
    }
  }
  std::sort(nodes.begin(), nodes.end());

  return absl::StrFormat("%d: ", collation_count_) + absl::StrJoin(nodes, ", ");
}

void Subgraph::ExtractForSignature(SigNodeMap* result) {
  // Mapping of nodes from the original graph to the new one.
  SigNode::TranslationMap full_to_new;

  for (auto node : id_) {
    auto newnode_ref = std::make_unique<SigNode>(node->node_def());
    auto newnode = newnode_ref.get();
    (*result)[node->name()] = std::move(newnode_ref);
    full_to_new[node] = newnode;
  }

  for (const auto& mapping : full_to_new) {
    mapping.second->CopyLinks(*mapping.first, full_to_new);
  }
}

//=== Subgraph

Subgraph::Subgraph(const Identity& parent_id, GenNode* add_node)
    : id_(parent_id) {
  id_.insert(add_node);
  hash_ = id_.Hash();
}

//=== SubgraphIterator

SubgraphIterator::SubgraphIterator(const Subgraph::Identity* id)
    : id_(id), id_it_(id_->begin()) {
  if (!id_->empty()) {
    link_map_it_ = (*id_it_)->links().begin();
    // In case if the node has no links.
    while (link_map_it_ == (*id_it_)->links().end()) {
      if (++id_it_ == id_->end()) {
        return;
      }
      link_map_it_ = (*id_it_)->links().begin();
    }
    link_idx_ = 0;
    // The LinkTargetVector should never be empty but just in case safeguard
    // against that too.
    PropagateNext();
  }
}

bool SubgraphIterator::Next() {
  if (AtEnd()) {
    return false;
  }
  ++link_idx_;
  return PropagateNext();
}

bool SubgraphIterator::NextIfSamePort() {
  if (AtEnd()) {
    return false;
  }
  const int64_t link_map_it_second_size = link_map_it_->second.size();
  if (link_idx_ + 1 < link_map_it_second_size) {
    ++link_idx_;
    return true;
  } else {
    return false;
  }
}

void SubgraphIterator::SkipPort() {
  if (AtEnd()) {
    return;
  }
  link_idx_ = link_map_it_->second.size() - 1;
}

void SubgraphIterator::SkipNode() {
  if (AtEnd()) {
    return;
  }
  for (auto next = link_map_it_; next != (*id_it_)->links().end(); ++next) {
    link_map_it_ = next;
  }
  link_idx_ = link_map_it_->second.size() - 1;
}

bool SubgraphIterator::PropagateNext() {
  // Loops are used to skip over the empty entries.
  const int64_t link_map_it_second_size = link_map_it_->second.size();
  while (link_idx_ >= link_map_it_second_size) {
    ++link_map_it_;
    while (link_map_it_ == (*id_it_)->links().end()) {
      if (++id_it_ == id_->end()) {
        return false;
      }
      link_map_it_ = (*id_it_)->links().begin();
    }
    link_idx_ = 0;
  }
  return true;
}

bool SubgraphIterator::operator==(const SubgraphIterator& other) const {
  if (id_ != other.id_) {
    return false;
  }
  if (id_it_ != other.id_it_) {
    return false;
  }
  // When AtEnd(), the rest of the fields are not valid.
  if (AtEnd()) {
    return true;
  }
  if (link_map_it_ != other.link_map_it_) {
    return false;
  }
  if (link_idx_ != other.link_idx_) {
    return false;
  }
  return true;
}

//=== SubgraphPtrSet

Subgraph* SubgraphPtrSet::ExtendParent(const Subgraph::Identity& parent_id,
                                       GenNode* node) {
  if (parent_id.find(node) != parent_id.end()) {
    // This was another link to the node that is already in the parent.
    return nullptr;
  }

  // Constructing an object just to check that an equivalent one is already
  // present is kind of ugly but storing the references rather than the objects
  // in the set avoids the need to make the object copyable.
  auto sg = std::make_unique<Subgraph>(parent_id, node);
  if (find(sg) != end()) {
    // This subgraph was already found by extending from a different path.
    return nullptr;
  }

  Subgraph* ptr = sg.get();
  insert(std::move(sg));
  return ptr;
}

}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow
