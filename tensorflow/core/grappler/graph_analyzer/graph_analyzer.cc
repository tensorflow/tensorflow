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

#include <deque>
#include <iostream>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/grappler/graph_analyzer/gen_node.h"
#include "tensorflow/core/grappler/graph_analyzer/graph_analyzer.h"
#include "tensorflow/core/grappler/graph_analyzer/sig_node.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {

GraphAnalyzer::GraphAnalyzer(const GraphDef& graph, int subgraph_size)
    : graph_(graph), subgraph_size_(subgraph_size) {}

GraphAnalyzer::~GraphAnalyzer() {}

Status GraphAnalyzer::Run() {
  // The signature computation code would detect this too, but better
  // to report it up front than spend time computing all the graphs first.
  if (subgraph_size_ > Signature::kMaxGraphSize) {
    return Status(absl::StatusCode::kInvalidArgument,
                  absl::StrFormat("Subgraphs of %d nodes are not supported, "
                                  "the maximal supported node count is %d.",
                                  subgraph_size_, Signature::kMaxGraphSize));
  }

  Status st = BuildMap();
  if (!st.ok()) {
    return st;
  }

  FindSubgraphs();
  DropInvalidSubgraphs();
  st = CollateResult();
  if (!st.ok()) {
    return st;
  }

  return absl::OkStatus();
}

Status GraphAnalyzer::BuildMap() {
  nodes_.clear();
  return GenNode::BuildGraphInMap(graph_, &nodes_);
}

void GraphAnalyzer::FindSubgraphs() {
  result_.clear();

  if (subgraph_size_ < 1) {
    return;
  }

  partial_.clear();
  todo_.clear();  // Just in case.

  // Start with all subgraphs of size 1.
  const Subgraph::Identity empty_parent;
  for (const auto& node : nodes_) {
    if (subgraph_size_ == 1) {
      result_.ExtendParent(empty_parent, node.second.get());
    } else {
      // At this point ExtendParent() is guaranteed to not return nullptr.
      todo_.push_back(partial_.ExtendParent(empty_parent, node.second.get()));
    }
  }

  // Then extend the subgraphs until no more extensions are possible.
  while (!todo_.empty()) {
    ExtendSubgraph(todo_.front());
    todo_.pop_front();
  }

  partial_.clear();
}

void GraphAnalyzer::ExtendSubgraph(Subgraph* parent) {
  const int next_parent_id = parent->id().size() + 1;
  bool will_complete = (next_parent_id == subgraph_size_);
  SubgraphPtrSet& sg_set = will_complete ? result_ : partial_;

  const GenNode* last_all_or_none_node = nullptr;
  for (SubgraphIterator sit(parent); !sit.AtEnd(); sit.Next()) {
    const GenNode* node = sit.GetNode();
    GenNode::Port port = sit.GetPort();
    const GenNode::LinkTarget& neighbor = sit.GetNeighbor();

    if (node->AllInputsOrNone() && port.IsInbound() && !port.IsControl()) {
      if (node != last_all_or_none_node) {
        ExtendSubgraphAllOrNone(parent, node);
        last_all_or_none_node = node;
      }
      sit.SkipPort();
    } else if (neighbor.node->AllInputsOrNone() && !port.IsInbound() &&
               !port.IsControl()) {
      if (parent->id().find(neighbor.node) == parent->id().end()) {
        // Not added yet.
        ExtendSubgraphAllOrNone(parent, neighbor.node);
      }
    } else if (node->IsMultiInput(port)) {
      ExtendSubgraphPortAllOrNone(parent, node, port);
      sit.SkipPort();
    } else if (neighbor.node->IsMultiInput(neighbor.port)) {
      // Would need to add all inputs of the neighbor node at this port at
      // once.
      if (parent->id().find(neighbor.node) != parent->id().end()) {
        continue;  // Already added.
      }
      ExtendSubgraphPortAllOrNone(parent, neighbor.node, neighbor.port);
    } else {
      Subgraph* sg = sg_set.ExtendParent(parent->id(), neighbor.node);
      if (!will_complete && sg != nullptr) {
        todo_.push_back(sg);
      }
    }
  }
}

void GraphAnalyzer::ExtendSubgraphAllOrNone(Subgraph* parent,
                                            const GenNode* node) {
  Subgraph::Identity id = parent->id();
  id.insert(node);

  auto range_end = node->links().end();

  for (auto nbit = node->links().begin(); nbit != range_end; ++nbit) {
    auto port = nbit->first;
    if (!port.IsInbound() || port.IsControl()) {
      continue;
    }

    // Since there might be multiple links to the same nodes,
    // have to add all links one-by-one to check whether the subgraph
    // would grow too large. But if it does grow too large, there is no
    // point in growing it more, can just skip over the rest of the links.
    for (const auto& link : nbit->second) {
      id.insert(link.node);
      const int id_size = id.size();
      if (id_size > subgraph_size_) {
        return;  // Too big.
      }
    }
  }

  AddExtendedSubgraph(parent, id);
}

void GraphAnalyzer::ExtendSubgraphPortAllOrNone(Subgraph* parent,
                                                const GenNode* node,
                                                GenNode::Port port) {
  auto nbit = node->links().find(port);
  if (nbit == node->links().end()) {
    return;  // Should never happen.
  }

  Subgraph::Identity id = parent->id();
  id.insert(node);

  // Since there might be multiple links to the same nodes,
  // have to add all links one-by-one to check whether the subgraph
  // would grow too large. But if it does grow too large, there is no
  // point in growing it more, can just skip over the rest of the links.
  for (const auto& link : nbit->second) {
    id.insert(link.node);
    const int id_size = id.size();
    if (id_size > subgraph_size_) {
      return;  // Too big.
    }
  }

  AddExtendedSubgraph(parent, id);
}

void GraphAnalyzer::AddExtendedSubgraph(Subgraph* parent,
                                        const Subgraph::Identity& id) {
  if (id.size() == parent->id().size()) {
    return;  // Nothing new was added.
  }

  auto sg = std::make_unique<Subgraph>(id);
  SubgraphPtrSet& spec_sg_set =
      (id.size() == subgraph_size_) ? result_ : partial_;
  if (spec_sg_set.find(sg) != spec_sg_set.end()) {
    // This subgraph was already found by extending from a different path.
    return;
  }
  const int id_size = id.size();
  if (id_size != subgraph_size_) {
    todo_.push_back(sg.get());
  }
  spec_sg_set.insert(std::move(sg));
}

void GraphAnalyzer::DropInvalidSubgraphs() {
  auto resit = result_.begin();
  while (resit != result_.end()) {
    if (HasInvalidMultiInputs(resit->get())) {
      auto delit = resit;
      ++resit;
      result_.erase(delit);
    } else {
      ++resit;
    }
  }
}

bool GraphAnalyzer::HasInvalidMultiInputs(Subgraph* sg) {
  // Do the all-or-none-input nodes.
  for (auto const& node : sg->id()) {
    if (!node->AllInputsOrNone()) {
      continue;
    }

    bool anyIn = false;
    bool anyOut = false;

    auto range_end = node->links().end();
    for (auto nbit = node->links().begin(); nbit != range_end; ++nbit) {
      auto port = nbit->first;
      if (!port.IsInbound() || port.IsControl()) {
        continue;
      }

      // Since there might be multiple links to the same nodes,
      // have to add all links one-by-one to check whether the subgraph
      // would grow too large. But if it does grow too large, there is no
      // point in growing it more, can just skip over the rest of the links.
      for (const auto& link : nbit->second) {
        if (sg->id().find(link.node) == sg->id().end()) {
          anyOut = true;
        } else {
          anyIn = true;
        }
      }
    }

    if (anyIn && anyOut) {
      return true;
    }
  }

  // Do the multi-input ports.
  for (SubgraphIterator sit(sg); !sit.AtEnd(); sit.Next()) {
    if (sit.GetNode()->IsMultiInput(sit.GetPort())) {
      bool anyIn = false;
      bool anyOut = false;
      do {
        GenNode* peer = sit.GetNeighbor().node;
        if (sg->id().find(peer) == sg->id().end()) {
          anyOut = true;
        } else {
          anyIn = true;
        }
      } while (sit.NextIfSamePort());

      if (anyIn && anyOut) {
        return true;
      }
    }
  }
  return false;
}

Status GraphAnalyzer::CollateResult() {
  ordered_collation_.clear();
  collation_map_.clear();

  // Collate by the signatures of the graphs.
  for (const auto& it : result_) {
    auto sig = std::make_unique<Signature>();
    it->ExtractForSignature(&sig->map);
    Status status = sig->Compute();
    if (!status.ok()) {
      return status;
    }

    auto& coll_entry = collation_map_[sig.get()];
    if (coll_entry.sig == nullptr) {
      coll_entry.sig = std::move(sig);
    }
    ++coll_entry.count;
  }

  // Then order them by the count.
  for (auto& entry : collation_map_) {
    ordered_collation_.insert(&entry.second);
  }

  result_.clear();  // Not needed after collation.

  return absl::OkStatus();
}

std::vector<string> GraphAnalyzer::DumpRawSubgraphs() {
  std::vector<string> result;
  for (const auto& it : result_) {
    result.emplace_back(it->Dump());
  }
  return result;
}

std::vector<string> GraphAnalyzer::DumpSubgraphs() {
  std::vector<string> result;
  for (auto ptr : ordered_collation_) {
    result.emplace_back(
        absl::StrFormat("%d %s", ptr->count, ptr->sig->ToString()));
  }
  return result;
}

Status GraphAnalyzer::OutputSubgraphs() {
  size_t total = 0;
  for (auto ptr : ordered_collation_) {
    std::cout << ptr->count << ' ' << ptr->sig->ToString() << '\n';
    total += ptr->count;
  }
  std::cout << "Total: " << total << '\n';
  if (std::cout.fail()) {
    return Status(absl::StatusCode::kDataLoss, "Failed to write to stdout");
  } else {
    return absl::OkStatus();
  }
}

}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow
