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

#include "tensorflow/compiler/aot/tfcompile_util.h"

#include <queue>
#include <set>
#include <unordered_map>

#include "tensorflow/compiler/aot/tfcompile.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace tfcompile {

namespace {

bool IsAlpha(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

bool IsAlphaNum(char c) { return IsAlpha(c) || (c >= '0' && c <= '9'); }

Status ValidateTensorId(const TensorId& id) {
  if (id.node_name().empty()) {
    return errors::InvalidArgument("TensorId node_name must be non-empty");
  }
  if (id.output_index() < 0) {
    return errors::InvalidArgument("TensorId output_index must be positive");
  }
  return Status::OK();
}

Status ValidateFeedFetchName(const string& kind, const string& name,
                             std::set<string>* names) {
  if (!name.empty()) {
    TF_RETURN_IF_ERROR(ValidateCppIdent(name, kind + " name"));
    if (!names->insert(name).second) {
      return errors::InvalidArgument("duplicate ", kind, " name: ", name);
    }
  }
  return Status::OK();
}

Status CheckFeedFetchNameConflicts(const string& kind,
                                   const std::set<string>& names) {
  // We don't allow the feeds or fetches to contain both "foo" and "foo_data",
  // since that will cause a collision in codegen symbols.
  for (const string& name : names) {
    const string name_data(name + "_data");
    if (names.find(name_data) != names.end()) {
      return errors::InvalidArgument("conflicting ", kind, " name: ", name,
                                     " and ", name_data);
    }
  }
  return Status::OK();
}

}  // namespace

Status ValidateCppIdent(StringPiece ident, StringPiece msg) {
  if (ident.empty()) {
    return errors::InvalidArgument("empty identifier: ", msg);
  }
  // Require that the identifier starts with a nondigit, and is composed of
  // nondigits and digits, as specified in section [2.11 Identifiers] of the
  // C++11 Standard.  Note that nondigit is defined as [_a-zA-Z] and digit is
  // defined as [0-9].
  //
  // Technically the standard also allows for `universal-character-name`, with a
  // table of allowed unicode ranges, as well as `other implementation-defined
  // characters`.  We disallow those here to give better error messages, at the
  // expensive of being more restrictive than the standard.
  if (ident[0] != '_' && !IsAlpha(ident[0])) {
    return errors::InvalidArgument("illegal leading char: ", msg);
  }
  for (size_t pos = 1; pos < ident.size(); ++pos) {
    if (ident[pos] != '_' && !IsAlphaNum(ident[pos])) {
      return errors::InvalidArgument("illegal char: ", msg);
    }
  }
  return Status::OK();
}

Status ValidateConfig(const Config& config) {
  std::set<string> names;
  for (const Feed& feed : config.feed()) {
    TF_RETURN_IF_ERROR(ValidateTensorId(feed.id()));
    TF_RETURN_IF_ERROR(TensorShape::IsValidShape(feed.shape()));
    TF_RETURN_IF_ERROR(ValidateFeedFetchName("feed", feed.name(), &names));
  }
  TF_RETURN_IF_ERROR(CheckFeedFetchNameConflicts("feed", names));
  names.clear();
  for (const Fetch& fetch : config.fetch()) {
    TF_RETURN_IF_ERROR(ValidateTensorId(fetch.id()));
    TF_RETURN_IF_ERROR(ValidateFeedFetchName("fetch", fetch.name(), &names));
  }
  TF_RETURN_IF_ERROR(CheckFeedFetchNameConflicts("fetch", names));
  if (config.feed().empty() || config.fetch().empty()) {
    return errors::InvalidArgument("feeds and fetches must be specified");
  }
  return Status::OK();
}

Status AddPlaceholdersForFeeds(
    const Config& config, const OpRegistryInterface* op_registry,
    std::unordered_map<string, string>* feed_remapping, GraphDef* graph_def) {
  struct PlaceholderInfo {
    const Feed* feed = nullptr;  // point to Feed in <config>.
    string placeholder_name;
    DataType data_type = DT_INVALID;
  };

  // Put each fed tensor into a map by name:port. A map is used for determinism
  // when creating placeholders (genrules want deterministic output).
  std::map<string, PlaceholderInfo> placeholder_info;
  for (int i = 0; i < config.feed_size(); ++i) {
    const Feed* feed = &config.feed(i);
    const string name_port = TensorIdToString(feed->id());
    auto& info = placeholder_info[name_port];
    info.feed = feed;
    info.placeholder_name = strings::StrCat(
        "aot_feed_", feed->id().output_index(), "/", feed->id().node_name());
    (*feed_remapping)[name_port] = info.placeholder_name;
  }

  // Verify node exists and determine data type.
  std::unordered_map<string, const NodeDef*> name_to_node;
  for (int i = 0; i < graph_def->node_size(); ++i) {
    name_to_node[graph_def->node(i).name()] = &graph_def->node(i);
  }
  for (auto it = placeholder_info.begin(); it != placeholder_info.end(); ++it) {
    PlaceholderInfo& info = it->second;
    const TensorId& feed_id = info.feed->id();

    // Find the existing node and determine data type.
    auto node_it = name_to_node.find(feed_id.node_name());
    if (node_it == name_to_node.end()) {
      return errors::NotFound("Can't find feed node: ",
                              TensorIdToString(feed_id));
    }
    const NodeDef* existing = node_it->second;

    if (info.feed->type() != DT_INVALID) {
      info.data_type = info.feed->type();
    } else {
      // Build the node in order to infer its type.

      // Must first add default attrs as well, so do this in a copied GraphDef.
      GraphDef gd;
      *gd.mutable_versions() = graph_def->versions();
      *gd.add_node() = *existing;
      TF_RETURN_IF_ERROR(
          AddDefaultAttrsToGraphDef(&gd, *op_registry, 0 /*node_offset*/));

      // Now build the node from the copied node def.
      Graph g(op_registry);
      g.set_versions(graph_def->versions());
      Status status;
      Node* feed_node = g.AddNode(gd.node(0), &status);
      TF_RETURN_IF_ERROR(status);
      info.data_type =
          BaseType(feed_node->output_type(info.feed->id().output_index()));
    }
  }

  // Create placeholders. Note that we could avoid creating a placeholder for
  // feeds which are already placeholders, but we omit that to avoid more cases
  // in this code.
  for (auto it = placeholder_info.begin(); it != placeholder_info.end(); ++it) {
    const PlaceholderInfo& info = it->second;
    NodeDef* d = graph_def->add_node();
    d->set_name(info.placeholder_name);
    d->set_op("PlaceholderV2");
    auto& attr_map = *d->mutable_attr();
    attr_map["dtype"].set_type(info.data_type);
    *attr_map["shape"].mutable_shape() = info.feed->shape();
  }

  // Rewrite references to the fed tensors to refer to the placeholder.
  for (int i = 0; i < graph_def->node_size(); ++i) {
    NodeDef* node_def = graph_def->mutable_node(i);
    for (int j = 0; j < node_def->input_size(); ++j) {
      auto id = ParseTensorName(node_def->input(j));
      auto it = placeholder_info.find(id.ToString());
      if (it != placeholder_info.end()) {
        node_def->set_input(j, it->second.placeholder_name);
      }
    }
  }

  return Status::OK();
}

Status PruneGraphDefInto(const Config& config, const GraphDef& in,
                         GraphDef* out) {
  *out = in;
  out->clear_node();

  // Tensors needed for feeding.
  std::set<std::pair<string, int>> feed_tensors;
  for (const auto& feed_config : config.feed()) {
    feed_tensors.insert(std::make_pair(feed_config.id().node_name(),
                                       feed_config.id().output_index()));
  }

  // Maps node name to reachability.
  std::unordered_map<string, std::pair<bool, const NodeDef*>> node_by_name;
  for (const NodeDef& node : in.node()) {
    node_by_name[node.name()] = std::pair<bool, const NodeDef*>(false, &node);
  }

  // Traverse.
  std::queue<string> name_queue;
  for (int i = 0; i < config.fetch_size(); ++i) {
    name_queue.push(config.fetch(i).id().node_name());
  }
  while (!name_queue.empty()) {
    const string name = name_queue.front();
    name_queue.pop();

    auto find_it = node_by_name.find(name);
    if (find_it == node_by_name.end()) {
      return errors::InvalidArgument("While pruning graph, node ", name,
                                     " needed but not found in the graph.");
    }
    auto& map_entry = find_it->second;
    if (map_entry.first) {
      continue;
    }
    map_entry.first = true;

    // Push input nodes of the currently visited node to name_queue.
    for (const string& in_edge : map_entry.second->input()) {
      auto id = ParseTensorName(in_edge);
      const string node_name = id.first.ToString();
      if (feed_tensors.find(std::make_pair(node_name, id.second)) ==
          feed_tensors.end()) {
        name_queue.push(node_name);
      } else {
        // The input tensor is from an edge that is being fed. Therefore,
        // we skip recursing down that edge, to avoid requiring nodes that
        // may not be needed (note that the input node may still be added
        // to name_queue later if one of its output edges is not being fed).
      }
    }
  }

  // Copy over, preserving order of original and only nodes that are reachable
  // from the fetches.
  out->mutable_node()->Reserve(in.node_size());
  for (const NodeDef& node : in.node()) {
    if (node_by_name[node.name()].first) {
      *out->add_node() = node;
    }
  }
  return Status::OK();
}

string TensorIdToString(const TensorId& id) {
  return strings::StrCat(id.node_name(), ":", id.output_index());
}

}  // namespace tfcompile
}  // namespace tensorflow
