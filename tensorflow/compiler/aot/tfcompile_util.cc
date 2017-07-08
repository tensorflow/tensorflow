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
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"

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

Status PruneGraphDefInto(const Config& config, const GraphDef& in,
                         GraphDef* out) {
  *out = in;
  out->clear_node();

  // Maps node name to reachability.
  std::unordered_map<string, std::pair<bool, const NodeDef*>> node_by_name;
  for (const NodeDef& node : in.node()) {
    node_by_name[node.name()] = std::pair<bool, const NodeDef*>(false, &node);
  }

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

    for (const string& in_edge : map_entry.second->input()) {
      name_queue.push(ParseTensorName(in_edge).first.ToString());
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

}  // namespace tfcompile
}  // namespace tensorflow
