/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_scope.h"

#include <stdio.h>
#include <utility>

#include "tensorflow/c/c_api.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_constants.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_tensor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace tfprof {
ScopeNode* TFScope::CreateParentNode(const string& name) {
  if (nodes_map_.find(name) != nodes_map_.end()) {
    return nodes_map_[name].get();
  }
  node_defs_.push_back(std::unique_ptr<NodeDef>(new NodeDef()));
  node_defs_.back()->set_name(name);
  node_defs_.back()->set_op(kTFScopeParent);
  parent_nodes_[name] =
      std::unique_ptr<TFNode>(new TFNode(node_defs_.back().get()));
  nodes_map_[name] =
      std::unique_ptr<ScopeNode>(new ScopeNode(parent_nodes_[name].get()));
  return nodes_map_[name].get();
}

void TFScope::AddNode(TFNode* node) {
  string name = node->node_def()->name();
  if (nodes_map_.find(node->node_def()->name()) == nodes_map_.end()) {
    nodes_map_[name] = std::unique_ptr<ScopeNode>(new ScopeNode(node));
  }

  auto last_slash = name.find_last_of("/");
  while (last_slash != name.npos) {
    name = name.substr(0, last_slash);
    if (nodes_map_.find(name) == nodes_map_.end()) {
      CHECK(CreateParentNode(name));
    }
    last_slash = name.find_last_of("/");
  }
}

void TFScope::Build() {
  if (!roots_.empty()) return;
  // Found roots, which are nodes without "/".
  for (auto it = nodes_map_.begin(); it != nodes_map_.end(); it++) {
    ScopeNode* node = it->second.get();
    auto last_slash = node->name().find_last_of("/");
    if (last_slash == string::npos) {
      roots_.push_back(node);
    } else {
      const string prefix = node->name().substr(0, last_slash);
      nodes_map_[prefix]->children.push_back(node);
    }
  }
}

const ShowNode* TFScope::ShowInternal(const Options& opts) {
  // Search from roots recursively to find start node, if start_name_regexes
  // is specified.
  std::vector<ScopeNode*> roots = roots_;
  if (opts.start_name_regexes.size() != 1 ||
      opts.start_name_regexes[0] != ".*") {
    roots = SearchRoot(roots, opts.start_name_regexes);
  }

  ScopeNode* root = CreateParentNode(kTFProfRoot);
  root->children.assign(roots.begin(), roots.end());
  Account({root}, opts);

  root = PrintScope({root}, opts, 1, 0)[0];
  return root;
}

std::vector<ScopeNode*> TFScope::SearchRoot(
    std::vector<ScopeNode*> roots, const std::vector<string>& regexes) {
  std::vector<ScopeNode*> res;
  if (roots.empty()) {
    return res;
  }
  for (ScopeNode* root : roots) {
    bool match_start_node = false;
    for (const string& regex : regexes) {
      if (RE2::FullMatch(root->name(), regex)) {
        res.push_back(root);
        match_start_node = true;
        break;
      }
    }
    if (match_start_node) {
      // Found a start node at this branch, no need to continue.
      continue;
    }
    std::vector<ScopeNode*> nroots = SearchRoot(root->children, regexes);
    res.insert(res.end(), nroots.begin(), nroots.end());
  }
  return res;
}

std::vector<ScopeNode*> TFScope::PrintScope(const std::vector<ScopeNode*> roots,
                                            const Options& opts, int depth,
                                            int last_ident) {
  std::vector<ScopeNode*> show_nodes;

  for (ScopeNode* node : roots) {
    int nlast_ident = last_ident;
    bool show = ShouldShow(node, opts, depth);
    if (show) {
      node->formatted_str.clear();
      if (opts.account_displayed_op_only) {
        node->ResetTotalStats();
        node->AddSelfToTotalStats();
      }
      nlast_ident += 2;
    }

    std::vector<ScopeNode*> show_cnodes;
    if (!ShouldTrim(node, opts.trim_name_regexes)) {
      show_cnodes = PrintScope(node->children, opts, depth + 1, nlast_ident);
    }
    if (show) {
      show_cnodes = SortNodes(show_cnodes, opts);
      string children_str;
      for (ScopeNode* sc : show_cnodes) {
        children_str += sc->formatted_str;
        node->mutable_proto()->add_children()->MergeFrom(sc->proto());
        if (opts.account_displayed_op_only) {
          node->AggregateTotalStats(sc);
        }
      }

      node->formatted_str =
          strings::Printf("%s%s\n", string(last_ident, ' ').c_str(),
                          node->Format(opts).c_str());

      if (opts.select.find(kShown[5]) != opts.select.end()) {
        std::unique_ptr<TFProfTensor> tfprof_tensor;
        if (LookUpCheckPoint(node->name(), &tfprof_tensor)) {
          string value_str;
          tfprof_tensor->Display(&value_str,
                                 node->mutable_proto()->mutable_tensor_value());
          node->formatted_str += value_str;
        }
      }

      node->formatted_str += children_str;
      show_nodes.push_back(node);
    } else {
      show_nodes.insert(show_nodes.end(), show_cnodes.begin(),
                        show_cnodes.end());
    }
  }
  return show_nodes;
}

void TFScope::Account(const std::vector<ScopeNode*>& roots,
                      const Options& opts) {
  if (roots.empty()) return;

  for (ScopeNode* node : roots) {
    node->ResetTotalStats();
    Account(node->children, opts);

    node->account = ShouldAccount(node, opts);
    if (node->account) {
      node->AddSelfToTotalStats();
    }
    for (ScopeNode* c : node->children) {
      node->AggregateTotalStats(c);
    }
  }
}
}  // namespace tfprof
}  // namespace tensorflow
