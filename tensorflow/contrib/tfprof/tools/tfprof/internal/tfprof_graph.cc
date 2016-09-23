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

#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_graph.h"

#include <stdio.h>
#include <utility>

#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_constants.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_tensor.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace tfprof {
GraphNode* TFGraph::CreateParentNode(const string& name) {
  node_defs_.push_back(std::unique_ptr<NodeDef>(new NodeDef()));
  node_defs_.back()->set_name(name);
  node_defs_.back()->set_op(kTFGraphParent);
  parent_nodes_[name] =
      std::unique_ptr<TFNode>(new TFNode(node_defs_.back().get()));
  nodes_map_[name] =
      std::unique_ptr<GraphNode>(new GraphNode(parent_nodes_[name].get()));
  return nodes_map_[name].get();
}

void TFGraph::AddNode(TFNode* node) {
  string name = node->node_def()->name();
  nodes_map_[name] = std::unique_ptr<GraphNode>(new GraphNode(node));
}

void TFGraph::Build() {
  if (!roots_.empty()) return;

  std::set<string> nonroots;
  // Filter out the root nodes (node not input of any other node).
  for (auto it = nodes_map_.begin(); it != nodes_map_.end(); it++) {
    GraphNode* node = it->second.get();
    const std::map<string, TFNode*>& inputs = node->node->inputs();
    for (auto inputs_it = inputs.cbegin(); inputs_it != inputs.cend();
         inputs_it++) {
      nonroots.insert(inputs_it->first);
      auto child_it = nodes_map_.find(inputs_it->first);
      if (child_it != nodes_map_.end()) {
        node->children.push_back(child_it->second.get());
      }
    }
  }
  for (auto it = nodes_map_.begin(); it != nodes_map_.end(); it++) {
    if (nonroots.find(it->first) == nonroots.end()) {
      roots_.push_back(it->second.get());
    }
  }
}

const ShowNode* TFGraph::ShowInternal(const Options& opts) {
  // Search the nodes to start from.
  std::vector<GraphNode*> roots = roots_;
  if (opts.start_name_regexes.size() != 1 ||
      opts.start_name_regexes[0] != ".*") {
    std::set<string> visited;
    roots = SearchRoot(roots, opts.start_name_regexes, &visited);
  }

  GraphNode* root = CreateParentNode(kTFProfRoot);
  root->children.assign(roots.begin(), roots.end());

  std::map<string, int64> account_visits;
  Account({root}, opts, &account_visits);

  if (opts.viz) {
    printf("Visualizing feature disabled...\n");
  }
  std::set<string> visits;
  return PrintGraph({root}, opts, 1, 0, 0, &visits)[0];
}

std::vector<GraphNode*> TFGraph::SearchRoot(
    const std::vector<GraphNode*>& roots, const std::vector<string>& regexes,
    std::set<string>* visited) {
  std::vector<GraphNode*> res;
  if (roots.empty()) {
    return res;
  }
  for (GraphNode* root : roots) {
    if (visited->find(root->name()) != visited->end()) continue;
    visited->insert(root->name());
    // If the parent is a start point, don't search its children.
    // Note that its children can still be added as start node through
    // another route.
    bool match_start_node = false;
    for (const string& regex : regexes) {
      if (RE2::FullMatch(root->name(), regex)) {
        res.push_back(root);
        match_start_node = true;
        break;
      }
    }
    if (match_start_node) {
      continue;
    }
    std::vector<GraphNode*> nroot =
        SearchRoot(root->children, regexes, visited);
    res.insert(res.end(), nroot.begin(), nroot.end());
  }
  return res;
}

std::vector<GraphNode*> TFGraph::PrintGraph(const std::vector<GraphNode*> roots,
                                            const Options& opts, int depth,
                                            int hidden, int last_ident,
                                            std::set<string>* visits) {
  std::vector<GraphNode*> show_nodes;

  for (GraphNode* node : roots) {
    if (visits->find(node->name()) != visits->end()) continue;
    visits->insert(node->name());

    int nhidden = hidden;
    int nlast_ident = last_ident;
    bool show = ShouldShow(node, opts, depth);
    if (show) {
      node->formatted_str.clear();
      if (opts.account_displayed_op_only) {
        node->ResetTotalStats();
        node->AddSelfToTotalStats();
      }
      nhidden = 0;
      nlast_ident = (hidden && opts.select.find(kShown[4]) != opts.select.end()
                         ? last_ident + 4
                         : last_ident + 2);
    } else {
      ++nhidden;
    }

    std::vector<GraphNode*> show_cnodes;
    if (!ShouldTrim(node, opts.trim_name_regexes)) {
      show_cnodes = PrintGraph(node->children, opts, depth + 1, nhidden,
                               nlast_ident, visits);
    }
    if (show) {
      show_cnodes = SortNodes(show_cnodes, opts);
      string children_str;
      for (GraphNode* sc : show_cnodes) {
        children_str += sc->formatted_str;
        node->mutable_proto()->add_children()->MergeFrom(sc->proto());
        if (opts.account_displayed_op_only) {
          node->AggregateTotalStats(sc);
        }
      }
      if (hidden && opts.select.find(kShown[4]) != opts.select.end()) {
        node->formatted_str = strings::Printf(
            "%s...hidden %d...\n", string(last_ident, ' ').c_str(), hidden);
        node->formatted_str +=
            strings::Printf("  %s%s\n", string(last_ident, ' ').c_str(),
                            node->Format(opts).c_str());
      } else {
        node->formatted_str =
            strings::Printf("%s%s\n", string(last_ident, ' ').c_str(),
                            node->Format(opts).c_str());
      }
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

void TFGraph::Account(const std::vector<GraphNode*>& roots, const Options& opts,
                      std::map<string, int64>* visits) {
  if (roots.empty()) return;

  for (GraphNode* node : roots) {
    if (visits->find(node->name()) != visits->end()) continue;
    (*visits)[node->name()] = 1;
    node->ResetTotalStats();
    // Depth-firsth.
    Account(node->children, opts, visits);

    node->account = ShouldAccount(node, opts);
    if (node->account) {
      node->AddSelfToTotalStats();
    }
    // Aggregate its children stats.
    for (GraphNode* c : node->children) {
      // A node can be visited from multiple parents. Only account once.
      // "visits==1" is when the node is visited through depth-first search.
      (*visits)[c->name()] += 1;
      if ((*visits)[c->name()] > 2) continue;

      node->AggregateTotalStats(c);
    }
  }
}
}  // namespace tfprof
}  // namespace tensorflow
