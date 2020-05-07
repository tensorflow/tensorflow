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

#include "tensorflow/core/profiler/internal/tfprof_graph.h"

#include <stdio.h>

#include <utility>

#include "absl/strings/str_format.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/profiler/internal/tfprof_constants.h"
#include "tensorflow/core/profiler/internal/tfprof_tensor.h"

namespace tensorflow {
namespace tfprof {
GraphNode* TFGraph::CreateParentNode(const string& name) {
  node_defs_.push_back(std::unique_ptr<NodeDef>(new NodeDef()));
  node_defs_.back()->set_name(name);
  node_defs_.back()->set_op(kTFGraphParent);
  parent_nodes_[name] = std::unique_ptr<TFGraphNode>(
      new TFGraphNode(node_defs_.back().get(), -1, nullptr));
  nodes_map_[name] =
      std::unique_ptr<GraphNode>(new GraphNode(parent_nodes_[name].get()));
  return nodes_map_[name].get();
}

void TFGraph::AddNode(TFGraphNode* node) {
  string name = node->name();
  nodes_map_[name] = std::unique_ptr<GraphNode>(new GraphNode(node));
}

void TFGraph::Build() {
  if (root_) return;

  std::set<string> nonroots;
  // Filter out the root nodes (node not input of any other node).
  for (auto it = nodes_map_.begin(); it != nodes_map_.end(); it++) {
    GraphNode* node = it->second.get();
    const std::map<int, string>& inputs = node->node->inputs();
    for (auto inputs_it = inputs.cbegin(); inputs_it != inputs.cend();
         inputs_it++) {
      nonroots.insert(inputs_it->second);
      auto child_it = nodes_map_.find(inputs_it->second);
      if (child_it != nodes_map_.end()) {
        node->children.push_back(child_it->second.get());
      }
    }
  }
  std::vector<GraphNode*> roots;
  for (auto it = nodes_map_.begin(); it != nodes_map_.end(); it++) {
    if (nonroots.find(it->first) == nonroots.end()) {
      roots.push_back(it->second.get());
    }
  }
  root_ = CreateParentNode(kTFProfRoot);
  root_->children.insert(root_->children.end(), roots.begin(), roots.end());
}

const ShowNode* TFGraph::ShowInternal(const Options& opts, Timeline* timeline) {
  root_->ResetTotalStats();
  root_->show_children.clear();

  if (opts.output_type == kOutput[3]) {
    absl::FPrintF(stderr, "Only 'code' view supports pprof output now.\n");
    return root_;
  }
  if (timeline && timeline->step() < 0) {
    // TODO(xpan): Maybe pick a default step for users.
    absl::FPrintF(
        stderr,
        "Must specify -step option to generate timeline in graph view.\n");
    return root_;
  }
  // 1. Account and aggregate the stats based on the graph structure.
  // Returns a graph consists of accounted nodes.
  std::set<string> visits;
  std::vector<GraphNode*> roots = Account(root_->children, opts, &visits);
  for (GraphNode* n : roots) {
    root_->AggregateTotalStats(n);
  }

  // 2. Trim the nodes before start_name_regexes.
  if (opts.start_name_regexes.size() != 1 ||
      opts.start_name_regexes[0] != ".*") {
    visits.clear();
    roots = SearchRoot(roots, opts.start_name_regexes, &visits);
  }

  // 3. Trim the nodes not matching show/hide/trim_name_regexes.
  // If account_displayed_op_only=true, redo the accounting.
  visits.clear();
  root_->show_children.assign(roots.begin(), roots.end());
  GraphNode* root = PrintGraph({root_}, opts, 1, 0, &visits)[0];

  // 4. Prepare output based on the final graphs.
  root->formatted_str = FormatLegend(opts) + root->formatted_str;
  Format(root->show_children, &root->formatted_str, root->mutable_proto());

  if (timeline) {
    timeline->GenerateGraphTimeline(root->show_children);
  }
  return root;
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
        SearchRoot(root->show_children, regexes, visited);
    res.insert(res.end(), nroot.begin(), nroot.end());
  }
  return res;
}

void TFGraph::Format(const std::vector<GraphNode*> roots, string* display_str,
                     GraphNodeProto* proto) {
  for (GraphNode* node : roots) {
    display_str->append(node->formatted_str);
    GraphNodeProto* child = proto->add_children();
    child->MergeFrom(node->proto());
    Format(node->show_children, display_str, child);
  }
}

std::vector<GraphNode*> TFGraph::PrintGraph(const std::vector<GraphNode*> roots,
                                            const Options& opts, int depth,
                                            int last_ident,
                                            std::set<string>* visits) {
  std::vector<GraphNode*> show_nodes;

  for (GraphNode* node : roots) {
    if (visits->find(node->name()) != visits->end()) continue;
    visits->insert(node->name());

    bool show = ShouldShow(node, opts, depth);
    int indent = last_ident;
    if (show) indent += 2;

    std::vector<GraphNode*> show_cnodes;
    if (!ShouldTrim(node, opts.trim_name_regexes) && depth <= opts.max_depth) {
      show_cnodes =
          PrintGraph(node->show_children, opts, depth + 1, indent, visits);
    }
    if (show) {
      node->show_children.clear();
      if (opts.account_displayed_op_only) {
        node->ResetTotalStats();
        node->AddSelfToTotalStats();
      }

      show_cnodes = SortNodes(show_cnodes, opts);
      for (GraphNode* sc : show_cnodes) {
        node->show_children.push_back(sc);
        if (opts.account_displayed_op_only) {
          node->AggregateTotalStats(sc);
        }
      }
      node->formatted_str = absl::StrFormat(
          "%s%s\n", std::string(last_ident, ' '), FormatNode(node, opts));

      if (opts.select.find(kShown[4]) != opts.select.end()) {
        std::unique_ptr<TFProfTensor> tfprof_tensor;
        if (LookUpCheckPoint(node->name(), &tfprof_tensor)) {
          string value_str;
          tfprof_tensor->Display(&value_str,
                                 node->mutable_proto()->mutable_tensor_value());
          node->formatted_str += value_str;
        }
      }
      show_nodes.push_back(node);
    } else {
      show_nodes.insert(show_nodes.end(), show_cnodes.begin(),
                        show_cnodes.end());
    }
  }
  return show_nodes;
}

std::vector<GraphNode*> TFGraph::Account(const std::vector<GraphNode*>& roots,
                                         const Options& opts,
                                         std::set<string>* visits) {
  std::vector<GraphNode*> act_nodes;
  for (GraphNode* node : roots) {
    if (visits->find(node->name()) != visits->end()) continue;
    visits->insert(node->name());
    // Depth-first.
    std::vector<GraphNode*> act_cnodes = Account(node->children, opts, visits);

    node->account = ReAccount(node, opts);
    if (node->account) {
      node->show_children.clear();
      node->ResetTotalStats();
      node->AddSelfToTotalStats();
      // Aggregate its accounted children stats.
      for (GraphNode* c : act_cnodes) {
        node->AggregateTotalStats(c);
        node->show_children.push_back(c);
      }
      act_nodes.push_back(node);
    } else {
      // If the current node is not accounted, pass the children to the
      // ancestor.
      act_nodes.insert(act_nodes.end(), act_cnodes.begin(), act_cnodes.end());
    }
  }
  return act_nodes;
}
}  // namespace tfprof
}  // namespace tensorflow
