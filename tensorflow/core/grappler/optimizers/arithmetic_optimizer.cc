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

#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer.h"
#include <unordered_map>
#include <unordered_set>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace grappler {

class UniqueNodes {
 public:
  NodeDef* FindOrAddRepresentative(NodeDef* node) {
    std::size_t sig = ComputeSignature(*node);
    std::vector<NodeDef*>& candidates = rep_[sig];
    for (auto& candidate : candidates) {
      if (SameNode(*candidate, *node)) {
        return candidate;
      }
    }
    candidates.push_back(node);
    return node;
  }

 private:
  std::size_t ComputeSignature(const NodeDef& node) const;
  bool SameNode(const NodeDef& node1, const NodeDef& node2) const;

  std::unordered_map<std::size_t, std::vector<NodeDef*>> rep_;
};

std::size_t UniqueNodes::ComputeSignature(const NodeDef& node) const {
  std::size_t h = std::hash<string>{}(node.op());
  h ^= std::hash<string>{}(node.device());
  for (const auto& input : node.input()) {
    int pos;
    string node_name = ParseNodeName(input, &pos);
    h ^= std::hash<string>{}(node_name);
    h ^= static_cast<std::size_t>(pos);
  }
  for (const auto& attr : node.attr()) {
    h ^= std::hash<string>{}(attr.first);
    string tmp;
    attr.second.AppendToString(&tmp);
    h ^= std::hash<string>{}(tmp);
  }
  return h;
}

bool UniqueNodes::SameNode(const NodeDef& node1, const NodeDef& node2) const {
  if (node1.op() != node2.op()) {
    return false;
  }
  if (node1.device() != node2.device()) {
    return false;
  }
  if (node1.input_size() != node2.input_size()) {
    return false;
  }
  if (node1.attr_size() != node2.attr_size()) {
    return false;
  }

  // Compare inputs.
  const OpDef* op_def = nullptr;
  Status status = OpRegistry::Global()->LookUpOpDef(node1.op(), &op_def);
  const bool is_commutative = status.ok() && op_def->is_commutative();
  if (is_commutative) {
    std::vector<string> inputs1(node1.input().begin(), node1.input().end());
    std::vector<string> inputs2(node2.input().begin(), node2.input().end());
    std::sort(inputs1.begin(), inputs1.end());
    std::sort(inputs2.begin(), inputs2.end());
    return inputs1 == inputs2;
  } else {
    std::vector<string> regular_inputs1;
    std::vector<string> regular_inputs2;
    std::vector<string> ctrl_inputs1;
    std::vector<string> ctrl_inputs2;
    for (int index = 0; index < node1.input_size(); ++index) {
      if (IsControlInput(node1.input(index))) {
        ctrl_inputs1.push_back(node1.input(index));
        ctrl_inputs2.push_back(node2.input(index));

      } else {
        regular_inputs1.push_back(node1.input(index));
        regular_inputs2.push_back(node2.input(index));
      }
    }
    if (regular_inputs1 != regular_inputs2) {
      return false;
    }
    std::sort(ctrl_inputs1.begin(), ctrl_inputs1.end());
    std::sort(ctrl_inputs2.begin(), ctrl_inputs2.end());
    if (ctrl_inputs1 != ctrl_inputs2) {
      return false;
    }
  }

  // Compare attributes.
  for (const auto& attr1 : node1.attr()) {
    auto it = node2.attr().find(attr1.first);
    if (it == node2.attr().end()) {
      return false;
    }
    const auto& attr2 = *it;
    string val1;
    attr1.second.AppendToString(&val1);
    string val2;
    attr2.second.AppendToString(&val2);
    if (val1 != val2) {
      return false;
    }
  }

  return true;
}

bool ArithmeticOptimizer::CanDedup(const NodeDef& node) const {
  if (nodes_to_preserve_.find(node.name()) != nodes_to_preserve_.end()) {
    return false;
  }
  if (IsEnter(node) || IsPlaceholder(node)) {
    return false;
  }
  const OpDef* op_def = nullptr;
  Status status = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);
  if (!status.ok()) {
    return false;
  }
  if (op_def->is_stateful()) {
    return false;
  }
  // Don't consolidate ops such as AssignAdd
  for (const auto& input : op_def->input_arg()) {
    if (input.is_ref()) {
      return false;
    }
  }
  return true;
}

void ArithmeticOptimizer::DedupComputations(GraphDef* optimized_graph) const {
  NodeMap map(optimized_graph);
  bool stop = true;
  std::set<int> duplicates;
  do {
    stop = true;
    UniqueNodes nodes;
    for (int i = 0; i < optimized_graph->node_size(); ++i) {
      if (duplicates.find(i) != duplicates.end()) {
        continue;
      }
      NodeDef* node = optimized_graph->mutable_node(i);
      if (!CanDedup(*node)) {
        continue;
      }
      NodeDef* rep = nodes.FindOrAddRepresentative(node);
      if (rep == node) {
        continue;
      }
      const std::set<NodeDef*>& fanouts = map.GetOutputs(node->name());
      for (NodeDef* fanout : fanouts) {
        for (string& name : *fanout->mutable_input()) {
          int position;
          string nodename = ParseNodeName(name, &position);
          if (nodename == node->name()) {
            if (position > 0) {
              name = strings::StrCat(rep->name(), ":", position);
            } else if (position == 0) {
              name = rep->name();
            } else {
              name = strings::StrCat("^", rep->name());
            }
            map.AddOutput(rep->name(), fanout->name());
          }
        }
      }
      duplicates.insert(i);
      stop = false;
    }
  } while (!stop);

  // Delete duplicates
  if (!duplicates.empty()) {
    int last = optimized_graph->node_size() - 1;
    for (auto it = duplicates.rbegin(); it != duplicates.rend(); ++it) {
      int index = *it;
      optimized_graph->mutable_node()->SwapElements(index, last);
      last--;
    }
    optimized_graph->mutable_node()->DeleteSubrange(last + 1,
                                                    duplicates.size());
  }
}

Status ArithmeticOptimizer::Optimize(Cluster* /*cluster*/,
                                     const GrapplerItem& item,
                                     GraphDef* optimized_graph) {
  *optimized_graph = item.graph;
  nodes_to_preserve_ = item.NodesToPreserve();

  // For now, only dedup computations.
  DedupComputations(optimized_graph);

  return Status::OK();
}

void ArithmeticOptimizer::Feedback(Cluster* /*cluster*/,
                                   const GrapplerItem& /*item*/,
                                   const GraphDef& /*optimized_graph*/,
                                   double /*result*/) {
  // Nothing to do for ArithmeticOptimizer.
}

}  // end namespace grappler
}  // end namespace tensorflow
