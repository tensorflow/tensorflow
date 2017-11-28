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

#include <memory>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/notification.h"

namespace tensorflow {
namespace grappler {

NodeMap::NodeMap(GraphDef* graph) {
  CHECK(graph != nullptr);
  for (int i = 0; i < graph->node_size(); i++) {
    NodeDef* node = graph->mutable_node(i);
    const string& node_name = node->name();
    auto rslt = nodes_.emplace(node_name, node);
    // Check that the graph doesn't contain multiple nodes with the same name.
    if (!rslt.second) {
      LOG(WARNING) << "Duplicated node in the graph: " << node_name;
    }
    for (const auto& input : node->input()) {
      outputs_[NodeName(input)].insert(nodes_[node_name]);
    }
  }
}

void NodeMap::RemoveNode(const string& name) {
  nodes_.erase(NodeName(name));
  outputs_.erase(NodeName(name));
}

NodeDef* NodeMap::GetNode(const string& name) const {
  const string node_name = NodeName(name);
  auto it = nodes_.find(node_name);
  if (it == nodes_.end()) {
    return nullptr;
  }
  return it->second;
}

bool NodeMap::NodeExists(const string& name) const {
  const string node_name = NodeName(name);
  return nodes_.find(node_name) != nodes_.end();
}

const std::set<NodeDef*>& NodeMap::GetOutputs(const string& node_name) const {
  auto it = outputs_.find(node_name);
  if (it == outputs_.end()) {
    return empty_set_;
  }
  return it->second;
}

void NodeMap::AddNode(const string& node_name, NodeDef* node) {
  auto ret = nodes_.emplace(node_name, CHECK_NOTNULL(node));
  CHECK(ret.second) << "Pair (" << node_name << "," << node
                    << ") is not inserted because the same key already exists.";
}

void NodeMap::AddOutput(const string& node_name, const string& output_name) {
  auto output_node = nodes_[NodeName(output_name)];
  CHECK(output_node) << "Output node " << output_name
                     << " is missing in NodeMap.";
  outputs_[node_name].insert(output_node);
}

void NodeMap::RemoveOutput(const string& node_name, const string& output_name) {
  outputs_[node_name].erase(nodes_[NodeName(output_name)]);
}

void NodeMap::UpdateInput(const string& node_name, const string& old_input_name,
                          const string& new_input_name) {
  RemoveOutput(NodeName(old_input_name), node_name);
  AddOutput(NodeName(new_input_name), node_name);
}

void NodeMap::RemoveInputs(const string& node_name) {
  auto node = nodes_[node_name];
  for (const auto& input : node->input()) {
    RemoveOutput(NodeName(input), node->name());
  }
}

void NodeMap::RemoveOutputs(const string& node_name) {
  outputs_.erase(node_name);
}

void NodeMap::UpdateOutput(const string& node_name,
                           const string& old_output_name,
                           const string& new_output_name) {
  std::set<NodeDef*>& outputs = outputs_[node_name];
  outputs.erase(nodes_[NodeName(old_output_name)]);
  outputs.insert(nodes_[NodeName(new_output_name)]);
}

OutputMap::OutputMap(GraphDef* graph) : graph_(graph) {
  for (int i = 0; i < graph_->node_size(); i++) {
    auto node = graph_->mutable_node(i);
    auto rslt = nodes_.emplace(node->name(), node);
    // Check that the graph doesn't contain multiple nodes with the same name.
    CHECK(rslt.second);
    for (const auto& input : node->input()) {
      string input_node = NodeName(input);
      if (outputs_[input_node].count(node) == 0) {
        outputs_[input_node].insert(std::make_pair(node, 1));
      } else {
        outputs_[input_node][node]++;
      }
    }
  }
}

NodeDef* OutputMap::GetNode(const string& name) const {
  string node_name = NodeName(name);
  auto it = nodes_.find(node_name);
  if (it == nodes_.end()) {
    return nullptr;
  }
  return it->second;
}

const std::unordered_map<NodeDef*, int>& OutputMap::GetOutputs(
    const string& node_name) const {
  auto it = outputs_.find(node_name);
  if (it == outputs_.end()) {
    return empty_map_;
  }
  return it->second;
}

bool IsSameInput(const string& name1, const string& name2) {
  if (name1 == name2) {
    return true;
  }
  int position1;
  string node1 = ParseNodeName(name1, &position1);
  int position2;
  string node2 = ParseNodeName(name2, &position2);
  return (position1 == position2) && (node1 == node2);
}

string ParseNodeName(const string& name, int* position) {
  // Strip the prefix '^' (if any), and strip the trailing ":{digits} (if any)
  // to get a node name.
  strings::Scanner scan(name);
  scan.ZeroOrOneLiteral("^")
      .RestartCapture()
      .One(strings::Scanner::LETTER_DIGIT_DOT)
      .Any(strings::Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);
  StringPiece capture;
  StringPiece remaining;
  if (scan.Peek(':') != ':' || !scan.GetResult(&remaining, &capture)) {
    *position = 0;
    return "";
  } else {
    if (name[0] == '^') {
      *position = -1;
    } else if (remaining.empty()) {
      *position = 0;
    } else {
      // Skip the first ':' character.
      CHECK(strings::safe_strto32(remaining.substr(1), position));
    }
    return capture.ToString();
  }
}

bool IsControlInput(const string& name) {
  return !name.empty() && name[0] == '^';
}

string NodeName(const string& name) {
  int position;
  return ParseNodeName(name, &position);
}

int NodePosition(const string& name) {
  int position;
  ParseNodeName(name, &position);
  return position;
}

string AddPrefixToNodeName(const string& name, const string& prefix,
                           const string& delimiter) {
  if (!name.empty()) {
    if (name[0] == '^') {
      return strings::StrCat("^", prefix, delimiter, name.substr(1));
    }
  }
  return strings::StrCat(prefix, delimiter, name);
}

string AddPrefixToNodeName(const string& name, const string& prefix) {
  return AddPrefixToNodeName(name, prefix, "/");
}

bool ExecuteWithTimeout(std::function<void()> fn, const int64 timeout_in_ms,
                        thread::ThreadPool* const thread_pool) {
  if (timeout_in_ms <= 0) {
    fn();
    return true;
  }
  auto done = std::make_shared<Notification>();
  thread_pool->Schedule([done, fn]() {
    fn();
    done->Notify();
  });
  const bool notified =
      WaitForNotificationWithTimeout(done.get(), timeout_in_ms * 1000);
  return notified;
}

string AsControlDependency(const NodeDef& node) {
  return strings::StrCat("^", node.name());
}

string AsControlDependency(const string& node_name) {
  CHECK(!node_name.empty());
  return (!node_name.empty() && node_name[0] == '^')
             ? node_name
             : strings::StrCat("^", node_name);
}

int NumOutputs(const NodeDef& node) {
  int num_outputs = 0;
  const OpDef* op_def = nullptr;
  auto status = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);
  if (status.ok()) {
    for (const auto& output : op_def->output_arg()) {
      if (!output.type_list_attr().empty()) {
        num_outputs +=
            node.attr().at(output.type_list_attr()).list().type_size();
      } else if (!output.number_attr().empty()) {
        num_outputs += node.attr().at(output.number_attr()).i();
      } else {
        num_outputs++;
      }
    }
  }
  return num_outputs;
}

int NumNonControlInputs(const NodeDef& node) {
  int num_inputs = node.input_size();
  for (const string& input : node.input()) {
    if (IsControlInput(input)) {
      --num_inputs;
    }
  }
  return num_inputs;
}

int NumNonControlOutputs(const NodeDef& node, const NodeMap& node_map) {
  int num_outputs = 0;
  for (const NodeDef* output : node_map.GetOutputs(node.name())) {
    for (const string& node_as_input : output->input()) {
      if (IsControlInput(node_as_input)) {
        break;
      }
      if (NodeName(node_as_input) == node.name()) {
        ++num_outputs;
      }
    }
  }
  return num_outputs;
}

// Returns the data type in attribute `attr_name` of `node`. If that attribute
// doesn't exist, returns DT_INVALID.
DataType GetDataTypeFromAttr(const NodeDef& node, const string& attr_name) {
  if (!node.attr().count(attr_name)) {
    return DT_INVALID;
  }
  const auto& attr = node.attr().at(attr_name);
  if (attr.value_case() != AttrValue::kType) {
    return DT_INVALID;
  }
  return attr.type();
}

NodeDef* GetTailOfChain(const NodeDef& source, const NodeMap& node_map,
                        bool follow_control_input,
                        const std::function<bool(const NodeDef&)>& pred_fn) {
  const NodeDef* current = &source;
  const NodeDef* next = current;
  while (next == &source || (next != nullptr && pred_fn(*next))) {
    current = next;
    if (current->input_size() == 0 ||
        (!follow_control_input && IsControlInput(current->input(0)))) {
      break;
    }
    next = node_map.GetNode(current->input(0));
    if (next == nullptr) {
      LOG(ERROR) << "Node not found: " << current->input(0);
    }
  }
  return const_cast<NodeDef*>(current);
}

}  // end namespace grappler
}  // end namespace tensorflow
