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

#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/notification.h"

namespace tensorflow {
namespace grappler {

NodeMap::NodeMap(GraphDef* graph) : graph_(graph) {
  for (int i = 0; i < graph_->node_size(); i++) {
    auto node = graph_->mutable_node(i);
    nodes_.insert(std::make_pair(node->name(), node));
    for (const auto& input : node->input()) {
      outputs_[NodeName(input)].insert(nodes_[node->name()]);
    }
  }
}

NodeDef* NodeMap::GetNode(const string& name) const {
  string node_name = NodeName(name);
  auto it = nodes_.find(node_name);
  if (it == nodes_.end()) {
    return nullptr;
  }
  return it->second;
}

const std::set<NodeDef*>& NodeMap::GetOutputs(const string& node_name) const {
  auto it = outputs_.find(node_name);
  if (it == outputs_.end()) {
    return empty_set_;
  }
  return it->second;
}

void NodeMap::AddNode(const string& name, NodeDef* node) {
  nodes_.insert(std::make_pair(name, node));
}

void NodeMap::AddOutput(const string& node, const string& output) {
  outputs_[node].insert(nodes_[output]);
}

void NodeMap::UpdateOutput(const string& node, const string& old_output,
                           const string& new_output) {
  outputs_[node].erase(nodes_[old_output]);
  outputs_[node].insert(nodes_[new_output]);
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

}  // end namespace grappler
}  // end namespace tensorflow
