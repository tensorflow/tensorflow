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

NodeDef* NodeMap::GetNode(const string& name) {
  string node_name = NodeName(name);
  return nodes_[node_name];
}

std::set<NodeDef*> NodeMap::GetOutputs(const string& node_name) {
  return outputs_[node_name];
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

string NodeName(const string& name) {
  int position;
  return ParseNodeName(name, &position);
}

int NodePosition(const string& name) {
  int position;
  ParseNodeName(name, &position);
  return position;
}

string AddPrefixToNodeName(const string& name, const string& prefix) {
  if (!name.empty()) {
    if (name[0] == '^') {
      return strings::StrCat("^", prefix, "/", name.substr(1));
    }
  }
  return strings::StrCat(prefix, "/", name);
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
  if (!notified) {
    return false;
  }
  return true;
}

}  // end namespace grappler
}  // end namespace tensorflow
