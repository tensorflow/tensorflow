/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/graph/equal_graph_def.h"

#include <unordered_map>
#include <unordered_set>
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

bool EqualGraphDef(const GraphDef& actual, const GraphDef& expected,
                   string* diff) {
  // Intentionally do not check that versions match so that this routine can
  // be used for less brittle golden file tests.

  std::unordered_map<string, const NodeDef*> actual_index;
  for (const NodeDef& node : actual.node()) {
    actual_index[node.name()] = &node;
  }

  for (const NodeDef& expected_node : expected.node()) {
    auto actual_iter = actual_index.find(expected_node.name());
    if (actual_iter == actual_index.end()) {
      if (diff != nullptr) {
        *diff = strings::StrCat("Did not find expected node '",
                                SummarizeNodeDef(expected_node), "'");
      }
      return false;
    }

    if (!EqualNodeDef(*actual_iter->second, expected_node, diff)) return false;

    actual_index.erase(actual_iter);
  }

  if (!actual_index.empty()) {
    if (diff != nullptr) {
      *diff = strings::StrCat("Found unexpected node '",
                              SummarizeNodeDef(*actual_index.begin()->second),
                              "' not in expected graph:\n",
                              SummarizeGraphDef(expected));
    }
    return false;
  }

  return true;
}

namespace {

string JoinStringField(const protobuf::RepeatedPtrField<string>& f) {
  string ret;
  for (int i = 0; i < f.size(); ++i) {
    if (i > 0) strings::StrAppend(&ret, ", ");
    strings::StrAppend(&ret, f.Get(i));
  }
  return ret;
}

}  // namespace

bool EqualNodeDef(const NodeDef& actual, const NodeDef& expected,
                  string* diff) {
  if (actual.name() != expected.name()) {
    if (diff != nullptr) {
      *diff = strings::StrCat("Actual node name '", actual.name(),
                              "' is not expected '", expected.name(), "'");
    }
    return false;
  }

  if (actual.op() != expected.op()) {
    if (diff != nullptr) {
      *diff = strings::StrCat("Node named '", actual.name(), "' has op '",
                              actual.op(), "' that is not expected '",
                              expected.op(), "'");
    }
    return false;
  }

  if (actual.device() != expected.device()) {
    if (diff != nullptr) {
      *diff = strings::StrCat("Node named '", actual.name(), "' has device '",
                              actual.device(), "' that is not expected '",
                              expected.device(), "'");
    }
    return false;
  }

  if (actual.input_size() != expected.input_size()) {
    if (diff != nullptr) {
      *diff = strings::StrCat("Node named '", actual.name(), "' has inputs '",
                              JoinStringField(actual.input()),
                              "' that don't match expected '",
                              JoinStringField(expected.input()), "'");
    }
    return false;
  }

  int first_control_input = actual.input_size();
  for (int i = 0; i < actual.input_size(); ++i) {
    if (StringPiece(actual.input(i)).starts_with("^")) {
      first_control_input = i;
      break;
    }
    if (actual.input(i) != expected.input(i)) {
      if (diff != nullptr) {
        *diff = strings::StrCat("Node named '", actual.name(), "' has input ",
                                i, " '", actual.input(i),
                                "' that doesn't match expected '",
                                expected.input(i), "'");
      }
      return false;
    }
  }

  std::unordered_set<string> actual_control;
  std::unordered_set<string> expected_control;
  for (int i = first_control_input; i < actual.input_size(); ++i) {
    actual_control.insert(actual.input(i));
    expected_control.insert(expected.input(i));
  }
  for (const auto& e : expected_control) {
    if (actual_control.erase(e) == 0) {
      if (diff != nullptr) {
        *diff = strings::StrCat("Node named '", actual.name(),
                                "' missing expected control input '", e, "'");
      }
      return false;
    }
  }
  if (!actual_control.empty()) {
    if (diff != nullptr) {
      *diff = strings::StrCat("Node named '", actual.name(),
                              "' has unexpected control input '",
                              *actual_control.begin(), "'");
    }
    return false;
  }

  std::unordered_set<string> actual_attr;
  for (const auto& a : actual.attr()) {
    if (!a.first.empty() && a.first[0] == '_') {
      continue;
    }
    actual_attr.insert(a.first);
  }
  for (const auto& e : expected.attr()) {
    if (!e.first.empty() && e.first[0] == '_') {
      continue;
    }

    if (actual_attr.erase(e.first) == 0) {
      if (diff != nullptr) {
        *diff = strings::StrCat("Node named '", actual.name(),
                                "' missing expected attr '", e.first,
                                "' with value: ", SummarizeAttrValue(e.second));
      }
      return false;
    }
    auto iter = actual.attr().find(e.first);
    if (!AreAttrValuesEqual(e.second, iter->second)) {
      if (diff != nullptr) {
        *diff = strings::StrCat(
            "Node named '", actual.name(), "' has attr '", e.first,
            "' with value: ", SummarizeAttrValue(iter->second),
            " that does not match expected: ", SummarizeAttrValue(e.second));
      }
      return false;
    }
  }
  if (!actual_attr.empty()) {
    if (diff != nullptr) {
      *diff = strings::StrCat(
          "Node named '", actual.name(), "' has unexpected attr '",
          *actual_attr.begin(), "' with value: ",
          SummarizeAttrValue(actual.attr().find(*actual_attr.begin())->second));
    }
    return false;
  }

  return true;
}

}  // namespace tensorflow
