/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/equal_graph_def.h"

#include <unordered_map>
#include <unordered_set>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

bool EqualGraphDef(const GraphDef& actual, const GraphDef& expected,
                   string* diff, const EqualGraphDefOptions& options) {
  // Intentionally do not check that versions match so that this routine can
  // be used for less brittle golden file tests.
  return EqualRepeatedNodeDef(actual.node(), expected.node(), diff, options);
}

uint64 GraphDefHash(const GraphDef& gdef, const EqualGraphDefOptions& options) {
  return RepeatedNodeDefHash(gdef.node(), options);
}

bool EqualRepeatedNodeDef(const protobuf::RepeatedPtrField<NodeDef>& actual,
                          const protobuf::RepeatedPtrField<NodeDef>& expected,
                          string* diff, const EqualGraphDefOptions& options) {
  std::unordered_map<string, const NodeDef*> actual_index;
  for (const NodeDef& node : actual) {
    actual_index[node.name()] = &node;
  }

  for (const NodeDef& expected_node : expected) {
    auto actual_iter = actual_index.find(expected_node.name());
    if (actual_iter == actual_index.end()) {
      if (diff != nullptr) {
        *diff = strings::StrCat("Did not find expected node '",
                                SummarizeNodeDef(expected_node), "'");
      }
      return false;
    }

    if (!EqualNodeDef(*actual_iter->second, expected_node, diff, options)) {
      return false;
    }

    actual_index.erase(actual_iter);
  }

  if (!actual_index.empty()) {
    if (diff != nullptr) {
      *diff =
          strings::StrCat("Found unexpected node '",
                          SummarizeNodeDef(*actual_index.begin()->second), "'");
    }
    return false;
  }

  return true;
}

uint64 RepeatedNodeDefHash(const protobuf::RepeatedPtrField<NodeDef>& ndefs,
                           const EqualGraphDefOptions& options) {
  uint64 h = 0xDECAFCAFFE;
  // Insert NodeDefs into map to deterministically sort by name
  std::map<string, const NodeDef*> nodes;
  for (const NodeDef& node : ndefs) {
    nodes[node.name()] = &node;
  }
  for (const auto& pair : nodes) {
    h = Hash64(pair.first.data(), pair.first.size(), h);
    h = Hash64Combine(NodeDefHash(*pair.second, options), h);
  }
  return h;
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

bool EqualNodeDef(const NodeDef& actual, const NodeDef& expected, string* diff,
                  const EqualGraphDefOptions& options) {
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
    if (options.ignore_internal_attrs && !a.first.empty() &&
        a.first[0] == '_') {
      continue;
    }
    actual_attr.insert(a.first);
  }
  for (const auto& e : expected.attr()) {
    if (options.ignore_internal_attrs && !e.first.empty() &&
        e.first[0] == '_') {
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

uint64 NodeDefHash(const NodeDef& ndef, const EqualGraphDefOptions& options) {
  uint64 h = Hash64(ndef.name());
  h = Hash64(ndef.op().data(), ndef.op().size(), h);
  h = Hash64(ndef.device().data(), ndef.device().size(), h);

  // Normal inputs. Order important.
  int first_control_input = ndef.input_size();
  for (int i = 0; i < ndef.input_size(); ++i) {
    if (StringPiece(ndef.input(i)).starts_with("^")) {
      first_control_input = i;
      break;
    }
    h = Hash64(ndef.input(i).data(), ndef.input(i).size(), h);
  }

  // Control inputs. Order irrelevant.
  std::set<string> ndef_control;
  for (int i = first_control_input; i < ndef.input_size(); ++i) {
    ndef_control.insert(ndef.input(i));
  }
  for (const string& s : ndef_control) {
    h = Hash64(s.data(), s.size(), h);
  }

  // Attributes
  std::map<string, AttrValue> ndef_attr;
  for (const auto& a : ndef.attr()) {
    if (options.ignore_internal_attrs && !a.first.empty() &&
        a.first[0] == '_') {
      continue;
    }
    ndef_attr[a.first] = a.second;
  }
  for (const auto& a : ndef_attr) {
    h = Hash64(a.first.data(), a.first.size(), h);
    h = Hash64Combine(AttrValueHash(a.second), h);
  }

  return h;
}

}  // namespace tensorflow
