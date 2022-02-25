/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/graph_utils.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace data {
namespace {

using tensorflow::protobuf::util::MessageDifferencer;

// Only compares GraphDef.node, NodeDef.name, NodeDef.op, NodeDef.input, and
// specific attrs listed below. We can't use MessageDifferencer::IgnoreField
// because it does not support ignoring specific keys in a map.
static const auto* const compare_attrs = new absl::flat_hash_set<std::string>{
    "Targuments", "output_shapes", "output_types", "value"};

NodeDef GetNodeForComparison(const NodeDef& node) {
  NodeDef result;
  result.set_name(node.name());
  result.set_op(node.op());
  *result.mutable_input() = node.input();
  for (const auto& attr : node.attr()) {
    if (compare_attrs->contains(attr.first)) {
      result.mutable_attr()->insert(attr);
    }
  }
  return result;
}

GraphDef GetGraphForComparison(const GraphDef& graph) {
  GraphDef result;
  for (const NodeDef& node : graph.node()) {
    *result.add_node() = GetNodeForComparison(node);
  }
  return result;
}
}  // namespace

std::pair<bool, std::string> HaveEquivalentStructures(const GraphDef& graph1,
                                                      const GraphDef& graph2) {
  MessageDifferencer differ;
  differ.set_message_field_comparison(MessageDifferencer::EQUIVALENT);
  differ.set_repeated_field_comparison(MessageDifferencer::AS_SET);

  std::string diff;
  differ.ReportDifferencesToString(&diff);
  bool equivalent = differ.Compare(GetGraphForComparison(graph1),
                                   GetGraphForComparison(graph2));
  return {equivalent, diff};
}

}  // namespace data
}  // namespace tensorflow
