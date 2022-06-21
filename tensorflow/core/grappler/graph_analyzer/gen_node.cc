/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/graph_analyzer/gen_node.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/grappler/graph_analyzer/hash_tools.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {

GenNode::GenNode(const NodeDef* node) : node_(node), op_(nullptr) {}

Status GenNode::BuildGraphInMap(const GraphDef& source, GenNodeMap* map) {
  for (const auto& n : source.node()) {
    const string& name = n.name();
    if (map->find(name) != map->end()) {
      // This error code looks more meaningful than ALREADY_EXISTS.
      return Status(error::INVALID_ARGUMENT,
                    "Duplicate node name '" + name + "'.");
    }
    (*map)[name] = absl::make_unique<GenNode>(&n);
  }
  // Now parse the links.
  for (const auto& mapit : *map) {
    Status st = mapit.second->ParseInputs(map);
    if (!st.ok()) {
      return st;
    }
  }
  return OkStatus();
}

Status GenNode::ParseInputs(const GenNodeMap* map) {
  all_inputs_or_none_ = false;
  Status st = OpRegistry::Global()->LookUpOpDef(opcode(), &op_);
  if (!st.ok()) {
    return Status(
        error::INVALID_ARGUMENT,
        absl::StrFormat("Node '%s' contains an undefined operation '%s': %s",
                        name(), opcode(), st.error_message()));
  }

  int n_inputs = node_->input_size();

  int n_named_inputs = op_->input_arg_size();

  int n_multi_inputs = 0;
  for (const auto& inarg : op_->input_arg()) {
    if (!inarg.number_attr().empty() || !inarg.type_list_attr().empty()) {
      ++n_multi_inputs;
    }
  }
  bool is_commutative = grappler::IsCommutative(*node_);

  if (n_multi_inputs > 1 || (n_multi_inputs > 0 && n_named_inputs > 1)) {
    // Can't handle more than one multi-input at a time.
    // And can't handle the commutativeness of only some arguments
    // rather than all of them.
    is_commutative = false;
  }

  if (is_commutative) {
    // If truly commutative, can treat all the inputs as one multi-input.
    // It's possible to just treat the commutative nodes as AllInputsOrNone
    // but (1) this way is a bit more efficient and (2) I want to preserve this
    // more efficient code path that does all-or-none by a single input and
    // perhaps extend its use in the future.
    n_named_inputs = 1;
    all_inputs_or_none_ = false;
  } else if (n_multi_inputs > 0) {
    all_inputs_or_none_ = true;
  }

  for (int i = 0; i < n_inputs; ++i) {
    int other_position;
    string other_name = ParseNodeName(node_->input(i), &other_position);
    auto other_it = map->find(other_name);
    if (other_it == map->end()) {
      return Status(
          error::INVALID_ARGUMENT,
          absl::StrFormat(
              "Node '%s' input %d refers to a non-existing node '%s'.", name(),
              i, other_name));
    }
    GenNode* other_node = other_it->second.get();

    int this_position = other_position < 0 ? -1 : (is_commutative ? 0 : i);

    if (this_position >= 0 && n_multi_inputs == 0 &&
        this_position >= n_named_inputs) {
      return Status(
          error::INVALID_ARGUMENT,
          absl::StrFormat(
              "Node '%s' has a non-control input from '%s' at index %d but its "
              "operation '%s' defines only %d inputs.",
              name(), other_name, this_position, op_->name(), n_named_inputs));
    }

    Port this_port(/*inbound=*/true, this_position);
    Port other_port(/*inbound=*/false, other_position);

    links_[this_port].emplace_back(LinkTarget(other_node, other_port));
    other_node->links_[other_port].emplace_back(LinkTarget(this, this_port));
  }
  return OkStatus();
}

bool GenNode::IsMultiInput(Port port) const {
  if (!port.IsInbound()) {
    return false;
  }
  auto it = links_.find(port);
  if (it == links_.end()) {
    return false;  // Shouldn't happen.
  }
  return (it->second.size() > 1);
}

GenNode::Port::operator string() const {
  string result = this->IsInbound() ? "i" : "o";
  if (this->IsControl()) {
    result.append("C");
  } else {
    result.append(absl::StrFormat("%d", this->Id()));
  }
  return result;
}

}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow
