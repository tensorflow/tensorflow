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
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
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
namespace {
template <typename T>
bool SafeSetScalarTensorValue(double value, Tensor* tensor) {
  using RealType = typename Eigen::NumTraits<T>::Real;
  if (value > std::numeric_limits<RealType>::max() ||
      value < std::numeric_limits<RealType>::min()) {
    return false;
  }
  tensor->flat<T>()(0) = static_cast<T>(value);
  return true;
}
}  // namespace

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
      .One(strings::Scanner::LETTER_DIGIT_DOT_UNDERSCORE)
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

int NumOutputs(const NodeDef& node, GraphDef* graph) {
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
  } else {
    FunctionLibraryDefinition fdef(OpRegistry::Global(), graph->library());
    auto status = fdef.LookUpOpDef(node.op(), &op_def);
    if (status.ok()) {
      num_outputs = op_def->output_arg_size();
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

// Every permutation is a product of one or more cycles. Iterate over the cycles
// in the permutation, and convert each of those into a product of
// transpositions (swaps): https://en.wikipedia.org/wiki/Cyclic_permutation
void PermuteNodesInPlace(GraphDef* graph, std::vector<int>* permutation,
                         bool invert_permutation) {
  CHECK_EQ(graph->node_size(), permutation->size());
  std::vector<int> inv_perm(permutation->size(), 0);
  if (invert_permutation) {
    for (size_t n = 0; n < permutation->size(); ++n) {
      inv_perm[(*permutation)[n]] = n;
    }
    permutation->swap(inv_perm);
  }
  for (std::size_t n = 0; n + 1 < permutation->size(); ++n) {
    while (n != (*permutation)[n]) {
      std::size_t r = (*permutation)[n];
      graph->mutable_node()->SwapElements(n, r);
      std::swap((*permutation)[n], (*permutation)[r]);
    }
  }
}

void DedupControlInputs(NodeDef* node) {
  std::unordered_set<string> inputs;
  int pos = 0;
  while (pos < node->input_size()) {
    const string& input = node->input(pos);
    if (!inputs.insert(NodeName(input)).second && IsControlInput(input)) {
      node->mutable_input()->SwapElements(pos, node->input_size() - 1);
      node->mutable_input()->RemoveLast();
    } else {
      ++pos;
    }
  }
}

namespace {
template <typename T>
inline void STLSortAndRemoveDuplicates(T* v) {
  std::sort(v->begin(), v->end());
  v->erase(std::unique(v->begin(), v->end()), v->end());
}
}  // namespace

Status SimpleGraphView::Initialize(const GraphDef& graph, bool dedup_inputs,
                                   bool dedup_outputs) {
  graph_ = &graph;
  const int num_nodes = graph.node_size();
  inputs_.clear();
  inputs_.resize(num_nodes);
  outputs_.clear();
  outputs_.resize(num_nodes);
  name_to_index_.clear();
  name_to_index_.reserve(num_nodes);
  index_to_name_.clear();
  index_to_name_.reserve(num_nodes);

  // Build map from name to index and vice versa.
  for (int node_idx = 0; node_idx < num_nodes; ++node_idx) {
    const NodeDef& node = graph.node(node_idx);
    name_to_index_.emplace(node.name(), node_idx);
    index_to_name_.push_back(node.name());
  }

  // Build forward and reverse adjacency lists.
  for (int node_idx = 0; node_idx < num_nodes; ++node_idx) {
    const NodeDef& node = graph.node(node_idx);
    inputs_[node_idx].reserve(node.input_size());
    for (const string& input : node.input()) {
      auto it = name_to_index_.find(NodeName(input));
      if (it == name_to_index_.end()) {
        return errors::InvalidArgument("Non-existent input ", input,
                                       " for node ", node.name());
      }
      const int input_idx = it->second;
      inputs_[node_idx].push_back(input_idx);
      outputs_[input_idx].push_back(node_idx);
    }
    if (dedup_inputs) {
      // Dedup the input list while it's still hot in cache.
      STLSortAndRemoveDuplicates(&inputs_[node_idx]);
    }
  }

  // Dedup outputs.
  if (dedup_outputs) {
    for (int node_idx = 0; node_idx < num_nodes; ++node_idx) {
      STLSortAndRemoveDuplicates(&outputs_[node_idx]);
    }
  }
  return Status::OK();
}

void SimpleGraphView::DepthFirstSearch(
    const std::unordered_set<string>& op_types_to_traverse, int node_idx,
    std::set<int>* nodes_found) const {
  if (nodes_found->find(node_idx) != nodes_found->end()) {
    return;
  }
  nodes_found->insert(node_idx);
  const string& op_type = graph_->node(node_idx).op();
  if (op_types_to_traverse.find(op_type) == op_types_to_traverse.end()) {
    return;
  }
  for (auto output_idx : this->outputs(node_idx)) {
    DepthFirstSearch(op_types_to_traverse, output_idx, nodes_found);
  }
}

string SimpleGraphView::PrintToString() const {
  string str;
  for (int i = 0; i < num_nodes(); ++i) {
    strings::StrAppend(&str, "Node ", i, "'", node_name(i), "'\n", "Inputs: [");
    for (int input : inputs(i)) {
      strings::StrAppend(&str, input, " '", node_name(input), "', ");
    }
    strings::StrAppend(&str, "]\n", "Outputs: [");
    for (int j = 0; j < outputs(i).size(); ++j) {
      const int output = outputs(i)[j];
      if (j > 0) {
        strings::StrAppend(&str, ", ");
      }
      strings::StrAppend(&str, output, " '", node_name(output), "'");
    }
    strings::StrAppend(&str, "]\n");
  }
  return str;
}

#define HANDLE_CASE(DTYPE)                                          \
  case DTYPE:                                                       \
    if (!SafeSetScalarTensorValue<EnumToDataType<DTYPE>::Type>(     \
            static_cast<double>(value), tensor)) {                  \
      return errors::InvalidArgument("Cannot store value ", value,  \
                                     " in tensor of type " #DTYPE); \
    }                                                               \
    break

Status SetTensorValue(DataType dtype, int value, Tensor* tensor) {
  // TODO(rmlarsen): Support more general shapes.
  if (tensor->NumElements() != 1) {
    return errors::InvalidArgument(
        "Expected scalar tensor, got num_elements = ", tensor->NumElements());
  }
  switch (dtype) {
    // TODO(rmlarsen): Handle DT_HALF.
    //    HANDLE_CASE(DT_HALF);
    HANDLE_CASE(DT_BOOL);
    HANDLE_CASE(DT_FLOAT);
    HANDLE_CASE(DT_DOUBLE);
    HANDLE_CASE(DT_UINT8);
    HANDLE_CASE(DT_INT8);
    HANDLE_CASE(DT_UINT16);
    HANDLE_CASE(DT_INT16);
    HANDLE_CASE(DT_INT32);
    HANDLE_CASE(DT_INT64);
    HANDLE_CASE(DT_COMPLEX64);
    HANDLE_CASE(DT_COMPLEX128);
    default:
      return errors::InvalidArgument("Unsupported type ",
                                     DataTypeString(dtype));
  }
  return Status::OK();
}

#undef HANDLE_CASE

}  // end namespace grappler
}  // end namespace tensorflow
