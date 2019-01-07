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

#include "tensorflow/core/grappler/utils.h"

#include <iterator>
#include <memory>
#include <queue>
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {
namespace {
template <typename T>
bool SafeSetScalarTensorValue(double value, Tensor* tensor) {
  using RealType = typename Eigen::NumTraits<T>::Real;
  if (value > static_cast<double>(Eigen::NumTraits<RealType>::highest()) ||
      value < static_cast<double>(Eigen::NumTraits<RealType>::lowest())) {
    return false;
  }
  tensor->flat<T>()(0) = static_cast<T>(value);
  return true;
}

// Is 'node' an operator that consumes only the shape of its input, not the
// data itself?
// TODO(ezhulenev): move to op_types.h. Requires to break circular dependency.
// TODO(ezhulenev): what about Identity passing tensor to Shape consumer?
bool IsShapeConsumer(const NodeDef& node) {
  const string& op = node.op();
  return op == "Shape" || op == "ShapeN" || op == "Rank" || op == "Size";
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

string TensorIdToString(const TensorId& tensor_id) {
  return tensor_id.index() == 0 ? string(tensor_id.node())
                                : tensor_id.ToString();
}

bool IsSameInput(const string& name1, const string& name2) {
  if (name1 == name2) return true;
  TensorId tensor1 = ParseTensorName(name1);
  TensorId tensor2 = ParseTensorName(name2);
  return tensor1 == tensor2;
}

bool IsControlInput(const string& name) {
  return !name.empty() && name[0] == '^';
}

bool IsControlInput(const TensorId& tensor_id) { return tensor_id.index() < 0; }

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

bool NodeIsOnCpu(const NodeDef* node) {
  string task, device;
  return DeviceNameUtils::SplitDeviceName(node->device(), &task, &device) &&
         str_util::StartsWith(device, DEVICE_CPU);
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

bool HasControlInputs(const NodeDef& node) {
  int num_inputs = node.input_size();
  if (num_inputs > 0 && IsControlInput(node.input(num_inputs - 1))) {
    return true;
  }
  return false;
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
      if (node_as_input == node.name()) {
        ++num_outputs;
      } else {
        const TensorId tensor = ParseTensorName(node_as_input);
        if (tensor.node() == node.name()) {
          ++num_outputs;
        }
      }
    }
  }
  return num_outputs;
}

int NumNonControlDataOutputs(const NodeDef& node, const NodeMap& node_map) {
  int num_data_outputs = 0;
  for (const NodeDef* output : node_map.GetOutputs(node.name())) {
    if (IsShapeConsumer(*output)) continue;

    for (int i = 0; i < output->input_size(); ++i) {
      const string& input = output->input(i);
      if (!IsControlInput(input) && NodeName(input) == node.name()) {
        ++num_data_outputs;
        break;
      }
    }
  }
  return num_data_outputs;
}

// Returns the data type in attribute `attr_name` of `node`. If that attribute
// doesn't exist, returns DT_INVALID.
DataType GetDataTypeFromAttr(const NodeDef& node, const string& type_attr) {
  if (!node.attr().count(type_attr)) {
    return DT_INVALID;
  }
  const auto& attr = node.attr().at(type_attr);
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

template <typename UniqueContainer>
void EraseNodesFromGraphImpl(const UniqueContainer& nodes_to_delete,
                             GraphDef* graph) {
  static_assert(std::is_same<typename UniqueContainer::value_type, int>::value,
                "Need to pass container of ints");

  int last = graph->node_size() - 1;
  for (auto it = nodes_to_delete.rbegin(); it != nodes_to_delete.rend(); ++it) {
    const int index = *it;
    graph->mutable_node()->SwapElements(index, last);
    last--;
  }
  graph->mutable_node()->DeleteSubrange(last + 1, nodes_to_delete.size());
}

template <typename T>
inline void STLSortAndRemoveDuplicates(T* v) {
  std::sort(v->begin(), v->end());
  v->erase(std::unique(v->begin(), v->end()), v->end());
}

}  // namespace

void EraseNodesFromGraph(const std::set<int>& nodes_to_delete,
                         GraphDef* graph) {
  EraseNodesFromGraphImpl(nodes_to_delete, graph);
}

void EraseNodesFromGraph(std::vector<int>&& nodes_to_delete, GraphDef* graph) {
  STLSortAndRemoveDuplicates(&nodes_to_delete);
  EraseNodesFromGraphImpl(nodes_to_delete, graph);
}

void EraseNodesFromGraph(const std::set<string>& nodes_to_delete,
                         GraphDef* graph) {
  std::vector<int> nodes_idx_to_delete;
  nodes_idx_to_delete.reserve(nodes_to_delete.size());
  for (int i = 0; i < graph->node_size(); ++i) {
    if (nodes_to_delete.count(graph->node(i).name()))
      nodes_idx_to_delete.push_back(i);
  }
  EraseNodesFromGraphImpl(nodes_idx_to_delete, graph);
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
    HANDLE_CASE(DT_HALF);
    HANDLE_CASE(DT_BFLOAT16);
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

Status CheckAttrExists(const NodeDef& node, const string& key) {
  if (!HasNodeAttr(node, key)) {
    return errors::InvalidArgument("Node '", node.name(), "' lacks '", key,
                                   "' attr: ", node.ShortDebugString());
  }
  return Status::OK();
}

Status CheckAttrsExist(const NodeDef& node, absl::Span<const string> keys) {
  for (const string& key : keys) {
    TF_RETURN_IF_ERROR(CheckAttrExists(node, key));
  }
  return Status::OK();
}

Status IsKernelRegisteredForNode(const NodeDef& node) {
  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(node.device(), &parsed_name)) {
    return errors::InvalidArgument("Could not parse device name: ",
                                   node.device());
  }
  return FindKernelDef(DeviceType(parsed_name.type), node, nullptr, nullptr);
}

}  // end namespace grappler
}  // end namespace tensorflow
