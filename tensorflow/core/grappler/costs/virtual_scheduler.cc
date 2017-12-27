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

#include "tensorflow/core/grappler/costs/virtual_scheduler.h"

#include <math.h>

#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/clusters/utils.h"
#include "tensorflow/core/grappler/costs/utils.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {
namespace {

Costs CombineCosts(const Costs& left, const Costs& right) {
  CHECK_NE(left.max_memory, kMemoryUnknown);
  CHECK_NE(left.max_per_op_buffers, kMemoryUnknown);
  CHECK_NE(left.max_per_op_streaming, kMemoryUnknown);

  Costs result = left;
  result.execution_time += right.execution_time;
  if (right.inaccurate) {
    result.inaccurate = true;
  }
  if (right.max_memory != kMemoryUnknown) {
    result.max_memory += right.max_memory;
  }
  if (right.max_per_op_buffers != kMemoryUnknown) {
    result.max_per_op_buffers =
        std::max(left.max_per_op_buffers, right.max_per_op_buffers);
  }
  if (right.max_per_op_streaming != kMemoryUnknown) {
    result.max_per_op_streaming =
        std::max(left.max_per_op_streaming, right.max_per_op_streaming);
  }
  VLOG(4) << "costs execution_time=" << result.execution_time.count()
          << " max_memory=" << result.max_memory
          << " max_per_op_buffers=" << result.max_per_op_buffers
          << " max_per_op_streaming=" << result.max_per_op_streaming;
  return result;
}

// Key to the cached _Recv ops map, and its hash and predicate structures.
struct RecvNodeDescriptor {
  const NodeDef* node;
  const int port_num;
  const string device;

  RecvNodeDescriptor(const NodeDef* node_, const int port_num_,
                     const string& device_)
      : node(node_), port_num(port_num_), device(device_) {}
};

struct RecvNodeDescriptorHash {
  std::size_t operator()(const RecvNodeDescriptor& recv_node) const {
    return std::hash<const NodeDef*>()(recv_node.node) ^
           std::hash<int>()(recv_node.port_num) ^
           std::hash<string>()(recv_node.device);
  }
};

struct RecvNodeDescriptorEqual {
  bool operator()(const RecvNodeDescriptor& a,
                  const RecvNodeDescriptor& b) const {
    return a.node == b.node && a.port_num == b.port_num && a.device == b.device;
  }
};
}  // namespace

// ReadyNodeManager
const NodeDef* LIFOManager::GetCurrNode() {
  CHECK(!nodes_.empty()) << "GetCurrNode(), but there's no ready node";
  if (curr_pos_ == nodes_.end()) {
    curr_pos_ = --(nodes_.rbegin().base());  // Last one in the list.
  }
  // Once curr_pos_ is set to a valid entry in the list, we keep using the
  // cached curr_pos_ until RemoveCurrNode() is called. AddNode() will not
  // change the GetCurrNode() return value.
  return *curr_pos_;
}

void LIFOManager::RemoveCurrNode() {
  // Make sure we have curr_pos_ ready to be removed.
  GetCurrNode();
  // Note curr_pos_ may not be pointing the last element if some nodes are
  // added.
  nodes_.erase(curr_pos_);

  curr_pos_ = nodes_.end();  // Reset curr_pos_.
}

FirstReadyManager::FirstReadyManager() : ReadyNodeManager() {
  std::make_heap(nodes_.begin(), nodes_.end());
}

void FirstReadyManager::Init(
    const std::unordered_map<const NodeDef*, NodeState>* node_state) {
  // Reset the node state since different instances of the scheduler can reuse
  // the same node_manager.
  node_state_ = node_state;
  nodes_.clear();
  waiting_queue_.clear();
  greater_ = [this](const NodeDef* a, const NodeDef* b) -> bool {
    if (node_state_->at(a).time_ready == node_state_->at(b).time_ready) {
      // Use Node name as tie-breaker for deterministic node scheduling.
      return a->name().compare(b->name()) > 0;
    } else {
      // Note: we need a node with minimum time_ready, not
      // maximum; hence, using a > b for comparison function.
      return node_state_->at(a).time_ready > node_state_->at(b).time_ready;
    }
  };
}

const NodeDef* FirstReadyManager::GetCurrNode() {
  if (nodes_.empty()) {
    // Nothing in the node_; probably, the very first call. Move
    // waiting_queue_ to node_.
    DrainWaitingQueue();
    CHECK(!nodes_.empty()) << "GetCurrNode(), but there's no ready node";
  }
  return nodes_.front();
}

void FirstReadyManager::RemoveCurrNode() {
  if (nodes_.empty()) {
    // Make sure that there is a node to be removed at the front of nodes_.
    GetCurrNode();
  }
  std::pop_heap(nodes_.begin(), nodes_.end(), greater_);
  nodes_.pop_back();
  DrainWaitingQueue();
}

bool FirstReadyManager::Empty() const {
  return nodes_.empty() && waiting_queue_.empty();
}

void FirstReadyManager::DrainWaitingQueue() {
  for (const auto* node : waiting_queue_) {
    // push_heap in AddNode() and pop_heap in RemoveCurrNode() guarantees that
    // the first element is the node with minimum time_ready.
    nodes_.push_back(node);
    std::push_heap(nodes_.begin(), nodes_.end(), greater_);
  }
  waiting_queue_.clear();
}

CompositeNodeManager::CompositeNodeManager()
    : ReadyNodeManager(), send_manager_(), recv_manager_() {}

void CompositeNodeManager::Init(
    const std::unordered_map<const NodeDef*, NodeState>* node_state) {
  node_state_ = node_state;
  send_manager_.Init(node_state);
  recv_manager_.Init(node_state);
  curr_node_ = nullptr;
}

void CompositeNodeManager::AddNode(const NodeDef* node) {
  if (IsSend(*node)) {
    send_manager_.AddNode(node);
  } else if (IsRecv(*node)) {
    recv_manager_.AddNode(node);
  } else {
    const auto& device = node_state_->at(node).device_name;
    ops_lifo_map_[device].AddNode(node);
  }
}

const NodeDef* CompositeNodeManager::GetCurrNode() {
  if (curr_node_) return curr_node_;

  // Locally (normal ops, not _Send / _Recv) LIFO,
  // Globally (among the LIFO-selected ops from each device and _Send and
  // _Recv) FirstReady.
  std::vector<std::pair<const NodeDef*, Costs::Duration>> candidates;
  for (auto& ops_lifo : ops_lifo_map_) {
    if (!ops_lifo.second.Empty()) {
      const auto* op = ops_lifo.second.GetCurrNode();
      candidates.emplace_back(op, node_state_->at(op).time_ready);
    }
  }
  if (!send_manager_.Empty()) {
    const auto* send = send_manager_.GetCurrNode();
    candidates.emplace_back(send, node_state_->at(send).time_ready);
  }
  if (!recv_manager_.Empty()) {
    const auto* recv = recv_manager_.GetCurrNode();
    candidates.emplace_back(recv, node_state_->at(recv).time_ready);
  }
  CHECK(!candidates.empty());
  auto first_ready =
      std::min_element(candidates.begin(), candidates.end(),
                       [](const std::pair<const NodeDef*, Costs::Duration>& a,
                          const std::pair<const NodeDef*, Costs::Duration>& b) {
                         return a.second < b.second;
                       });
  // Next time we call GetCurrNode(), it just returns the cached one,
  // curr_node_ until we call RemovCurrNode().
  curr_node_ = first_ready->first;

  return curr_node_;
}

void CompositeNodeManager::RemoveCurrNode() {
  const auto* node = GetCurrNode();
  if (IsSend(*node)) {
    send_manager_.RemoveCurrNode();
  } else if (IsRecv(*node)) {
    recv_manager_.RemoveCurrNode();
  } else {
    const auto device = node_state_->at(node).device_name;
    ops_lifo_map_[device].RemoveCurrNode();
  }
  // Reset curr_node_ so that GetCurrNode() finds another node.
  curr_node_ = nullptr;
}

bool CompositeNodeManager::Empty() const {
  // Empty if all the ready managers are empty.
  bool empty = true;
  for (const auto& ops_lifo : ops_lifo_map_) {
    empty &= ops_lifo.second.Empty();
  }
  return empty && send_manager_.Empty() && recv_manager_.Empty();
}

VirtualScheduler::VirtualScheduler(const GrapplerItem* grappler_item,
                                   const bool use_static_shapes,
                                   Cluster* cluster,
                                   ReadyNodeManager* ready_nodes)
    : ready_nodes_(ready_nodes),
      graph_costs_(Costs::ZeroCosts()),
      graph_properties_(*grappler_item),
      cluster_(cluster),
      grappler_item_(grappler_item),
      use_static_shapes_(use_static_shapes),
      placer_(cluster) {
  initialized_ = false;
}

ReadyNodeManager* VirtualScheduler::ReadyNodeManagerFactory(
    const string& ready_node_manager) {
  if (ready_node_manager == "FIFO") {
    return new FIFOManager();
  } else if (ready_node_manager == "LIFO") {
    return new LIFOManager();
  } else if (ready_node_manager == "FirstReady") {
    return new FirstReadyManager();
  } else if (ready_node_manager == "Composite") {
    return new CompositeNodeManager();
  }
  LOG(FATAL) << "Not a valid ready node manager: " << ready_node_manager;
}

Status VirtualScheduler::Init() {
  // Init() preprocesses the input grappler_item and graph_properties to extract
  // necessary information for emulating tensorflow op scheduling and
  // construct internal data structures (NodeState and DeviceState) for virtual
  // scheduling.
  ready_nodes_->Init(GetNodeStates());
  // Construct graph properties.
  Status status;
  if (use_static_shapes_) {
    status = graph_properties_.InferStatically(true);
  } else {
    status = graph_properties_.InferDynamically(cluster_);
  }
  if (!status.ok()) {
    return status;
  }

  const auto& graph = grappler_item_->graph;
  const auto& fetch_nodes = grappler_item_->fetch;
  std::set<string> feed_nodes;
  for (const auto& f : grappler_item_->feed) {
    auto iter_and_inserted_flag = feed_nodes.insert(f.first);
    QCHECK(iter_and_inserted_flag.second)
        << "Duplicate feed node found: " << f.first;
  }

  // Get the nodes that would run to output fetch_nodes.
  std::vector<const NodeDef*> nodes =
      ComputeTransitiveFanin(graph, fetch_nodes);

  // TODO(dyoon): this is a bit inefficient as name_to_node is already built in
  // ComputeTransitiveFanin().
  // Once ComputeTransitiveFanin is complete, only the nodes that can be reached
  // from the fetch nodes are scheduled. So the scheduled nodes should be
  // exactly the same as those executed for real. One possible discrepancy could
  // be the control flow nodes, where tf only executes one path.
  std::unordered_map<string, const NodeDef*> name_to_node;
  for (const auto& node : nodes) {
    name_to_node[node->name()] = node;
  }

  // TODO(dyoon): Instead of identifying _Send node here manually, add _Send
  // to _Recv as control dependency when creating GrapplerItem.
  std::unordered_map<string, const NodeDef*> name_to_send;
  for (const auto& node : graph.node()) {
    if (node.op() == "_Send") {
      const auto& attr = node.attr();
      name_to_send[attr.at("tensor_name").s()] = &node;
    }
  }

  // To reuse _Recv ops.
  std::unordered_map<RecvNodeDescriptor, const NodeDef*, RecvNodeDescriptorHash,
                     RecvNodeDescriptorEqual>
      cached_recv_nodes;

  // Build node_map; for each node, create its NodeState and connect its inputs
  // and outputs.
  for (const auto* curr_node : nodes) {
    auto& curr_node_state = GetNodeStateOrCreateIt(curr_node);
    const string curr_node_device = DeviceName(curr_node);
    std::vector<string> inputs;
    if (IsRecv(*curr_node)) {
      const auto& attr = curr_node->attr();
      const NodeDef* send = name_to_send[attr.at("tensor_name").s()];
      inputs = {send->name()};
    } else {
      for (const string& input : curr_node->input()) {
        inputs.push_back(input);
      }
    }
    for (const string& input_node_name : inputs) {
      // Note that input_node_name may be in <prefix><node_name>:<port_num>
      // format, where <prefix> (e.g., "^" for control dependency) and
      // ":<port_num>" may be omitted. NodeName() extracts only the node_name.
      const NodeDef* input_node = name_to_node[NodeName(input_node_name)];

      CHECK(input_node);
      const string in_device = DeviceName(input_node);
      const auto input_node_port_num = NodePosition(input_node_name);

      if (curr_node_device == in_device) {
        // Same device: connect input_node and curr_node directly.
        curr_node_state.inputs.push_back(
            std::make_pair(input_node, input_node_port_num));
        auto& input_node_state = GetNodeStateOrCreateIt(input_node);
        input_node_state.outputs[input_node_port_num].push_back(curr_node);
      } else {
        RecvNodeDescriptor recv_node(input_node, input_node_port_num,
                                     curr_node_device);
        auto it = cached_recv_nodes.find(recv_node);
        if (it != cached_recv_nodes.end()) {
          // Different device, but found an already-cached copy (a _Recv op);
          // connect the _Recv to curr_node.
          const NodeDef* recv_op = it->second;
          // recv_op's output port is hard-coded to zero.
          curr_node_state.inputs.push_back(std::make_pair(recv_op, 0));
          auto& input_node_state = node_map_.at(recv_op);
          input_node_state.outputs[0].push_back(curr_node);
        } else {
          // Different device, no cached copy; transfer input_node to the
          // curr_node's device.
          auto send_and_recv =
              CreateSendRecv(input_node, curr_node, input_node_name);
          // Note that CreateSendRecv() already connected input/output between
          // _Send and _Recv ops.
          const auto* send = send_and_recv.first;
          const auto* recv = send_and_recv.second;
          // recv_op's output port is hard-coded to zero.
          curr_node_state.inputs.push_back(std::make_pair(recv, 0));
          auto& input_node_state = GetNodeStateOrCreateIt(input_node);
          input_node_state.outputs[input_node_port_num].push_back(send);

          // Cache the _Recv op for future use.
          cached_recv_nodes[recv_node] = recv;
        }
      }
    }

    // Special case: given feed nodes are ready at time 0.
    const bool given_as_feed =
        feed_nodes.find(curr_node->name()) != feed_nodes.end();

    // Default case: node without inputs are ready at time 0.
    const bool has_no_inputs = curr_node->input().empty();

    if (!IsRecv(*curr_node) && (given_as_feed || has_no_inputs)) {
      curr_node_state.time_ready = Costs::Duration();
      ready_nodes_->AddNode(curr_node);
      VLOG(3) << "Added ready node: " << curr_node->name();
    }

    feed_nodes.erase(curr_node->name());

    if (IsPersistentNode(curr_node)) {
      auto& device_state = device_[curr_node_device];
      for (int port_num = 0;
           port_num < curr_node_state.output_properties.size(); ++port_num) {
        device_state.persistent_nodes.insert(
            std::make_pair(curr_node, port_num));
      }
    }
  }

  if (ready_nodes_->Empty()) {
    return Status(error::UNAVAILABLE, "No ready nodes in the graph.");
  }

  if (!feed_nodes.empty())
    LOG(ERROR) << "Some feed nodes were not found in the graph: "
               << str_util::Join(feed_nodes, ",");

  initialized_ = true;
  return Status::OK();
}

void VirtualScheduler::MaybeUpdateInputOutput(const NodeDef* node) {
  CHECK(!initialized_) << "MaybeUpdateInputOutput is called after Init().";
  // This method is called when NodeState is created and adds input and output
  // properties for a few exceptional cases that GraphProperties cannot provide
  // input/output properties.
  if ((IsSend(*node) || IsRecv(*node)) && node->attr().count(kAttrInputSrc)) {
    // _Send and _Recv ops created from VirtualScheduler have kAttrInputSrc
    // attr; normal _Send and _Recv ops (from the input graph) do not have that
    // attr.
    auto& node_state = node_map_[node];
    auto& inputs = node_state.input_properties;
    auto& outputs = node_state.output_properties;

    // _Send and _Recv ops are created from VirtualScheduler, so
    // there should be no inputs TensorProperties.
    CHECK(inputs.empty());
    CHECK(outputs.empty());
    const auto& attr = node->attr();
    // This is the original input source to the _Send and _Recv, and this
    // string includes "^" if it was control dependency, and output port
    /// (e.g., ":2") if the input source had multiple outputs.
    const auto& input_source_name = attr.at(kAttrInputSrc).s();
    if (IsControlInput(input_source_name)) {
      // Control dependency; regardless of the input source tensor size,
      // send 4B.
      OpInfo::TensorProperties control_message;
      control_message.set_dtype(DT_FLOAT);
      control_message.mutable_shape()->add_dim()->set_size(1);
      auto* value = control_message.mutable_value();
      value->add_float_val(1);
      inputs.push_back(control_message);
      outputs.push_back(control_message);
    } else {
      auto output_properties =
          graph_properties_.GetOutputProperties(NodeName(input_source_name));
      // Like with HasInputProperties, if a node does not have output
      // properties, it's likely it was pruned during the shape inference run.
      if (!output_properties.empty()) {
        const auto input_node_port_num = NodePosition(input_source_name);
        // Use the input source's output property as _Send and _Recv's input
        // property.
        CHECK_GT(output_properties.size(), input_node_port_num);
        inputs.push_back(output_properties[input_node_port_num]);
        outputs.push_back(output_properties[input_node_port_num]);
      }
    }
  }
}

float VirtualScheduler::Round2(const float x) const {
  // Not using std::round from <cmath> here because not all platforms seem to
  // support that (specifically Android).
  return ::round(100.0 * x) / 100.0;
}

bool VirtualScheduler::IsPersistentNode(const NodeDef* node) const {
  // Variables are persistent nodes.
  return IsVariable(*node);
}

string VirtualScheduler::DeviceName(const NodeDef* node) const {
  return placer_.get_canonical_device_name(*node);
}

string VirtualScheduler::SanitizedDeviceName(const NodeDef* node) const {
  // Replace the ":" characters that may be present in the device name with "_".
  // This makes it possible to then use the resulting string in a node name.
  return str_util::StringReplace(placer_.get_canonical_device_name(*node), ":",
                                 "_", true);
}

string VirtualScheduler::ChannelDeviceName(const NodeDef* from,
                                           const NodeDef* to) const {
  CHECK(!initialized_) << "ChannelDeviceName is called after Init().";
  return kChannelDevice + "_from_" + SanitizedDeviceName(from) + "_to_" +
         SanitizedDeviceName(to);
}

std::pair<const NodeDef*, const NodeDef*> VirtualScheduler::CreateSendRecv(
    const NodeDef* from, const NodeDef* to, const string& input_name) {
  CHECK(!initialized_) << "CreateSendRecv is called after Init().";

  // Connect "from" node to "to" node with _Send and _Recv such that
  // from -> _Send -> _Recv -> to.
  // _Send is placed on "Channel" device, and _Recv is on the same device
  // as "to" node.
  // input_node_name is the string from the "to" node to identify which output
  // we get from the "from" node.

  // Note that we use NodeState for scheduling, so _Send and _Recv
  // NodeDefs created here need not be correct: in terms of name,
  // input names, attrs, etc.

  auto input_node_port_num = NodePosition(input_name);
  string src_name;
  if (input_node_port_num >= 0) {
    src_name = strings::StrCat(from->name(), "_", input_node_port_num);
  } else {
    src_name = strings::StrCat(from->name(), "_minus1");
  }

  // _Send op.
  auto* send = new NodeDef();
  send->set_name("Send_" + src_name + "_from_" + SanitizedDeviceName(from) +
                 "_to_" + SanitizedDeviceName(to));
  send->set_op("_Send");
  send->add_input(from->name());
  send->set_device(ChannelDeviceName(from, to));
  auto& send_attr = *(send->mutable_attr());
  send_attr[kAttrInputSrc].set_s(input_name);
  send_attr[kAttrSrcDevice].set_s(DeviceName(from));
  send_attr[kAttrDstDevice].set_s(DeviceName(to));

  // _Recv op.
  auto* recv = new NodeDef();
  recv->set_name("Recv_" + src_name + "_on_" + SanitizedDeviceName(to));
  recv->set_op("_Recv");
  recv->add_input(send->name());
  recv->set_device(DeviceName(to));
  auto& recv_attr = *(recv->mutable_attr());
  recv_attr[kAttrInputSrc].set_s(input_name);

  // NodeState for _Send op.
  auto& send_node_state = GetNodeStateOrCreateIt(send);
  send_node_state.device_name = send->device();  // Set Channel device.
  send_node_state.inputs.push_back(std::make_pair(from, input_node_port_num));
  send_node_state.outputs[0].push_back(recv);

  // NodeState for _Recv op.
  auto& recv_node_state = GetNodeStateOrCreateIt(recv);
  recv_node_state.inputs.push_back(std::make_pair(send, 0));
  recv_node_state.outputs[0].push_back(to);

  // Keep the created nodes.
  additional_nodes_.emplace_back(std::unique_ptr<NodeDef>(send));
  additional_nodes_.emplace_back(std::unique_ptr<NodeDef>(recv));

  // Return _Send and _Recv.
  return std::make_pair(send, recv);
}

OpContext VirtualScheduler::GetCurrNode() const {
  const NodeDef* node = ready_nodes_->GetCurrNode();

  // Get the device from the placer.
  DeviceProperties device;
  device = placer_.get_device(*node);

  // Special case for _Send op.
  if (IsSend(*node)) {
    device.set_type(kChannelDevice);
  }

  // Construct OpContext.
  OpContext op_context;
  const auto& node_state = node_map_.at(node);
  op_context.name = node->name();
  op_context.device_name = node_state.device_name;
  auto& op_info = op_context.op_info;
  op_info.set_op(node->op());
  *op_info.mutable_attr() = node->attr();
  for (auto& input : node_state.input_properties) {
    *op_info.add_inputs() = input;
  }
  for (auto& output : node_state.output_properties) {
    *op_info.add_outputs() = output;
  }
  op_info.mutable_device()->Swap(&device);

  if (grappler_item_->graph.has_library()) {
    op_context.function_library = &grappler_item_->graph.library();
  }
  return op_context;
}

NodeState& VirtualScheduler::GetNodeStateOrCreateIt(const NodeDef* node) {
  CHECK(!initialized_) << "GetNodeStateOrCreateIt is called after Init().";

  auto it = node_map_.find(node);
  if (it == node_map_.end()) {
    // Not found; create a NodeState for this node.
    it = node_map_.emplace(node, NodeState()).first;
    auto& node_state = it->second;
    node_state.input_properties =
        graph_properties_.GetInputProperties(node->name());
    node_state.output_properties =
        graph_properties_.GetOutputProperties(node->name());

    // Some ops may need further processing to the input / output properties:
    // _Send and _Recv.
    MaybeUpdateInputOutput(node);

    if (!IsSend(*node)) {
      node_state.device_name = DeviceName(node);
      // For _Send op, device_name will be set to Channel in CreateSendRecv().
    }

    // Initialize output port related data:
    // Assume the size of OutputProperties represents the number of output ports
    // of this node.
    for (size_t i = 0; i < node_state.output_properties.size(); ++i) {
      node_state.time_no_references[i] = Costs::Duration::max();
      node_state.num_outputs_executed[i] = 0;
      // Populate an empty vector for each port. The caller will add nodes
      // that use this port as input.
      node_state.outputs[i] = {};
    }
    // Port_num -1 is for control dependency.
    node_state.time_no_references[-1] = Costs::Duration::max();
    node_state.num_outputs_executed[-1] = 0;
    node_state.outputs[-1] = {};
  }
  return it->second;
}

int64 VirtualScheduler::CalculateOutputSize(
    const std::vector<OpInfo::TensorProperties>& output_properties,
    const int port_num) const {
  if (port_num < 0) {
    return 4;  // 4B for control dependency.
  }

  if (port_num >= output_properties.size()) {
    VLOG(3) << "VirtualScheduler::CalculateOutputSize() -- "
            << "port_num: " << port_num
            << " >= output_properties.size(): " << output_properties.size();
    return 0;
  }

  const auto& output = output_properties[port_num];
  int64 output_size = DataTypeSize(BaseType(output.dtype()));

  for (const auto& dim : output.shape().dim()) {
    auto dim_size = dim.size();
    if (dim_size < 0) {
      // Zero output size if there's any unknown dim.
      output_size = 0;
      VLOG(3) << "VirtualScheduler::CalculateOutputSize() -- "
              << "unknown dim: " << output_size;
      break;
    }
    output_size *= dim_size;
  }

  return output_size;
}

Costs& VirtualScheduler::FindOrCreateZero(const string& op_name,
                                          std::map<string, Costs>* op_cost) {
  auto it = op_cost->find(op_name);
  if (it == op_cost->end()) {
    // Note that default constructor of Costs sets some memory related fields
    // to unknown values so we should explicitly initialize it with ZeroCosts.
    it = op_cost->emplace(op_name, Costs::ZeroCosts()).first;
  }
  return it->second;
}

bool VirtualScheduler::MarkCurrNodeExecuted(const Costs& node_costs) {
  // Update graph_costs_ and per-op costs.
  graph_costs_ = CombineCosts(graph_costs_, node_costs);
  const NodeDef* node = ready_nodes_->GetCurrNode();
  const string& op_name = node->op();

  // Also keep track of op counts and times per op (with their shapes).
  OpContext op_context = GetCurrNode();
  string node_description = GetOpDescription(op_context.op_info);
  op_counts_[node_description] += 1;
  op_costs_[node_description] =
      std::make_pair(node_costs.execution_time.asMicroSeconds().count(),
                     !node_costs.inaccurate);

  auto& op_cost = FindOrCreateZero(op_name, &op_to_cost_);
  op_cost = CombineCosts(op_cost, node_costs);

  // Update node and device states.
  auto& node_state = node_map_[node];
  auto& device = device_[node_state.device_name];
  device.nodes_executed.push_back(node);
  // Node is scheduled when the device is available AND all the inputs are
  // ready; hence, time_scheduled is time_ready if time_ready > device curr
  // time.
  node_state.time_scheduled =
      std::max(device.GetCurrTime(), node_state.time_ready);
  // Override device curr time with the time_scheduled.
  device.device_costs.execution_time = node_state.time_scheduled;
  device.device_costs = CombineCosts(device.device_costs, node_costs);
  auto curr_time = device.GetCurrTime();
  node_state.time_finished = curr_time;

  // Update device memory usage.
  if (!IsPersistentNode(node)) {
    for (const auto& port_num_output_pair : node_state.outputs) {
      int port_num = port_num_output_pair.first;
      // There's a chance that a specific output is not used at all.
      if (node_state.outputs[port_num].empty()) {
        node_state.time_no_references[port_num] = curr_time;
      } else {
        device.memory_usage +=
            CalculateOutputSize(node_state.output_properties, port_num);
        device.nodes_in_memory.insert(std::make_pair(node, port_num));
      }
    }
  }

  // Update device's per-op cost.
  auto& device_op_cost = FindOrCreateZero(op_name, &device.op_to_cost);
  device_op_cost = CombineCosts(device_op_cost, node_costs);

  VLOG(3) << "Op scheduled -- name: " << node->name() << ", op: " << node->op()
          << ", device: " << node->device()
          << ", ready: " << node_state.time_ready.count()
          << ", scheduled: " << node_state.time_scheduled.count()
          << ", finished: " << node_state.time_finished.count();

  // Increment num_inputs_ready of the output nodes
  for (const auto& port_num_output_pair : node_state.outputs) {
    for (auto* output_node : port_num_output_pair.second) {
      auto& output_state = node_map_[output_node];
      output_state.num_inputs_ready++;
      // Execute a node as soon as all its inputs are ready. Merge nodes are
      // special since they run as soon as one of their inputs becomes
      // available.
      if (output_state.num_inputs_ready == output_state.inputs.size() ||
          IsMerge(*output_node)) {
        // This output node is now ready.
        output_state.time_ready = curr_time;
        ready_nodes_->AddNode(output_node);
      }
    }
  }

  // Increment num_outputs_executed of the input nodes.
  for (const auto& input_port : node_state.inputs) {
    auto* input = input_port.first;
    auto port = input_port.second;
    auto& input_state = node_map_[input];
    input_state.num_outputs_executed[port]++;
    if (input_state.num_outputs_executed[port] ==
            input_state.outputs[port].size() &&
        !IsPersistentNode(input)) {
      // All the outputs are executed; no reference to this output port of
      // input node.
      input_state.time_no_references[port] = curr_time;
      auto& input_device = device_[input_state.device_name];
      input_device.memory_usage -=
          CalculateOutputSize(input_state.output_properties, port);

      input_device.nodes_in_memory.erase(std::make_pair(input, port));
    }
  }

  if (!IsPersistentNode(node)) {
    // Now that output memory is added and used up nodes are deallocated,
    // check max memory usage.
    if (device.memory_usage > device.max_memory_usage) {
      device.max_memory_usage = device.memory_usage;
      device.mem_usage_snapshot_at_peak = device.nodes_in_memory;
    }
  }

  // Remove the current node; assume FIFO.
  ready_nodes_->RemoveCurrNode();

  return !ready_nodes_->Empty();
}

Costs VirtualScheduler::Summary() const {
  // Print out basic execution summary.
  VLOG(1) << "Expected execution time: " << graph_costs_.execution_time.count();
  VLOG(1) << "Expected max memory: " << graph_costs_.max_memory;
  VLOG(1) << "Expected max per-op buffers: " << graph_costs_.max_per_op_buffers;
  VLOG(1) << "Expected max per-op streaming buffers: "
          << graph_costs_.max_per_op_streaming;

  VLOG(1) << "Per-op execution time:";
  for (const auto& op_cost_pair : op_to_cost_) {
    const auto& op = op_cost_pair.first;
    const auto& cost = op_cost_pair.second.execution_time.count();
    const bool is_op_cost_accurate = !op_cost_pair.second.inaccurate;
    if (cost) {  // Skip printing out zero-cost ops.
      VLOG(1) << " + " << op << " : " << (is_op_cost_accurate ? "" : "~")
              << cost;
    }
  }

  // Print per device summary
  VLOG(1) << "Devices:";
  Costs critical_path_costs = Costs::ZeroCosts();

  for (const auto& device : device_) {
    const auto& name = device.first;
    const auto& state = device.second;

    std::map<string, int64> op_to_memory;
    // First profile only persistent memory usage.
    int64 persistent_memory_usage = 0;
    std::set<string> persisent_ops;
    for (const auto& node_port : state.persistent_nodes) {
      const auto* node = node_port.first;
      const auto port = node_port.second;
      const auto output_size =
          CalculateOutputSize(node_map_.at(node).output_properties, port);
      persistent_memory_usage += output_size;
      op_to_memory[node->op()] += output_size;
      persisent_ops.insert(node->op());
    }
    int64 max_memory_usage = persistent_memory_usage + state.max_memory_usage;
    critical_path_costs.estimated_max_memory_per_device[name] =
        max_memory_usage;

    const Costs::NanoSeconds wall_time_ns = state.GetCurrTime();
    VLOG(1) << "Device = " << name
            << ", num_nodes = " << state.nodes_executed.size()
            << ", wall_time_ns = " << wall_time_ns.count() << ", memory usage: "
            << "persistent = "
            << strings::HumanReadableNumBytes(persistent_memory_usage)
            << ", peak = "
            << strings::HumanReadableNumBytes(state.max_memory_usage)
            << ", total = " << strings::HumanReadableNumBytes(max_memory_usage)
            << ", at the end: "
            << strings::HumanReadableNumBytes(state.memory_usage);

    VLOG(1) << "Per-op execution time (and memory usage at peak memory usage):";

    // Profile non-persistent op memory usage.
    for (const auto& node_port : state.mem_usage_snapshot_at_peak) {
      const auto* node = node_port.first;
      const auto port = node_port.second;
      op_to_memory[node->op()] +=
          CalculateOutputSize(node_map_.at(node).output_properties, port);
    }
    Costs::NanoSeconds total_compute_time_ns;
    bool is_total_cost_accurate = true;
    for (const auto& op_cost_pair : state.op_to_cost) {
      const auto& op = op_cost_pair.first;
      const auto& cost = op_cost_pair.second.execution_time.count();
      total_compute_time_ns += op_cost_pair.second.execution_time;
      const bool is_op_cost_accurate = !op_cost_pair.second.inaccurate;
      if (!is_op_cost_accurate) {
        is_total_cost_accurate = false;
      }

      int64 op_mem_usage = 0;
      auto it = op_to_memory.find(op);
      if (it != op_to_memory.end()) {
        op_mem_usage = it->second;
      }

      const float mem_usage_percent =
          max_memory_usage > 0 ? Round2(100.0 * op_mem_usage / max_memory_usage)
                               : 0.0;
      if (cost || mem_usage_percent > 1.0) {
        // Print out only non-zero cost ops or ops with > 1% memory usage.
        VLOG(1) << " + " << op << " : " << (is_op_cost_accurate ? "" : "~")
                << cost << " (" << strings::HumanReadableNumBytes(op_mem_usage)
                << " [" << mem_usage_percent << "%] "
                << (persisent_ops.count(op) > 0 ? ": persistent op)" : ")");
      }
    }

    int utilization = 0;
    if (wall_time_ns.count() > 0) {
      utilization = total_compute_time_ns.count() * 100 / wall_time_ns.count();
    }
    VLOG(1) << "Device = " << name << ", total_compute_time_ns = "
            << (is_total_cost_accurate ? "" : "~")
            << total_compute_time_ns.count()
            << ", utilization = " << utilization << "%";

    if (critical_path_costs.execution_time <= state.GetCurrTime()) {
      critical_path_costs = state.device_costs;
    }
  }

  if (VLOG_IS_ON(2)) {
    // Also log the op description and their corresponding counts.
    VLOG(2) << "Node description, counts, cost:";
    for (const auto& item : op_counts_) {
      int cost;
      bool is_cost_accurate;
      std::tie(cost, is_cost_accurate) = op_costs_.at(item.first);
      VLOG(2) << "Node: " << item.first << ", Count: " << item.second
              << ", Individual Cost: " << (is_cost_accurate ? "" : "~") << cost;
    }
  }

  VLOG(1) << "Critical path execution time: "
          << critical_path_costs.execution_time.count();
  return critical_path_costs;
}

Costs VirtualScheduler::Summary(RunMetadata* metadata) {
  if (metadata != nullptr) {
    StepStats* stepstats = metadata->mutable_step_stats();
    for (const auto& device : device_) {
      GraphDef* device_partition_graph = metadata->add_partition_graphs();
      DeviceStepStats* device_stepstats = stepstats->add_dev_stats();
      device_stepstats->set_device(device.first);
      for (const auto& node_def : device.second.nodes_executed) {
        const NodeState& nodestate = node_map_.at(node_def);
        NodeExecStats* node_stats = device_stepstats->add_node_stats();
        uint64 total_output_size = 0;
        for (int slot = 0; slot < nodestate.output_properties.size(); slot++) {
          const auto& properties = nodestate.output_properties[slot];
          NodeOutput* no = node_stats->add_output();
          no->set_slot(slot);
          TensorDescription* tensor_descr = no->mutable_tensor_description();
          tensor_descr->set_dtype(properties.dtype());
          *tensor_descr->mutable_shape() = properties.shape();
          // Optional allocation description.
          const auto tensor_size =
              CalculateOutputSize(nodestate.output_properties, slot);
          total_output_size += tensor_size;
          tensor_descr->mutable_allocation_description()->set_requested_bytes(
              tensor_size);
          tensor_descr->mutable_allocation_description()->set_allocated_bytes(
              tensor_size);
        }
        node_stats->set_timeline_label(node_def->op());
        node_stats->set_node_name(node_def->name());
        node_stats->set_op_start_rel_micros(0);
        node_stats->set_all_start_micros(
            nodestate.time_scheduled.asMicroSeconds().count());
        node_stats->set_op_end_rel_micros(
            nodestate.time_finished.asMicroSeconds().count() -
            nodestate.time_scheduled.asMicroSeconds().count());
        node_stats->set_all_end_rel_micros(
            nodestate.time_finished.asMicroSeconds().count() -
            nodestate.time_scheduled.asMicroSeconds().count());
        auto* mem_stats = node_stats->mutable_memory_stats();
        // VirtualScheduler does not specify scratch pad memory usage.
        mem_stats->set_host_temp_memory_size(0);
        mem_stats->set_device_temp_memory_size(0);
        int64 host_persistent_memory_size = 0;
        int64 device_persistent_memory_size = 0;
        if (IsPersistentNode(node_def)) {
          if (device.first.find("cpu") != string::npos ||
              device.first.find("CPU") != string::npos) {
            host_persistent_memory_size = total_output_size;
          } else {
            device_persistent_memory_size = total_output_size;
          }
        }
        mem_stats->set_host_persistent_memory_size(host_persistent_memory_size);
        mem_stats->set_device_persistent_memory_size(
            device_persistent_memory_size);
        *device_partition_graph->add_node() = *node_def;
      }
    }
  }
  return Summary();
}

const std::unordered_map<string, int64> VirtualScheduler::GetPeakMemoryUsage()
    const {
  std::unordered_map<string, int64> result;
  for (const auto& device : device_) {
    const string& name = device.first;
    const DeviceState& state = device.second;
    result[name] = state.max_memory_usage;
  }
  return result;
}

}  // end namespace grappler
}  // end namespace tensorflow
