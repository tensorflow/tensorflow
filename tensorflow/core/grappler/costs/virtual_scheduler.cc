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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {

namespace {

using ::absl::StrCat;
using ::tensorflow::strings::HumanReadableNumBytes;

constexpr char kAttrInputSrc[] = "input_source_";
constexpr char kAttrSrcDevice[] = "send_device";
constexpr char kAttrDstDevice[] = "recv_device";
constexpr char kAttrTensorName[] = "tensor_name";
constexpr char kChannelDevice[] = "Channel";

float Round2(const float x) {
  // Not using std::round from <cmath> here because not all platforms seem to
  // support that (specifically Android).
  return ::round(100.0 * x) / 100.0;
}

Costs& FindOrCreateZero(const string& op_name,
                        std::map<string, Costs>* op_cost) {
  auto it = op_cost->find(op_name);
  if (it == op_cost->end()) {
    // Note that default constructor of Costs sets some memory related fields
    // to unknown values so we should explicitly initialize it with ZeroCosts.
    it = op_cost->emplace(op_name, Costs::ZeroCosts()).first;
  }
  return it->second;
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

void UpdateDeviceAnnotationState(const NodeDef* node,
                                 const NodeState& node_state,
                                 DeviceState* device) {
  if (node->attr().count(kOutputShapes) == 0) return;

  int64 execution_count = node->attr().count(kExecutionCount) == 0
                              ? 1
                              : node->attr().at(kExecutionCount).i();

  auto& shape_annotation_stats = device->shape_annotation_stats;
  shape_annotation_stats.num_ops_annotated += 1;
  shape_annotation_stats.num_ops_executed += execution_count;
  shape_annotation_stats.num_ops_executed_more_than_once +=
      execution_count > 1 ? 1 : 0;
  shape_annotation_stats.num_ops_with_incompatible_shapes +=
      node_state.shape_incompatible ? 1 : 0;
  shape_annotation_stats.num_ops_with_dynamic_shapes +=
      (execution_count > 1 && node->attr().count(kOutputSame) == 0) ? 1 : 0;
}

}  // namespace

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

HeapReadyManager::HeapReadyManager() : ReadyNodeManager() {
  std::make_heap(nodes_.begin(), nodes_.end());
}

Status HeapReadyManager::Init(
    const std::unordered_map<const NodeDef*, NodeState>* node_map) {
  // Resets the node state since different instances of the scheduler can reuse
  // the same node_manager.
  node_map_ = node_map;
  nodes_.clear();
  waiting_queue_.clear();

  // Sets up the comparator for the heap.
  greater_ = Greater();

  return Status::OK();
}

const NodeDef* HeapReadyManager::GetCurrNode() {
  if (nodes_.empty()) {
    // Nothing in the node_; probably, the very first call. Move waiting_queue_
    // to node_.
    DrainWaitingQueue();
    CHECK(!nodes_.empty()) << "GetCurrNode(), but there's no ready node";
  }
  return nodes_.front();
}

void HeapReadyManager::RemoveCurrNode() {
  if (nodes_.empty()) {
    // Make sure that there is a node to be removed at the front of nodes_.
    GetCurrNode();
  }
  std::pop_heap(nodes_.begin(), nodes_.end(), greater_);
  nodes_.pop_back();
  DrainWaitingQueue();
}

bool HeapReadyManager::Empty() const {
  return nodes_.empty() && waiting_queue_.empty();
}

void HeapReadyManager::DrainWaitingQueue() {
  for (const auto* node : waiting_queue_) {
    // push_heap in AddNode() and pop_heap in RemoveCurrNode() guarantees that
    // the first element is the node with minimum time_ready.
    nodes_.push_back(node);
    std::push_heap(nodes_.begin(), nodes_.end(), greater_);
  }
  waiting_queue_.clear();
}

bool FirstReadyCmp(
    const std::unordered_map<const NodeDef*, NodeState>* node_map,
    const NodeDef* a, const NodeDef* b) {
  if (node_map->at(a).time_ready == node_map->at(b).time_ready) {
    // Use Node name as tie-breaker for deterministic node scheduling.
    return a->name().compare(b->name()) > 0;
  } else {
    // Note: we need a node with minimum time_ready, not maximum; hence, using
    // a > b for comparison function.
    return node_map->at(a).time_ready > node_map->at(b).time_ready;
  }
}

std::function<bool(const NodeDef*, const NodeDef*)>
FirstReadyManager::Greater() {
  auto greater = [this](const NodeDef* a, const NodeDef* b) -> bool {
    return FirstReadyCmp(node_map_, a, b);
  };
  return greater;
}

std::function<bool(const NodeDef*, const NodeDef*)>
PriorityReadyManager::Greater() {
  auto greater = [this](const NodeDef* a, const NodeDef* b) -> bool {
    auto pri_a = node_priority_.at(a->name());
    auto pri_b = node_priority_.at(b->name());
    if (pri_a == pri_b) {
      // Fallback to default (FirstReady) behaviour.
      return FirstReadyCmp(node_map_, a, b);
    }
    return pri_a > pri_b;
  };
  return greater;
}

void PriorityReadyManager::AddNode(const NodeDef* node) {
  if (node_priority_.count(node->name()) == 0) {
    VLOG(3) << "Priority of node " << node->name() << " not found.";
    node_priority_[node->name()] = 0;
  }
  HeapReadyManager::AddNode(node);
}

Status PriorityReadyManager::SetPriority(
    const std::unordered_map<string, int>& node_priority) {
  node_priority_ = node_priority;
  return Status::OK();
}

CompositeNodeManager::CompositeNodeManager()
    : ReadyNodeManager(), send_manager_(), recv_manager_() {}

Status CompositeNodeManager::Init(
    const std::unordered_map<const NodeDef*, NodeState>* node_map) {
  node_map_ = node_map;
  TF_RETURN_IF_ERROR(send_manager_.Init(node_map));
  TF_RETURN_IF_ERROR(recv_manager_.Init(node_map));
  curr_node_ = nullptr;
  return Status::OK();
}

void CompositeNodeManager::AddNode(const NodeDef* node) {
  if (IsSend(*node)) {
    send_manager_.AddNode(node);
  } else if (IsRecv(*node)) {
    recv_manager_.AddNode(node);
  } else {
    const auto& device = node_map_->at(node).device_name;
    ops_lifo_map_[device].AddNode(node);
  }
}

const NodeDef* CompositeNodeManager::GetCurrNode() {
  if (curr_node_) return curr_node_;

  // Per-device LIFO for normal ops (not _Send / _Recv),
  // FirstReady for _Send and _Recv (separately),
  // Globally (among the LIFO-selected ops from each device and _Send and
  // _Recv) FirstReady,
  // Priorty order: _Send, _Recv, and then the rest, if time_ready is equal.
  std::vector<std::pair<const NodeDef*, Costs::Duration>> candidates;
  for (auto& ops_lifo : ops_lifo_map_) {
    if (!ops_lifo.second.Empty()) {
      const auto* op = ops_lifo.second.GetCurrNode();
      candidates.emplace_back(op, node_map_->at(op).time_ready);
    }
  }
  if (!send_manager_.Empty()) {
    const auto* send = send_manager_.GetCurrNode();
    candidates.emplace_back(send, node_map_->at(send).time_ready);
  }
  if (!recv_manager_.Empty()) {
    const auto* recv = recv_manager_.GetCurrNode();
    candidates.emplace_back(recv, node_map_->at(recv).time_ready);
  }
  CHECK(!candidates.empty());
  auto first_ready = std::min_element(
      candidates.begin(), candidates.end(),
      [](const std::pair<const NodeDef*, Costs::Duration>& a,
         const std::pair<const NodeDef*, Costs::Duration>& b) {
        if (a.second == b.second) {
          // Note that there can be only 1 Send and only 1 Recv in candidates,
          // at most; hence, score is 2 for Send, 1 for Recv, and 0 for a
          // normap op, and a_score and b_score are equal only if both are
          // normal ops.
          int a_score = 2 * IsSend(*a.first) + IsRecv(*a.first);
          int b_score = 2 * IsSend(*b.first) + IsRecv(*b.first);
          if (a_score == b_score) {
            // Both are normal ops; use node name as tie breaker.
            return a.first->name().compare(b.first->name()) < 0;
          } else {
            // Priortize by op type: _Send, _Recv, and normap ops.
            return a_score > b_score;
          }
        } else {
          return a.second < b.second;
        }
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
    const auto device = node_map_->at(node).device_name;
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

std::unique_ptr<ReadyNodeManager> ReadyNodeManagerFactory(
    const string& ready_node_manager) {
  if (ready_node_manager == "FIFO") {
    return absl::make_unique<FIFOManager>();
  } else if (ready_node_manager == "LIFO") {
    return absl::make_unique<LIFOManager>();
  } else if (ready_node_manager == "FirstReady") {
    return absl::make_unique<FirstReadyManager>();
  } else if (ready_node_manager == "Composite") {
    return absl::make_unique<CompositeNodeManager>();
  }
  LOG(FATAL) << "Not a valid ready node manager: " << ready_node_manager;
  return nullptr;
}

VirtualScheduler::VirtualScheduler(const bool use_static_shapes,
                                   const bool use_aggressive_shape_inference,
                                   Cluster* cluster,
                                   ReadyNodeManager* ready_nodes,
                                   std::unique_ptr<VirtualPlacer> placer)
    : ready_nodes_(ready_nodes),
      graph_costs_(Costs::ZeroCosts()),
      cluster_(cluster),
      use_static_shapes_(use_static_shapes),
      use_aggressive_shape_inference_(use_aggressive_shape_inference),
      placer_(std::move(placer)) {
  DCHECK(placer_);  // check if the pointer is valid.
  graph_costs_.num_ops_total = 0;
  initialized_ = false;
  track_mem_usage_snapshot_ = VLOG_IS_ON(1);
}

Status VirtualScheduler::Init(const GrapplerItem* item) {
  initialized_ = false;

  // Clear all internal states so that the VirtualScheduler is reusable for
  // different GrapplerItems
  node_map_.clear();
  device_.clear();
  additional_nodes_.clear();

  graph_costs_ = Costs::ZeroCosts();
  graph_costs_.num_ops_total = 0;
  op_to_cost_.clear();

  op_counts_.clear();
  op_costs_.clear();

  // Init() preprocesses the input grappler_item and graph_properties to extract
  // necessary information for emulating tensorflow op scheduling and
  // construct internal data structures (NodeState and DeviceState) for virtual
  // scheduling.
  TF_RETURN_IF_ERROR(ready_nodes_->Init(GetNodeStates()));

  // Constructs graph properties and performs shape inference.
  graph_properties_ = absl::make_unique<GraphProperties>(*item);
  if (use_static_shapes_) {
    TF_RETURN_IF_ERROR(graph_properties_->InferStatically(
        true, use_aggressive_shape_inference_, true));
  } else {
    TF_RETURN_IF_ERROR(graph_properties_->InferDynamically(cluster_));
  }

  grappler_item_ = item;
  const auto& graph = grappler_item_->graph;
  const auto& fetch_nodes = grappler_item_->fetch;
  std::set<string> feed_nodes;
  for (const auto& f : grappler_item_->feed) {
    auto iter_and_inserted_flag = feed_nodes.insert(f.first);
    QCHECK(iter_and_inserted_flag.second)
        << "Duplicate feed node found: " << f.first;
  }

  // Get the nodes that would run to output fetch_nodes.
  bool ill_formed = false;
  std::unordered_map<string, const NodeDef*> name_to_node;
  const std::vector<const NodeDef*> fetch_fanin_nodes =
      ComputeTransitiveFanin(graph, fetch_nodes, &name_to_node, &ill_formed);
  if (ill_formed) {
    return errors::InvalidArgument(
        "Ill formed graph or invalid set of fetch nodes specified");
  }

  // Once ComputeTransitiveFanin is complete, only the nodes that can be reached
  // from the fetch nodes are scheduled. So the scheduled nodes should be
  // exactly the same as those executed for real. One possible discrepancy could
  // be the control flow nodes, where tf only executes one path.

  // Traverses the graph to record _Send nodes.
  // TODO(dyoon): Instead of identifying _Send node here manually, add _Send
  // to _Recv as control dependency when creating GrapplerItem.
  std::unordered_map<string, const NodeDef*> name_to_send;
  for (const auto& node : graph.node()) {
    if (IsSend(node)) {
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
  for (const auto* curr_node : fetch_fanin_nodes) {
    auto& curr_node_state = GetNodeStateOrCreateIt(curr_node);
    const string curr_node_device = DeviceName(curr_node);
    std::vector<string> inputs;
    if (IsRecv(*curr_node)) {
      const auto& attr = curr_node->attr();
      if (attr.count("tensor_name")) {
        const auto& send_node_name = attr.at("tensor_name").s();
        auto it = name_to_send.find(send_node_name);
        // If there is a _Send associated with the curr_node (_Recv), add it as
        // input.
        if (it != name_to_send.end()) {
          const NodeDef* send = it->second;
          inputs = {send->name()};
        }
      }
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
          auto send_and_recv = CreateSendRecv(input_node, curr_node, input_node,
                                              input_node_name);
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
    // Note that we check inputs vector which may be different to
    // curr_node->input(); e.g., we add Send as input to Recv.
    const bool has_no_inputs = inputs.empty();

    if (given_as_feed || has_no_inputs) {
      curr_node_state.time_ready = Costs::Duration();
      ready_nodes_->AddNode(curr_node);
      VLOG(3) << "Added ready node: " << curr_node->name();
    }

    feed_nodes.erase(curr_node->name());

    if (IsPersistent(*curr_node)) {
      auto& device_state = device_[curr_node_device];
      for (int port_num = 0;
           port_num < curr_node_state.output_properties.size(); ++port_num) {
        device_state.persistent_nodes.insert(
            std::make_pair(curr_node, port_num));
      }
    }
  }

  if (ready_nodes_->Empty()) {
    return errors::InvalidArgument("No ready nodes in the graph.");
  }

  if (!feed_nodes.empty()) {
    // This isn't always a bug: when the caller hasn't specified the exact list
    // of feed and fetch nodes, by default we consider all placeholders as feed
    // nodes, but some of them may not be needed for the default fetch node.
    VLOG(1) << "Some feed nodes were not consumed by the fetch fanin: "
            << absl::StrJoin(feed_nodes, ",");
  }

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
      const auto& output_properties =
          graph_properties_->GetOutputProperties(NodeName(input_source_name));
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

string VirtualScheduler::DeviceName(const NodeDef* node) const {
  return placer_->get_canonical_device_name(*node);
}

string VirtualScheduler::SanitizedDeviceName(const NodeDef* node) const {
  // Replace the ":" characters that may be present in the device name with "_".
  // This makes it possible to then use the resulting string in a node name.
  return str_util::StringReplace(placer_->get_canonical_device_name(*node), ":",
                                 "_", true);
}

string VirtualScheduler::ChannelDeviceName(const NodeDef* from,
                                           const NodeDef* to) const {
  CHECK(!initialized_) << "ChannelDeviceName is called after Init().";
  return StrCat(kChannelDevice, "_from_", SanitizedDeviceName(from), "_to_",
                SanitizedDeviceName(to));
}

std::pair<const NodeDef*, const NodeDef*> VirtualScheduler::CreateSendRecv(
    const NodeDef* from, const NodeDef* to, const NodeDef* input_node,
    const string& input_name) {
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
    src_name = StrCat(from->name(), "_", input_node_port_num);
  } else {
    src_name = StrCat(from->name(), "_minus1");
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
  // GraphDef generated by AutoGrappler has tensor_name field when removing
  // _Send/_Recv nodes.
  if (input_node->attr().count(kAttrTensorName)) {
    send_attr[kAttrTensorName].set_s(
        input_node->attr().at(kAttrTensorName).s());
  }

  // _Recv op.
  auto* recv = new NodeDef();
  recv->set_name("Recv_" + src_name + "_on_" + SanitizedDeviceName(to));
  recv->set_op("_Recv");
  recv->add_input(send->name());
  recv->set_device(DeviceName(to));
  auto& recv_attr = *(recv->mutable_attr());
  recv_attr[kAttrInputSrc].set_s(input_name);
  if (input_node->attr().count(kAttrTensorName)) {
    recv_attr[kAttrTensorName].set_s(
        input_node->attr().at(kAttrTensorName).s());
  }

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
  device = placer_->get_device(*node);

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
  if (it != node_map_.end()) {
    return it->second;
  }

  // Not found; create a NodeState for this node.
  it = node_map_.emplace(node, NodeState()).first;
  auto& node_state = it->second;
  node_state.input_properties =
      graph_properties_->GetInputProperties(node->name());
  node_state.output_properties =
      graph_properties_->GetOutputProperties(node->name());
  node_state.shape_incompatible =
      graph_properties_->CheckShapeIncompatible(node->name());

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

  return it->second;
}

void VirtualScheduler::AddOutputNodesToReadyQueue(
    const NodeDef* node, const Costs::Duration& curr_time) {
  // Checks whether the Switch's output slots change over iterations.
  int slot = -1;
  if (IsSwitch(*node) && node->attr().count(kOutputSlots) > 0 &&
      node->attr().at(kOutputSlots).list().i_size() > 0) {
    slot = node->attr().at(kOutputSlots).list().i(0);
    for (int i = 1; i < node->attr().at(kOutputSlots).list().i_size(); ++i) {
      if (slot != node->attr().at(kOutputSlots).list().i(i)) {
        slot = -1;
        break;
      }
    }
  }

  // Increment num_inputs_ready of the output nodes and maybe add to ready
  // nodes.
  auto& node_state = node_map_[node];
  for (const auto& port_num_output_pair : node_state.outputs) {
    // If Switch is annotated and its output slots are always the same, we only
    // schedule the slot that was executed. Otherwise, scheduler both slots.
    if (slot >= 0 && port_num_output_pair.first != slot) continue;

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
        VLOG(3) << "  Add output: " << output_node->name();
      }
    }
  }
}

bool VirtualScheduler::MarkCurrNodeExecuted(const Costs& node_costs) {
  // Update graph_costs_ and per-op costs.
  const NodeDef* node = ready_nodes_->GetCurrNode();
  auto& node_state = node_map_[node];
  // TODO(dyoon, andiryxu): Consider to revisit node execution w.r.t. Switch and
  // Merge -- it can create a loop which may include loop-carried dependency,
  // diverge-merge, and other complex execution patterns.
  bool previously_executed_merge =
      IsMerge(*node) && (node_state.time_finished != Costs::Duration::max());

  // If there is annotation in the graph about execution times, we use that
  // number, otherwise, we assume the node is executed once.
  node_state.execution_count = node->attr().count(kExecutionCount) == 0
                                   ? 1
                                   : node->attr().at(kExecutionCount).i();

  node_state.node_costs = node_costs;
  // TotalNodeCosts() Should be called after node_costs and execution_count.
  Costs total_node_costs = node_state.TotalNodeCosts();

  graph_costs_ = CombineCosts(graph_costs_, total_node_costs);
  const string& op_name = node->op();

  auto& op_cost = FindOrCreateZero(op_name, &op_to_cost_);
  op_cost = CombineCosts(op_cost, total_node_costs);

  if (VLOG_IS_ON(2)) {
    // Also keep track of op counts and costs per op (with their shapes).
    OpContext op_context = GetCurrNode();

    string node_description = GetOpDescription(op_context.op_info);
    op_counts_[node_description] += 1;
    op_costs_[node_description] =
        std::make_pair(total_node_costs.execution_time.asMicroSeconds().count(),
                       !node_costs.inaccurate);
  }

  // Update node and device states.
  auto& device = device_[node_state.device_name];
  device.nodes_executed.push_back(node);
  // Node is scheduled when the device is available AND all the inputs are
  // ready; hence, time_scheduled is time_ready if time_ready > device curr
  // time.
  node_state.time_scheduled =
      std::max(device.GetCurrTime(), node_state.time_ready);
  // Override device curr time with the time_scheduled.
  device.device_costs.execution_time = node_state.time_scheduled;
  device.device_costs = CombineCosts(device.device_costs, total_node_costs);
  auto curr_time = device.GetCurrTime();
  node_state.time_finished = curr_time;

  // Update shape annotation states.
  UpdateDeviceAnnotationState(node, node_state, &device);

  // Update device memory usage.
  if (!IsPersistent(*node)) {
    for (const auto& port_num_output_pair : node_state.outputs) {
      int port_num = port_num_output_pair.first;
      // There's a chance that a specific output is not used at all.
      if (node_state.outputs[port_num].empty()) {
        node_state.time_no_references[port_num] = curr_time;
      } else {
        device.memory_usage +=
            CalculateOutputSize(node_state.output_properties, port_num) *
            node_state.execution_count;
        device.nodes_in_memory.insert(std::make_pair(node, port_num));
      }
    }
  }

  // Update device's per-op cost.
  auto& device_op_cost = FindOrCreateZero(op_name, &device.op_to_cost);
  device_op_cost = CombineCosts(device_op_cost, total_node_costs);

  VLOG(3) << "Op scheduled -- name: " << node->name() << ", op: " << node->op()
          << ", device: " << node->device()
          << ", execution_count: " << node_state.execution_count
          << ", ready: " << node_state.time_ready.count()
          << ", scheduled: " << node_state.time_scheduled.count()
          << ", finished: " << node_state.time_finished.count();

  if (previously_executed_merge) {
    // Skip AddOutputNodesToReadyQueue; this is due to Switch-Merge.
    VLOG(1) << "node [ " << node->name() << ", " << node->op() << " ] "
            << "is executed more than once. "
            << "Skip scheduling its output nodes.";
  } else {
    // Checks outputs, and adds ready nodes to queue.
    AddOutputNodesToReadyQueue(node, curr_time);
  }

  // Increment num_outputs_executed of the input nodes and maybe update memory.
  for (const auto& input_port : node_state.inputs) {
    auto* input = input_port.first;
    auto port = input_port.second;
    auto& input_state = node_map_[input];
    input_state.num_outputs_executed[port]++;
    if (input_state.num_outputs_executed[port] ==
            input_state.outputs[port].size() &&
        !IsPersistent(*input)) {
      // All the outputs are executed; no reference to this output port of
      // input node.
      input_state.time_no_references[port] = curr_time;
      auto& input_device = device_[input_state.device_name];
      input_device.memory_usage -=
          CalculateOutputSize(input_state.output_properties, port) *
          node_state.execution_count;

      input_device.nodes_in_memory.erase(std::make_pair(input, port));
    }
  }

  if (!IsPersistent(*node)) {
    // Now that output memory is added and used up nodes are deallocated,
    // check max memory usage.
    if (device.memory_usage > device.max_memory_usage) {
      device.max_memory_usage = device.memory_usage;

      if (track_mem_usage_snapshot_) {
        device.mem_usage_snapshot_at_peak = device.nodes_in_memory;
      }
    }
  }

  ready_nodes_->RemoveCurrNode();

  return !ready_nodes_->Empty();
}

Costs VirtualScheduler::Summary() const {
  // Overall statement about accuracy
  VLOG(1) << graph_costs_.num_ops_total << " ops processed in total, with "
          << graph_costs_.num_ops_with_unknown_shapes
          << " having unknown shapes";

  // Print out basic execution summary.
  VLOG(1) << "Expected execution time: " << graph_costs_.execution_time.count();
  VLOG(1) << "Expected compute time: " << graph_costs_.compute_time.count();
  VLOG(1) << "Expected memory time: " << graph_costs_.memory_time.count();
  VLOG(1) << "Expected intermediate memory time: "
          << graph_costs_.intermediate_memory_time.count();
  VLOG(1) << "Expected max memory: " << graph_costs_.max_memory;
  VLOG(1) << "Expected max per-op buffers: " << graph_costs_.max_per_op_buffers;
  VLOG(1) << "Expected max per-op streaming buffers: "
          << graph_costs_.max_per_op_streaming;

  VLOG(1) << "Per-op execution time / compute time / memory time"
          << " / intermediate memory time:";
  for (const auto& op_cost_pair : op_to_cost_) {
    const auto& op = op_cost_pair.first;
    const auto& cost = op_cost_pair.second.execution_time.count();
    const auto& compute_cost = op_cost_pair.second.compute_time.count();
    const auto& memory_cost = op_cost_pair.second.memory_time.count();
    const auto& intermediate_memory_cost =
        op_cost_pair.second.intermediate_memory_time.count();
    const bool is_op_cost_accurate = !op_cost_pair.second.inaccurate;
    if (cost) {  // Skip printing out zero-cost ops.
      VLOG(1) << strings::Printf(
          " + %30s : %c %10lld / %10lld / %10lld / %10lld", op.c_str(),
          (is_op_cost_accurate ? ' ' : '~'), static_cast<int64>(cost),
          static_cast<int64>(compute_cost), static_cast<int64>(memory_cost),
          static_cast<int64>(intermediate_memory_cost));
    }
  }

  // Print per device summary
  VLOG(1) << "Devices:";
  Costs critical_path_costs = Costs::ZeroCosts();
  std::vector<string> device_names;
  device_names.reserve(device_.size());
  for (auto& it : device_) {
    device_names.push_back(it.first);
  }
  std::sort(device_names.begin(), device_names.end());

  for (const auto& name : device_names) {
    const auto& state = device_.at(name);

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
            << "persistent = " << HumanReadableNumBytes(persistent_memory_usage)
            << ", peak = " << HumanReadableNumBytes(state.max_memory_usage)
            << ", total = " << HumanReadableNumBytes(max_memory_usage)
            << ", at the end: " << HumanReadableNumBytes(state.memory_usage);

    // Overall statement about accuracy
    VLOG(1) << state.device_costs.num_ops_total
            << " ops processed in total, with "
            << state.device_costs.num_ops_with_unknown_shapes
            << " having unknown shapes";

    // Device shape annotation statistics.
    const auto& device_annotation_stats = state.shape_annotation_stats;
    if (device_annotation_stats.num_ops_annotated > 0) {
      VLOG(1) << device_annotation_stats.num_ops_annotated
              << " ops with shape annotation, with "
              << device_annotation_stats.num_ops_executed_more_than_once
              << " executed more than once, "
              << device_annotation_stats.num_ops_with_dynamic_shapes
              << " with dynamic shapes, "
              << device_annotation_stats.num_ops_with_incompatible_shapes
              << " with incompatible shapes, "
              << device_annotation_stats.num_ops_executed
              << " ops executed in total.";
    }

    VLOG(1) << "Per-op execution time / compute time / memory time "
            << " / intermediate memory time"
            << " (and memory usage at peak memory usage):";

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
      const auto& compute_cost = op_cost_pair.second.compute_time.count();
      const auto& memory_cost = op_cost_pair.second.memory_time.count();
      const auto& intermediate_memory_cost =
          op_cost_pair.second.intermediate_memory_time.count();
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
        VLOG(1) << strings::Printf(
                       " + %30s : %c %10lld / %10lld / %10lld / %10lld",
                       op.c_str(), (is_op_cost_accurate ? ' ' : '~'),
                       static_cast<int64>(cost),
                       static_cast<int64>(compute_cost),
                       static_cast<int64>(memory_cost),
                       static_cast<int64>(intermediate_memory_cost))
                << " (" << HumanReadableNumBytes(op_mem_usage) << " ["
                << mem_usage_percent << "%] "
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
              << ", Individual Cost: " << (is_cost_accurate ? "" : "~") << cost
              << " us";
    }
  }

  VLOG(1) << "Critical path execution time: "
          << critical_path_costs.execution_time.count();
  return critical_path_costs;
}

Costs VirtualScheduler::Summary(RunMetadata* metadata) {
  if (metadata) GenerateRunMetadata(metadata);
  return Summary();
}

void VirtualScheduler::GenerateRunMetadata(RunMetadata* metadata) {
  // Fill RunMetadata's step_stats and partition_graphs fields.
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
      if (node_def->op() != "HloGenericOp") {
        node_stats->set_timeline_label(node_def->op());
      } else {
        // For HloGenericOp, display hlo_opcode as timeline label.
        string timeline_label;
        if (node_def->attr().count("hlo_opcode") > 0) {
          absl::StrAppend(&timeline_label,
                          node_def->attr().at("hlo_opcode").s());
        }
        if (node_def->attr().count("_hlo_metadata_op_type") > 0) {
          absl::StrAppend(&timeline_label, "/",
                          node_def->attr().at("_hlo_metadata_op_type").s());
        }
        node_stats->set_timeline_label(timeline_label);
      }
      node_stats->set_node_name(node_def->name());
      // Timestamps in microseconds (can be used by timeline_server).
      node_stats->set_op_start_rel_micros(0);
      node_stats->set_all_start_micros(
          nodestate.time_scheduled.asMicroSeconds().count());
      node_stats->set_op_end_rel_micros(
          nodestate.time_finished.asMicroSeconds().count() -
          nodestate.time_scheduled.asMicroSeconds().count());
      node_stats->set_all_end_rel_micros(
          nodestate.time_finished.asMicroSeconds().count() -
          nodestate.time_scheduled.asMicroSeconds().count());
      // Timestamps in nanoseconds (can be used by xprof trace).
      node_stats->set_op_start_rel_nanos(0);
      node_stats->set_all_start_nanos(nodestate.time_scheduled.count());
      node_stats->set_op_end_rel_nanos(nodestate.time_finished.count() -
                                       nodestate.time_scheduled.count());
      node_stats->set_all_end_rel_nanos(nodestate.time_finished.count() -
                                        nodestate.time_scheduled.count());

      auto* mem_stats = node_stats->mutable_memory_stats();
      // VirtualScheduler does not specify scratch pad memory usage.
      mem_stats->set_temp_memory_size(0);
      int64 persistent_memory_size = 0;
      if (IsPersistent(*node_def)) {
        persistent_memory_size = total_output_size;
      }
      mem_stats->set_persistent_memory_size(persistent_memory_size);
      *device_partition_graph->add_node() = *node_def;
    }
  }
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

const std::unordered_map<string, int64>
VirtualScheduler::GetPersistentMemoryUsage() const {
  std::unordered_map<string, int64> result;
  for (const auto& device : device_) {
    const string& name = device.first;
    const DeviceState& state = device.second;
    int64 persistent_memory_usage = 0;
    for (const auto& node_port : state.persistent_nodes) {
      const auto* node = node_port.first;
      const auto port = node_port.second;
      const auto output_size =
          CalculateOutputSize(node_map_.at(node).output_properties, port);
      persistent_memory_usage += output_size;
    }
    result[name] = persistent_memory_usage;
  }
  return result;
}
}  // end namespace grappler
}  // end namespace tensorflow
