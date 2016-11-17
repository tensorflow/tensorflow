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

#include "tensorflow/core/graph/graph_partition.h"

#include <deque>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

namespace {

struct DupRecvKey {
  int src_node_id;           // Edge's src node id
  int src_output_slot;       // Edge's src node output slot
  GraphDef* dst_graph;       // Edge's dst node is in this subgraph
  bool recv_output_on_host;  // The output of recv is on host
};

struct DupRecvKeyHash {
  size_t operator()(const DupRecvKey& k) const {
    size_t h = Hash64(reinterpret_cast<const char*>(&k.src_node_id),
                      sizeof(k.src_node_id), k.src_output_slot);
    h = Hash64(reinterpret_cast<const char*>(&k.dst_graph), sizeof(k.dst_graph),
               h);
    h = Hash64(reinterpret_cast<const char*>(&k.recv_output_on_host),
               sizeof(k.recv_output_on_host), h);
    return h;
  }
};

struct DupRecvKeyEq {
  bool operator()(const DupRecvKey& x, const DupRecvKey& y) const {
    return (x.src_node_id == y.src_node_id) &&
           (x.src_output_slot == y.src_output_slot) &&
           (x.dst_graph == y.dst_graph) &&
           (x.recv_output_on_host == y.recv_output_on_host);
  }
};

// struct used to store the recvs, so that start times can be properly updated
struct RecvInfo {
  NodeDef* recv;
  NodeDef* real_recv;
  int64 start_time;
};

typedef std::unordered_map<DupRecvKey, RecvInfo, DupRecvKeyHash, DupRecvKeyEq>
    DupRecvTable;

struct PairIntHash {
 public:
  std::size_t operator()(const std::pair<int, int>& x) const {
    return std::hash<int>()(x.first) ^ std::hash<int>()(x.second);
  }
};
// A map used to store memory types for the inputs/outputs of every node.
// The key is a pair of ints consisting of a node id and input/output index.
typedef std::unordered_map<std::pair<int, int>, MemoryType, PairIntHash>
    MemoryTypeMap;

// We collect the following information about the graph before performing
// graph partitioning.
struct GraphInfo {
  std::vector<DeviceType> device_types;
  MemoryTypeMap input_types;
  MemoryTypeMap output_types;
  std::vector<ControlFlowInfo> cf_info;
};

DataType EdgeType(const Edge* e) {
  if (e->IsControlEdge()) {
    return DT_FLOAT;
  } else {
    return e->dst()->input_type(e->dst_input());
  }
}

// Return true iff we need to add a same device send/recv for 'edge'.
bool NeedSameDeviceSendRecv(const Edge* edge, const GraphInfo& info) {
  if (edge->IsControlEdge()) {
    return false;
  }

  Node* src = edge->src();
  Node* dst = edge->dst();
  if (src->assigned_device_name() == dst->assigned_device_name()) {
    int src_port = edge->src_output();
    int dst_port = edge->dst_input();
    if (info.device_types[src->id()] != DEVICE_CPU) {
      auto src_it = info.output_types.find({src->id(), src_port});
      DCHECK(src_it != info.output_types.end());
      auto dst_it = info.input_types.find({dst->id(), dst_port});
      DCHECK(dst_it != info.input_types.end());
      return src_it->second != dst_it->second;
    }
  }
  return false;
}

// Return true iff (dst, dst_input) is specified on host memory.
bool IsDstInputOnHost(const Edge* edge, const GraphInfo& info) {
  Node* dst = edge->dst();
  int dst_port = edge->dst_input();
  if (info.device_types[dst->id()] != DEVICE_CPU) {
    if (edge->IsControlEdge()) return false;
    auto dst_it = info.input_types.find({dst->id(), dst_port});
    DCHECK(dst_it != info.input_types.end());
    return dst_it->second == HOST_MEMORY;
  }
  return true;
}

// Add an input to dst that comes from the "src_slot" output of the
// node named by "src_name".
void AddInput(NodeDef* dst, StringPiece src_name, int src_slot) {
  if (src_slot == Graph::kControlSlot) {
    dst->add_input(strings::StrCat("^", src_name));
  } else if (src_slot == 0) {
    dst->add_input(src_name.data(), src_name.size());
  } else {
    dst->add_input(strings::StrCat(src_name, ":", src_slot));
  }
}

// Add a control edge from each input to each recv.
void AddReadControl(const std::vector<NodeDef*>& recvs,
                    const std::vector<string>& inputs) {
  for (NodeDef* recv : recvs) {
    for (const string& input : inputs) {
      recv->add_input(strings::StrCat("^", input));
    }
  }
}

void SetSendRecvAttrs(const PartitionOptions& opts, const Edge* edge,
                      NodeDefBuilder* builder) {
  builder->Attr("tensor_name",
                strings::StrCat("edge_", edge->id(), "_", edge->src()->name()));
  builder->Attr("send_device", edge->src()->assigned_device_name());
  builder->Attr("send_device_incarnation",
                static_cast<int64>(
                    opts.get_incarnation(edge->src()->assigned_device_name())));
  builder->Attr("recv_device", edge->dst()->assigned_device_name());
  builder->Attr("client_terminated", false);
}

NodeDef* AddSend(const PartitionOptions& opts, const GraphInfo& g_info,
                 GraphDef* gdef, const Edge* edge,
                 NodeDefBuilder::NodeOut send_from, int64 start_time,
                 Status* status) {
  const DataType dtype = send_from.data_type;
  const DataType cast_dtype = opts.should_cast ? opts.should_cast(edge) : dtype;
  const Node* src = edge->src();
  const int src_port = edge->src_output();

  // host_memory = true iff we need to use HostSend/HostCast.
  bool host_memory = false;
  if (!edge->IsControlEdge()) {
    auto src_it = g_info.output_types.find({src->id(), src_port});
    DCHECK(src_it != g_info.output_types.end());
    host_memory = (src_it->second == HOST_MEMORY);
  }

  // Add a cast node that casts dtype to cast_dtype.
  // NOTE(yuanbyu): Only cast for cross-device send/recv.
  if (dtype != cast_dtype && !NeedSameDeviceSendRecv(edge, g_info)) {
    const string cast_op = (host_memory) ? "_HostCast" : "Cast";
    NodeDefBuilder cast_builder(opts.new_name(src->name()), cast_op);
    cast_builder.Device(src->assigned_device_name()).Input(send_from);
    if (opts.scheduling_for_recvs) {
      cast_builder.Attr("_start_time", start_time);
    }
    cast_builder.Attr("DstT", cast_dtype);
    NodeDef* cast = gdef->add_node();
    *status = cast_builder.Finalize(cast);
    if (!status->ok()) return nullptr;

    // Connect the Send op to the cast.
    send_from.Reset(cast->name(), 0, cast_dtype);
  }

  // Add the send node.
  const string send_op = (host_memory) ? "_HostSend" : "_Send";
  NodeDefBuilder send_builder(opts.new_name(src->name()), send_op);
  SetSendRecvAttrs(opts, edge, &send_builder);
  send_builder.Device(src->assigned_device_name()).Input(send_from);
  if (opts.scheduling_for_recvs) {
    send_builder.Attr("_start_time", start_time);
  }
  NodeDef* send = gdef->add_node();
  *status = send_builder.Finalize(send);
  return send;
}

NodeDef* AddRecv(const PartitionOptions& opts, const GraphInfo& g_info,
                 GraphDef* gdef, const Edge* edge, NodeDef** real_recv,
                 Status* status) {
  const DataType dtype = EdgeType(edge);
  const Node* src = edge->src();
  const Node* dst = edge->dst();
  const int dst_port = edge->dst_input();
  DataType cast_dtype = dtype;

  // NOTE(yuanbyu): Only cast for cross-device send/recv.
  if (opts.should_cast && !NeedSameDeviceSendRecv(edge, g_info)) {
    cast_dtype = opts.should_cast(edge);
  }

  // host_memory = true iff we need to use HostRecv/HostCast.
  bool host_memory = false;
  if (!edge->IsControlEdge()) {
    auto dst_it = g_info.input_types.find({dst->id(), dst_port});
    DCHECK(dst_it != g_info.input_types.end());
    host_memory = (dst_it->second == HOST_MEMORY);
  }

  // Add the recv node.
  const string recv_op = (host_memory) ? "_HostRecv" : "_Recv";
  NodeDefBuilder recv_builder(opts.new_name(src->name()), recv_op);
  SetSendRecvAttrs(opts, edge, &recv_builder);
  recv_builder.Device(dst->assigned_device_name())
      .Attr("tensor_type", cast_dtype);
  NodeDef* recv = gdef->add_node();
  *status = recv_builder.Finalize(recv);
  if (!status->ok()) return nullptr;
  *real_recv = recv;

  // Add the cast node (from cast_dtype to dtype) or an Identity node.
  if (dtype != cast_dtype) {
    const string cast_op = (host_memory) ? "_HostCast" : "Cast";
    NodeDefBuilder cast_builder(opts.new_name(src->name()), cast_op);
    cast_builder.Attr("DstT", dtype);
    cast_builder.Device(dst->assigned_device_name())
        .Input(recv->name(), 0, cast_dtype);
    NodeDef* cast = gdef->add_node();
    *status = cast_builder.Finalize(cast);
    if (!status->ok()) return nullptr;
    return cast;
  } else if (edge->IsControlEdge()) {
    // An Identity is only needed for control edges.
    NodeDefBuilder id_builder(opts.new_name(src->name()), "Identity");
    id_builder.Device(dst->assigned_device_name())
        .Input(recv->name(), 0, cast_dtype);
    NodeDef* id = gdef->add_node();
    *status = id_builder.Finalize(id);
    if (!status->ok()) return nullptr;
    return id;
  } else {
    return recv;
  }
}

NodeDef* AddDummyConst(const PartitionOptions& opts, GraphDef* gdef,
                       const Edge* edge, Status* status) {
  const Node* src = edge->src();
  Tensor tensor(DT_FLOAT, TensorShape({0}));
  NodeDef* result = gdef->add_node();
  *status = NodeDefBuilder(opts.new_name(src->name()), "Const")
                .Device(src->assigned_device_name())
                .Attr("dtype", DT_FLOAT)
                .Attr("value", tensor)
                .Finalize(result);
  return result;
}

// A dummy node for scheduling.
NodeDef* AddControlTrigger(const PartitionOptions& opts, GraphDef* gdef,
                           const string& assigned_device_name, int64 epoch,
                           int64 starttime, Status* status) {
  NodeDef* result = gdef->add_node();
  *status = NodeDefBuilder(opts.new_name(strings::StrCat("synch_", epoch)),
                           "ControlTrigger")
                .Device(assigned_device_name)
                .Attr("_start_time", starttime)
                .Finalize(result);
  return result;
}

// Optimize colocation for control flow nodes. For cond, we want the
// switch nodes to colocate with its data input. This is particularly
// needed for conditional reading of a remote variable. It may also
// reduce the number of devices involved in a loop.
// TODO(yuanbyu): In this case, we don't respect the requested device in
// the GraphDef for these nodes. Ideally, the placer would enforce the
// colocation to render this unnecessary.
void OptimizeControlFlowColocation(Graph* graph) {
  auto visit = [](Node* node) {
    if (IsSwitch(node)) {
      for (const Edge* in_edge : node->in_edges()) {
        if (in_edge->dst_input() == 0) {
          // Colocate with the data input.
          node->set_assigned_device_name(
              in_edge->src()->assigned_device_name());
          return;
        }
      }
    } else if (IsExit(node)) {
      for (const Edge* in_edge : node->in_edges()) {
        if (!in_edge->IsControlEdge()) {
          // Colocate with upstream node.
          node->set_assigned_device_name(
              in_edge->src()->assigned_device_name());
          return;
        }
      }
    } else {
      if ((IsEnter(node) && !IsRefType(node->input_type(0))) ||
          IsNextIteration(node)) {
        const Edge* data_edge = nullptr;
        for (const Edge* out_edge : node->out_edges()) {
          if (!out_edge->IsControlEdge()) {
            if (data_edge) {
              data_edge = nullptr;
              return;
            }
            data_edge = out_edge;
          }
        }
        // Colocate if there is only one downstream data node.
        if (data_edge) {
          node->set_assigned_device_name(
              data_edge->dst()->assigned_device_name());
        }
      }
    }
  };
  DFS(*graph, visit, {});
}

string ControlLoopName(const string& name) {
  return strings::StrCat("_cloop", name);
}

bool IsControlLoop(const Node* node) {
  const string& name = node->def().name();
  return StringPiece(name).starts_with("_cloop");
}

// An enter node for control flow.
Node* AddControlEnter(Graph* g, const string& node_name,
                      const string& device_name, const string& frame_name,
                      const int parallel_iterations, Status* status) {
  NodeBuilder node_builder(node_name, "Enter", g->op_registry());
  node_builder.Input({"dummy", 0, DT_FLOAT});
  node_builder.Attr("frame_name", frame_name);
  node_builder.Attr("parallel_iterations", parallel_iterations);
  Node* res_node;
  *status = node_builder.Finalize(g, &res_node);
  if (!status->ok()) return nullptr;
  res_node->set_assigned_device_name(device_name);
  return res_node;
}

// A merge node for control flow.
Node* AddControlMerge(const string& in_name1, const string& in_name2, Graph* g,
                      const string& node_name, const string& device_name,
                      Status* status) {
  NodeBuilder node_builder(node_name, "Merge", g->op_registry());
  node_builder.Input({{in_name1, 0, DT_FLOAT}, {in_name2, 0, DT_FLOAT}});
  Node* res_node;
  *status = node_builder.Finalize(g, &res_node);
  if (!status->ok()) return nullptr;
  res_node->set_assigned_device_name(device_name);
  return res_node;
}

// A switch node for control flow.
Node* AddControlSwitch(NodeBuilder::NodeOut input1, NodeBuilder::NodeOut input2,
                       const string& device_name,
                       const GraphDefBuilder::Options& bopts) {
  Node* res_node = ops::BinaryOp("Switch", input1, input2, bopts);
  if (bopts.HaveError()) return nullptr;
  res_node->set_assigned_device_name(device_name);
  return res_node;
}

// A next_iteration node for control flow.
Node* AddControlNext(NodeBuilder::NodeOut input, const string& device_name,
                     const GraphDefBuilder::Options& bopts) {
  Node* res_node = ops::UnaryOp("NextIteration", input, bopts);
  if (bopts.HaveError()) return nullptr;
  res_node->set_assigned_device_name(device_name);
  return res_node;
}

Node* EmptyConst(const GraphDefBuilder::Options& options) {
  if (options.HaveError()) return nullptr;
  NodeBuilder node_builder(options.GetNameForOp("Const"), "Const",
                           options.op_registry());
  const DataType dt = DataTypeToEnum<float>::v();
  TensorProto proto;
  proto.set_dtype(dt);
  TensorShape empty_shape({0});
  empty_shape.AsProto(proto.mutable_tensor_shape());
  node_builder.Attr("dtype", dt).Attr("value", proto);
  return options.FinalizeBuilder(&node_builder);
}

// A dummy const node for control flow.
Node* AddControlConst(const string& device_name,
                      const GraphDefBuilder::Options& bopts) {
  Node* res_node = EmptyConst(bopts);
  if (bopts.HaveError()) return nullptr;
  res_node->set_assigned_device_name(device_name);
  return res_node;
}

// A synthetic loop, made up of dummy nodes. It performs control-flow actions
// on behalf of a leader on a different device.
struct ControlLoop {
  Node* enter = nullptr;
  Node* merge = nullptr;
  Node* switch_node = nullptr;
};

// Add the control flow info of a new node added during partitioning.
// The new node has the same control flow info as src.
void AddControlFlowInfo(const Node* node, const Node* src,
                        std::vector<ControlFlowInfo>* cf_info) {
  int id = node->id();
  if (static_cast<size_t>(id) >= cf_info->size()) {
    cf_info->resize(id + 1);
  }
  const ControlFlowInfo& src_info = (*cf_info)[src->id()];
  ControlFlowInfo* info = &(*cf_info)[id];
  info->frame = src_info.frame;
  info->parent_frame = src_info.parent_frame;
  info->frame_name = src_info.frame_name;
}

// Constructs a control loop. Returns a struct containing the newly created
// enter, merge, and switch nodes. The enter and merge nodes are used in the
// recursive construction of control loops for nested frames (loops). The
// switch node will be connected to the LoopCond node. The merge node will
// be connected to all the recvs of the same frame by control edges when
// the actual partitioning happens.
Status AddControlLoop(const PartitionOptions& opts, Graph* g, const Node* src,
                      const Edge* edge, Node* loop_cond,
                      std::vector<ControlFlowInfo>* cf_info,
                      ControlLoop* loop) {
  Status status;
  GraphDefBuilder::Options bopts(g, &status);
  const ControlFlowInfo& src_info = (*cf_info)[src->id()];
  const string& device_name = edge->dst()->assigned_device_name();
  const string& frame_name = src_info.frame_name;
  int parallel_iterations;
  status = GetNodeAttr(src_info.frame->def(), "parallel_iterations",
                       &parallel_iterations);
  if (!status.ok()) return status;

  // The names of the nodes to be added.
  const string& enter_name =
      ControlLoopName(opts.new_name(edge->dst()->name()));
  const string& merge_name =
      ControlLoopName(opts.new_name(edge->dst()->name()));
  const string& switch_name =
      ControlLoopName(opts.new_name(edge->dst()->name()));
  const string& next_name = ControlLoopName(opts.new_name(edge->dst()->name()));

  // Add the nodes to the graph g.
  Node* enter = AddControlEnter(g, enter_name, device_name, frame_name,
                                parallel_iterations, &status);
  if (!status.ok()) return status;
  Node* merge = AddControlMerge(enter_name, next_name, g, merge_name,
                                device_name, &status);
  if (!status.ok()) return status;
  Node* switch_node = AddControlSwitch(merge, loop_cond, device_name,
                                       bopts.WithName(switch_name));
  if (!status.ok()) return status;
  Node* next =
      AddControlNext({switch_node, 1}, device_name, bopts.WithName(next_name));
  if (!status.ok()) return status;

  // Add control flow info for these new nodes:
  AddControlFlowInfo(enter, src, cf_info);
  AddControlFlowInfo(merge, src, cf_info);
  AddControlFlowInfo(switch_node, src, cf_info);
  AddControlFlowInfo(next, src, cf_info);

  // Add input edges for the newly created merge node:
  g->AddEdge(enter, 0, merge, 0);
  g->AddEdge(next, 0, merge, 1);

  loop->enter = enter;
  loop->merge = merge;
  loop->switch_node = switch_node;
  return Status::OK();
}

// Build memory and device type info for every node in the graph.
// TODO(yuanbyu): It might be simpler if we convert MemoryType to
// DeviceType for the inputs/outputs of each node.
Status BuildMemoryDeviceInfo(const Graph& g, GraphInfo* info) {
  MemoryTypeVector input_memory_types;
  MemoryTypeVector output_memory_types;

  info->device_types.resize(g.num_node_ids(), DEVICE_CPU);
  for (const Node* node : g.nodes()) {
    if (!node->IsOp()) continue;  // Skip Sink/Source nodes.

    DeviceNameUtils::ParsedName parsed;
    if (!DeviceNameUtils::ParseFullName(node->assigned_device_name(),
                                        &parsed)) {
      return errors::Internal("Malformed assigned device '",
                              node->assigned_device_name(), "'");
    }

    TF_RETURN_IF_ERROR(MemoryTypesForNode(
        g.op_registry(), DeviceType(parsed.type), node->def(),
        &input_memory_types, &output_memory_types));

    int node_id = node->id();
    info->device_types[node_id] = DeviceType(parsed.type);
    for (size_t i = 0; i < input_memory_types.size(); ++i) {
      info->input_types[{node_id, i}] = input_memory_types[i];
    }
    for (size_t i = 0; i < output_memory_types.size(); ++i) {
      info->output_types[{node_id, i}] = output_memory_types[i];
    }
  }
  return Status::OK();
}

const Node* InputFrame(const Node* node,
                       const std::vector<ControlFlowInfo>& cf_info) {
  // An input is in the same frame as the node except for Enter nodes.
  // The input of Enter is in the parent frame of the Enter node.
  if (!node->IsEnter()) {
    return node;
  }
  return cf_info[node->id()].parent_frame;
}

const Node* OutputFrame(const Node* node,
                        const std::vector<ControlFlowInfo>& cf_info) {
  // An output is in the same frame as the node except for Exit nodes.
  // The output of Exit is in the parent frame of the Exit node.
  if (!node->IsExit()) {
    return node;
  }
  return cf_info[node->id()].parent_frame;
}

// Each participating device needs to decide a) if there is a next iteration,
// and b) if the loop terminates. We take the approach to encode this control
// flow logic in the dataflow graph. There are at least two possible encodings.
// In a completely decentralized encoding, the participants communicate peer
// to peer. The other encoding uses a frame leader (the participant who owns
// the pivot termination predicate) to broadcast the termination condition to
// all the participants. For now we take the latter because it is simpler.
//
// TODO(yuanbyu): The correctness of this construction is rather subtle. I got
// it wrong many times so it would be nice to write a proof to be sure.
Status AddControlFlow(const PartitionOptions& opts, Graph* g,
                      GraphInfo* g_info) {
  Status status;
  GraphDefBuilder::Options bopts(g, &status);
  std::vector<ControlFlowInfo>& cf_info = g_info->cf_info;

  // Build the control flow info for every node.
  status = BuildControlFlowInfo(g, &cf_info);
  if (!status.ok()) return status;

  OptimizeControlFlowColocation(g);

  // The map from frames to their LoopCond nodes.
  std::unordered_map<string, Node*> frame_cond_map;
  int num_node_ids = g->num_node_ids();
  for (int i = 0; i < num_node_ids; ++i) {
    Node* node = g->FindNodeId(i);
    if (node == nullptr) continue;

    if (IsLoopCond(node)) {
      const string& frame_name = cf_info[node->id()].frame_name;
      DCHECK(!frame_name.empty());
      frame_cond_map[frame_name] = node;
    }
  }

  // Add all control loops for cross-device frames.
  // A control loop is added only when there is a cross-device edge in a
  // non-root frame. Nothing is added if there is no loops. We also don't
  // add anything for a frame that is completely local to a device. For
  // nested loops, we stack the control loops together by connecting
  // the merge of the outer loop to the enter of the inner loop.
  //
  // A map from <frame_name, device_name> to ControlLoop.
  std::unordered_map<string, ControlLoop> control_loops;
  int num_edge_ids = g->num_edge_ids();
  for (int i = 0; i < num_edge_ids; ++i) {
    const Edge* edge = g->FindEdgeId(i);
    if (edge == nullptr) continue;

    const Node* src = edge->src();
    const Node* dst = edge->dst();
    // Skip Sink/Source nodes.
    if (!src->IsOp() || !dst->IsOp()) continue;

    const string& src_device = src->assigned_device_name();
    const string& dst_device = dst->assigned_device_name();
    // Skip local edges.
    if (src_device == dst_device) continue;

    const Node* src_frame = OutputFrame(src, cf_info);
    const Node* dst_frame = InputFrame(dst, cf_info);
    const string& src_frame_name = cf_info[src_frame->id()].frame_name;
    const string& dst_frame_name = cf_info[dst_frame->id()].frame_name;
    // Skip if src and dst are not in the same frame.
    if (src_frame_name.empty() || src_frame_name != dst_frame_name) {
      continue;
    }

    // Add the control loop. Start by adding the control loop for the
    // current frame if needed, and recursively adding the control loop
    // for its outer frame when nested.
    ControlLoop child_loop;
    while (true) {
      const string& curr_frame_name = cf_info[src_frame->id()].frame_name;
      if (curr_frame_name.empty()) {
        // We have reached the root frame.
        if (child_loop.merge != nullptr) {
          const string& node_name = opts.new_name(edge->dst()->name());
          const string& device_name = edge->dst()->assigned_device_name();
          Node* const_node =
              AddControlConst(device_name, bopts.WithName(node_name));
          if (!status.ok()) return status;
          AddControlFlowInfo(const_node, src_frame, &cf_info);
          g->AddEdge(const_node, 0, child_loop.enter, 0);
        }
        break;
      }

      const string& cl_key = strings::StrCat(curr_frame_name, "$$", dst_device);
      auto it = control_loops.find(cl_key);
      if (it != control_loops.end()) {
        if (child_loop.enter != nullptr) {
          g->AddEdge(it->second.merge, 0, child_loop.enter, 0);
        }
        break;
      }

      // Get the frame's LoopCond.
      auto cond_it = frame_cond_map.find(curr_frame_name);
      if (cond_it == frame_cond_map.end()) {
        return errors::InvalidArgument(
            "A cross-device loop must have a pivot predicate: ",
            curr_frame_name);
      }
      Node* loop_cond = cond_it->second;

      // Add the control loop.
      ControlLoop curr_loop;
      status = AddControlLoop(opts, g, src_frame, edge, loop_cond, &cf_info,
                              &curr_loop);
      if (!status.ok()) return status;
      control_loops[cl_key] = curr_loop;

      if (child_loop.enter != nullptr) {
        // Connect the merge of the outer loop to the enter of the inner.
        g->AddEdge(curr_loop.merge, 0, child_loop.enter, 0);
      }
      src_frame = cf_info[src_frame->id()].parent_frame;
      child_loop = curr_loop;
    }
  }

  // For a cross-device edge, on the dst device, add a control edge
  // from the merge node of the control loop to dst. If a send/recv is
  // introduced for this edge in future partitioning, we delete this
  // control edge and add a new control edge from the merge to the recv.
  num_edge_ids = g->num_edge_ids();
  for (int i = 0; i < num_edge_ids; ++i) {
    const Edge* edge = g->FindEdgeId(i);
    if (edge == nullptr) continue;

    const Node* src = edge->src();
    Node* dst = edge->dst();
    // Skip Sink/Source nodes.
    if (!src->IsOp() || !dst->IsOp()) continue;

    const string& src_device = src->assigned_device_name();
    const string& dst_device = dst->assigned_device_name();
    if (src_device != dst_device) {
      const Node* src_frame = OutputFrame(src, cf_info);
      const Node* dst_frame = InputFrame(dst, cf_info);
      const string& src_frame_name = cf_info[src_frame->id()].frame_name;
      const string& dst_frame_name = cf_info[dst_frame->id()].frame_name;
      if (!src_frame_name.empty() && src_frame_name == dst_frame_name) {
        const string& cl_key =
            strings::StrCat(dst_frame_name, "$$", dst_device);
        ControlLoop loop = control_loops[cl_key];
        DCHECK(loop.enter != nullptr);
        g->AddControlEdge(loop.merge, dst);
      }
    }
  }
  return Status::OK();
}

}  // end namespace

Status AddControlEdges(const PartitionOptions& opts,
                       std::unordered_map<string, GraphDef>* partitions) {
  Status status;
  // TODO(yuanbyu): Very naive for now. To be improved.
  const int num_epochs = 100;
  const int prefetch = 6;

  typedef std::pair<const NodeDef*, int64> NodeStartTime;
  for (auto& part : *partitions) {
    GraphDef* gdef = &part.second;

    std::vector<NodeStartTime> start_times;
    start_times.resize(gdef->node_size());
    for (int n = 0; n < gdef->node_size(); ++n) {
      const NodeDef& ndef = gdef->node(n);
      int64 start_time;
      status = GetNodeAttr(ndef, "_start_time", &start_time);
      if (!status.ok()) {
        return status;
      }
      start_times[n] = std::make_pair(&ndef, start_time);
    }

    // Sort the nodes based on their start times.
    std::sort(
        start_times.begin(), start_times.end(),
        [](NodeStartTime x, NodeStartTime y) { return x.second < y.second; });

    // Add a dummy node for every epoch, and add a control edge from the
    // "last" node in the preceding epoch to the dummy node.
    string device_name = gdef->node(0).device();
    int64 makespan = start_times.back().second;
    int64 resolution = (makespan / num_epochs) + 1;

    int i = 0;
    int j = 0;
    std::vector<NodeDef*> dummys;
    while (i < num_epochs && static_cast<size_t>(j) < start_times.size()) {
      if (i * resolution > start_times[j].second) {
        j++;
      } else {
        NodeDef* dummy = AddControlTrigger(opts, gdef, device_name, i,
                                           i * resolution, &status);
        if (!status.ok()) {
          return status;
        }
        dummys.push_back(dummy);
        if (j > 0) {
          string src_name = start_times[j - 1].first->name();
          AddInput(dummy, src_name, Graph::kControlSlot);
        }
        i++;
      }
    }

    // Finally, add the control edges to recvs.
    for (int n = 0; n < gdef->node_size(); ++n) {
      NodeDef* ndef = gdef->mutable_node(n);
      if (ndef->op() == "_Recv") {
        int64 start_time;
        status = GetNodeAttr(*ndef, "_start_time", &start_time);
        if (!status.ok()) {
          return status;
        }
        int recv_epoch = start_time / resolution;
        if (recv_epoch >= prefetch) {
          NodeDef* dummy = dummys[recv_epoch - prefetch];
          AddInput(ndef, dummy->name(), Graph::kControlSlot);
        }
      }
    }
  }
  return Status::OK();
}

Status Partition(const PartitionOptions& opts, Graph* g,
                 std::unordered_map<string, GraphDef>* partitions) {
  Status status;
  partitions->clear();

  GraphInfo g_info;
  if (!opts.control_flow_added) {
    // Add the "code" for distributed execution of control flow. Code is
    // added only for the frames that are placed on multiple devices. The
    // new graph is an equivalent transformation of the original graph and
    // has the property that it can be subsequently partitioned arbitrarily
    // (down to the level of individual device) for distributed execution.
    status = AddControlFlow(opts, g, &g_info);
    if (!status.ok()) return status;
  }

  // At this point, all the graph mutations have been done. Build memory
  // and device type info for every node and edge in the graph.
  status = BuildMemoryDeviceInfo(*g, &g_info);
  if (!status.ok()) return status;

  string dstp;
  std::vector<const Edge*> inputs;
  DupRecvTable dup_recv(3);
  // For a node dst, 'ref_recvs' remembers the recvs introduced by a ref
  // edge to dst. 'ref_control_inputs' remembers the inputs by a non-ref
  // edge to dst. We will add a control edge for every pair in
  // (ref_recvs x ref_control_inputs).
  std::vector<NodeDef*> ref_recvs;
  std::vector<string> ref_control_inputs;

  int32 num_data = 0;
  int32 num_control = 0;
  for (const Node* dst : g->nodes()) {
    if (!dst->IsOp()) continue;  // Skip Sink/Source nodes.

    dstp = opts.node_to_loc(dst);
    GraphDef* dst_graph = &(*partitions)[dstp];
    NodeDef* dst_def = dst_graph->add_node();
    *dst_def = dst->def();
    dst_def->set_device(dst->assigned_device_name());
    dst_def->clear_input();  // Inputs are filled below
    if (opts.need_to_record_start_times) {
      int64 start_time = opts.start_times[dst->id()].value();
      AddNodeAttr("_start_time", start_time, dst_def);
    }

    // Arrange the incoming edges to dst so that input[i] holds the
    // input flowing into slot numbered i. Trailing entries in input[]
    // hold control edges.
    inputs.clear();
    inputs.resize(dst->num_inputs(), nullptr);
    ref_recvs.clear();
    ref_control_inputs.clear();
    const Edge* control_flow_edge = nullptr;
    int32 num_control_flow_edges = 0;
    int32 num_input_edges = 0;
    for (const Edge* edge : dst->in_edges()) {
      if (edge->IsControlEdge()) {
        if (IsMerge(edge->src()) && IsControlLoop(edge->src())) {
          // This is one of the control edges added for control flow. There
          // can be multiple such edges as the dest node may have multiple
          // remote inputs. We keep track of the number of such edges.
          control_flow_edge = edge;
          ++num_control_flow_edges;
        } else {
          inputs.push_back(edge);
        }
      } else {
        DCHECK(inputs[edge->dst_input()] == nullptr);
        inputs[edge->dst_input()] = edge;
        ++num_input_edges;
      }
    }

    if (num_input_edges != dst->num_inputs()) {
      return errors::InvalidArgument("Incomplete graph, missing ",
                                     (dst->num_inputs() - num_input_edges),
                                     " inputs for ", dst->name());
    }

    // Process in order so that all data edges are added as inputs to
    // dst in Edge::dst_input() order.
    for (const Edge* edge : inputs) {
      const Node* src = edge->src();
      if (!src->IsOp()) continue;  // Skip Sink/Source nodes.

      GraphDef* src_graph = &(*partitions)[opts.node_to_loc(src)];
      if (src_graph == dst_graph && !NeedSameDeviceSendRecv(edge, g_info)) {
        // Same partition and compatible memory types:
        AddInput(dst_def, src->name(), edge->src_output());
        if (edge->IsControlEdge() ||
            !IsRefType(src->output_type(edge->src_output()))) {
          ref_control_inputs.push_back(src->name());
        }
        continue;
      }

      int64 send_start_time = 0;
      int64 recv_start_time = 0;
      if (opts.scheduling_for_recvs) {
        if (opts.need_to_record_start_times) {
          send_start_time = opts.start_times[src->id()].value();
          recv_start_time = opts.start_times[dst->id()].value();
        } else {
          status = GetNodeAttr(src->def(), "_start_time", &send_start_time);
          if (!status.ok()) {
            return status;
          }
          status = GetNodeAttr(dst->def(), "_start_time", &recv_start_time);
          if (!status.ok()) {
            return status;
          }
        }
      }

      // Check whether there is already a send/recv pair transferring
      // the same tensor/control from the src to dst partition.
      const bool on_host = IsDstInputOnHost(edge, g_info);
      DupRecvKey key{src->id(), edge->src_output(), dst_graph, on_host};
      auto iter = dup_recv.find(key);
      if (iter != dup_recv.end()) {
        // We found one. Reuse the data/control transferred already.
        const string& recv_node_name = iter->second.recv->name();
        if (edge->IsControlEdge()) {
          AddInput(dst_def, recv_node_name, Graph::kControlSlot);
        } else {
          AddInput(dst_def, recv_node_name, 0);
        }
        ref_control_inputs.push_back(recv_node_name);

        // We want the start_time for the recv to be the smallest of the start
        // times of it's consumers. So we update this whenever we use a recv,
        // and write it out to the attribute at the end of the subroutine
        if (iter->second.start_time > recv_start_time) {
          iter->second.start_time = recv_start_time;
        }
        continue;
      }

      NodeDefBuilder::NodeOut send_from;
      if (edge->IsControlEdge()) {
        // Insert a dummy const node that will generate a tiny
        // data element to be sent from send to recv.
        VLOG(1) << "Send/Recv control: " << src->assigned_device_name() << "["
                << src->name() << "] -> " << dst->assigned_device_name() << "["
                << dst->name() << "]";
        NodeDef* dummy = AddDummyConst(opts, src_graph, edge, &status);
        if (!status.ok()) return status;
        // Set the start time for this dummy node.
        if (opts.scheduling_for_recvs) {
          AddNodeAttr("_start_time", send_start_time, dummy);
        }
        AddInput(dummy, src->name(), Graph::kControlSlot);
        send_from.Reset(dummy->name(), 0, DT_FLOAT);
      } else {
        send_from.Reset(src->name(), edge->src_output(), EdgeType(edge));
      }

      // Need to split edge by placing matching send/recv nodes on
      // the src/dst sides of the edge.
      NodeDef* send = AddSend(opts, g_info, src_graph, edge, send_from,
                              send_start_time, &status);
      if (!status.ok()) return status;

      NodeDef* real_recv = nullptr;
      NodeDef* recv =
          AddRecv(opts, g_info, dst_graph, edge, &real_recv, &status);
      if (!status.ok()) return status;

      // Fix up the control flow edge.
      // NOTE(yuanbyu): 'real_recv' must be the real recv node.
      if (src_graph == dst_graph) {
        // For same device send/recv, add a control edge from send to recv.
        // This prevents the asynchronous recv kernel from being scheduled
        // before the data is available.
        AddInput(real_recv, send->name(), Graph::kControlSlot);
      } else if (control_flow_edge != nullptr) {
        // Redirect control edge to the real recv since this is not a same
        // device send/recv.
        --num_control_flow_edges;
        AddInput(real_recv, control_flow_edge->src()->name(),
                 Graph::kControlSlot);
      }

      if (!edge->IsControlEdge() &&
          IsRefType(src->output_type(edge->src_output()))) {
        AddNodeAttr("_start_time", recv_start_time, recv);
        if (real_recv != recv) {
          AddNodeAttr("_start_time", recv_start_time, real_recv);
        }
        // If src is of ref type and the edge is not a control edge, dst has
        // read semantics and therefore we must control the recv.
        ref_recvs.push_back(real_recv);
      } else {
        // Memorize the send/recv pair, only if this is not a "ref" edge.
        // NOTE(yuanbyu): Collapsing ref edges requires extreme care so
        // for now we don't do it.
        dup_recv[key] = {recv, real_recv, recv_start_time};
        ref_control_inputs.push_back(recv->name());
      }

      if (edge->IsControlEdge()) {
        ++num_control;
        AddInput(dst_def, recv->name(), Graph::kControlSlot);
      } else {
        ++num_data;
        AddInput(dst_def, recv->name(), 0);
      }
    }

    // Add control edges from 'ref_control_inputs' to 'ref_recvs'.
    // NOTE(yuanbyu): Adding these control edges should not introduce
    // deadlocks. 'dst' has implicit "read" nodes that, when we split
    // across devices, are made explicit; Retargettig the dependencies
    // to 'dst' to those nodes would not introduce cycles if there isn't
    // one before the transformation.
    // NOTE(yuanbyu): This may impact performance because it defers the
    // execution of recvs until all the other inputs become available.
    AddReadControl(ref_recvs, ref_control_inputs);

    // Add back the control edges for control flow that are not used.
    if (control_flow_edge != nullptr) {
      for (int i = 0; i < num_control_flow_edges; ++i) {
        AddInput(dst_def, control_flow_edge->src()->name(),
                 Graph::kControlSlot);
      }
    }
  }

  // Set versions
  for (auto& it : *partitions) {
    it.second.mutable_versions()->CopyFrom(g->versions());
  }

  // Set the start times for recvs at the very end.
  if (opts.scheduling_for_recvs) {
    for (auto& it : dup_recv) {
      AddNodeAttr("_start_time", it.second.start_time, it.second.recv);
      if (it.second.real_recv != it.second.recv) {
        AddNodeAttr("_start_time", it.second.start_time, it.second.real_recv);
      }
    }
  }

  VLOG(1) << "Added send/recv: controls=" << num_control
          << ", data=" << num_data;
  return Status::OK();
}

}  // namespace tensorflow
