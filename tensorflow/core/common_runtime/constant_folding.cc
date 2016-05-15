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

#include <algorithm>
#include <atomic>
#include <set>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/constant_folding.h"

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace {

bool IsConstantFoldable(const Node* n,
                        std::function<bool(const Node*)> consider) {
  if (n->op_def().is_stateful()) {
    return false;
  }
  if (consider && !consider(n)) {
    return false;
  }
  if (n->IsControlFlow() || n->IsSend() || n->IsRecv()) {
    return false;
  }
  // TODO(yuanbyu): For now disable these session handle operations.
  if (n->IsGetSessionHandle() || n->IsGetSessionTensor() ||
      n->IsDeleteSessionTensor()) {
    return false;
  }
  if (n->IsSource()) {
    return false;
  }
  if (n->IsSink()) {
    return false;
  }
  return true;
}

// Returns the constant foldable nodes in `nodes_result` in data flow order.
void FindConstantFoldableNodes(const Graph* graph, ConstantFoldingOptions opts,
                               std::vector<Node*>* nodes_result) {
  std::set<const Node*> node_set;
  std::vector<Node*>& nodes = *nodes_result;
  bool internal_node_inserted = false;
  // Walk the nodes in data flow order
  ReverseDFS(*graph, nullptr,
             [&nodes, &node_set, &internal_node_inserted, opts](Node* n) {
               if (n->IsConstant()) {
                 // Constants with no control inputs (except from _SOURCE node)
                 // are definitely constant foldable.
                 if (n->in_edges().size() == 0 ||
                     (n->in_edges().size() == 1 &&
                      (*n->in_edges().begin())->src()->IsSource())) {
                   node_set.insert(n);
                   nodes.push_back(n);
                 }
               } else if (IsConstantFoldable(n, opts.consider)) {
                 // Check whether the set of this node's in_nodes is completely
                 // included in the set of constant foldable nodes. If true,
                 // then this node is also constant foldable.
                 bool all_parents_constant = true;
                 for (const Node* parent : n->in_nodes()) {
                   if (node_set.count(parent) == 0 && !parent->IsSource()) {
                     all_parents_constant = false;
                     break;
                   }
                 }
                 if (all_parents_constant) {
                   node_set.insert(n);
                   nodes.push_back(n);
                   internal_node_inserted = true;
                 }
               }
             });
  // If we have inserted just leaf level nodes, then there is nothing to fold.
  if (!internal_node_inserted) {
    nodes.clear();
  }
}

typedef std::pair<Node*, int> NodeAndOutput;

// Given the constant foldable nodes in 'nodes', returns a new graph 'g'. 'g'
// will contain copies of the nodes in 'nodes'. In addition, if there is an edge
// going from a node 'n' in 'nodes' to another node in 'orig_graph' but not in
// 'nodes', then 'tensors_to_fetch' will contain the mapping from the
// corresponding copy of 'n' and the edge number in 'g' to 'n'.
Graph* GetConstantGraph(const Graph* orig_graph,
                        const std::vector<Node*>& nodes,
                        std::map<NodeAndOutput, Node*>* tensors_to_fetch) {
  Graph* constant_graph = new Graph(orig_graph->op_registry());
  std::unordered_map<Node*, Node*> node_map;
  std::set<Node*> already_added;
  already_added.insert(constant_graph->source_node());
  already_added.insert(constant_graph->sink_node());
  node_map[orig_graph->source_node()] = constant_graph->source_node();
  node_map[orig_graph->sink_node()] = constant_graph->sink_node();
  for (Node* n : nodes) {
    Node* added = constant_graph->CopyNode(n);
    node_map[n] = added;
    already_added.insert(added);
    for (const Edge* in_edge : n->in_edges()) {
      Node* in = in_edge->src();
      CHECK_GT(node_map.count(in), size_t{0}) << n->DebugString() << " <-"
                                              << in->DebugString();
      CHECK_GT(already_added.count(node_map[in]), size_t{0})
          << in->DebugString();
      constant_graph->AddEdge(node_map[in], in_edge->src_output(), added,
                              in_edge->dst_input());
    }
  }

  for (auto const& added_nodes : node_map) {
    for (const Edge* out_edge : added_nodes.first->out_edges()) {
      if (node_map.count(out_edge->dst()) == 0) {
        if (out_edge->IsControlEdge()) continue;
        tensors_to_fetch->insert(
            {{added_nodes.second, out_edge->src_output()}, added_nodes.first});
      }
    }
  }

  return constant_graph;
}

int64 UniqueConstantId() {
  static std::atomic_int_fast64_t id;
  return id.fetch_add(1);
}

Device* GetCPUDevice() {
  static Device* device = nullptr;
  if (!device) {
    std::vector<Device*> devices;
    DeviceFactory::GetFactory(DEVICE_CPU)
        ->CreateDevices(SessionOptions{}, "", &devices);
    if (devices.size() > 0) {
      device = devices[0];
    }
  }
  return device;
}

thread::ThreadPool* GetThreadPool() {
  static thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "Compute", 1);
  return thread_pool;
}

// A simple rendezvous class.
// Assumes a single sender and a single receiver, no duplicate sends, and no
// sends of dead tensors.
class SimpleRendezvous : public Rendezvous {
 public:
  explicit SimpleRendezvous() {}

  Status Send(const string& key, const Args& send_args, const Tensor& val,
              const bool is_dead) override {
    if (is_dead) {
      return errors::Internal("Send of a dead tensor");
    }
    ParsedKey parsed;
    TF_RETURN_IF_ERROR(ParseKey(key, &parsed));

    mutex_lock l(mu_);
    if (table_.count(parsed.edge_name) > 0) {
      return errors::Internal("Send of an already sent tensor");
    }
    table_[parsed.edge_name] = val;
    return Status::OK();
  }

  void RecvAsync(const string& key, const Args& recv_args,
                 DoneCallback done) override {
    Tensor tensor;
    Status status = Status::OK();
    {
      mutex_lock l(mu_);
      if (table_.count(key) <= 0) {
        status = errors::Internal("Did not find key ", key);
      } else {
        tensor = table_[key];
      }
    }
    done(status, Args{}, recv_args, tensor, false);
  }

  void StartAbort(const Status& status) override {}

 private:
  typedef std::unordered_map<string, Tensor> Table;

  mutex mu_;
  Table table_ GUARDED_BY(mu_);
};

}  // namespace

bool ReplaceTensorWithConstant(Graph* graph, Device* partition_device,
                               NodeAndOutput tensor, const Tensor& constant) {
  // Be conservative when replacing a tensor with a constant, when not
  // running on CPU.
  // 1) If the destination tensor is not an int32 tensor, and has HOST_MEMORY
  // constraint, do not replace it.
  // 2) If the destination tensor is an int32 tensor, but has DEVICE_MEMORY
  // constraint, do not replace it.
  // 3) If the constant op created does not have a kernel implementation
  // for the device, do not use it.
  // 4) If the size of the constant in bytes is too large (> 10M), do not
  // replace it. This prevents the size of the Graph from growing too large.
  // TODO(keveman): Consider adding a new constant op that has a kernel
  // implementation for all types, but with HostMemory constraint on it's
  // output.
  // 5) Do not replace another constant.
  if (tensor.first->IsConstant()) {
    return false;
  }
  DeviceType device_type = partition_device
                               ? DeviceType{partition_device->device_type()}
                               : DEVICE_CPU;
  if (partition_device && device_type != DEVICE_CPU) {
    MemoryType memory_type;
    if (!MemoryTypeForOutput(device_type, graph, tensor.first, tensor.second,
                             &memory_type)
             .ok()) {
      return false;
    }
    bool is_int32 = tensor.first->output_type(tensor.second) == DT_INT32;
    if ((memory_type == HOST_MEMORY && !is_int32) ||
        (memory_type == DEVICE_MEMORY && is_int32)) {
      return false;
    }
  }
  if (constant.TotalBytes() > 10 * 1024 * 1024) {
    return false;
  }

  Node* n = tensor.first;
  std::vector<const Edge*> edges_to_remove;
  for (const Edge* out_edge : n->out_edges()) {
    if (out_edge->src_output() == tensor.second) {
      edges_to_remove.push_back(out_edge);
    }
  }
  string node_name = n->name();
  Node* constant_node;
  auto builder = NodeDefBuilder(strings::StrCat(graph->NewName(node_name),
                                                "__cf__", UniqueConstantId()),
                                "Const")
                     .Attr("dtype", constant.dtype())
                     .Attr("value", constant);
  NodeDef def;
  if (!builder.Finalize(&def).ok()) {
    return false;
  }
  const KernelDef* kdef;
  if (!FindKernelDef(device_type, def, &kdef, nullptr).ok()) {
    return false;
  }

  VLOG(1) << "Replacing " << tensor.first->DebugString()
          << " :: " << tensor.second << " with a constant";

  if (!NodeBuilder(builder).Finalize(graph, &constant_node).ok()) {
    return false;
  }
  for (auto edge : edges_to_remove) {
    graph->AddEdge(constant_node, 0, edge->dst(), edge->dst_input());
    graph->RemoveEdge(edge);
  }
  graph->AddEdge(graph->source_node(), -1, constant_node, -1);
  return true;
}

bool DoConstantFolding(const ConstantFoldingOptions& opts,
                       Device* partition_device, Graph* graph) {
  DumpGraph("Before", graph);
  Device* device = GetCPUDevice();
  thread::ThreadPool* thread_pool = GetThreadPool();
  if (!device || !thread_pool) {
    VLOG(1) << "Cannot find a device and/or a thread pool to do constant "
               "folding on";
    return false;
  }

  std::vector<Node*> constant_foldable_nodes;
  FindConstantFoldableNodes(graph, opts, &constant_foldable_nodes);
  if (constant_foldable_nodes.empty()) {
    VLOG(1) << "No constant foldable nodes found";
    return false;
  }

  std::map<NodeAndOutput, Node*> tensors_to_fetch;
  Graph* constant_graph =
      GetConstantGraph(graph, constant_foldable_nodes, &tensors_to_fetch);
  DumpGraph("Constant graph", constant_graph);

  if (tensors_to_fetch.empty()) {
    VLOG(1) << "No constant nodes found that feed into the original graph.";
    delete constant_graph;
    return false;
  }
  VLOG(1) << "Constant foldable " << constant_graph->num_node_ids() << " : "
          << graph->num_node_ids();

  // Create a local executor and evaluate the constant foldable nodes.
  subgraph::NameIndex name_index;
  for (Node* n : constant_graph->nodes()) {
    name_index[n->name()] = n;
  }

  std::vector<Node*> fetch_nodes;
  std::vector<string> tensors_to_fetch_names;
  std::vector<NodeAndOutput> tensors_to_replace;
  for (auto n : tensors_to_fetch) {
    tensors_to_fetch_names.push_back(
        strings::StrCat(n.first.first->name(), ":", n.first.second));
    tensors_to_replace.push_back({n.second, n.first.second});
  }
  // For nodes that need to be fetched back from the constant_graph, attach Send
  // nodes.
  Status s =
      subgraph::FetchOutputs(constant_graph, device->attributes(),
                             tensors_to_fetch_names, &name_index, &fetch_nodes);
  if (!s.ok()) {
    delete constant_graph;
    VLOG(1) << "Could not fetch constants: " << s;
    return false;
  }

  CHECK_EQ(fetch_nodes.size(), tensors_to_fetch.size());

  // Create the local executor and the Rendezvous for fetching back the
  // constants.
  auto runner = [thread_pool](Executor::Args::Closure c) {
    thread_pool->Schedule(c);
  };
  LocalExecutorParams params;
  params.device = device;
  params.create_kernel = [device, constant_graph](const NodeDef& ndef,
                                                  OpKernel** kernel) {
    return CreateNonCachedKernel(device, nullptr, ndef,
                                 constant_graph->versions().producer(), kernel);
  };
  params.delete_kernel = [](OpKernel* kernel) { delete kernel; };
  Executor* executor;
  if (!NewLocalExecutor(params, constant_graph, &executor).ok()) {
    return false;
  }

  std::unique_ptr<Executor> executor_unref(executor);

  SimpleRendezvous* rendez = new SimpleRendezvous;
  core::ScopedUnref rendez_unref(rendez);

  Executor::Args args;
  args.step_id = LogMemory::CONSTANT_FOLDING_STEP_ID;
  args.runner = runner;
  args.rendezvous = rendez;

  // Run the constant_graph.
  Notification executor_done;
  Status executor_done_status;
  ExecutorBarrier* barrier = new ExecutorBarrier(
      1, rendez, [&executor_done, &executor_done_status](const Status& ret) {
        executor_done_status = ret;
        executor_done.Notify();
      });

  executor->RunAsync(args, barrier->Get());

  executor_done.WaitForNotification();

  if (!executor_done_status.ok()) {
    return false;
  }

  // Fetch the constant tensors and replace the corresponding tensors in the
  // original graph with those constants.
  int32 num_nodes_replaced = 0;
  for (size_t c = 0; c < fetch_nodes.size(); ++c) {
    Tensor output;
    bool is_dead;
    string tensor_name;
    if (!GetNodeAttr(fetch_nodes[c]->def(), "tensor_name", &tensor_name).ok()) {
      // We successfully replaced some nodes previously, but had a problem with
      // this node. Don't bother processing the rest of the nodes.
      return c > 0;
    }
    Status s = rendez->Recv(tensor_name, Rendezvous::Args(), &output, &is_dead);
    if (!s.ok() || is_dead) {
      return c > 0;
    }
    if (ReplaceTensorWithConstant(graph, partition_device,
                                  tensors_to_replace[c], output)) {
      ++num_nodes_replaced;
    }
  }

  DumpGraph("After", graph);

  return num_nodes_replaced > 0;
}

}  // namespace tensorflow
