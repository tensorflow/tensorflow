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
#include "tensorflow/core/common_runtime/partitioning_utils.h"

#include <algorithm>
#include <utility>

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_partition.h"

namespace tensorflow {

Status PartitionFunctionGraph(
    const DeviceSet& device_set, std::unique_ptr<Graph> graph,
    std::unordered_map<string, std::unique_ptr<Graph>>* subgraphs,
    std::function<string(const Edge*)> get_tensor_name_attr) {
  PartitionOptions partition_options;
  partition_options.node_to_loc = [](const Node* node) {
    // TODO(iga): To support the distributed case, first split the graph by
    // worker (e.g,. using the master session's `SplitByWorker` policy), and
    // then recursively partition the per-worker shards at the remote worker(s).
    // Currently, we simply split the graph at device boundaries.
    return node->assigned_device_name();
  };
  int64_t edge_name_counter = 0;
  partition_options.new_name = [&edge_name_counter](const string& prefix) {
    return strings::StrCat(prefix, "/_", ++edge_name_counter);
  };
  partition_options.get_incarnation =
      [&device_set](const string& name) -> int64 {
    const Device* d = device_set.FindDeviceByName(name);
    if (d == nullptr) {
      return PartitionOptions::kIllegalIncarnation;
    } else {
      return d->attributes().incarnation();
    }
  };
  partition_options.control_flow_added = false;
  partition_options.get_tensor_name_attr = get_tensor_name_attr;
  std::unordered_map<string, GraphDef> partitions;
  TF_RETURN_IF_ERROR(Partition(partition_options, graph.get(), &partitions));

  for (auto& partition : partitions) {
    const string& device = partition.first;
    GraphDef& graph_def = partition.second;
    // Each partition gets a new graph.
    std::unique_ptr<Graph> subgraph(
        new Graph(graph->flib_def().default_registry()));
    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    opts.expect_device_spec = true;
    TF_RETURN_IF_ERROR(
        ConvertGraphDefToGraph(opts, std::move(graph_def), subgraph.get()));
    subgraphs->emplace(device, std::move(subgraph));
  }

  return Status::OK();
}

Status UpdateArgAndRetvalMetadata(
    Graph* graph, std::vector<FunctionArgIndex>* arg_indices,
    std::vector<int>* ret_indices,
    std::vector<AllocatorAttributes>* arg_alloc_attrs,
    std::vector<AllocatorAttributes>* ret_alloc_attrs, bool ints_on_device) {
  std::vector<std::pair<Node*, FunctionArgIndex>> arg_nodes;
  std::vector<std::pair<Node*, int>> ret_nodes;
  const AttrValue* attr_value;

  // Find the Arg and Retval nodes, along with their corresponding indices
  // in the original function.
  for (Node* node : graph->op_nodes()) {
    if (node->IsArg()) {
      TF_RETURN_IF_ERROR(node->attrs().Find("index", &attr_value));
      int index = static_cast<int>(attr_value->i());
      int sub_index = -1;
      if (node->attrs().Find("sub_index", &attr_value).ok()) {
        sub_index = static_cast<int>(attr_value->i());
      }
      arg_nodes.emplace_back(node, FunctionArgIndex(index, sub_index));
    } else if (node->IsRetval()) {
      TF_RETURN_IF_ERROR(node->attrs().Find("index", &attr_value));
      int index = static_cast<int>(attr_value->i());
      ret_nodes.emplace_back(node, index);
    }
  }

  // Sort the nodes by index so that the order is stable.
  //
  // In particular, this enables calling a single-partition function with
  // the same signature as the original unpartitioned function.
  auto arg_comparator = [](std::pair<Node*, FunctionArgIndex> a,
                           std::pair<Node*, FunctionArgIndex> b) {
    return std::tie(a.second.index, a.second.sub_index) <
           std::tie(b.second.index, b.second.sub_index);
  };
  std::sort(arg_nodes.begin(), arg_nodes.end(), arg_comparator);
  auto ret_comparator = [](std::pair<Node*, int> a, std::pair<Node*, int> b) {
    return a.second < b.second;
  };
  std::sort(ret_nodes.begin(), ret_nodes.end(), ret_comparator);

  arg_indices->reserve(arg_nodes.size());
  for (const auto& pair : arg_nodes) arg_indices->push_back(pair.second);
  ret_indices->reserve(ret_nodes.size());
  for (const auto& pair : ret_nodes) ret_indices->push_back(pair.second);

  for (int i = 0; i < arg_nodes.size(); ++i) {
    Node* arg = arg_nodes[i].first;
    arg->AddAttr("index", i);
    TF_RETURN_IF_ERROR(arg->attrs().Find("T", &attr_value));
    if (arg_alloc_attrs != nullptr) {
      AllocatorAttributes alloc_attr;
      DataType type = attr_value->type();
      MemoryType mtype = ints_on_device ? MTypeFromDTypeIntsOnDevice(type)
                                        : MTypeFromDType(type);
      if (mtype == HOST_MEMORY) {
        alloc_attr.set_on_host(true);
      }
      arg_alloc_attrs->push_back(alloc_attr);
    }
  }
  for (int i = 0; i < ret_nodes.size(); ++i) {
    Node* ret = ret_nodes[i].first;
    ret->AddAttr("index", i);
    TF_RETURN_IF_ERROR(ret->attrs().Find("T", &attr_value));
    if (ret_alloc_attrs) {
      AllocatorAttributes alloc_attr;
      DataType type = attr_value->type();
      MemoryType mtype = ints_on_device ? MTypeFromDTypeIntsOnDevice(type)
                                        : MTypeFromDType(type);
      if (mtype == HOST_MEMORY) {
        alloc_attr.set_on_host(true);
      }
      ret_alloc_attrs->push_back(alloc_attr);
    }
  }

  return Status::OK();
}

string FunctionNameGenerator::GetName() {
  while (true) {
    const string candidate = strings::StrCat(name_, "_", counter_++);
    if (flib_def_->Find(candidate) == nullptr) {
      return candidate;
    }
  }
}

}  // namespace tensorflow
