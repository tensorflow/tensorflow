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

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_partition.h"

namespace tensorflow {

Status PartitionFunctionGraph(
    const DeviceSet& device_set, std::unique_ptr<Graph> graph,
    std::unordered_map<string, std::unique_ptr<Graph>>* subgraphs) {
  PartitionOptions partition_options;
  partition_options.node_to_loc = [](const Node* node) {
    // TODO(iga): To support the distributed case, first split the graph by
    // worker (e.g,. using the master session's `SplitByWorker` policy), and
    // then recursively partition the per-worker shards at the remote worker(s).
    // Currently, we simply split the graph at device boundaries.
    return node->assigned_device_name();
  };
  int64 edge_name_counter = 0;
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
  std::unordered_map<string, GraphDef> partitions;
  TF_RETURN_IF_ERROR(Partition(partition_options, graph.get(), &partitions));

  for (const auto& partition : partitions) {
    const string& device = partition.first;
    const GraphDef& graph_def = partition.second;
    // Each partition gets a copy of all the
    // std::unique_ptr<Graph> subgraph(new Graph(graph->flib_def()));
    std::unique_ptr<Graph> subgraph(
        new Graph(graph->flib_def().ReachableDefinitions(graph_def)));
    FunctionLibraryDefinition global_flib(OpRegistry::Global(), {});
    TF_CHECK_OK(subgraph->AddFunctionLibrary(global_flib.ToProto()));
    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    opts.expect_device_spec = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, graph_def, subgraph.get()));
    subgraphs->emplace(device, std::move(subgraph));
  }

  return Status::OK();
}

Status UpdateArgAndRetvalMetadata(
    Graph* subgraph, std::vector<int>* arg_indices,
    std::vector<int>* ret_indices,
    std::vector<AllocatorAttributes>* arg_alloc_attrs,
    std::vector<AllocatorAttributes>* ret_alloc_attrs) {
  std::vector<std::pair<Node*, int>> arg_nodes;
  std::vector<std::pair<Node*, int>> ret_nodes;
  const AttrValue* attr_value;

  // Find the Arg and Retval nodes, along with their corresponding indices
  // in the original function.
  for (Node* node : subgraph->op_nodes()) {
    string node_type = node->type_string();
    if (node_type == FunctionLibraryDefinition::kArgOp) {
      TF_RETURN_IF_ERROR(node->attrs().Find("index", &attr_value));
      int index = static_cast<int>(attr_value->i());
      arg_indices->push_back(index);
      arg_nodes.push_back(std::make_pair(node, index));
    } else if (node_type == FunctionLibraryDefinition::kRetOp) {
      TF_RETURN_IF_ERROR(node->attrs().Find("index", &attr_value));
      int index = static_cast<int>(attr_value->i());
      ret_indices->push_back(index);
      ret_nodes.push_back(std::make_pair(node, index));
    }
  }

  for (int i = 0; i < arg_nodes.size(); ++i) {
    Node* arg = arg_nodes[i].first;
    arg->AddAttr("index", i);
    TF_RETURN_IF_ERROR(arg->attrs().Find("T", &attr_value));
    AllocatorAttributes alloc_attr;
    DataType type = attr_value->type();
    if (MTypeFromDType(type) == HOST_MEMORY) {
      alloc_attr.set_on_host(true);
    }
    arg_alloc_attrs->push_back(alloc_attr);
  }
  for (int i = 0; i < ret_nodes.size(); ++i) {
    Node* ret = ret_nodes[i].first;
    ret->AddAttr("index", i);
    TF_RETURN_IF_ERROR(ret->attrs().Find("T", &attr_value));
    AllocatorAttributes alloc_attr;
    DataType type = attr_value->type();
    if (MTypeFromDType(type) == HOST_MEMORY) {
      alloc_attr.set_on_host(true);
    }
    ret_alloc_attrs->push_back(alloc_attr);
  }

  return Status::OK();
}

std::vector<Tensor> GetArgsForIndices(const std::vector<int>& indices,
                                      gtl::ArraySlice<Tensor> arguments) {
  std::vector<Tensor> args;
  args.reserve(indices.size());
  for (int i : indices) {
    args.push_back(arguments[i]);
  }
  return args;
}

string FunctionNameGenerator::GetName() {
  for (;; ++counter_) {
    const string candidate = strings::StrCat(name_, "_", counter_);
    if (flib_def_->Find(candidate) == nullptr) {
      return candidate;
    }
  }
}

}  // namespace tensorflow
