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

#include "tensorflow/core/grappler/optimizers/auto_parallel.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace grappler {
const char kAutoParallelPrefix[] = "AutoParallel";

NodeDef* AutoParallel::AddNodeNumReplicasConst(bool is_float, GraphDef* graph) {
  NodeDef* node = graph->add_node();
  node->set_op("Const");

  AttrValue attr_data_type;
  if (is_float) {
    attr_data_type.set_type(DT_FLOAT);
    node->set_name(
        strings::StrCat(kAutoParallelPrefix, "-NumReplicas-Float-Const"));
  } else {
    attr_data_type.set_type(DT_INT32);
    node->set_name(
        strings::StrCat(kAutoParallelPrefix, "-NumReplicas-Int-Const"));
  }
  node->mutable_attr()->insert({"dtype", attr_data_type});

  AttrValue attr_tensor;
  auto tensor = attr_tensor.mutable_tensor();
  if (is_float) {
    tensor->add_float_val(static_cast<float>(num_replicas_));
    tensor->set_dtype(DT_FLOAT);
  } else {
    tensor->add_int_val(num_replicas_);
    tensor->set_dtype(DT_INT32);
  }
  node->mutable_attr()->insert({"value", attr_tensor});
  return node;
}

NodeDef* AutoParallel::AddNodeMaxLongConst(const string& name,
                                           GraphDef* graph) {
  NodeDef* node = graph->add_node();
  node->set_name(name);
  node->set_op("Const");

  AttrValue attr_data_type;
  attr_data_type.set_type(DT_INT64);
  node->mutable_attr()->insert({"dtype", attr_data_type});

  AttrValue attr_tensor;
  auto tensor = attr_tensor.mutable_tensor();
  tensor->add_int64_val(std::numeric_limits<long>::max());
  tensor->set_dtype(DT_INT64);
  node->mutable_attr()->insert({"value", attr_tensor});
  return node;
}

NodeDef* AutoParallel::AddNodeDiv(const string& name, const string& input_a,
                                  const string& input_b, GraphDef* graph) {
  NodeDef* node = graph->add_node();
  node->set_name(strings::StrCat(kAutoParallelPrefix, "-Div-", name));
  node->set_op("RealDiv");
  node->add_input(input_a);
  node->add_input(input_b);
  AttrValue attr_type;
  attr_type.set_type(DT_FLOAT);
  node->mutable_attr()->insert({"T", attr_type});
  return node;
}

NodeDef* AutoParallel::AddNodeAdd(const string& name,
                                  const std::set<string>& inps,
                                  GraphDef* graph) {
  NodeDef* add_node = graph->add_node();
  add_node->set_name(strings::StrCat(kAutoParallelPrefix, "-Add-", name));
  add_node->set_op("AddN");

  for (auto input_name : inps) {
    add_node->add_input(input_name);
  }

  AttrValue attr_type;
  attr_type.set_type(DT_FLOAT);
  add_node->mutable_attr()->insert({"T", attr_type});
  AttrValue attr_numbers;
  attr_numbers.set_i(inps.size());
  add_node->mutable_attr()->insert({"N", attr_numbers});
  return add_node;
}

NodeDef* AutoParallel::AddNodeSparseAccumulator(const string& name,
                                                GraphDef* graph) {
  NodeDef* accumulator = graph->add_node();
  accumulator->set_name(strings::StrCat(kAutoParallelPrefix, "-Accum-", name));
  accumulator->set_op("SparseConditionalAccumulator");
  AttrValue attr_type;
  attr_type.set_type(DT_FLOAT);
  accumulator->mutable_attr()->insert({"dtype", attr_type});
  return accumulator;
}

NodeDef* AutoParallel::AddNodeCast(const string& name, const string& input,
                                   const DataType& src_dtype,
                                   const DataType& dst_dtype, GraphDef* graph) {
  NodeDef* cast_node = graph->add_node();
  cast_node->set_name(name);
  cast_node->set_op("Cast");
  cast_node->add_input(input);
  AttrValue attr_src_type;
  attr_src_type.set_type(src_dtype);
  AttrValue attr_dst_type;
  attr_dst_type.set_type(dst_dtype);
  cast_node->mutable_attr()->insert({"SrcT", attr_src_type});
  cast_node->mutable_attr()->insert({"DstT", attr_dst_type});
  return cast_node;
}

NodeDef* AutoParallel::AddNodeSparseAccumApply(
    const string& name, const string& accumulator, const string& max_long,
    const string& indices, const string& values, GraphDef* graph) {
  NodeDef* sparse_accum_apply_node = graph->add_node();
  sparse_accum_apply_node->set_op("SparseAccumulatorApplyGradient");
  sparse_accum_apply_node->set_name(name);
  sparse_accum_apply_node->add_input(accumulator);
  sparse_accum_apply_node->add_input(max_long);
  sparse_accum_apply_node->add_input(indices);
  sparse_accum_apply_node->add_input(values);
  sparse_accum_apply_node->add_input(values);  // dummy shape node
  AttrValue attr_type;
  attr_type.set_type(DT_FLOAT);
  sparse_accum_apply_node->mutable_attr()->insert({"dtype", attr_type});
  AttrValue attr_bool;
  attr_bool.set_b(false);
  sparse_accum_apply_node->mutable_attr()->insert(
      {"has_known_shape", attr_bool});
  return sparse_accum_apply_node;
}

NodeDef* AutoParallel::AddNodeSparseAccumTakeGrad(const string& name,
                                                  const string& accumulator,
                                                  const string& num_replicas,
                                                  const string& control,
                                                  GraphDef* graph) {
  NodeDef* take_grad = graph->add_node();
  take_grad->set_name(strings::StrCat(kAutoParallelPrefix, "-TakeGrad-", name));
  take_grad->set_op("SparseAccumulatorTakeGradient");
  take_grad->add_input(accumulator);
  take_grad->add_input(num_replicas);
  take_grad->add_input(strings::StrCat("^", control));
  AttrValue attr_type;
  attr_type.set_type(DT_FLOAT);
  take_grad->mutable_attr()->insert({"dtype", attr_type});
  return take_grad;
}

NodeDef* AutoParallel::AddNodeControl(const string& name,
                                      const std::set<string>& deps,
                                      GraphDef* graph) {
  NodeDef* node = graph->add_node();
  node->set_name(name);
  node->set_op("NoOp");
  for (const auto& dep : deps) {
    node->add_input(strings::StrCat("^", dep));
  }
  return node;
}

Status AutoParallel::Initialize(const GrapplerItem& item) {
  num_gpus_ = GetNumAvailableGPUs();
  LOG(INFO) << "Number of GPUs: " << num_gpus_;
  item_ = &item;
  graph_ = item.graph;
  LOG(INFO) << "Original graph size: " << graph_.node_size();
  if (item.fetch.empty()) {
    return Status(error::INVALID_ARGUMENT, "No fetch nodes provided.");
  }

  if (item.MainVariables().empty()) {
    return Status(error::INVALID_ARGUMENT, "No variables provided.");
  }

  for (const auto& init : item.init_ops) {
    VLOG(1) << "Init node: " << init;
  }

  for (const auto& fetch : item.fetch) {
    VLOG(1) << "Fetch node: " << fetch;
  }

  for (const auto& var : item.MainVariables()) {
    VLOG(2) << "Variable: " << var->name();
  }

  for (int i = 0; i < graph_.node_size(); i++) {
    all_nodes_.insert(
        std::make_pair(graph_.node(i).name(), graph_.mutable_node(i)));
  }

  // Replicate operations that are necessary for computing gradients
  // of trainable variables.
  std::vector<std::string> gradient_nodes;
  for (const auto& grad_info : item.gradients_info) {
    auto target_tensor_info = grad_info.first;
    if (std::find(item.trainable_variables.begin(),
                  item.trainable_variables.end(),
                  NodeName(target_tensor_info.values_tensor_name())) !=
        item.trainable_variables.end()) {
      auto grad_tensor_info = grad_info.second;
      auto indices = grad_tensor_info.indices_tensor_name();
      auto values = grad_tensor_info.values_tensor_name();

      gradients_.insert(std::make_pair(indices, values));
      if (!indices.empty()) {
        gradient_nodes.push_back(NodeName(indices));
      }
      gradient_nodes.push_back(NodeName(values));
    }
  }

  auto train_nodes = ComputeTransitiveFanin(graph_, gradient_nodes);
  LOG(INFO) << "Number of training nodes: " << train_nodes.size();

  const NodeDef* dequeue_node;
  for (const auto& train_node : train_nodes) {
    if (IsDequeueOp(*train_node)) {
      dequeue_node = train_node;
      break;
    }
  }

  std::vector<const NodeDef*> input_nodes;
  if (dequeue_node) {
    LOG(INFO) << "Dequeue node: " << dequeue_node->name();
    input_nodes = ComputeTransitiveFanin(graph_, {dequeue_node->name()});
  }
  LOG(INFO) << "Number of input nodes: " << input_nodes.size();

  std::set<string> dont_replicate_nodes;
  for (const auto& variable : item.MainVariables()) {
    dont_replicate_nodes.insert(variable->name());
  }

  for (const auto& init : item.init_ops) {
    dont_replicate_nodes.insert(NodeName(init));
  }

  // Don't replicate all input nodes, except the dequeue node.
  for (const auto& input_node : input_nodes) {
    if (input_node->name() != dequeue_node->name()) {
      dont_replicate_nodes.insert(input_node->name());
    }
  }

  for (const auto& node : train_nodes) {
    if (dont_replicate_nodes.find(node->name()) == dont_replicate_nodes.end()) {
      replica_nodes_.insert(node->name());
    }
  }
  LOG(INFO) << "Number of replica nodes: " << replica_nodes_.size();

  for (const auto& node : all_nodes_) {
    if (replica_nodes_.find(node.first) == replica_nodes_.end()) {
      shared_nodes_.insert(node.first);
    }
  }
  LOG(INFO) << "Number of shared nodes: " << shared_nodes_.size();
  return Status::OK();
}

bool AutoParallel::NotSharedNode(const string& name) {
  return shared_nodes_.find(name) == shared_nodes_.end();
}

void AutoParallel::AddSharedNodes(GraphDef* graph) {
  string prefix = strings::StrCat(kAutoParallelPrefix, "-Replica-", 0);
  for (const auto& node : shared_nodes_) {
    auto new_node = graph->add_node();
    *new_node = *all_nodes_[node];
    for (int i = 0; i < new_node->input_size(); i++) {
      if (NotSharedNode(NodeName(new_node->input(i)))) {
        string new_name = AddPrefixToNodeName(new_node->input(i), prefix);
        *new_node->mutable_input(i) = new_name;
      }
    }
  }
}

void AutoParallel::AddOneReplica(GraphDef* graph, int number) {
  string prefix = strings::StrCat(kAutoParallelPrefix, "-Replica-", number);
  for (const auto& node : replica_nodes_) {
    auto new_node = graph->add_node();
    *new_node = *all_nodes_[node];
    if (NotSharedNode(new_node->name())) {
      new_node->set_name(AddPrefixToNodeName(new_node->name(), prefix));
      if (num_gpus_ > 0) {
        new_node->set_device(strings::StrCat("/gpu:", number % num_gpus_));
      }
      for (int i = 0; i < new_node->input_size(); i++) {
        if (NotSharedNode(NodeName(new_node->input(i)))) {
          string new_name = AddPrefixToNodeName(new_node->input(i), prefix);
          *new_node->mutable_input(i) = new_name;
        }
      }
    }
  }
}

void AutoParallel::AddDenseAggregatedGrad(GraphDef* graph,
                                          NodeDef* num_replicas,
                                          const std::string& grad_name,
                                          std::string* new_grad_name) {
  // make unique grad name
  const auto& unique_grad_name =
      str_util::StringReplace(grad_name, ":", "_", false);

  std::set<std::string> inputs;
  for (int i = 0; i < num_replicas_; i++) {
    auto prefix = strings::StrCat(kAutoParallelPrefix, "-Replica-", i);
    inputs.insert(AddPrefixToNodeName(grad_name, prefix));
  }

  auto add_node = AddNodeAdd(unique_grad_name, inputs, graph);

  // divide the aggregated grad by num_replicas
  auto div_node = AddNodeDiv(unique_grad_name, add_node->name(),
                             num_replicas->name(), graph);

  *new_grad_name = div_node->name();
}

void AutoParallel::AddSparseAggregatedGrad(GraphDef* graph,
                                           NodeDef* num_replicas_node,
                                           const std::string& indices_name,
                                           const std::string& values_name,
                                           std::string* new_indices_name,
                                           std::string* new_grad_name) {
  const auto& unique_indices_name =
      str_util::StringReplace(indices_name, ":", "_", false);
  const auto& unique_values_name =
      str_util::StringReplace(values_name, ":", "_", false);

  auto sparse_accum_node = AddNodeSparseAccumulator(unique_values_name, graph);

  int indices_index = 0;
  if (str_util::StrContains(indices_name, ":")) {
    auto str_splits = str_util::Split(indices_name, ":");
    strings::safe_strto32(str_splits.back(), &indices_index);
  }
  auto indices_node = all_nodes_[NodeName(indices_name)];
  const auto& indices_op_def = item_->op_def.at(indices_node->op());
  auto type_name = indices_op_def.output_arg(indices_index).type_attr();
  auto indices_dtype = indices_node->attr().at(type_name).type();

  // Create SparseAccumulatorApplyGradientOp per replica
  std::set<std::string> sparse_accum_apply_nodes;
  for (int i = 0; i < num_replicas_; i++) {
    auto prefix = strings::StrCat(kAutoParallelPrefix, "-Replica-", i);
    auto indices_replica_name = AddPrefixToNodeName(indices_name, prefix);
    if (indices_dtype != DT_INT64) {
      assert(indices_dtype == DT_INT32);
      auto indices_cast_node = AddNodeCast(
          AddPrefixToNodeName(strings::StrCat("Cast-", unique_indices_name),
                              prefix),
          indices_replica_name, DT_INT32, DT_INT64, graph);
      indices_replica_name = indices_cast_node->name();
    }

    auto values_replica_name = AddPrefixToNodeName(values_name, prefix);
    auto max_long_node = AddNodeMaxLongConst(
        AddPrefixToNodeName("MaxLong-Const", prefix), graph);
    auto sparse_accum_apply_node = AddNodeSparseAccumApply(
        AddPrefixToNodeName(strings::StrCat("AccumApply-", unique_values_name),
                            prefix),
        sparse_accum_node->name(), max_long_node->name(), indices_replica_name,
        values_replica_name, graph);
    sparse_accum_apply_nodes.insert(sparse_accum_apply_node->name());
  }

  auto control_node =
      AddNodeControl(strings::StrCat(kAutoParallelPrefix, "-AccumApplyControl-",
                                     unique_values_name),
                     sparse_accum_apply_nodes, graph);

  auto take_grad = AddNodeSparseAccumTakeGrad(
      unique_values_name, sparse_accum_node->name(), num_replicas_node->name(),
      control_node->name(), graph);

  *new_indices_name = strings::StrCat(take_grad->name(), ":0");
  *new_grad_name = strings::StrCat(take_grad->name(), ":1");

  if (indices_dtype != DT_INT64) {
    assert(indices_dtype == DT_INT32);
    auto indices_cast_node =
        AddNodeCast(strings::StrCat(take_grad->name(), "-Cast"),
                    *new_indices_name, DT_INT64, DT_INT32, graph);
    *new_indices_name = indices_cast_node->name();
  }
}

void AutoParallel::UpdateConsumers(
    const std::vector<std::pair<NodeDef*, int>>& grad_consumers,
    const std::string& new_grad_name) {
  for (const auto& consumer : grad_consumers) {
    auto consumer_node = consumer.first;
    int input_index = consumer.second;
    if (str_util::StartsWith(consumer_node->input(input_index), "^")) {
      consumer_node->set_input(input_index,
                               strings::StrCat("^", new_grad_name));
    } else {
      consumer_node->set_input(input_index, new_grad_name);
    }
  }
}

void AutoParallel::AddGradientAggregation(GraphDef* graph) {
  // Find gradient consumers.
  // Gradient name -> list of (consumer node, index of the gradient in the
  // conumser inputs).
  std::map<std::string, std::vector<std::pair<NodeDef*, int>>> grad_consumers;
  string prefix = strings::StrCat(kAutoParallelPrefix, "-Replica-");
  for (int i = 0; i < graph->node_size(); i++) {
    auto node = graph->mutable_node(i);
    for (int j = 0; j < node->input_size(); j++) {
      auto input = node->input(j);
      if (str_util::StartsWith(input, prefix)) {
        // Remove prefix(e.g'AutoParallel-Replica-15/')
        input = input.substr(input.find("/") + 1);
      }
      for (auto gradient : gradients_) {
        auto indices_name = gradient.first;
        auto values_name = gradient.second;
        if (!indices_name.empty() &&
            (!input.compare(indices_name) ||
             !input.compare(strings::StrCat("^", indices_name)))) {
          grad_consumers[indices_name].push_back(std::make_pair(node, j));
        }
        if (!input.compare(values_name) ||
            !input.compare(strings::StrCat("^", values_name))) {
          grad_consumers[values_name].push_back(std::make_pair(node, j));
        }
      }
    }
  }

  NodeDef* num_replicas_node_f = AddNodeNumReplicasConst(true, graph);
  NodeDef* num_replicas_node_i = NULL;
  for (const auto& gradient : gradients_) {
    const auto& indices_name = gradient.first;
    const auto& values_name = gradient.second;
    bool is_sparse_values = !indices_name.empty();
    std::string new_indices_name;
    std::string new_values_name;
    if (!is_sparse_values) {
      AddDenseAggregatedGrad(graph, num_replicas_node_f, values_name,
                             &new_values_name);
      UpdateConsumers(grad_consumers[values_name], new_values_name);
    } else {
      if (num_replicas_node_i == NULL) {
        num_replicas_node_i = AddNodeNumReplicasConst(false, graph);
      }
      AddSparseAggregatedGrad(graph, num_replicas_node_i, indices_name,
                              values_name, &new_indices_name, &new_values_name);
      UpdateConsumers(grad_consumers[indices_name], new_indices_name);
      UpdateConsumers(grad_consumers[values_name], new_values_name);
    }
  }
}

void AutoParallel::BuildGraph(GraphDef* graph) {
  AddSharedNodes(graph);
  for (int i = 0; i < num_replicas_; i++) {
    AddOneReplica(graph, i);
  }

  AddGradientAggregation(graph);

  *graph->mutable_library() = item_->graph.library();
  *graph->mutable_versions() = item_->graph.versions();
  LOG(INFO) << "Parallelized graph size: " << graph->node_size();
}

Status AutoParallel::Optimize(Cluster* cluster, const GrapplerItem& item,
                              GraphDef* output) {
  TF_RETURN_IF_ERROR(Initialize(item));
  BuildGraph(output);
  return Status::OK();
}

void AutoParallel::Feedback(Cluster* cluster, const GrapplerItem& item,
                            const GraphDef& optimize_output, double result) {
  // TODO(yaozhang): Add feedback.
}

}  // end namespace grappler
}  // end namespace tensorflow
