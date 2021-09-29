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
#include "tensorflow/core/grappler/utils/transitive_fanin.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace grappler {
const char kAutoParallelPrefix[] = "AutoParallel";

NodeDef* AutoParallel::AddNodeDivConst() {
  NodeDef* node = graph_.add_node();
  node->set_name(strings::StrCat(kAutoParallelPrefix, "-Div-Const"));
  node->set_op("Const");

  AttrValue attr_data_type;
  attr_data_type.set_type(DT_FLOAT);
  node->mutable_attr()->insert({"dtype", attr_data_type});

  AttrValue attr_tensor;
  auto tensor = attr_tensor.mutable_tensor();
  tensor->add_float_val(static_cast<float>(num_replicas_));
  tensor->set_dtype(DT_FLOAT);
  node->mutable_attr()->insert({"value", attr_tensor});
  return node;
}

NodeDef* AutoParallel::AddNodeDiv(const string& name, const string& input_a,
                                  const string& input_b) {
  NodeDef* node = graph_.add_node();
  node->set_name(strings::StrCat(kAutoParallelPrefix, "-Div-", name));
  node->set_op("RealDiv");
  node->add_input(input_a);
  node->add_input(input_b);
  AttrValue attr_type;
  attr_type.set_type(DT_FLOAT);
  node->mutable_attr()->insert({"T", attr_type});
  return node;
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

  const std::set<string> apply_gradients_ops = {"ApplyGradientDescent",
                                                "ApplyProximalGradientDescent",
                                                "ApplyAdadelta",
                                                "ApplyAdagrad",
                                                "ApplyProximalAdagrad",
                                                "ApplyAdagradDA",
                                                "ApplyFtrl",
                                                "ApplyMomentum",
                                                "ApplyAdam",
                                                "ApplyRMSProp",
                                                "ApplyCenteredRMSProp"};
  for (int i = 0; i < graph_.node_size(); i++) {
    all_nodes_.insert(
        std::make_pair(graph_.node(i).name(), graph_.mutable_node(i)));
    if (apply_gradients_ops.find(graph_.node(i).op()) !=
        apply_gradients_ops.end()) {
      apply_gradients_nodes_.insert(graph_.node(i).name());
      VLOG(2) << "Apply gradients node: " << graph_.node(i).name();
    }
  }

  auto div_const_node = AddNodeDivConst();
  all_nodes_.insert(std::make_pair(div_const_node->name(), div_const_node));
  std::map<string, int> gradient_pos = {{"ApplyGradientDescent", 2},
                                        {"ApplyProximalGradientDescent", 4},
                                        {"ApplyAdadelta", 6},
                                        {"ApplyAdagrad", 3},
                                        {"ApplyProximalAdagrad", 5},
                                        {"ApplyAdagradDA", 3},
                                        {"ApplyFtrl", 3},
                                        {"ApplyMomentum", 3},
                                        {"ApplyAdam", 9},
                                        {"ApplyRMSProp", 7},
                                        {"ApplyCenteredRMSProp", 8}};
  for (const auto& apply_gradient_node_name : apply_gradients_nodes_) {
    auto apply_gradients_op = all_nodes_[apply_gradient_node_name]->op();
    auto apply_gradients_node = all_nodes_[apply_gradient_node_name];

    auto div_node = AddNodeDiv(
        apply_gradient_node_name,
        apply_gradients_node->input(gradient_pos[apply_gradients_op]),
        div_const_node->name());
    all_nodes_.insert(std::make_pair(div_node->name(), div_node));
    *apply_gradients_node->mutable_input(gradient_pos[apply_gradients_op]) =
        div_node->name();
  }
  LOG(INFO) << "Graph size after adding div nodes: " << all_nodes_.size();

  std::vector<const NodeDef*> train_nodes;
  TF_RETURN_IF_ERROR(ComputeTransitiveFanin(graph_, item.fetch, &train_nodes));
  LOG(INFO) << "Number of training nodes: " << train_nodes.size();

  const NodeDef* dequeue_node = nullptr;
  for (const auto& train_node : train_nodes) {
    if (IsDequeueOp(*train_node)) {
      dequeue_node = train_node;
      break;
    }
  }

  std::vector<const NodeDef*> input_nodes;
  if (dequeue_node) {
    LOG(INFO) << "Dequeue node: " << dequeue_node->name();
    TF_RETURN_IF_ERROR(ComputeTransitiveFanin(graph_, {dequeue_node->name()},
                                              {}, &input_nodes));
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

void AutoParallel::BuildGraph(GraphDef* graph) {
  AddSharedNodes(graph);
  for (int i = 0; i < num_replicas_; i++) {
    AddOneReplica(graph, i);
  }
  std::set<string> fetches;
  for (size_t i = 0; i < item_->fetch.size(); i++) {
    for (int j = 0; j < num_replicas_; j++) {
      string prefix = strings::StrCat(kAutoParallelPrefix, "-Replica-", j);
      string fetch = AddPrefixToNodeName(item_->fetch[i], prefix);
      fetches.insert(fetch);
    }
  }
  string name_control =
      strings::StrCat(kAutoParallelPrefix, "-Control-", "Fetch");
  auto control = AddNodeControl(name_control, fetches, graph);

  for (const auto& fetch : item_->fetch) {
    AddNodeControl(fetch, {control->name()}, graph);
  }
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

}  // end namespace grappler
}  // end namespace tensorflow
