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
#include "tensorflow/lite/toco/tensorflow_graph_matching/resolve_cluster.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/lite/toco/tensorflow_graph_matching/cluster.h"
#include "tensorflow/lite/toco/tensorflow_graph_matching/cluster_utils.h"
#include "tensorflow/lite/toco/tensorflow_graph_matching/resolve_svdf.h"

namespace toco {

using tensorflow::GraphDef;
using tensorflow::NodeDef;

void AddNodeToGraph(const NodeDef& node,
                    const std::vector<std::string>& cluster_names,
                    GraphDef* graph) {
  NodeDef* new_node = graph->add_node();
  new_node->set_op(node.op());
  new_node->set_name(node.name());
  new_node->set_device(node.device());
  // If the inputs are coming from a node which belongs to another cluster, then
  // those inputs are renamed to the source cluster name. Otherwise the original
  // input name is used.
  for (const std::string& node_input : node.input()) {
    bool input_from_cluster = false;
    for (const std::string& cluster_name : cluster_names) {
      if (StrContains(node_input, cluster_name) &&
          !StrContains(node.name(), cluster_name)) {
        new_node->add_input(cluster_name);
        input_from_cluster = true;
        break;
      }
    }
    if (!input_from_cluster) {
      new_node->add_input(node_input);
    }
  }
  for (const auto& attr : node.attr()) {
    (*new_node->mutable_attr())[attr.first] = attr.second;
  }
}

bool FindCluster(const ClusterFactoryInterface& cluster_factory,
                 const GraphDef& graph_def,
                 std::unordered_map<std::string, bool>* is_node_in_cluster,
                 std::vector<std::unique_ptr<Cluster>>* clusters) {
  for (const NodeDef& node : graph_def.node()) {
    // If the node is not assigned to any cluster, then we check if it belong to
    // the cluster_factory.
    bool node_in_cluster = (*is_node_in_cluster)[node.name()];
    if (!node_in_cluster) {
      std::unique_ptr<Cluster> cluster =
          cluster_factory.CreateCluster(node, graph_def);
      if (cluster) {
        // Label all the nodes in is_node_in_cluster which are in this cluster
        // as belonged to this cluster.
        for (const NodeDef* cluster_node : cluster->GetNodes()) {
          (*is_node_in_cluster)[cluster_node->name()] = true;
        }
        clusters->push_back(std::move(cluster));
      }
    }
  }
  return (!clusters->empty());
}

std::unique_ptr<GraphDef> MaybeResolveClusters(
    const GraphDef& graph_def,
    const std::vector<ClusterFactoryInterface*>& cluster_factories) {
  std::unique_ptr<GraphDef> pruned_graph(new GraphDef);
  // The structure to keep track of which cluster each node is assigned to, and
  // to initialize them to all un-assigned,
  std::unordered_map<std::string, bool> is_node_in_cluster;
  for (const NodeDef& node : graph_def.node()) {
    is_node_in_cluster[node.name()] = false;
  }

  std::vector<std::string> cluster_names;
  std::vector<std::unique_ptr<Cluster>> all_clusters;
  // Find the clusters for all available cluster factories.
  for (const ClusterFactoryInterface* cluster_factory : cluster_factories) {
    std::vector<std::unique_ptr<Cluster>> clusters;
    if (FindCluster(*cluster_factory, graph_def, &is_node_in_cluster,
                    &clusters)) {
      for (auto itr = clusters.begin(); itr != clusters.end(); ++itr) {
        cluster_names.push_back((*itr)->GetName());
        (*itr)->CreateNodes();
        all_clusters.push_back(std::move(*itr));
      }
    }
  }

  for (const std::unique_ptr<Cluster>& cluster : all_clusters) {
    for (const std::unique_ptr<tensorflow::NodeDef>& src_node :
         cluster->GetNewNodes()) {
      // Add it to the output GraphDef.
      AddNodeToGraph(*src_node, cluster_names, pruned_graph.get());
    }
  }

  // Add any node which is not part of a cluster.
  for (const NodeDef& node : graph_def.node()) {
    bool node_in_cluster = is_node_in_cluster[node.name()];
    if (!node_in_cluster) {
      AddNodeToGraph(node, cluster_names, pruned_graph.get());
    }
  }

  if (pruned_graph->node_size() == 0) {
    return nullptr;
  } else {
    return pruned_graph;
  }
}

std::unique_ptr<GraphDef> MaybeReplaceCompositeSubgraph(
    const GraphDef& tf_graph) {
  SvdfClusterFactory svdf_cluster_factory;

  std::vector<ClusterFactoryInterface*> cluster_factories;
  cluster_factories.push_back(&svdf_cluster_factory);

  std::unique_ptr<GraphDef> pruned_graph =
      MaybeResolveClusters(tf_graph, cluster_factories);

  // Copy function definitions
  if (pruned_graph) {
    *(pruned_graph->mutable_library()) = tf_graph.library();
  }
  return pruned_graph;
}

}  // end namespace toco
