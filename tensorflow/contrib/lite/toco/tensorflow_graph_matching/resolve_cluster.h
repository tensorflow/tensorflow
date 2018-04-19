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
#ifndef TENSORFLOW_CONTRIB_LITE_TOCO_TENSORFLOW_GRAPH_MATCHING_RESOLVE_CLUSTER_H
#define TENSORFLOW_CONTRIB_LITE_TOCO_TENSORFLOW_GRAPH_MATCHING_RESOLVE_CLUSTER_H

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/lite/toco/tensorflow_graph_matching/cluster.h"
#include "tensorflow/contrib/lite/toco/tensorflow_graph_matching/resolve_svdf.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"

namespace toco {

// Given a graph info and a list of cluster classes (cluster_factories), it
// partitions the graph to clusters, and then collapses each cluster into their
// corresponding composite ops. It generates a new graph using the newly
// generated composite ops. Each cluster factory is responsible to recognize a
// cluster of nodes into a cluster using a name-based pattern matching approach.
std::unique_ptr<tensorflow::GraphDef> MaybeResolveClusters(
    const tensorflow::GraphDef& graph_def,
    const std::vector<ClusterFactoryInterface*>& cluster_factories);

// Adds a node to a given graph. The added node will be a copy of a given source
// node, except for the inputs. If the inputs are coming from a node which
// belongs to another cluster, then those inputs are renamed to the source
// cluster name.
void AddNodeToGraph(const tensorflow::NodeDef& node,
                    const std::vector<string>& cluster_names,
                    tensorflow::GraphDef* graph);

// Given a graph and a cluster class, it finds all the nodes which belong to a
// given class factory, encapsulate them inside a cluster of the given type and
// returns a vector of those clusters. It also labels the nodes in that graph if
// they belong to the generated clusters.
bool FindCluster(const ClusterFactoryInterface& cluster_factory,
                 const tensorflow::GraphDef& graph_def,
                 std::unordered_map<string, bool>* is_node_in_cluster,
                 std::vector<std::unique_ptr<Cluster>>* clusters);

// Receives a graph and generates another graph by replacing the cluster of
// nodes which matches a given composite op. Each composite op is represented
// using a class factory.
std::unique_ptr<tensorflow::GraphDef> MaybeReplaceCompositeSubgraph(
    const tensorflow::GraphDef& tf_graph);

}  // end namespace toco

#endif  // CONTRIB_LITE_TOCO_TENSORFLOW_GRAPH_MATCHING_RESOLVE_CLUSTER_H
