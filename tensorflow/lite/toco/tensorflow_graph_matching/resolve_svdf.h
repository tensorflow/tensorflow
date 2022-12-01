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
#ifndef TENSORFLOW_LITE_TOCO_TENSORFLOW_GRAPH_MATCHING_RESOLVE_SVDF_H_
#define TENSORFLOW_LITE_TOCO_TENSORFLOW_GRAPH_MATCHING_RESOLVE_SVDF_H_

#include <string>
#include <vector>

#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tensorflow_graph_matching/cluster.h"
#include "tensorflow/lite/toco/tensorflow_graph_matching/cluster_utils.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"

namespace toco {

class SvdfCluster : public Cluster {
 public:
  // For this cluster, it collapses all the nodes in nodes_ into a composite op
  // and it returns all the newly generated ops in new_nodes_.
  void CreateNodes() override;

  // A helper function to set the pattern of Const nodes which CreateNodes()
  // should handle specially.
  void AddConstNodePattern(const std::string& const_pattern) {
    const_node_patterns_.push_back(const_pattern);
  }

  ~SvdfCluster() override {}

 private:
  // The main function which is used to create Const nodes for this cluster.
  // These Const nodes are the inputs to the composite op generated for this
  // cluster.
  void CreateConstNode(const std::string& const_pattern);

  // Receives a vector of Const nodes, merge them (if necessary) and returns
  // only one Const node holding all the arrays contents. It transposes it if
  // needed.
  void MaybeMergeConstNodes(
      const std::vector<const tensorflow::NodeDef*>& const_node_parts,
      bool transpose_tensor_value,
      const std::unique_ptr<tensorflow::NodeDef>& merged_node);

  // Infer the value of Svdf filter rank, by looking up a reshape operator which
  // is used for 'output' which reshapes output from [num_filters, batch, 1]
  // shape to [num_units, rank, batch] shape. The 2nd shape element is rank.
  int InferFilterRank();

  std::vector<std::string> const_node_patterns_;
};

class SvdfClusterFactory : public ClusterFactoryInterface {
 public:
  // Creates a cluster of nodes using a name-based pattern matching approach. It
  // uses a node as a seed and if its name matches a certain pattern, then it
  // builds the cluster around that node.
  // This factory expects nodes which have "SVDF_weights_feature" and
  // "SVDF_weights_time" pattern in their names (and optionally "SVDF_bias")
  // and it creates an SVDF Op from them.
  std::unique_ptr<Cluster> CreateCluster(
      const tensorflow::NodeDef& node,
      const tensorflow::GraphDef& graph_def) const override;
};

}  // end namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TENSORFLOW_GRAPH_MATCHING_RESOLVE_SVDF_H_
