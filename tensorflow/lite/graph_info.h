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
#ifndef TENSORFLOW_LITE_GRAPH_INFO_H_
#define TENSORFLOW_LITE_GRAPH_INFO_H_

#include <stddef.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/common.h"

namespace tflite {

// Basic information about an inference graph, where execution nodes
// are connected via tensors.
class GraphInfo {
 public:
  virtual ~GraphInfo() {}

  // Total number of tensors in the graph.
  virtual size_t num_tensors() const = 0;

  // Returns a tensor given its index which is expected to be between 0 and
  // num_tensors().
  virtual TfLiteTensor* tensor(size_t index) = 0;

  // Number of nodes in the current execution plan.
  virtual size_t num_execution_nodes() const = 0;

  // Total number of known nodes, which may include nodes that are no longer in
  // the execution plan. This happens in case of applying multiple delegates.
  // Should be >= num_execution_nodes()
  virtual size_t num_total_nodes() const = 0;

  // Returns a node given its index in the execution plan, which is expected to
  // be between 0 and num_execution_nodes().
  virtual const TfLiteNode& node(size_t index) const = 0;

  // Returns an implementation-specific node index which may be different from
  // execution-plan index.
  // Expected to be between 0 and num_total_nodes().
  virtual size_t node_index(size_t index) const = 0;

  // Returns the indices of the input tensors.
  virtual const std::vector<int>& inputs() const = 0;

  // Returns the indices of the output tensors.
  virtual const std::vector<int>& outputs() const = 0;

  // Returns the indices of the variable tensors.
  virtual const std::vector<int>& variables() const = 0;
};

// Represents a subset of nodes in a TensorFlow Lite graph.
struct NodeSubset {
  enum Type {
    kTfUnexplored = 0,  // temporarily used during creation
    kTfPartition,
    kTfNonPartition
  };
  Type type = kTfUnexplored;
  // Nodes within the node sub set
  std::vector<int> nodes;
  // Tensors that stride output from another node sub set that this depends on,
  // or global inputs to the TensorFlow Lite full graph.
  std::vector<int> input_tensors;
  // Outputs that are consumed by other node sub sets or are global output
  // tensors. All output tensors of the nodes in the node sub set that do not
  // appear in this list are intermediate results that can be potentially
  // elided.
  std::vector<int> output_tensors;
};

// Node edge.second depends on node edge.first.
using ControlEdge = std::pair<int32_t, int32_t>;
using ControlEdges = std::vector<ControlEdge>;

// Partitions a list of node indices `nodes_to_partition` into node subsets.
// Each node subset is in dependency order (i.e. all members of the node subsets
// can be executed in the order they occur). Maintains the relative ordering of
// nodes that have their `might_have_side_effects` attribute set. `node_subsets`
// is assumed to be empty.
TfLiteStatus PartitionGraphIntoIndependentNodeSubsets(
    const GraphInfo* info, const TfLiteIntArray* nodes_to_partition,
    std::vector<NodeSubset>* node_subsets);

// Partitions a list of node indices `nodes_to_partition` into node subsets.
// Each node subset is in dependency order (i.e. all members of the node subset
// can be executed in the order they occur). `control_edges` specified a control
// dependency DAG on the nodes contained in `info`. The resulting partitioning
// will respect these control dependencies. This way, restrictions (in addition
// to the nodes' data dependencies) can be imposed on the ultimate execution
// order of the graph.
//
// (Example: with `control_edges.empty()` and `nodes_to_partition == {2, 3}`,
// the graph
//                    /------------\
//                    |            v
// 0 --> 1 --> 2* --> 3*     4 --> 5
//       |                   ^
//       \-------------------/
//
// will be partitioned as {{0, 1, 4}, {2, 3}, {5}}, since data dependencies
// (notated '-->') allow for execution of 4 immediately after 1.
//
// With an additional control dependency `control_edges == {{3, 4}}` (notated
// '==>'), execution of node 4 requires prior execution of node 3:
//
//                    /------------\
//                    |            v
// 0 --> 1 --> 2* --> 3* ==> 4 --> 5
//       |                   ^
//       \-------------------/
//
// and the partitioning will be {{0, 1}, {2, 3}, {4, 5}}.)
//
// `node_subsets` is assumed to be empty.
TfLiteStatus PartitionGraphIntoIndependentNodeSubsets(
    const GraphInfo* info, const TfLiteIntArray* nodes_to_partition,
    const ControlEdges& control_edges, std::vector<NodeSubset>* node_subsets);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_GRAPH_INFO_H_
