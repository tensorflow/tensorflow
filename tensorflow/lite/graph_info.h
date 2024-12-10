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

#include "tensorflow/lite/core/c/common.h"

namespace tflite {

// Basic information about an inference graph, where execution nodes
// are connected via tensors.
class GraphInfo {
 public:
  virtual ~GraphInfo() {}

  // Total number of tensors in the graph. This should be cached when possible.
  virtual size_t num_tensors() const = 0;

  // Returns a tensor given its index which is expected to be between 0 and
  // num_tensors(). Use tensors() below for iteration as it is much faster.
  virtual TfLiteTensor* tensor(size_t index) = 0;

  // Returns all tensors in the graph
  virtual TfLiteTensor* tensors() = 0;

  // Number of nodes in the current execution plan.
  virtual size_t num_execution_nodes() const = 0;

  // Total number of known nodes, which may include nodes that are no longer in
  // the execution plan. This happens in case of applying multiple delegates.
  // Should be >= num_execution_nodes()
  virtual size_t num_total_nodes() const = 0;

  // Returns a node given its index in the execution plan, which is expected to
  // be between 0 and num_execution_nodes().
  virtual const TfLiteNode& node(size_t index) const = 0;

  // Returns a node registration given its index which is expected to be between
  // 0 and num_nodes().
  virtual const TfLiteRegistration& registration(size_t index) const = 0;

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

// LINT.IfChange
// Node edge.second depends on node edge.first.
using ControlEdge = std::pair<int32_t, int32_t>;
using ControlEdges = std::vector<ControlEdge>;
// LINT.ThenChange(//tensorflow/compiler/mlir/lite/utils/control_edges.h)

// Partitions a list of node indices `nodes_to_partition` into node subsets.
// Each node subset is in dependency order internally (i.e. all members of the
// node subsets can be executed in the order they occur) and externally (i.e.,
// node subsets are executable in the order they occur.) The function assumes
// that the nodes of the graph represented in *info are in dependency order.
//
// Depending on the value of `greedily`, the function behaves
//
// - greedily: while a node_set is generated whose members are (aren't) members
// of
//   `*nodes_to_partition`, it will add nodes to this subset, as long as they
//   are (aren't) members of *nodes_to_partition and they are schedulable (i.e.,
//   all nodes they depend have already be added to `*node_subsets`.)
//
// - non-greedily: this preserves the original execution order, i.e. the node
//   subsets generated will be of the form [ [0..i_1), [i1..i2), ... ].
//
// `control_edges` specifies a control dependency DAG on the nodes contained in
// `info`. The resulting partitioning will respect these control
// dependencies. This way, restrictions (in addition to the nodes' data
// dependencies) can be imposed on the ultimate execution order of the graph
// (naturally, this is relevant only if ordering greedily.)
//
// (Example: with `greedily`, `control_edges.empty()`, and `nodes_to_partition
// == {2, 3}`, the graph
//
//                    ▼------------▼
//                    |            v
// 0 --> 1 --> 2* --> 3*     4 --> 5
//       |                   ^
//       ▲-------------------▲
//
// will be partitioned as {{0, 1, 4}, {2, 3}, {5}}, since data dependencies
// (notated '-->') allow for execution of 4 immediately after 1.
//
// With an additional control dependency `control_edges == {{3, 4}}` (notated
// '==>'), execution of node 4 requires prior execution of node 3:
//
//                    ▼------------▼
//                    |            v
// 0 --> 1 --> 2* --> 3* ==> 4 --> 5
//       |                   ^
//       ▲-------------------▲
//
// and the partitioning will be {{0, 1}, {2, 3}, {4, 5}}.)
//
// If control_edges == nullptr, the algorithm preserves the relative ordering of
// nodes that have their `might_have_side_effects` attribute set, i.e., it
// behaves as if `*control_dependencies` of the form `{ {n_1, n_2}, {n_2, n_3},
// ... }` had been handed in, where the n_i are the (sorted) indices of nodes
// with `might_have_side_effects` attribute set.
//
// The function assumes that `*node_subsets` is initially empty.
TfLiteStatus PartitionGraphIntoIndependentNodeSubsets(
    const GraphInfo* info, const TfLiteIntArray* nodes_to_partition,
    std::vector<NodeSubset>* node_subsets, bool greedily,
    const ControlEdges* control_edges = nullptr,
    bool disable_node_fusion = false);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_GRAPH_INFO_H_
