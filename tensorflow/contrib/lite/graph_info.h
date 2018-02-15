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
#ifndef TENSORFLOW_CONTRIB_LITE_GRAPH_INFO_H_
#define TENSORFLOW_CONTRIB_LITE_GRAPH_INFO_H_

#include <vector>

#include "tensorflow/contrib/lite/context.h"

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

  // Total number of nodes in the graph.
  virtual size_t num_nodes() const = 0;

  // Returns a node given its index which is expected to be between 0 and
  // num_nodes().
  virtual const TfLiteNode& node(size_t index) const = 0;

  // Returns the indices of the input tensors.
  virtual const std::vector<int>& inputs() const = 0;

  // Returns the indices of the output tensors.
  virtual const std::vector<int>& outputs() const = 0;
};

// Represents a subgraph of a TensorFlow Lite graph.
struct Subgraph {
  enum Type {
    kTfUnexplored = 0,  // temporarily used during creation
    kTfPartition,
    kTfNonPartition
  };
  Type type = kTfUnexplored;
  // Nodes within the subgraph
  std::vector<int> nodes;
  // Tensors that stride output from another subgraph that this depends on,
  // or global inputs to the TensorFlow Lite full graph.
  std::vector<int> input_tensors;
  // Outputs that are consumed by other subgraphs or are global output tensors.
  // All output tensors of the nodes in the subgraph that do not appear in this
  // list are intermediate results that can be potentially elided.
  std::vector<int> output_tensors;
};

// Partitions a list of node indices `nodes_to_partition` into subgraphs.
// Each subgraph is in dependency order (i.e. all members of the subgraph).
// `subgraphs` is assumed to be empty.
TfLiteStatus PartitionGraphIntoIndependentSubgraphs(
    const GraphInfo* info, const TfLiteIntArray* nodes_to_partition,
    std::vector<Subgraph>* subgraphs);

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_GRAPH_INFO_H_
