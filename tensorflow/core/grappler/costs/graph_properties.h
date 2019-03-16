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

#ifndef TENSORFLOW_CORE_GRAPPLER_COSTS_GRAPH_PROPERTIES_H_
#define TENSORFLOW_CORE_GRAPPLER_COSTS_GRAPH_PROPERTIES_H_

#include <unordered_map>
#include <vector>
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {

namespace grappler {

// Optional attributes that tell about node output information.
// We use these side information, if provided, for static shape inference
// and VirtualScheduler scheduling.

// Switch op attribute as a vector of int that tells which branch the
// Switch output is taken on every round of execution.
// Used for scheduling ops after Switch correctly (e.g., While loop).
ABSL_CONST_INIT const char kOutputSlots[] = "_output_slot_vector";

// Example:
// Assume a node has two outputs and iterated for three times. Then it has:
// _execution_count = 3
// _output_sizes_vector = [2, 2, 2]
// _output_dtype_vector.size = 6
// _output_shape_vector.size = 6

// If all the iterations have same output shapes, then
// _execution_count = 3
// _same_output_for_iterations = true
// _output_sizes_vector = [2]
// _output_dtype_vector.size = 2
// _output_shape_vector.size = 2

// How many times this node has been executed.
ABSL_CONST_INIT const char kExecutionCount[] = "_execution_count";

// Records the output sizes for each round of execution.
ABSL_CONST_INIT const char kOutputSizes[] = "_output_sizes_vector";

// The node has been scheduled multiple times with outputs that have the same
// shape.
ABSL_CONST_INIT const char kOutputSame[] = "_same_output_for_iterations";

// Outputs DataType vector.
ABSL_CONST_INIT const char kOutputTypes[] = "_output_dtype_vector";

// Outputs TensorShapeProto vector.
ABSL_CONST_INIT const char kOutputShapes[] = "_output_shape_vector";

class SymbolicShapeRefiner;
class TopoQueue;

// Infer OpInfo::TensorProperties for graph nodes inputs/outputs.
//
// Typical use case, is to infer tensor properties from a graph, before doing
// optimization pass. Nodes modified during optimization pass have to be
// invalidated, to prevent further incorrect optimizations based on wrong shape
// and data type properties.
class GraphProperties {
 public:
  // The item must outlive the properties
  explicit GraphProperties(const GrapplerItem& item) : item_(item) {}

  // Infer the shapes through abstract interpretation. Feed information can be
  // incorrect so it should be discarded to ensure correctness of the analysis.
  // However, it can help infer shapes in the fanout of fed nodes (even though
  // the correctness of these shapes can't be guaranteed), so in some cases
  // (such as simulation or scheduling) it makes sense of keep these shapes.
  // aggressive_shape_inference option executes nodes on the host to identify
  // output values when possible and does other aggressive strategies.
  // Similar to assuming_valid_feeds, this may cause incorrectness in graph
  // analyses, but is useful for simulation or scheduling.
  Status InferStatically(bool assume_valid_feeds,
                         bool aggressive_shape_inference);
  Status InferStatically(bool assume_valid_feeds) {
    return InferStatically(assume_valid_feeds,
                           /*aggressive_shape_inference=*/false);
  }
  // Infer the shape by running the graph on the specified cluster and recording
  // the shapes of the processed tensors.
  Status InferDynamically(Cluster* cluster);
  // Extract the properties from a cost graph. For testing only since there is
  // no way to ensure that the cost graph match the item.
  Status InferFromCostGraph(const CostGraphDef& cost_graph);

  // Stores `item_.graph` with the inferred output shapes to `output_graph_def`.
  Status AnnotateOutputShapes(GraphDef* output_graph_def) const;

  // Return the properties of node inputs/outputs, including data types and
  // shapes. Note that the dimensions in the shapes can be negative. We use the
  // -1 value to denote that we don't know anything about a dimension. We use
  // values strictly less than -1 to encode symbolic dimensions: although we
  // don't know the actual value of the symbolic dimension, we know that all the
  // dimensions denoted by the same negative value are the equal.
  bool HasInputProperties(const string& node_name) const;
  bool HasOutputProperties(const string& node_name) const;
  const std::vector<OpInfo::TensorProperties>& GetInputProperties(
      const string& node_name) const;
  const std::vector<OpInfo::TensorProperties>& GetOutputProperties(
      const string& node_name) const;
  // Invalidate input/output properties for nodes modified during graph
  // optimization pass, to prevent potential optimizations, based on incorrect
  // shape information.
  void ClearInputProperties(const string& node_name);
  void ClearOutputProperties(const string& node_name);
  // Returns true if we have *any* properties.
  bool has_properties() const {
    return input_properties_.size() > 0 || output_properties_.size() > 0;
  }

 private:
  // Relaxes shapes <shapes_and_types>, determined from an EnqueueV2 node, into
  // <*queue_shapes_and_types>.
  static Status RelaxEnqueueShapesAndMergeTypes(
      SymbolicShapeRefiner* shape_refiner, const NodeDef* qnode,
      const std::vector<shape_inference::ShapeAndType>& shapes_and_types,
      std::vector<shape_inference::ShapeAndType>* queue_shapes_and_types);

  // Update the shapes of the enqueue node, port them over to the corresponding
  // queue, and schedule the reprocessing of the queue if needed.
  static Status UpdateEnqueue(
      const NodeDef* enqueue_node,
      const std::unordered_map<const NodeDef*, const NodeDef*>&
          resource_handles,
      SymbolicShapeRefiner* shape_refiner, bool* new_shapes);

  // Update the shapes and types of the Queue node, if not set by Enqueue node.
  static Status UpdateQueue(const NodeDef* queue_node,
                            SymbolicShapeRefiner* shape_refiner,
                            bool* new_shapes);

  // Update the output shapes of a Merge node, and enqueue its fanout in
  // new_shapes if needed.
  Status UpdateMerge(SymbolicShapeRefiner* shape_refiner, const NodeDef* node,
                     bool* new_shapes) const;
  // Process the Enter node, and enqueue its fanout in new_shapes if needed.
  static Status UpdateEnter(SymbolicShapeRefiner* shape_refiner,
                            const NodeDef* node, bool* new_shapes);
  // Update the shapes for node 'n'. If output shapes for n have changed,
  // enqueue its fanout in 'new_shapes'.
  Status UpdateShapes(SymbolicShapeRefiner* shape_refiner,
                      const std::unordered_map<const NodeDef*, const NodeDef*>&
                          resource_handles,
                      const NodeDef* n, bool* new_shapes) const;
  // Propagate the shapes for the nodes enqueued in new_shapes and their
  // transitive fanout until a fixed point is reached.
  Status PropagateShapes(
      SymbolicShapeRefiner* shape_refiner, TopoQueue* new_shapes,
      const std::unordered_map<const NodeDef*, const NodeDef*>&
          resource_handles,
      int num_loops) const;

  // Data members
  const GrapplerItem& item_;
  std::unordered_map<string, std::vector<OpInfo::TensorProperties>>
      input_properties_;
  std::unordered_map<string, std::vector<OpInfo::TensorProperties>>
      output_properties_;
  const std::vector<OpInfo::TensorProperties> missing_properties_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_COSTS_GRAPH_PROPERTIES_H_
