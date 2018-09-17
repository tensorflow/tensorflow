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
  Status InferStatically(bool assume_valid_feeds);
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
  bool HasInputProperties(const string& name) const;
  bool HasOutputProperties(const string& name) const;
  const std::vector<OpInfo::TensorProperties>& GetInputProperties(
      const string& node_name) const;
  const std::vector<OpInfo::TensorProperties>& GetOutputProperties(
      const string& node_name) const;
  // Invalidate input/output properties for nodes modified during graph
  // optimization pass, to prevent potential optimizations, based on incorrect
  // shape information.
  void ClearInputProperties(const string& node_name);
  void ClearOutputProperties(const string& node_name);

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
  Status UpdateMergeNode(SymbolicShapeRefiner* shape_refiner,
                         const NodeDef* node, bool* new_shapes) const;
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
  std::map<string, std::vector<OpInfo::TensorProperties>> input_properties_;
  std::map<string, std::vector<OpInfo::TensorProperties>> output_properties_;
  const std::vector<OpInfo::TensorProperties> missing_properties_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_COSTS_GRAPH_PROPERTIES_H_
