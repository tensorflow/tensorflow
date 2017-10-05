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

#ifndef TENSORFLOW_GRAPPLER_COSTS_GRAPH_PROPERTIES_H_
#define TENSORFLOW_GRAPPLER_COSTS_GRAPH_PROPERTIES_H_

#include <unordered_map>
#include <vector>
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {
namespace grappler {

// A TensorFlow model to optimize.
// Models are represented by the combination of a graph, one of more fetch
// nodes, and potentially a set of nodes to feed.
class GraphProperties {
 public:
  // Factory method for creating a GrapplerShapes from a MetaGraphDef.
  // Returns nullptr if the given meta_graph cannot be converted.
  explicit GraphProperties(const GrapplerItem& item) : item_(item) {}

  Status InferStatically();
  Status InferDynamically(Cluster* cluster);
  Status InferFromCostGraph(const CostGraphDef& cost_graph);

  // Stores `item_.graph` with the inferred output shapes to `output_graph_def`.
  Status AnnotateOutputShapes(GraphDef* output_graph_def);

  bool HasInputProperties(const string& name) const;
  bool HasOutputProperties(const string& name) const;
  const std::vector<OpInfo::TensorProperties>& GetInputProperties(
      const string& node_name) const;
  const std::vector<OpInfo::TensorProperties>& GetOutputProperties(
      const string& node_name) const;

  static void FillTensorPropertiesFromContext(
      const shape_inference::ShapeHandle&, const DataType&,
      shape_inference::InferenceContext*, OpInfo::TensorProperties*);

 private:
  // Inputs
  GrapplerItem item_;
  std::map<string, std::vector<OpInfo::TensorProperties>> input_properties_;
  std::map<string, std::vector<OpInfo::TensorProperties>> output_properties_;
  const std::vector<OpInfo::TensorProperties> missing_properties_;

  // Merges shapes <shapes_and_types>, determined from an EnqueueV2 node, into
  // <*queue_shapes_and_types>.
  Status MergeEnqueueShapesAndTypes(
      const std::vector<shape_inference::ShapeAndType>& shapes_and_types,
      shape_inference::InferenceContext* qctx,
      std::vector<shape_inference::ShapeAndType>* queue_shapes_and_types);
  // Relaxes shapes <shapes_and_types>, determined from an EnqueueV2 node, into
  // <*queue_shapes_and_types>.
  Status RelaxEnqueueShapesAndMergeTypes(
      const std::vector<shape_inference::ShapeAndType>& shapes_and_types,
      shape_inference::InferenceContext* qctx,
      std::vector<shape_inference::ShapeAndType>* queue_shapes_and_types);

  // This gives access to private function of InferenceContext.
  static void Relax(shape_inference::InferenceContext* c,
                    shape_inference::ShapeHandle s0,
                    shape_inference::ShapeHandle s1,
                    shape_inference::ShapeHandle* out);

  // These give access to private functions of ShapeRefiner.
  static bool SameDefinedShape(shape_inference::InferenceContext* c,
                               shape_inference::ShapeHandle s0,
                               shape_inference::ShapeHandle s1);
  static bool IsUpdatedShapesOrTypes(
      shape_inference::InferenceContext* c,
      const std::vector<shape_inference::ShapeAndType>& existing,
      const std::vector<shape_inference::ShapeAndType>& updated);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_COSTS_GRAPH_PROPERTIES_H_
