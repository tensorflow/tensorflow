/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_RESOURCE_HANDLE_OP(FertileStatsResource);

REGISTER_RESOURCE_HANDLE_OP(DecisionTreeResource);

REGISTER_OP("FertileStatsIsInitializedOp")
    .Input("stats_handle: resource")
    .Output("is_initialized: bool")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Checks whether a stats has been initialized.
)doc");

REGISTER_OP("CreateFertileStatsVariable")
    .Attr("params: string")
    .Input("stats_handle: resource")
    .Input("stats_config: string")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .Doc(R"doc(
Creates a stats model and returns a handle to it.

params: A serialized TensorForestParams proto.
stats_handle: handle to the stats resource to be created.
stats_config: Serialized proto of the stats.
)doc");

REGISTER_OP("FertileStatsSerialize")
    .Attr("params: string")
    .Input("stats_handle: resource")
    .Output("stats_config: string")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Serializes the stats to a proto.

params: A serialized TensorForestParams proto.
stats_handle: The handle to the stats.
stats_config: Serialized proto of the stats.
)doc");

REGISTER_OP("FertileStatsDeserialize")
    .Attr("params: string")
    .Input("stats_handle: resource")
    .Input("stats_config: string")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .Doc(R"doc(
Deserializes a serialized stats config and replaces current stats.

params: A serialized TensorForestParams proto.
stats_handle: The handle to the stats.
stats_config: Serialized proto of the stats.
)doc");

REGISTER_OP("GrowTree")
    .Attr("params: string")
    .Input("tree_handle: resource")
    .Input("stats_handle: resource")
    .Input("finished_nodes: int32")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .Doc(R"doc(
Grows the tree for finished nodes and allocates waiting nodes.

params: A serialized TensorForestParams proto.
tree_handle: The handle to the tree.
stats_handle: The handle to the stats.
finished_nodes: A 1-d Tensor of finished node ids from ProcessInput.
)doc");

REGISTER_OP("ProcessInput")
    .Attr("random_seed: int")
    .Attr("input_spec: string")
    .Attr("params: string")
    .Input("tree_handle: resource")
    .Input("stats_handle: resource")
    .Input("input_data: float")
    .Input("input_labels: float")
    .Input("leaf_ids: int32")
    .Output("finished_nodes: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(c->UnknownDim()));
      return Status::OK();
    })
    .Doc(R"doc(
Add labels to stats after traversing the tree for each example.

Outputs node ids that are finished.

params: A serialized TensorForestParams proto.
tree_handle: The handle to the tree.
stats_handle: The handle to the stats.
input_data: The training batch's features as a 2-d tensor; `input_data[i][j]`
   gives the j-th feature of the i-th input.
input_labels: The training batch's labels as a 1 or 2-d tensor.
  'input_labels[i][j]' gives the j-th label/target for the i-th input.
finished_nodes: A 1-d tensor of node ids that have finished and are ready to
  grow.
leaf_ids: `leaf_ids[i]` is the leaf id for input i.
)doc");

REGISTER_OP("FinalizeTree")
    .Attr("params: string")
    .Input("tree_handle: resource")
    .Input("stats_handle: resource")
    .SetShapeFn([](shape_inference::InferenceContext* c) { return Status::OK(); })
    .Doc(R"doc(
Puts the Leaf models inside the tree into their final form.

If drop_final_class is true, the per-class probability prediction of the
last class is not stored in the leaf models.

params: A serialized TensorForestParams proto.
tree_handle: The handle to the tree.
stats_handle: The handle to the stats.
)doc");

REGISTER_OP("TreeIsInitializedOp")
    .Input("tree_handle: resource")
    .Output("is_initialized: bool")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Checks whether a tree has been initialized.
)doc");

REGISTER_OP("CreateTreeVariable")
    .Attr("params: string")
    .Input("tree_handle: resource")
    .Input("tree_config: string")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .Doc(R"doc(
Creates a tree  model and returns a handle to it.

params: A serialized TensorForestParams proto.
tree_handle: handle to the tree resource to be created.
tree_config: Serialized proto of the tree.
)doc");

REGISTER_OP("TreeSerialize")
    .Input("tree_handle: resource")
    .Output("tree_config: string")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Serializes the tree  to a proto.

tree_handle: The handle to the tree.
tree_config: Serialized proto of the tree.
)doc");

REGISTER_OP("TreeDeserialize")
    .Attr("params: string")
    .Input("tree_handle: resource")
    .Input("tree_config: string")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .Doc(R"doc(
Deserializes a serialized tree config and replaces current tree.

params: A serialized TensorForestParams proto.
tree_handle: The handle to the tree .
tree_config: Serialized proto of the .
)doc");

REGISTER_OP("TreeSize")
    .Input("tree_handle: resource")
    .Output("tree_size: int32")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Outputs the size of the tree, including leaves.

tree_handle: The handle to the tree.
tree_size: Size scalar.
)doc");

REGISTER_OP("TreePredictions")
    .Attr("input_spec: string")
    .Attr("params: string")
    .Input("tree_handle: resource")
    .Input("input_data: float")
    .Input("sparse_input_indices: int64")
    .Input("sparse_input_values: float")
    .Input("sparse_input_shape: int64")
    .Output("predictions: float")
    .Output("tree_paths: string")
    .SetShapeFn([](InferenceContext* c) {
		    shape_inference::DimensionHandle num_points = c->UnknownDim();

      if (c->RankKnown(c->input(1)) && c->Rank(c->input(1)) > 0 &&
          c->Value(c->Dim(c->input(1), 0)) > 0) {
        num_points = c->Dim(c->input(1), 0);
      }

      c->set_output(0, c->Matrix(num_points, c->UnknownDim()));
      c->set_output(1, c->Vector(c->UnknownDim()));
      return Status::OK();
    })
    .Doc(R"doc(
Outputs the predictions for the given input data.

params: A serialized TensorForestParams proto.
tree_handle: The handle to the tree.
input_data: The training batch's features as a 2-d tensor; `input_data[i][j]`
   gives the j-th feature of the i-th input.
sparse_input_indices: The indices tensor from the SparseTensor input.
sparse_input_values: The values tensor from the SparseTensor input.
sparse_input_shape: The shape tensor from the SparseTensor input.
predictions: `predictions[i][j]` is the probability that input i is class j.
tree_paths: `tree_paths[i]` is a serialized TreePath proto for example i.
)doc");

REGISTER_OP("TraverseTree")
    .Attr("input_spec: string")
    .Attr("params: string")
    .Input("tree_handle: resource")
    .Input("input_data: float")
    .Input("sparse_input_indices: int64")
    .Input("sparse_input_values: float")
    .Input("sparse_input_shape: int64")
    .Output("leaf_ids: int32")
    .SetShapeFn([](InferenceContext* c) {
		    shape_inference::DimensionHandle num_points = c->UnknownDim();

      if (c->RankKnown(c->input(1)) && c->Rank(c->input(1)) > 0 &&
          c->Value(c->Dim(c->input(1), 0)) > 0) {
        num_points = c->Dim(c->input(1), 0);
      }

      c->set_output(0, c->Vector(num_points));
      return Status::OK();
    })
    .Doc(R"doc(
Outputs the leaf ids for the given input data.

params: A serialized TensorForestParams proto.
tree_handle: The handle to the tree.
input_data: The training batch's features as a 2-d tensor; `input_data[i][j]`
   gives the j-th feature of the i-th input.
sparse_input_indices: The indices tensor from the SparseTensor input.
sparse_input_values: The values tensor from the SparseTensor input.
sparse_input_shape: The shape tensor from the SparseTensor input.
leaf_ids: `leaf_ids[i]` is the leaf id for input i.
)doc");

REGISTER_OP("UpdateModel")
    .Attr("params: string")
    .Input("tree_handle: resource")
    .Input("leaf_ids: int32")
    .Input("input_labels: float")
    .Input("input_weights: float")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .Doc(R"doc(
Updates the given leaves for each example with the new labels.

params: A serialized TensorForestParams proto.
tree_handle: The handle to the tree.
leaf_ids: `leaf_ids[i]` is the leaf id for input i.
input_labels: The training batch's labels as a 1 or 2-d tensor.
  'input_labels[i][j]' gives the j-th label/target for the i-th input.
input_weights: The training batch's weights as a 1-d tensor.
  'input_weights[i]' gives the weight for the i-th input.
)doc");

REGISTER_OP("FeatureUsageCounts")
    .Attr("params: string")
    .Input("tree_handle: resource")
    .Output("feature_counts: int32")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(c->UnknownDim()));
      return Status::OK();
    })
    .Doc(R"doc(
Outputs the number of times each feature was used in a split.

params: A serialized TensorForestParams proto.
tree_handle: The handle to the tree.
feature_counts: `feature_counts[i]` is the number of times feature i was used
    in a split.
)doc");
}  // namespace tensorflow
