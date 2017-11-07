// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;

namespace tensorforest {

REGISTER_RESOURCE_HANDLE_OP(DecisionTreeResource);

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

REGISTER_OP("TreePredictionsV4")
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
      DimensionHandle num_points = c->UnknownDim();

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

REGISTER_OP("TraverseTreeV4")
    .Attr("input_spec: string")
    .Attr("params: string")
    .Input("tree_handle: resource")
    .Input("input_data: float")
    .Input("sparse_input_indices: int64")
    .Input("sparse_input_values: float")
    .Input("sparse_input_shape: int64")
    .Output("leaf_ids: int32")
    .SetShapeFn([](InferenceContext* c) {
      DimensionHandle num_points = c->UnknownDim();

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

REGISTER_OP("UpdateModelV4")
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
input_weights: The training batch's eample weights as a 1-d tensor.
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

}  // namespace tensorforest
}  // namespace tensorflow
