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

#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_RESOURCE_HANDLE_OP(DecisionTreeResource);

REGISTER_OP("TreeIsInitializedOp")
    .Input("tree_handle: resource")
    .Output("is_initialized: bool")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Checks whether a tree has been initialized.
)doc");

REGISTER_OP("CreateTreeVariable")
    .Attr("leaf_model_type: int32")
    .Attr("num_output: int32")
    .Input("tree_handle: resource")
    .Input("tree_config: string")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .Doc(R"doc(
Creates a tree  model and returns a handle to it.
leaf_model_type: A type for classification.
num_output: the output dimension.
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
    .Input("tree_handle: resource")
    .Input("tree_config: string")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .Doc(R"doc(
Deserializes a serialized tree config and replaces current tree.
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
    .Attr("leaf_model_type: int32")
    .Attr("num_output: int32")
    .Input("tree_handle: resource")
    .Input("input_data: float")
    .Output("predictions: float")
    .Output("tree_paths: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
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
leaf_model_type: A type for classification.
num_output: the output dimension.
tree_handle: The handle to the tree.
input_data: The training batch's features as a 2-d tensor; `input_data[i][j]`
   gives the j-th feature of the i-th input.
predictions: `predictions[i][j]` is the probability that input i is class j.
tree_paths: `tree_paths[i]` is a serialized TreePath proto for example i.
)doc");
}  // namespace tensorflow
