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
namespace gtflow {

REGISTER_RESOURCE_HANDLE_OP(DecisionTreeEnsembleResource);

REGISTER_OP("TreeEnsembleIsInitializedOp")
    .Input("tree_ensemble_handle: resource")
    .Output("is_initialized: bool")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Checks whether a tree ensemble has been initialized.
)doc");

REGISTER_OP("CreateTreeEnsembleVariable")
    .Input("tree_ensemble_handle: resource")
    .Input("stamp_token: int64")
    .Input("tree_ensemble_config: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused_input));
      return Status::OK();
    })
    .Doc(R"doc(
Creates a tree ensemble model and returns a handle to it.

tree_ensemble_handle: Handle to the tree ensemble resource to be created.
stamp_token: Token to use as the initial value of the resource stamp.
tree_ensemble_config: Serialized proto of the tree ensemble.
)doc");

REGISTER_OP("TreeEnsembleStampToken")
    .Input("tree_ensemble_handle: resource")
    .Output("stamp_token: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Retrieves the tree ensemble resource stamp token.

tree_ensemble_handle: Handle to the tree ensemble.
stamp_token: Stamp token of the tree ensemble resource.
)doc");

REGISTER_OP("TreeEnsembleSerialize")
    .Input("tree_ensemble_handle: resource")
    .Output("stamp_token: int64")
    .Output("tree_ensemble_config: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Serializes the tree ensemble to a proto.

tree_ensemble_handle: Handle to the tree ensemble.
stamp_token: Stamp token of the tree ensemble resource.
tree_ensemble_config: Serialized proto of the ensemble.
)doc");

REGISTER_OP("TreeEnsembleDeserialize")
    .Input("tree_ensemble_handle: resource")
    .Input("stamp_token: int64")
    .Input("tree_ensemble_config: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused_input));
      return Status::OK();
    })
    .Doc(R"doc(
Deserializes a serialized tree ensemble config and replaces current tree
ensemble.

tree_ensemble_handle: Handle to the tree ensemble.
stamp_token: Token to use as the new value of the resource stamp.
tree_ensemble_config: Serialized proto of the ensemble.
)doc");

}  // namespace gtflow
}  // namespace tensorflow
