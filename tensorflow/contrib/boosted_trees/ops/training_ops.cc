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
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace boosted_trees {

REGISTER_OP("CenterTreeEnsembleBias")
    .Attr("learner_config: string")
    .Attr("centering_epsilon: float = 0.01")
    .Input("tree_ensemble_handle: resource")
    .Input("stamp_token: int64")
    .Input("next_stamp_token: int64")
    .Input("delta_updates: float")
    .Output("continue_centering: bool")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &unused_input));
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Centers the tree ensemble bias before adding trees based on feature splits.

learner_config: Config for the learner of type LearnerConfig proto.
tree_ensemble_handle: Handle to the ensemble variable.
stamp_token: Stamp token for validating operation consistency.
next_stamp_token: Stamp token to be used for the next iteration.
delta_updates: Rank 1 Tensor containing delta updates per bias dimension.
continue_centering: Scalar indicating whether more centering is needed.
)doc");

REGISTER_OP("GrowTreeEnsemble")
    .Attr("learner_config: string")
    .Attr("num_handlers: int >= 0")
    .Attr("center_bias: bool")
    .Input("tree_ensemble_handle: resource")
    .Input("stamp_token: int64")
    .Input("next_stamp_token: int64")
    .Input("learning_rate: float")
    .Input("dropout_seed: int64")
    .Input("max_tree_depth: int32")
    .Input("weak_learner_type: int32")
    .Input("partition_ids: num_handlers * int32")
    .Input("gains: num_handlers * float")
    .Input("splits: num_handlers * string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused_input));
      // Dropout seed.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused_input));
      // Maximum tree depth.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused_input));
      return Status::OK();
    })
    .Doc(R"doc(
Grows the tree ensemble by either adding a layer to the last tree being grown
or by starting a new tree.

learner_config: Config for the learner of type LearnerConfig proto.
num_handlers: Number of handlers generating candidates.
tree_ensemble_handle: Handle to the ensemble variable.
stamp_token: Stamp token for validating operation consistency.
next_stamp_token: Stamp token to be used for the next iteration.
learning_rate: Scalar learning rate.
weak_learner_type: The type of weak learner to use.
partition_ids: List of Rank 1 Tensors containing partition Id per candidate.
gains: List of Rank 1 Tensors containing gains per candidate.
splits: List of Rank 1 Tensors containing serialized SplitInfo protos per candidate.
)doc");

REGISTER_OP("TreeEnsembleStats")
    .Input("tree_ensemble_handle: resource")
    .Input("stamp_token: int64")
    .Output("num_trees: int64")
    .Output("num_layers: int64")
    .Output("active_tree: int64")
    .Output("active_layer: int64")
    .Output("attempted_trees: int64")
    .Output("attempted_layers: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused_input));
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      c->set_output(3, c->Scalar());
      c->set_output(4, c->Scalar());
      c->set_output(5, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Retrieves stats related to the tree ensemble.

tree_ensemble_handle: Handle to the ensemble variable.
stamp_token: Stamp token for validating operation consistency.
num_trees: Scalar indicating the number of finalized trees in the ensemble.
num_layers: Scalar indicating the number of layers in the ensemble.
active_tree: Scalar indicating the active tree being trained.
active_layer: Scalar indicating the active layer being trained.
)doc");

}  // namespace boosted_trees
}  // namespace tensorflow
