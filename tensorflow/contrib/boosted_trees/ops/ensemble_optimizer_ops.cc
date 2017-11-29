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

REGISTER_OP("AddTreesToEnsemble")
    .Input("tree_ensemble_handle: resource")
    .Input("ensemble_to_add: string")
    .Input("feature_column_usage_counts_handle: Ref(int64)")
    .Input("feature_column_usage_counts_to_add: int64")
    .Input("feature_column_gains_handle: Ref(float)")
    .Input("feature_column_gains_to_add: float")
    .Input("drop_out_tree_indices_weights: float")
    .Input("learning_rate: float")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Synchronously adds a tree ensemble to a an existing tree ensemble variable.
tree_ensemble_handle: Handle to the ensemble variable.
ensemble_to_add: Serialized DecisionTreeConfig proto of the tree.
feature_column_usage_counts_handle: Handle to the feature column usage counts variable.
feature_column_usage_counts_to_add: Rank 1 Tensor holding feature column usage counts to add.
feature_column_gains_handle: Handle to the feature column gains variable.
feature_column_gains_to_add: Rank 1 Tensor holding feature column gains to add.
drop_out_tree_indices_weights: Rank 2 Tensor containing dropped trees indices
and original weights of those trees during prediction.
learning_rate: The learning rate that the tuner found for this iteration.
)doc");

}  // namespace tensorflow
