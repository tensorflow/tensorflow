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
#include "tensorflow/contrib/boosted_trees/proto/learner.pb.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using tensorflow::boosted_trees::learner::LearnerConfig;

namespace tensorflow {

using shape_inference::InferenceContext;

static Status ApplyGradientTreesPredictionShapeFn(InferenceContext* c) {
  string learner_config_str;
  // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!
  c->GetAttr("learner_config", &learner_config_str).IgnoreError();
  LearnerConfig learner_config;
  ParseProtoUnlimited(&learner_config, learner_config_str);

  bool reduce_dim;
  c->GetAttr("reduce_dim", &reduce_dim).IgnoreError();
  // Sets the shape of the output as a matrix.
  c->set_output(0, {c->Matrix(InferenceContext::kUnknownDim,
                              reduce_dim ? learner_config.num_classes() - 1
                                         : learner_config.num_classes())});
  c->set_output(1, {c->Matrix(InferenceContext::kUnknownDim,
                              reduce_dim ? learner_config.num_classes() - 1
                                         : learner_config.num_classes())});
  c->set_output(2, {c->Vector(InferenceContext::kUnknownDim)});
  return Status::OK();
}

REGISTER_OP("GradientTreesPrediction")
    .Attr("learner_config: string")
    .Attr("num_dense_float_features: int >= 0")
    .Attr("num_sparse_float_features: int >= 0")
    .Attr("num_sparse_int_features: int >= 0")
    .Attr("use_locking: bool = false")
    .Attr("apply_dropout: bool")
    .Attr("apply_averaging: bool")
    .Attr("center_bias: bool")
    .Attr("reduce_dim: bool")
    .Input("tree_ensemble_handle: resource")
    .Input("seed: int64")
    .Input("dense_float_features: num_dense_float_features * float")
    .Input("sparse_float_feature_indices: num_sparse_float_features * int64")
    .Input("sparse_float_feature_values: num_sparse_float_features * float")
    .Input("sparse_float_feature_shapes: num_sparse_float_features * int64")
    .Input("sparse_int_feature_indices: num_sparse_int_features * int64")
    .Input("sparse_int_feature_values: num_sparse_int_features * int64")
    .Input("sparse_int_feature_shapes: num_sparse_int_features * int64")
    .Output("predictions: float")
    .Output("no_dropout_predictions: float")
    .Output("drop_out_tree_indices_weights: float")
    .SetShapeFn(ApplyGradientTreesPredictionShapeFn)
    .Doc(R"doc(
Runs multiple additive regression forests predictors on input instances
and computes the final prediction for each class.

learner_config: Config for the learner of type LearnerConfig proto. Prediction
ops for now uses only LearningRateDropoutDrivenConfig config from the learner.
num_dense_float_features: Number of dense float features.
num_sparse_float_features: Number of sparse float features.
num_sparse_int_features: Number of sparse int features.
use_locking: Whether to use locking.
seed: random seed to be used for dropout.
reduce_dim: whether to reduce the dimension (legacy impl) or not.
apply_dropout: whether to apply dropout during prediction.
apply_averaging: whether averaging of tree ensembles should take place. If set
to true, will be based on AveragingConfig from learner_config.
tree_ensemble_handle: The handle to the tree ensemble.
dense_float_features: Rank 2 Tensors containing dense float feature values.
sparse_float_feature_indices: Rank 2 Tensors containing sparse float indices.
sparse_float_feature_values: Rank 1 Tensors containing sparse float values.
sparse_float_feature_shapes: Rank 1 Tensors containing sparse float shapes.
sparse_int_feature_indices: Rank 2 Tensors containing sparse int indices.
sparse_int_feature_values: Rank 1 Tensors containing sparse int values.
sparse_int_feature_shapes: Rank 1 Tensors containing sparse int shapes.
predictions: Rank 2 Tensor containing predictions per example per class.
no_dropout_predictions: The same as predictions, but using all trees (even
those that were dropped due to dropout).
drop_out_tree_indices_weights: Tensor of Rank 2 containing dropped trees indices
and original weights of those trees during prediction.
)doc");

REGISTER_OP("GradientTreesPartitionExamples")
    .Attr("num_dense_float_features: int >= 0")
    .Attr("num_sparse_float_features: int >= 0")
    .Attr("num_sparse_int_features: int >= 0")
    .Attr("use_locking: bool = false")
    .Input("tree_ensemble_handle: resource")
    .Input("dense_float_features: num_dense_float_features * float")
    .Input("sparse_float_feature_indices: num_sparse_float_features * int64")
    .Input("sparse_float_feature_values: num_sparse_float_features * float")
    .Input("sparse_float_feature_shapes: num_sparse_float_features * int64")
    .Input("sparse_int_feature_indices: num_sparse_int_features * int64")
    .Input("sparse_int_feature_values: num_sparse_int_features * int64")
    .Input("sparse_int_feature_shapes: num_sparse_int_features * int64")
    .Output("partition_ids: int32")
    .SetShapeFn([](InferenceContext* c) {
      return c->set_output("partition_ids",
                           {c->Vector(InferenceContext::kUnknownDim)});
    })
    .Doc(R"doc(
Splits input examples into the leaves of the tree.

num_dense_float_features: Number of dense float features.
num_sparse_float_features: Number of sparse float features.
num_sparse_int_features: Number of sparse int features.
use_locking: Whether to use locking.
tree_ensemble_handle: The handle to the tree ensemble.
dense_float_features: Rank 2 Tensors containing dense float feature values.
sparse_float_feature_indices: Rank 2 Tensors containing sparse float indices.
sparse_float_feature_values: Rank 1 Tensors containing sparse float values.
sparse_float_feature_shapes: Rank 1 Tensors containing sparse float shapes.
sparse_int_feature_indices: Rank 2 Tensors containing sparse int indices.
sparse_int_feature_values: Rank 1 Tensors containing sparse int values.
sparse_int_feature_shapes: Rank 1 Tensors containing sparse int shapes.
partition_ids: Rank 1 Tensor containing partition ids per example.
)doc");

}  // namespace tensorflow
