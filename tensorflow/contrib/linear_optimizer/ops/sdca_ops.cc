/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"
namespace tensorflow {

// --------------------------------------------------------------------------

REGISTER_OP("SdcaSolver")
    .Attr("loss_type: {'logistic_loss', 'squared_loss', 'hinge_loss'}")
    .Attr("num_sparse_features: int >= 0")
    .Attr("num_dense_features: int >= 0")
    .Attr("l1: float >= 0")
    .Attr("l2: float >= 1")
    .Attr("num_inner_iterations: int >= 1")
    .Attr("container: string")
    .Attr("solver_uuid: string")
    .Input("sparse_features_indices: num_sparse_features * int64")
    .Input("sparse_features_values: num_sparse_features * float")
    .Input("dense_features: num_dense_features * float")
    .Input("example_weights: float")
    .Input("example_labels: float")
    .Input("example_ids: string")
    .Input("sparse_weights: Ref(num_sparse_features * float)")
    .Input("dense_weights: Ref(num_dense_features * float)")
    .Doc(R"doc(
Stochastic Dual Coordinate Ascent (SDCA) optimizer for linear models with
L1 + L2 regularization. As global optimization objective is strongly-convex, the
optimizer optimizes the dual objective at each step. The optimizer applies each
update one example at a time. Examples are sampled uniformly, and the optimizer
is learning rate free and enjoys linear convergence rate.

Proximal Stochastic Dual Coordinate Ascent, Shalev-Shwartz, Shai; Zhang, Tong.
2012arXiv1211.2717S: http://arxiv.org/pdf/1211.2717v1.pdf

  Loss objective = \sum f_{i}(wx_{i}) + l2 * |w|^2 + l1 * |w|

loss_type: Type of the primal loss. Currently SdcaSolver supports logistic,
  squared and hinge losses.
num_sparse_features: Number of sparse feature groups to train on.
num_dense_features: Number of dense feature groups to train on.
l1: Symmetric l1 regularization strength.
l2: Symmetric l2 regularization strength.
num_inner_iterations: Number of iterations per mini-batch.
container: Name of the Container that stores data across invocations of this
  Kernel. Together with SolverUUID form an isolation unit for this solver.
solver_uuid: Universally Unique Identifier for this solver.
sparse_features_indices: a list of matrices with two columns that contain
  example_indices, and feature_indices.
sparse_features_values: a list of vectors which contains feature value
  associated with each feature group.
dense_features: a list of vectors which contains the dense feature values.
example_weights: a vector which contains the weight associated with each
  example.
example_labels: a vector which contains the label/target associated with each
  example.
example_ids: a vector which contains the unique identifier associated with each
  example.
sparse_weights: a list of vectors where each value is the weight associated with
  a feature group.
dense_weights: a list of vectors where the value is the weight associated with
  a dense feature group.
)doc");

REGISTER_OP("SdcaShrinkL1")
    .Attr("num_sparse_features: int >= 0")
    .Attr("num_dense_features: int >= 0")
    .Attr("l1: float >= 0")
    .Attr("l2: float >= 1")
    .Input("sparse_weights: Ref(num_sparse_features * float)")
    .Input("dense_weights: Ref(num_dense_features * float)")
    .Doc(R"doc(
Applies L1 regularization shrink step on the parameters.

num_sparse_features: Number of sparse feature groups to train on.
num_dense_features: Number of dense feature groups to train on.
l1: Symmetric l1 regularization strength.
l2: Symmetric l2 regularization strength.
sparse_weights: a list of vectors where each value is the weight associated with
  a feature group.
dense_weights: a list of vectors where the value is the weight associated with
  a dense feature group.
)doc");

REGISTER_OP("SdcaTrainingStats")
    .Attr("container: string")
    .Attr("solver_uuid: string")
    .Output("primal_loss: float64")
    .Output("dual_loss: float64")
    .Output("example_weights: float64")
    .Doc(R"doc(
Computes statistics over all examples seen by the optimizer.

container: Name of the Container that stores data across invocations of this
  Kernel. Together with SolverUUID form an isolation unit for this solver.
solver_uuid: Universally Unique Identifier for this solver.
primal_loss: total primal loss of all examples seen by the optimizer.
dual_loss: total dual loss of all examples seen by the optimizer.
example_weights: total example weights of all examples seen by the optimizer
  (guaranteed to be positive; otherwise returns FAILED_PRECONDITION as it
   probably indicates a bug in the training data).
)doc");

}  // namespace tensorflow
