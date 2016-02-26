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
    .Attr("LossType: {'logistic_loss'}")
    .Attr("NumSparseFeatures: int >= 0")
    .Attr("NumDenseFeatures: int >= 0")
    .Attr("L1: float >= 0")
    .Attr("L2: float >= 0")
    .Attr("DualityGapThreshold: float = 0.01")
    .Attr("Container: string")
    .Attr("SolverUUID: string")
    .Input("sparse_features_indices: NumSparseFeatures * int64")
    .Input("sparse_features_values: NumSparseFeatures * float")
    .Input("dense_features: NumDenseFeatures * float")
    .Input("example_weights: float")
    .Input("example_labels: float")
    .Input("sparse_weights: Ref(NumSparseFeatures * float)")
    .Input("dense_weights: Ref(NumDenseFeatures * float)")
    .Input("primal_loss: Ref(double)")
    .Doc(R"doc(
Stochastic Dual Coordinate Ascent (SDCA) optimizer for linear models with
L1 + L2 regularization. As global optimization objective is strongly-convex, the
optimizer optimizes the dual objective at each step. The optimizer applies each
update one example at a time. Examples are sampled uniformly, and the optimizer
is learning rate free and enjoys linear convergence rate.

Proximal Stochastic Dual Coordinate Ascent, Shalev-Shwartz, Shai; Zhang, Tong.
2012arXiv1211.2717S: http://arxiv.org/pdf/1211.2717v1.pdf

LossType: Type of the primal loss. Only logistic_loss is supported.
NumSparseFeatures: Number of sparse feature groups to train on.
NumDenseFeatures: Number of dense feature groups to train on.
L1: Per example symmetric l1 regularization strength.
L2: Per example symmetric l2 regularization strength.
DualityGapThreshold: Gap threshold at which we should stop training.
Container: Name of the Container that stores data across invocations of this
  Kernel. Together with SolverUUID form an isolation unit for this solver.
SolverUUID: Universally Unique Identifier for this solver.
sparse_features_indices: a list of matrices with two columns that contain
  example_indices, and feature_indices.
sparse_features_values: a list of vectors which contains feature value
  associated with each feature group.
dense_features: a list of vectors which contains the dense feature values.
example_weights: a vector which contains the example weight associated with
  each example.
example_labels: a vector which contains the example label/target asscociated
  with each example.
sparse_weights: a list of vectors where each value is the weight associated with
  a feature index.
dense_weights: a list of vectors where the value is the weight associated with
  a dense feature group.
)doc");

}  // namespace tensorflow
