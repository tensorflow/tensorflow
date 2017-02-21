// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/contrib/tensor_forest/hybrid/core/ops/utils.h"
#include "tensorflow/contrib/tensor_forest/kernels/tree_utils.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/top_n.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

using tensorforest::LeftProbabilityK;

REGISTER_OP("KFeatureGradient")
    .Attr("layer_num: int")
    .Attr("random_seed: int")
    .Input("input_data: float")
    .Input("tree_parameters: float")
    .Input("tree_biases: float")
    .Input("routes: float")
    .Output("routing_gradient: float")
    .Output("data_gradient: float")
    .Output("weight_gradient: float")
    .Doc(R"doc(
    Computes the derivative of the routing loss with respect to each decision
    node.  Each decision node is constrained to make a decision based on only
    k features.

    layer_num: The layer number of this tree.
    random_seed: The base random seed.

    input_data: The training batch's features as a 2-d tensor;
     `input_data[i][j]` gives the j-th feature of the i-th input.
    tree_parameters: `tree_parameters[i]` gives the weight of
     the logistic regression model that translates from node features to
     probabilities.
    tree_biases: `tree_biases[i]` gives the bias of the logistic
     regression model that translates from node features to
     probabilities.
    routes: The routes computed by routing_function_op.

    routing_gradient: `routing_gradient` provides du / df, where u is the
     routing function and f is the (vector of) decision functions.  A decision
     function f_i computes the routing decision at node i.

    data_gradient: `data_gradient` provides df / dx, where f is the (vector
     of) decision functions and x is a batch of data.

    weights_gradient: `weights_gradient` provides df / dw, where f is the
     (vector of) decision functions and w is the matrix of parameters that
     determine how instances are routed through a tree.

    f_i, the decision function at node i, is parameterized by t_i (parameters)
    and b_i (bias) and takes data x as input.  This op is called in
    training_ops.py to compute du / df, and we use that to compute

    du / dx = du / df * df / dx,
    du / dt = du / df * df / dt, and
    du / db = du / df * df / db.
)doc");

class KFeatureGradient : public OpKernel {
 public:
  explicit KFeatureGradient(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("layer_num",
                                             &layer_num_));
    OP_REQUIRES_OK(context, context->GetAttr("random_seed",
                                             &random_seed_));
  }

  void Compute(OpKernelContext* context) override {
    // Gather input.
    const Tensor& input_data_tensor = context->input(0);
    const Tensor& tree_parameters_tensor = context->input(1);
    const Tensor& tree_biases_tensor = context->input(2);
    const Tensor& routing_tensor = context->input(3);

    // Extract dimensions from input tensors.
    const int32 num_data = static_cast<int32>(
        input_data_tensor.shape().dim_size(0));
    const int32 num_features = static_cast<int32>(
        input_data_tensor.shape().dim_size(1));
    const int32 num_nodes = static_cast<int32>(
        tree_parameters_tensor.shape().dim_size(0));
    const int32 num_features_per_node = static_cast<int32>(
        tree_parameters_tensor.shape().dim_size(1));

    // Construct output tensors.
    Tensor* out_routes = nullptr;
    TensorShape out_routes_shape;
    out_routes_shape.AddDim(num_data);
    out_routes_shape.AddDim(num_nodes);

    Tensor* out_data = nullptr;
    TensorShape out_data_shape;
    out_data_shape.AddDim(num_nodes);
    out_data_shape.AddDim(num_features);

    Tensor* out_weights = nullptr;
    TensorShape out_weights_shape;
    out_weights_shape.AddDim(num_data);
    out_weights_shape.AddDim(num_nodes);
    out_weights_shape.AddDim(num_features_per_node);

    OP_REQUIRES_OK(context, context->allocate_output(
        0, out_routes_shape, &out_routes));
    OP_REQUIRES_OK(context, context->allocate_output(
        1, out_data_shape, &out_data));
    OP_REQUIRES_OK(context, context->allocate_output(
        2, out_weights_shape, &out_weights));

    tensorforest::Initialize(*out_data, 0.0f);

    // Compute output.
    const auto input_data = input_data_tensor.tensor<float, 2>();
    const auto tree_parameters = tree_parameters_tensor.tensor<float, 2>();
    const auto tree_biases = tree_biases_tensor.tensor<float, 1>();
    const auto routes = routing_tensor.tensor<float, 2>();

    auto routes_grad = out_routes->tensor<float, 2>();
    auto data_grad = out_data->tensor<float, 2>();
    auto weights_grad = out_weights->tensor<float, 3>();

    std::vector<int32> feature_set;
    for (int i = 0; i < num_data; i++) {
      const Tensor point = input_data_tensor.Slice(i, i+1);
      feature_set.clear();

      // Traverse the tree from the bottom up.
      for (int j = num_nodes - 1; j >= 0; j--) {
        tensorforest::GetFeatureSet(
            layer_num_,
            j,
            random_seed_,
            num_features,
            num_features_per_node,
            &feature_set);

        // Compute routing gradient.
        // j is a leaf node.
        if (j >= num_nodes / 2) {
          routes_grad(i, j) = routes(i, j);
        } else {  // j is not a leaf node
          int32 left_child = 2 * j + 1;
          int32 right_child = left_child + 1;

          float left_prob = LeftProbabilityK(
              point,
              feature_set,
              tree_parameters_tensor.Slice(j, j+1),
              tree_biases(j),
              num_features,
              num_features_per_node);

          float right_prob = 1.0f - left_prob;

          routes_grad(i, j) = (right_prob * routes(i, left_child) +
                               left_prob * routes(i, right_child));
        }
        // Compute data and weight gradient.
        for (int k = 0; k < num_features_per_node; k++) {
          CHECK_LT(feature_set[k], num_features);
          data_grad(j, feature_set[k]) = tree_parameters(j, k);
          weights_grad(i, j, k) = input_data(i, feature_set[k]);
        }
      }
    }
  }

 private:
  int32 layer_num_;
  int32 random_seed_;
};

REGISTER_KERNEL_BUILDER(Name("KFeatureGradient").Device(DEVICE_CPU),
                        KFeatureGradient);
}  // namespace tensorflow
