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
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/top_n.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

using tensorforest::LeftProbability;

// This op computes the derivative of the routing loss with respect to each
// decision node.
REGISTER_OP("StochasticHardRoutingGradient")
    .Attr("tree_depth: int")
    .Input("input_data: float")
    .Input("tree_parameters: float")
    .Input("tree_biases: float")
    .Input("path_probability: float")
    .Input("path: int32")
    .Output("routing_gradient: float")
    .Output("data_gradient: float")
    .Output("parameter_gradient: float")
    .Output("bias_gradient: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input, params;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &params));

      auto num_points = c->Dim(input, 0);
      auto num_features = c->Dim(input, 1);
      auto num_nodes = c->Dim(params, 0);

      c->set_output(0, c->Matrix(num_points, num_nodes));
      c->set_output(1, c->Matrix(num_nodes, num_features));
      c->set_output(2, c->MakeShape({num_points, num_nodes, num_features}));
      c->set_output(3, c->Vector(num_nodes));
      return Status::OK();
    })
    .Doc(R"doc(
  Computes the derivative of the routing loss with respect to each decision
  node.

  tree_depth: The depth of the decision tree.

  input_data: The training batch's features as a 2-d tensor; `input_data[i][j]`
   gives the j-th feature of the i-th input
  tree_parameters: `tree_parameters[i]` gives the weight of
   the logistic regression model that translates from node features to
   probabilities.
  tree_biases: `tree_biases[i]` gives the bias of the logistic
   regression model that translates from node features to
   probabilities.
  path_probability: `path_probability[i]` gives the probability of reaching each
   node in `path[i]`.
  path: `path[i][j]` gives the jth node in the path taken by the ith data
   instance.

  routing_gradient: `routing_gradient` provides du / df, where u is the routing
   function and f is the (vector of) decision functions.  A decision function
   f_i computes the routing decision at node i.
  data_gradient: `data_gradient` provides df / dx, where f is the (vector
   of) decision functions and x is a batch of data.
  parameter_gradient: `parameter_gradient` provides df / dw, where f is the
   (vector of) decision functions and w is the matrix of parameters that
   determine how instances are routed through a tree.
  bias_gradient: `bias_gradient` provides df / db, where f is the
   (vector of) decision functions and b is the vector of bias parameters that
   determine how instances are routed through a tree.

  f_i is parameterized by t_i (parameters) and b_i (bias) and takes data x as
  input.  This op is called in training_ops.py to compute du / df, and we use
  that to compute

     du / dx = du / df * df / dx,
     du / dt = du / df * df / dt, and
     du / db = du / df * df / db.
)doc");

class StochasticHardRoutingGradient : public OpKernel {
 public:
  explicit StochasticHardRoutingGradient(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("tree_depth", &tree_depth_));
  }

  void Compute(OpKernelContext* context) override {
    VLOG(1) << "stochastic gradient start";
    const Tensor& input_data = context->input(0);
    const Tensor& tree_parameters_tensor = context->input(1);
    const Tensor& tree_biases_tensor = context->input(2);

    const Tensor& path_probability_tensor = context->input(3);
    const Tensor& path_tensor = context->input(4);

    const int32 num_data = static_cast<int32>(input_data.shape().dim_size(0));
    const int32 num_features =
        static_cast<int32>(input_data.shape().dim_size(1));
    const int32 num_nodes =
        static_cast<int32>(tree_parameters_tensor.shape().dim_size(0));

    Tensor* output_routing = nullptr;
    TensorShape output_routing_shape;
    output_routing_shape.AddDim(num_data);
    output_routing_shape.AddDim(num_nodes);

    Tensor* output_data = nullptr;
    TensorShape output_data_shape;
    output_data_shape.AddDim(num_nodes);
    output_data_shape.AddDim(num_features);

    Tensor* output_parameters = nullptr;
    TensorShape output_parameters_shape;
    output_parameters_shape.AddDim(num_data);
    output_parameters_shape.AddDim(num_nodes);
    output_parameters_shape.AddDim(num_features);

    Tensor* output_bias = nullptr;
    TensorShape output_bias_shape;
    output_bias_shape.AddDim(num_data);

    OP_REQUIRES_OK(context, context->allocate_output(0, output_routing_shape,
                                                     &output_routing));
    OP_REQUIRES_OK(
        context, context->allocate_output(1, output_data_shape, &output_data));
    OP_REQUIRES_OK(context, context->allocate_output(2, output_parameters_shape,
                                                     &output_parameters));
    OP_REQUIRES_OK(
        context, context->allocate_output(3, output_bias_shape, &output_bias));

    tensorforest::Initialize(*output_routing, 0.0);
    tensorforest::Initialize(*output_data, 0.0);
    tensorforest::Initialize(*output_parameters, 0.0);
    tensorforest::Initialize(*output_bias, 0.0);

    auto out_routing = output_routing->tensor<float, 2>();
    auto out_data = output_data->tensor<float, 2>();
    auto out_parameters = output_parameters->tensor<float, 3>();
    auto out_bias = output_bias->tensor<float, 1>();

    const auto data = input_data.tensor<float, 2>();
    const auto tree_parameters = tree_parameters_tensor.tensor<float, 2>();
    const auto tree_biases = tree_biases_tensor.tensor<float, 1>();
    const auto path_probability = path_probability_tensor.tensor<float, 2>();
    const auto path = path_tensor.tensor<int32, 2>();

    for (int i = 0; i < num_data; i++) {
      const Tensor point = input_data.Slice(i, i + 1);

      // Traverses the tree from the bottom up.
      for (int j = tree_depth_ - 1; j > -1; j--) {
        int32 node = path(i, j);

        CHECK_LT(node, num_nodes);
        CHECK_GT(node, -1);

        // Compute data, parameter, and bias gradients.
        // TODO(atwoodj): Should these be normalized?  Loss looks pretty large.
        for (int k = 0; k < num_features; k++) {
          out_data(node, k) = tree_parameters(node, k);
          out_parameters(i, node, k) = out_parameters(i, node, k) + data(i, k);
        }
        out_bias(node) = out_bias(node) + 1.0;

        // Compute decision gradient.
        // node is a leaf
        if (node >= num_nodes / 2) {
          CHECK_LT(node, num_nodes);
          out_routing(i, node) = path_probability(i, j);
        } else {  // node is not a leaf
          int32 left_child = 2 * j + 1;

          float left_prob =
              LeftProbability(point, tree_parameters_tensor.Slice(j, j + 1),
                              tree_biases(j), num_features);

          float right_prob = 1 - left_prob;

          CHECK_GT(j - 1, -1);
          if (path(i, j - 1) == left_child) {
            CHECK_LT(node, num_nodes);
            out_routing(i, node) = right_prob * path_probability(i, j - 1);
          } else {
            CHECK_LT(node, num_nodes);
            out_routing(i, node) = left_prob * path_probability(i, j - 1);
          }
        }
      }
    }
    VLOG(1) << "stochastic gradient end";
  }

 private:
  int32 tree_depth_;
};

REGISTER_KERNEL_BUILDER(
    Name("StochasticHardRoutingGradient").Device(DEVICE_CPU),
    StochasticHardRoutingGradient);
}  // namespace tensorflow
