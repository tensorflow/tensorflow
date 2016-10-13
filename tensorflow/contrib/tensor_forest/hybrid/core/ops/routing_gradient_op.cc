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

#include "tensorflow/contrib/tensor_forest/core/ops/tree_utils.h"
#include "tensorflow/contrib/tensor_forest/hybrid/core/ops/utils.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/top_n.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

using tensorforest::LeftProbability;

// This op computes the derivative of the routing loss with respect to each
// decision node.
REGISTER_OP("RoutingGradient")
    .Attr("max_nodes: int")
    .Input("input_data: float")
    .Input("tree_parameters: float")
    .Input("tree_biases: float")
    .Input("routes: float")
    .Output("routing_gradient: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input, params;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &input));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &params));

      c->set_output(0, c->Matrix(c->Dim(input, 0), c->Dim(params, 0)));
      return Status::OK();
    })
    .Doc(R"doc(
  Computes the derivative of the routing loss with respect to each decision
  node.

  max_nodes: The number of nodes in the tree.

  tree_parameters: `tree_parameters[i]` gives the weight of
   the logistic regression model that translates from node features to
   probabilities.
  tree_biases: `tree_biases[i]` gives the bias of the logistic
   regression model that translates from node features to
   probabilities.
  routes: The routes computed by routing_function_op.

  routing_gradient: `routing_gradient` provides du / df, where u is the routing
   function and f is the (vector of) decision functions.  A decision function
   f_i computes the routing decision at node i.

   f_i is parameterized by t_i (parameters) and b_i (bias) and takes data x as
   input.  This op is called in training_ops.py to compute du / df, and we use
   that to compute

     du / dx = du / df * df / dx,
     du / dt = du / df * df / dt, and
     du / db = du / df * df / db.
)doc");

class RoutingGradient : public OpKernel {
 public:
  explicit RoutingGradient(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("max_nodes", &max_nodes_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_data = context->input(0);
    const Tensor& tree_parameters_tensor = context->input(1);
    const Tensor& tree_biases_tensor = context->input(2);
    const Tensor& routing_tensor = context->input(3);

    // TODO(atwoodj): Add dimension checks.

    const int32 num_data = static_cast<int32>(input_data.shape().dim_size(0));
    const int32 num_features =
        static_cast<int32>(input_data.shape().dim_size(1));

    Tensor* output = nullptr;
    TensorShape output_shape;
    output_shape.AddDim(num_data);
    output_shape.AddDim(max_nodes_);

    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    auto out = output->tensor<float, 2>();
    const auto tree_biases = tree_biases_tensor.tensor<float, 1>();
    const auto routes = routing_tensor.tensor<float, 2>();

    // A derivation of the gradient can be found at go/routingderivation.
    for (int i = 0; i < num_data; i++) {
      const Tensor point = input_data.Slice(i, i + 1);

      // Traverses the tree from the bottom up.
      for (int j = max_nodes_ - 1; j >= 0; j--) {
        // j is a leaf node
        if (j >= max_nodes_ / 2) {
          out(i, j) = routes(i, j);
        } else {  // j is not a leaf node
          int32 left_child = 2 * j + 1;
          int32 right_child = left_child + 1;
          float left_prob =
              LeftProbability(point, tree_parameters_tensor.Slice(j, j + 1),
                              tree_biases(j), num_features);

          float right_prob = 1 - left_prob;

          out(i, j) = (right_prob * routes(i, left_child) +
                       left_prob * routes(i, right_child));
        }
      }
    }
  }

 private:
  int32 max_nodes_;
};

REGISTER_KERNEL_BUILDER(Name("RoutingGradient").Device(DEVICE_CPU),
                        RoutingGradient);
}  // namespace tensorflow
