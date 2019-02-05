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
// RoutingFunction returns the probability of reaching each leaf node
// in a soft decision tree.

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

using tensorforest::CheckTensorBounds;
using tensorforest::LeftProbability;

// The term 'routing function' is synonymous with 'the probability
// that an instance is routed to each leaf node.'  It is defined in
// 'Deep Neural Decision Forests' by Kontschieder et al.
REGISTER_OP("RoutingFunction")
    .Attr("max_nodes: int")
    .Input("input_data: float")
    .Input("tree_parameters: float")
    .Input("tree_biases: float")
    .Output("probabilities: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input, params;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &input));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &params));

      c->set_output(0, c->Matrix(c->Dim(input, 0), c->Dim(params, 0)));
      return Status::OK();
    })
    .Doc(R"doc(
  Returns the probability that each input will reach each leaf node.

  max_nodes: The number of nodes in the tree.

  input_data: The training batch's features as a 2-d tensor; `input_data[i][j]`
   gives the j-th feature of the i-th input.
  tree_parameters: `tree_parameters[i]` gives the weight of
   the logistic regression model that translates from node features to
   probabilities.
  tree_biases: `tree_biases[i]` gives the bias of the logistic
   regression model that translates from node features to
   probabilities.

  probabilities: `probabilities[i][j]` is the probability that input i
   will reach node j.
)doc");

class RoutingFunction : public OpKernel {
 public:
  explicit RoutingFunction(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("max_nodes", &max_nodes_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_data = context->input(0);
    const Tensor& tree_parameters_tensor = context->input(1);
    const Tensor& tree_biases_tensor = context->input(2);

    if (input_data.shape().dim_size(0) > 0) {
      OP_REQUIRES(
          context, input_data.shape().dims() == 2,
          errors::InvalidArgument("input_data should be two-dimensional"));
    }

    // Check tensor bounds.
    if (!CheckTensorBounds(context, input_data)) return;

    const int32 num_data = static_cast<int32>(input_data.shape().dim_size(0));
    const int32 num_features =
        static_cast<int32>(input_data.shape().dim_size(1));

    Tensor* output_probabilities = nullptr;
    TensorShape output_shape;
    output_shape.AddDim(num_data);
    output_shape.AddDim(max_nodes_);

    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_probabilities));

    auto out_probs = output_probabilities->tensor<float, 2>();
    const auto tree_biases = tree_biases_tensor.tensor<float, 1>();

    // Iteratively compute the probability of reaching each leaf.
    for (int i = 0; i < num_data; i++) {
      const Tensor point = input_data.Slice(i, i + 1);

      out_probs(i, 0) = 1.0;

      for (int j = 0; j < max_nodes_ / 2; j++) {
        int32 left_child = 2 * j + 1;
        int32 right_child = left_child + 1;

        float prob = out_probs(i, j);
        float left_prob =
            LeftProbability(point, tree_parameters_tensor.Slice(j, j + 1),
                            tree_biases(j), num_features);

        out_probs(i, left_child) = prob * left_prob;
        out_probs(i, right_child) = prob * (1.0 - left_prob);
      }
    }
  }

 private:
  int32 max_nodes_;
};

REGISTER_KERNEL_BUILDER(Name("RoutingFunction").Device(DEVICE_CPU),
                        RoutingFunction);
}  // namespace tensorflow
