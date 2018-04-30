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
#include <random>
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
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
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
REGISTER_OP("StochasticHardRoutingFunction")
    .Attr("tree_depth: int")
    .Attr("random_seed: int")
    .Input("input_data: float")
    .Input("tree_parameters: float")
    .Input("tree_biases: float")
    .Output("path_probability: float")
    .Output("path: int32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &input));
      int64 tree_depth;
      TF_RETURN_IF_ERROR(c->GetAttr("tree_depth", &tree_depth));

      auto out = c->Matrix(c->Dim(input, 0), tree_depth);
      c->set_output(0, out);
      c->set_output(1, out);
      return Status::OK();
    })
    .Doc(R"doc(
  Samples a path for each instance in `input_data` and returns the
  probability of the path and the path taken.

  tree_depth: The depth of the decision tree.
  random_seed: The base random seed.

  input_data: The training batch's features as a 2-d tensor; `input_data[i][j]`
   gives the j-th feature of the i-th input.
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
)doc");

class StochasticHardRoutingFunction : public OpKernel {
 public:
  explicit StochasticHardRoutingFunction(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("tree_depth", &tree_depth_));
    OP_REQUIRES_OK(context, context->GetAttr("random_seed", &random_seed_));
    single_rand_ = std::unique_ptr<random::PhiloxRandom>(
        new random::PhiloxRandom(random_seed_));
    rng_ = std::unique_ptr<random::SimplePhilox>(
        new random::SimplePhilox(single_rand_.get()));
  }

  void Compute(OpKernelContext* context) override {
    VLOG(1) << "stochastic routing start";
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
    const int32 num_nodes =
        static_cast<int32>(tree_parameters_tensor.shape().dim_size(0));

    Tensor* output_probability = nullptr;
    TensorShape output_probability_shape;
    output_probability_shape.AddDim(num_data);
    output_probability_shape.AddDim(tree_depth_);

    Tensor* output_path = nullptr;
    TensorShape output_path_shape;
    output_path_shape.AddDim(num_data);
    output_path_shape.AddDim(tree_depth_);

    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_probability_shape,
                                            &output_probability));
    OP_REQUIRES_OK(
        context, context->allocate_output(1, output_path_shape, &output_path));

    auto out_probability = output_probability->tensor<float, 2>();
    auto out_path = output_path->tensor<int32, 2>();
    const auto tree_biases = tree_biases_tensor.tensor<float, 1>();

    // Stochastically traverse the tree to a leaf.

    for (int i = 0; i < num_data; i++) {
      const Tensor point = input_data.Slice(i, i + 1);

      int32 node = 0;
      out_probability(i, 0) = 1.0;
      out_path(i, 0) = 0;
      for (int j = 0; j < tree_depth_ - 1; j++) {
        int32 left_child = 2 * node + 1;
        int32 right_child = left_child + 1;

        float left_prob =
            LeftProbability(point, tree_parameters_tensor.Slice(j, j + 1),
                            tree_biases(j), num_features);

        if (left_prob < rng_->RandFloat()) {
          CHECK_LT(i, num_data);
          CHECK_LT(j + 1, tree_depth_);
          CHECK_LT(left_child, num_nodes);

          out_probability(i, j + 1) = left_prob * out_probability(i, j);
          out_path(i, j + 1) = left_child;
          node = left_child;
        } else {
          CHECK_LT(i, num_data);
          CHECK_LT(j + 1, tree_depth_);
          CHECK_LT(right_child, num_nodes);

          out_probability(i, j + 1) = (1.0 - left_prob) * out_probability(i, j);
          out_path(i, j + 1) = right_child;
          node = right_child;
        }
      }
    }
    VLOG(1) << "stochastic routing end";
  }

 private:
  std::unique_ptr<random::PhiloxRandom> single_rand_;
  std::unique_ptr<random::SimplePhilox> rng_;
  int32 tree_depth_;
  int32 random_seed_;
};

REGISTER_KERNEL_BUILDER(
    Name("StochasticHardRoutingFunction").Device(DEVICE_CPU),
    StochasticHardRoutingFunction);
}  // namespace tensorflow
