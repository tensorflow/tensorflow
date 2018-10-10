/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/grappler/optimizers/data/vectorization/vectorizer_registry.h"

namespace tensorflow {
namespace grappler {

namespace {

const char* const kExpandDimsPrefix = "vectorized/expanddims/";

// Reshapes stacked inputs for broadcast. Stacked inputs have an extra leading
// dimension, which may cause automatic broadcasting rules to expand the
// input dimensions wrongly when the unstacked shapes have different ranks.
// To avoid that, we reshape stacked inputs to the maximum rank they need
// to be broadcasted to.
//
// For example, suppose we have inputs A and B, where A is a stacked tensor with
// shape [n, 5] (where n is the stack size) and B is an unstacked tensor with
// shape [12, 7, 5]. If we added them directly, tensorflow broadcasting rules
// would expand the dimensions of A to [1, n, 5], then (incorrectly) check that
// the dimensions n and 7 are compatible, and if so, create an output of shape
// [12, 7, 5]. However, correct addition of these inputs would create an output
// with shape [n, 12, 7, 5]: we need to manually expand the dimensions of A
// *after* the leading dimension, i.e. expand A to the shape [n, 1, 1, 5] before
// broadcasting.
Status ExpandDimsForBroadcast(std::vector<WrappedTensor>* inputs, Graph* g) {
  Status status;
  Scope parent = NewInternalScope(g, &status, nullptr);
  Scope s = parent.NewSubScope(kExpandDimsPrefix);

  // TODO(rachelim): We can potentially get rid of all these ops if shapes are
  // known statically

  Output const_0 = ops::Const(s, 0);
  Output const_1 = ops::Const(s, 1);

  std::vector<Output> ranks;
  ranks.reserve(inputs->size());

  // Get the stacked rank of each input
  for (const auto& input : *inputs) {
    Output rank = ops::Rank(s, Output(input.node, input.output_index));

    if (!input.stacked) {
      // If the input is unstacked, add 1
      rank = ops::Add(s, rank, const_1);
    }

    ranks.push_back(rank);
  }

  // Pack the ranks into one tensor to get the max
  Output packed_ranks = ops::Stack(s, ranks);

  Output max_rank =
      ops::Max(s, packed_ranks, const_0, ops::Max::Attrs().KeepDims(true));

  std::vector<WrappedTensor> expanded_inputs;
  expanded_inputs.reserve(inputs->size());

  // For all inputs that are stacked, expand dimensions after dim 0.
  for (size_t i = 0; i < inputs->size(); ++i) {
    if (!inputs->at(i).stacked) {
      expanded_inputs.push_back(inputs->at(i));
      continue;
    }

    Output input(inputs->at(i).node, inputs->at(i).output_index);

    // Number of dimensions to expand
    Output rank_diff = ops::Sub(s, max_rank, ranks[i]);

    // [1] * rank_diff
    Output ones = ops::Tile(s, ops::Const(s, {1}), rank_diff);

    Output const_vec_1 = ops::Const(s, {1});

    Output shape = ops::Shape(s, input);

    // shape[:1]
    Output concat_pre =
        ops::StridedSlice(s, shape, const_vec_1, const_vec_1, const_vec_1,
                          ops::StridedSlice::Attrs().BeginMask(1));

    // shape[1:]
    Output concat_post =
        ops::StridedSlice(s, shape, const_vec_1, const_vec_1, const_vec_1,
                          ops::StridedSlice::Attrs().EndMask(1));

    // tf.concat([shape[:1], ones, shape[1:]], 0)
    Output new_shape = ops::Concat(s, {concat_pre, ones, concat_post}, const_0);

    Output result = ops::Reshape(s, input, new_shape);

    expanded_inputs.push_back({result.node(), 0, true});
  }

  inputs->swap(expanded_inputs);
  return status;
}

class AddVectorizer : public Vectorizer {
 public:
  Status Vectorize(const Node& node, Graph* outer_scope,
                   std::vector<WrappedTensor>&& inputs,
                   std::vector<WrappedTensor>* outputs) override {
    if (node.num_inputs() != 2) {
      return errors::Internal("Add op should only have two inputs.");
    }

    TF_RETURN_IF_ERROR(ExpandDimsForBroadcast(&inputs, outer_scope));

    // Add new Add node with the same op and attrs as the original node
    Node* new_add_node;
    TF_RETURN_IF_ERROR(NodeBuilder("Add", "Add")
                           .Input(inputs[0].node, inputs[0].output_index)
                           .Input(inputs[1].node, inputs[1].output_index)
                           .Finalize(outer_scope, &new_add_node));

    // Add output mappings
    outputs->push_back({new_add_node, 0, true});
    return Status::OK();
  }
};

REGISTER_VECTORIZER("Add", AddVectorizer);

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
