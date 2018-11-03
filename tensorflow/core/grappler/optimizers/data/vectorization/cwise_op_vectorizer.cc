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

// Vectorizer for component-wise ops. Since these operations act component-wise,
// the vectorized op is the same as the original, with additional
// instrumentation to support correct broadcasting for binary ops.
class CwiseOpVectorizer : public Vectorizer {
 public:
  Status Vectorize(const Node& node, Graph* outer_scope,
                   std::vector<WrappedTensor>&& inputs,
                   std::vector<WrappedTensor>* outputs) override {
    if (inputs.size() > 1) {
      // Binary ops support broadcasting
      TF_RETURN_IF_ERROR(ExpandDimsForBroadcast(&inputs, outer_scope));
    }

    // Add new node with the same op type and attrs as the original node
    Node* new_node;
    auto node_builder = NodeBuilder(strings::StrCat("vectorized/", node.name()),
                                    node.type_string());
    for (const auto& input : inputs) {
      node_builder = node_builder.Input(input.node, input.output_index);
    }
    for (const auto& attr_slice : node.attrs()) {
      node_builder = node_builder.Attr(attr_slice.first, attr_slice.second);
    }
    TF_RETURN_IF_ERROR(node_builder.Finalize(outer_scope, &new_node));

    // Add output mappings
    outputs->push_back({new_node, 0, true});
    return Status::OK();
  }
};

// Bitwise unary
REGISTER_VECTORIZER("Invert", CwiseOpVectorizer);

// Logical unary
REGISTER_VECTORIZER("LogicalNot", CwiseOpVectorizer);

// Complex unary
REGISTER_VECTORIZER("Angle", CwiseOpVectorizer);
REGISTER_VECTORIZER("ComplexAbs", CwiseOpVectorizer);
REGISTER_VECTORIZER("Conj", CwiseOpVectorizer);
REGISTER_VECTORIZER("Imag", CwiseOpVectorizer);
REGISTER_VECTORIZER("Real", CwiseOpVectorizer);

// Real unary
REGISTER_VECTORIZER("Abs", CwiseOpVectorizer);
REGISTER_VECTORIZER("Acos", CwiseOpVectorizer);
REGISTER_VECTORIZER("Acosh", CwiseOpVectorizer);
REGISTER_VECTORIZER("Asin", CwiseOpVectorizer);
REGISTER_VECTORIZER("Asinh", CwiseOpVectorizer);
REGISTER_VECTORIZER("Atan", CwiseOpVectorizer);
REGISTER_VECTORIZER("Atanh", CwiseOpVectorizer);
REGISTER_VECTORIZER("BesselI0e", CwiseOpVectorizer);
REGISTER_VECTORIZER("BesselI1e", CwiseOpVectorizer);
REGISTER_VECTORIZER("Ceil", CwiseOpVectorizer);
REGISTER_VECTORIZER("Cos", CwiseOpVectorizer);
REGISTER_VECTORIZER("Cosh", CwiseOpVectorizer);
REGISTER_VECTORIZER("Digamma", CwiseOpVectorizer);
REGISTER_VECTORIZER("Elu", CwiseOpVectorizer);
REGISTER_VECTORIZER("Erf", CwiseOpVectorizer);
REGISTER_VECTORIZER("Erfc", CwiseOpVectorizer);
REGISTER_VECTORIZER("Exp", CwiseOpVectorizer);
REGISTER_VECTORIZER("Expm1", CwiseOpVectorizer);
REGISTER_VECTORIZER("Floor", CwiseOpVectorizer);
REGISTER_VECTORIZER("Inv", CwiseOpVectorizer);
REGISTER_VECTORIZER("IsFinite", CwiseOpVectorizer);
REGISTER_VECTORIZER("IsInf", CwiseOpVectorizer);
REGISTER_VECTORIZER("Lgamma", CwiseOpVectorizer);
REGISTER_VECTORIZER("Log", CwiseOpVectorizer);
REGISTER_VECTORIZER("Log1p", CwiseOpVectorizer);
REGISTER_VECTORIZER("Neg", CwiseOpVectorizer);
REGISTER_VECTORIZER("Reciprocal", CwiseOpVectorizer);
REGISTER_VECTORIZER("Relu", CwiseOpVectorizer);
REGISTER_VECTORIZER("Relu6", CwiseOpVectorizer);
REGISTER_VECTORIZER("Rint", CwiseOpVectorizer);
REGISTER_VECTORIZER("Round", CwiseOpVectorizer);
REGISTER_VECTORIZER("Rsqrt", CwiseOpVectorizer);
REGISTER_VECTORIZER("Selu", CwiseOpVectorizer);
REGISTER_VECTORIZER("Sigmoid", CwiseOpVectorizer);
REGISTER_VECTORIZER("Sign", CwiseOpVectorizer);
REGISTER_VECTORIZER("Sin", CwiseOpVectorizer);
REGISTER_VECTORIZER("Sinh", CwiseOpVectorizer);
REGISTER_VECTORIZER("Softplus", CwiseOpVectorizer);
REGISTER_VECTORIZER("Softsign", CwiseOpVectorizer);
REGISTER_VECTORIZER("Sqrt", CwiseOpVectorizer);
REGISTER_VECTORIZER("Square", CwiseOpVectorizer);
REGISTER_VECTORIZER("Tanh", CwiseOpVectorizer);
REGISTER_VECTORIZER("Tan", CwiseOpVectorizer);

// Bitwise binary
REGISTER_VECTORIZER("BitwiseAnd", CwiseOpVectorizer);
REGISTER_VECTORIZER("BitwiseOr", CwiseOpVectorizer);
REGISTER_VECTORIZER("BitwiseXor", CwiseOpVectorizer);
REGISTER_VECTORIZER("LeftShift", CwiseOpVectorizer);
REGISTER_VECTORIZER("RightShift", CwiseOpVectorizer);

// Logical binary
REGISTER_VECTORIZER("LogicalAnd", CwiseOpVectorizer);
REGISTER_VECTORIZER("LogicalOr", CwiseOpVectorizer);

// Real binary
REGISTER_VECTORIZER("Add", CwiseOpVectorizer);
REGISTER_VECTORIZER("AddV2", CwiseOpVectorizer);
REGISTER_VECTORIZER("Atan2", CwiseOpVectorizer);
REGISTER_VECTORIZER("Complex", CwiseOpVectorizer);
REGISTER_VECTORIZER("Div", CwiseOpVectorizer);
REGISTER_VECTORIZER("DivNoNan", CwiseOpVectorizer);
REGISTER_VECTORIZER("Equal", CwiseOpVectorizer);
REGISTER_VECTORIZER("FloorDiv", CwiseOpVectorizer);
REGISTER_VECTORIZER("FloorMod", CwiseOpVectorizer);
REGISTER_VECTORIZER("Greater", CwiseOpVectorizer);
REGISTER_VECTORIZER("GreaterEqual", CwiseOpVectorizer);
REGISTER_VECTORIZER("Igamma", CwiseOpVectorizer);
REGISTER_VECTORIZER("Igammac", CwiseOpVectorizer);
REGISTER_VECTORIZER("IgammaGradA", CwiseOpVectorizer);
REGISTER_VECTORIZER("Less", CwiseOpVectorizer);
REGISTER_VECTORIZER("LessEqual", CwiseOpVectorizer);
REGISTER_VECTORIZER("Maximum", CwiseOpVectorizer);
REGISTER_VECTORIZER("Minimum", CwiseOpVectorizer);
REGISTER_VECTORIZER("Mod", CwiseOpVectorizer);
REGISTER_VECTORIZER("Mul", CwiseOpVectorizer);
REGISTER_VECTORIZER("NotEqual", CwiseOpVectorizer);
REGISTER_VECTORIZER("Polygamma", CwiseOpVectorizer);
REGISTER_VECTORIZER("Pow", CwiseOpVectorizer);
REGISTER_VECTORIZER("RealDiv", CwiseOpVectorizer);
REGISTER_VECTORIZER("SquaredDifference", CwiseOpVectorizer);
REGISTER_VECTORIZER("Sub", CwiseOpVectorizer);
REGISTER_VECTORIZER("TruncateDiv", CwiseOpVectorizer);
REGISTER_VECTORIZER("TruncateMod", CwiseOpVectorizer);
REGISTER_VECTORIZER("Zeta", CwiseOpVectorizer);
}  // namespace
}  // namespace grappler
}  // namespace tensorflow
