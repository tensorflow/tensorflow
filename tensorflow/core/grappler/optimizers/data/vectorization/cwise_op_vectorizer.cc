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
Status ExpandDimsForBroadcast(VectorizerInput* inputs, Graph* g) {
  Status status;
  Scope parent = NewInternalScope(g, &status, nullptr);
  Scope scope = parent.NewSubScope(kExpandDimsPrefix);

  // TODO(rachelim): We can potentially get rid of all these ops if shapes are
  // known statically

  // Get the stacked rank of each input
  auto get_stacked_rank = [&scope](const WrappedTensor& input) {
    Output rank = ops::Rank(scope, Output(input.node, input.output_index));

    if (!input.stacked) {
      // If the input is unstacked, add 1
      rank = ops::Add(scope, rank, ops::Const(scope, 1));
    }

    return rank;
  };

  Output rank_0 = get_stacked_rank(inputs->at(0));
  Output rank_1 = get_stacked_rank(inputs->at(1));

  Output max_rank = ops::Maximum(scope, rank_0, rank_1);

  // For all inputs that are stacked, expand dimensions after dim 0.
  auto expand_dims_if_unstacked =
      [&scope, &max_rank](const WrappedTensor& tensor, const Output& rank) {
        if (!tensor.stacked)
          return WrappedTensor(tensor.node, tensor.output_index, false);

        Output input(tensor.node, tensor.output_index);

        Output rank_diff = ops::Sub(scope, max_rank, rank);

        // [1] * rank_diff
        Output ones = ops::Fill(
            scope, ops::ExpandDims(scope, rank_diff, ops::Const(scope, 0)),
            ops::Const(scope, 1));

        Output shape = ops::Shape(scope, input);

        Output const_vec_1 = ops::Const(scope, {1});
        // shape[:1]
        Output concat_pre = ops::StridedSlice(
            scope, shape, const_vec_1, const_vec_1, const_vec_1,
            ops::StridedSlice::Attrs().BeginMask(1));

        // shape[1:]
        Output concat_post = ops::StridedSlice(
            scope, shape, const_vec_1, const_vec_1, const_vec_1,
            ops::StridedSlice::Attrs().EndMask(1));

        // tf.concat([shape[:1], ones, shape[1:]], 0)
        Output new_shape = ops::Concat(scope, {concat_pre, ones, concat_post},
                                       ops::Const(scope, 0));

        Output reshaped = ops::Reshape(scope, input, new_shape);

        return WrappedTensor(reshaped.node(), 0, true);
      };

  *inputs = VectorizerInput({expand_dims_if_unstacked(inputs->at(0), rank_0),
                             expand_dims_if_unstacked(inputs->at(1), rank_1)});
  return Status::OK();
}

// Vectorization helper for component-wise ops. Since these operations act
// component-wise, the vectorized op is the same as the original.
Status CwiseVectorizeHelper(const Node& node, Graph* outer_scope,
                            VectorizerInput&& inputs,
                            VectorizerOutput* outputs) {
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

class UnaryCwiseOpVectorizer : public Vectorizer {
 public:
  Status Vectorize(const Node& node, Graph* outer_scope,
                   VectorizerInput&& inputs,
                   VectorizerOutput* outputs) override {
    if (inputs.size() != 1) {
      return errors::Internal("Failed to vectorize ", node.type_string(),
                              ". The op should have 1 input, but has ",
                              inputs.size());
    }

    return CwiseVectorizeHelper(node, outer_scope, std::move(inputs), outputs);
  }
};

class BinaryCwiseOpVectorizer : public Vectorizer {
 public:
  Status Vectorize(const Node& node, Graph* outer_scope,
                   VectorizerInput&& inputs,
                   VectorizerOutput* outputs) override {
    if (inputs.size() != 2) {
      return errors::Internal("Failed to vectorize ", node.type_string(),
                              ". The op should have 2 input, but has ",
                              inputs.size());
    }
    // Binary ops support broadcasting
    TF_RETURN_IF_ERROR(ExpandDimsForBroadcast(&inputs, outer_scope));

    return CwiseVectorizeHelper(node, outer_scope, std::move(inputs), outputs);
  }
};

// Bitwise unary
REGISTER_VECTORIZER("Invert", UnaryCwiseOpVectorizer);

// Logical unary
REGISTER_VECTORIZER("LogicalNot", UnaryCwiseOpVectorizer);

// Complex unary
REGISTER_VECTORIZER("Angle", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("ComplexAbs", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Conj", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Imag", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Real", UnaryCwiseOpVectorizer);

// Real unary
REGISTER_VECTORIZER("Abs", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Acos", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Acosh", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Asin", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Asinh", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Atan", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Atanh", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("BesselI0e", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("BesselI1e", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Ceil", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Cos", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Cosh", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Digamma", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Elu", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Erf", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Erfc", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Exp", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Expm1", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Floor", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Inv", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("IsFinite", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("IsInf", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Lgamma", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Log", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Log1p", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Neg", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Reciprocal", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Relu", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Relu6", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Rint", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Round", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Rsqrt", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Selu", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Sigmoid", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Sign", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Sin", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Sinh", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Softplus", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Softsign", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Sqrt", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Square", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Tanh", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Tan", UnaryCwiseOpVectorizer);

// Miscellaneous unary
REGISTER_VECTORIZER("Cast", UnaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Identity", UnaryCwiseOpVectorizer);

// Bitwise binary
REGISTER_VECTORIZER("BitwiseAnd", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("BitwiseOr", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("BitwiseXor", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("LeftShift", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("RightShift", BinaryCwiseOpVectorizer);

// Logical binary
REGISTER_VECTORIZER("LogicalAnd", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("LogicalOr", BinaryCwiseOpVectorizer);

// Real binary
REGISTER_VECTORIZER("Add", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("AddV2", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Atan2", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Complex", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Div", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("DivNoNan", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Equal", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("FloorDiv", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("FloorMod", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Greater", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("GreaterEqual", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Igamma", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Igammac", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("IgammaGradA", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Less", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("LessEqual", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Maximum", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Minimum", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Mod", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Mul", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("NotEqual", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Polygamma", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Pow", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("RealDiv", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("SquaredDifference", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Sub", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("TruncateDiv", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("TruncateMod", BinaryCwiseOpVectorizer);
REGISTER_VECTORIZER("Zeta", BinaryCwiseOpVectorizer);
}  // namespace
}  // namespace grappler
}  // namespace tensorflow
