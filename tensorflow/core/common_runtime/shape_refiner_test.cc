/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/shape_refiner.h"

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

#define EXPECT_SHAPE(EXPECTED, M, OP, IDX)                            \
  do {                                                                \
    shape_inference::InferenceContext* ctx = M.GetContext(OP.node()); \
    EXPECT_EQ(EXPECTED, ctx->DebugString(ctx->output(IDX)));          \
  } while (0);

TEST(ShapeRefinerTest, Constant) {
  // Create a constant node and validate that adding it is successful
  // and that its shape is correct.
  Scope root = Scope::NewRootScope();
  auto c = ops::Const(root, 42.0f);
  ShapeRefiner m(OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(c.node()));

  EXPECT_SHAPE("[]", m, c, 0);
}

TEST(ShapeRefinerTest, MatMul) {
  ShapeRefiner m(OpRegistry::Global());

  Scope root = Scope::NewRootScope();
  auto a = ops::Const(root, {{1.0f}, {2.0f}});
  auto b = ops::Const(root, {{1.0f, 2.0f}});
  auto mm = ops::MatMul(root, a, b);

  TF_ASSERT_OK(m.AddNode(a.node()));
  TF_ASSERT_OK(m.AddNode(b.node()));
  TF_ASSERT_OK(m.AddNode(mm.node()));

  EXPECT_SHAPE("[2,1]", m, a, 0);
  EXPECT_SHAPE("[1,2]", m, b, 0);
  EXPECT_SHAPE("[2,2]", m, mm, 0);
}

TEST(ShapeRefinerTest, InvalidOrder) {
  ShapeRefiner m(OpRegistry::Global());
  Scope root = Scope::NewRootScope();
  auto a = ops::Const(root, {{1.0f}, {2.0f}});
  auto b = ops::Const(root, {{1.0f, 2.0f}});
  auto mm = ops::MatMul(root, a, b);

  Status s = m.AddNode(mm.node());
  ASSERT_FALSE(s.ok());
  ASSERT_EQ(
      "Input 0 ('Const') for 'MatMul' was not previously added to "
      "ShapeRefiner.",
      s.error_message());
}

TEST(ShapeRefinerTest, BadShapes) {
  ShapeRefiner m(OpRegistry::Global());
  Scope root = Scope::NewRootScope();
  auto a = ops::Const(root, {{1.0f}, {2.0f}});
  auto b = ops::Const(root, {{1.0f}, {2.0f}});
  auto mm = ops::MatMul(root, a, b);

  TF_ASSERT_OK(m.AddNode(a.node()));
  TF_ASSERT_OK(m.AddNode(b.node()));
  // The shape of the inputs are not compatible, so we should expect
  // an error.
  Status s = m.AddNode(mm.node());
  ASSERT_FALSE(s.ok());
  ASSERT_TRUE(StringPiece(s.error_message())
                  .contains("Dimensions must be equal, but are 1 and 2"));
}

TEST(ShapeRefinerTest, SetShape) {
  ShapeRefiner m(OpRegistry::Global());

  Scope root = Scope::NewRootScope();
  auto a = ops::Placeholder(root, DT_FLOAT);

  TF_ASSERT_OK(m.AddNode(a.node()));

  auto ic = m.GetContext(a.node());
  ASSERT_NE(nullptr, ic);
  shape_inference::ShapeHandle h = ic->MakeShape({2, ic->UnknownDim()});
  TF_ASSERT_OK(m.SetShape(a.node(), 0, h));
  EXPECT_SHAPE("[2,?]", m, a, 0);

  // Check that shapes are merged with the existing shape.
  shape_inference::ShapeHandle h2 = ic->MakeShape({ic->UnknownDim(), 2});
  TF_ASSERT_OK(m.SetShape(a.node(), 0, h2));
  EXPECT_SHAPE("[2,2]", m, a, 0);

  // Out of range.
  ASSERT_FALSE(m.SetShape(a.node(), 1, h).ok());
  ASSERT_FALSE(m.SetShape(a.node(), -1, h).ok());

  auto b = ops::Const(root, {{1.0f}, {2.0f}});
  // Forget to add node first.
  ASSERT_FALSE(m.SetShape(b.node(), 0, h).ok());

  // Set an incompatible shape (3 vs 2)
  h = ic->MakeShape({3, ic->UnknownDim()});
  ASSERT_FALSE(m.SetShape(a.node(), 0, h).ok());
}

TEST(ShapeRefinerTest, PropagateConstants) {
  // Reduction dimension is a variable, so we don't know its value.
  // So the output shape value is unknown (though its rank is known).
  {
    Scope root = Scope::NewRootScope();
    // 3x2 input
    auto input = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    // Reduce along unspecified dimension
    auto dim = ops::Variable(root, {}, DT_INT32);

    auto am = ops::ArgMax(root, input, dim);
    ShapeRefiner m(OpRegistry::Global());
    TF_ASSERT_OK(m.AddNode(input.node()));
    TF_ASSERT_OK(m.AddNode(dim.node()));
    TF_ASSERT_OK(m.AddNode(am.node()));
    EXPECT_SHAPE("[?]", m, am, 0);
  }

  // Constant is used as dimension, which can be materialized,
  // so the shape function can be more precise about the output.
  {
    Scope root = Scope::NewRootScope();
    // 3x2 input
    auto input = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    // Reduce along 2nd dimension
    auto dim = ops::Const(root, 1);

    auto am = ops::ArgMax(root, input, dim);
    ShapeRefiner m(OpRegistry::Global());
    TF_ASSERT_OK(m.AddNode(input.node()));
    TF_ASSERT_OK(m.AddNode(dim.node()));
    TF_ASSERT_OK(m.AddNode(am.node()));
    EXPECT_SHAPE("[3]", m, am, 0);
  }

  // Reduce along known first dimension.
  {
    Scope root = Scope::NewRootScope();
    // 3x2 input
    auto input = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    // Reduce along 1st dimension
    auto dim = ops::Const(root, 0);

    auto am = ops::ArgMax(root, input, dim);
    ShapeRefiner m(OpRegistry::Global());
    TF_ASSERT_OK(m.AddNode(input.node()));
    TF_ASSERT_OK(m.AddNode(dim.node()));
    TF_ASSERT_OK(m.AddNode(am.node()));
    EXPECT_SHAPE("[2]", m, am, 0);
  }
}

namespace {

// An op with a shape function whose outputs depend in a complex
// way on whether input tensors are available.
REGISTER_OP("TestOp")
    .Input("a: float")
    .Input("b: float")
    .Output("o: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      if (c->input_tensor(0)) {
        if (c->input_tensor(1)) {
          c->set_output(0, c->Matrix(10, 10));
          return Status::OK();
        }
        return shape_inference::ScalarShape(c);
      }
      return shape_inference::UnknownShape(c);
    });

}  // namespace

TEST(ShapeRefinerTest, InputTensorDependencies) {
  ShapeRefiner m(OpRegistry::Global());
  Graph graph(OpRegistry::Global());
  Node* node;

  Tensor a(DT_FLOAT, TensorShape({}));
  a.scalar<float>()() = 1.0;

  Tensor b(DT_FLOAT, TensorShape({}));
  b.scalar<float>()() = 2.0;

  Node* input_a = test::graph::Constant(&graph, a);
  Node* input_b = test::graph::Constant(&graph, b);
  TF_ASSERT_OK(NodeBuilder("Test", "TestOp")
                   .Input(input_a)
                   .Input(input_b)
                   .Finalize(&graph, &node));

  TF_ASSERT_OK(m.AddNode(input_a));
  TF_ASSERT_OK(m.AddNode(input_b));
  TF_ASSERT_OK(m.AddNode(node));
  shape_inference::InferenceContext* ctx = m.GetContext(node);
  EXPECT_EQ("[10,10]", ctx->DebugString(ctx->output(0)));
}

namespace {

// An op with a shape function that looks at its input tensor
// data and makes a Shape out of it.
REGISTER_OP("ShapeData")
    .Input("a: int32")
    .Output("o: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      const Tensor* shape_data = c->input_tensor(0);
      if (shape_data == nullptr) {
        return shape_inference::UnknownShape(c);
      }

      std::vector<shape_inference::DimensionHandle> dims;
      for (int i = 0; i < shape_data->NumElements(); ++i) {
        dims.emplace_back(c->MakeDim(shape_data->flat<int32>()(i)));
      }

      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    });

}  // namespace

TEST(ShapeRefinerTest, PropagateShape) {
  Scope root = Scope::NewRootScope();
  // 3x2 input
  auto input = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});

  // Shape is a vector of 2 elements (3,2)
  auto shape = ops::Shape(root, input);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeData")
                   .Input(shape.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(input.node()));
  TF_ASSERT_OK(m.AddNode(shape.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));

  shape_inference::InferenceContext* ctx = m.GetContext(shape_data);
  EXPECT_EQ("[3,2]", ctx->DebugString(ctx->output(0)));
}

TEST(ShapeRefinerTest, PropagateSize) {
  Scope root = Scope::NewRootScope();
  // 3x2 input
  auto input = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});

  auto size = ops::Size(root, input);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeData")
                   .Input(size.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(input.node()));
  TF_ASSERT_OK(m.AddNode(size.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));

  shape_inference::InferenceContext* ctx = m.GetContext(shape_data);
  EXPECT_EQ("[6]", ctx->DebugString(ctx->output(0)));
}

TEST(ShapeRefinerTest, PropagateRank) {
  Scope root = Scope::NewRootScope();
  // 3x2 input
  auto input = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});

  auto rank = ops::Rank(root, input);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeData")
                   .Input(rank.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(input.node()));
  TF_ASSERT_OK(m.AddNode(rank.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));

  shape_inference::InferenceContext* ctx = m.GetContext(shape_data);
  EXPECT_EQ("[2]", ctx->DebugString(ctx->output(0)));
}

TEST(ShapeRefinerTest, PropagateRange) {
  Scope root = Scope::NewRootScope();
  auto begin = ops::Const(root, 1);
  auto limit = ops::Const(root, 11);
  auto delta = ops::Const(root, 3);
  auto range = ops::Range(root, begin, limit, delta);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeData")
                   .Input(range.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(begin.node()));
  TF_ASSERT_OK(m.AddNode(limit.node()));
  TF_ASSERT_OK(m.AddNode(delta.node()));
  TF_ASSERT_OK(m.AddNode(range.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));

  shape_inference::InferenceContext* ctx = m.GetContext(shape_data);
  EXPECT_EQ("[1,4,7,10]", ctx->DebugString(ctx->output(0)));
}

TEST(ShapeRefinerTest, ConstantValueTwoInputsToSameNode) {
  Scope root = Scope::NewRootScope();
  // This node is used as two inputs to 'range'.
  auto begin_and_delta = ops::Const(root, 1);
  auto limit = ops::Const(root, 4);
  auto range = ops::Range(root, begin_and_delta, limit, begin_and_delta);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeData")
                   .Input(range.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(begin_and_delta.node()));
  TF_ASSERT_OK(m.AddNode(limit.node()));
  TF_ASSERT_OK(m.AddNode(range.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));

  shape_inference::InferenceContext* ctx = m.GetContext(shape_data);
  EXPECT_EQ("[1,2,3]", ctx->DebugString(ctx->output(0)));
}

// Creates a graph where 'begin' is attempted to be visited during
// constant value evaluation after having been processed once.
TEST(ShapeRefinerTest, ConstantValueVisitNodeTwice) {
  Scope root = Scope::NewRootScope();
  auto begin = ops::Const(root, 1);
  auto limit = ops::Const(root, 8);
  auto delta = ops::Const(root, 3);

  auto d1 = ops::Add(root, begin, limit);  // 9
  auto d2 = ops::Add(root, begin, delta);  // 4
  // Visiting flimit's children will visit 'begin' before 'd1'.
  // It will then visit d1, whose child is 'begin'.  That edge still
  // must be visited.
  auto flimit = ops::Sub(root, begin, d1);  // 1-9=-8
  auto fdelta = ops::Sub(root, begin, d2);  // 1-4=-3
  auto nl = ops::Abs(root, flimit);         // 8
  auto nd = ops::Abs(root, fdelta);         // 3

  auto range = ops::Range(root, begin, nl, nd);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeData")
                   .Input(range.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(begin.node()));
  TF_ASSERT_OK(m.AddNode(limit.node()));
  TF_ASSERT_OK(m.AddNode(delta.node()));
  TF_ASSERT_OK(m.AddNode(d1.node()));
  TF_ASSERT_OK(m.AddNode(d2.node()));
  TF_ASSERT_OK(m.AddNode(flimit.node()));
  TF_ASSERT_OK(m.AddNode(fdelta.node()));
  TF_ASSERT_OK(m.AddNode(nl.node()));
  TF_ASSERT_OK(m.AddNode(nd.node()));
  TF_ASSERT_OK(m.AddNode(range.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));

  shape_inference::InferenceContext* ctx = m.GetContext(shape_data);
  EXPECT_EQ("[1,4,7]", ctx->DebugString(ctx->output(0)));
}

namespace {

Status TensorAsShapeShapeFn(shape_inference::InferenceContext* c) {
  shape_inference::ShapeHandle out;
  TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0 /* input_idx */, &out));
  c->set_output(0, out);
  return Status::OK();
}

// Register ops used by the ConstantValueAsShape* tests.

REGISTER_OP("TensorAsShapeInt32")
    .Input("a: int32")
    .Output("o: int32")
    .SetShapeFn(TensorAsShapeShapeFn);

REGISTER_OP("TensorAsShapeInt64")
    .Input("a: int64")
    .Output("o: int64")
    .SetShapeFn(TensorAsShapeShapeFn);

REGISTER_OP("NonConstScalarInt32")
    .Output("o: int32")
    .SetIsStateful()  // prevents constant folding
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("NonConstScalarInt64")
    .Output("o: int64")
    .SetIsStateful()  // prevents constant folding
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("WithEmptyVectorShape")
    .Output("o: int32")
    .SetIsStateful()  // prevents constant folding
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(0));
      return Status::OK();
    });

REGISTER_OP("WithPartialShape")
    .Output("o: int32")
    .SetIsStateful()  // prevents constant folding
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(
          0, c->MakeShape({1, shape_inference::InferenceContext::kUnknownDim, 3,
                           shape_inference::InferenceContext::kUnknownDim, 5}));
      return Status::OK();
    });

REGISTER_OP("WithPartialShape2")
    .Output("o: int32")
    .SetIsStateful()  // prevents constant folding
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(
          0,
          c->MakeShape({6, shape_inference::InferenceContext::kUnknownDim, 8}));
      return Status::OK();
    });

REGISTER_OP("WithUnknownShape")
    .Output("o: int32")
    .SetIsStateful()  // prevents constant folding
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    });

}  // namespace

TEST(ShapeRefinerTest, ConstantValueAsShape_EmptyVector) {
  Scope root = Scope::NewRootScope();
  Node* input;
  TF_ASSERT_OK(
      NodeBuilder("in", "WithEmptyVectorShape").Finalize(root.graph(), &input));
  Node* result;
  TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt32")
                   .Input(input)
                   .Finalize(root.graph(), &result));

  ShapeRefiner m(OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(input));
  TF_ASSERT_OK(m.AddNode(result));

  shape_inference::InferenceContext* ctx = m.GetContext(result);
  EXPECT_EQ("[]", ctx->DebugString(ctx->output(0)));
}

TEST(ShapeRefinerTest, ConstantValueAsShape_Shape) {
  for (int pass = 0; pass < 2; ++pass) {
    Scope root = Scope::NewRootScope();
    Node* input;
    TF_ASSERT_OK(
        NodeBuilder("in", pass == 0 ? "WithPartialShape" : "WithUnknownShape")
            .Finalize(root.graph(), &input));
    auto shape = ops::Shape(root, ops::Output(input));
    Node* result;
    TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt32")
                     .Input(shape.node())
                     .Finalize(root.graph(), &result));

    ShapeRefiner m(OpRegistry::Global());
    TF_ASSERT_OK(m.AddNode(input));
    TF_ASSERT_OK(m.AddNode(shape.node()));
    TF_ASSERT_OK(m.AddNode(result));

    shape_inference::InferenceContext* ctx = m.GetContext(result);
    if (pass == 0) {
      EXPECT_EQ("[1,?,3,?,5]", ctx->DebugString(ctx->output(0)));
    } else {
      EXPECT_EQ("?", ctx->DebugString(ctx->output(0)));
    }
  }
}

TEST(ShapeRefinerTest, ConstantValueAsShape_PackInt32) {
  Scope root = Scope::NewRootScope();
  Node* scalar_non_const;
  TF_ASSERT_OK(NodeBuilder("in", "NonConstScalarInt32")
                   .Finalize(root.graph(), &scalar_non_const));

  ops::InputList inputs{
      ops::Input(ops::Const<int32>(root, 10)),
      ops::Input(ops::Const<int32>(root, 20)),
      ops::Input(ops::Output(scalar_non_const)),
      ops::Input(ops::Const<int32>(root, 40)),
  };
  auto pack = ops::Pack(root, inputs);
  TF_ASSERT_OK(root.status());

  Node* result;
  TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt32")
                   .Input(pack.node())
                   .Finalize(root.graph(), &result));

  ShapeRefiner m(OpRegistry::Global());
  for (auto input : inputs) {
    TF_ASSERT_OK(m.AddNode(input.node()));
  }
  TF_ASSERT_OK(m.AddNode(pack.node()));
  TF_ASSERT_OK(m.AddNode(result));

  shape_inference::InferenceContext* ctx = m.GetContext(result);
  EXPECT_EQ("[10,20,?,40]", ctx->DebugString(ctx->output(0)));
}

TEST(ShapeRefinerTest, ConstantValueAsShape_PackInt64) {
  Scope root = Scope::NewRootScope();
  Node* scalar_non_const;
  TF_ASSERT_OK(NodeBuilder("in", "NonConstScalarInt64")
                   .Finalize(root.graph(), &scalar_non_const));

  ops::InputList inputs{
      ops::Input(ops::Const<int64>(root, 10LL)),
      ops::Input(ops::Const<int64>(root, 20LL)),
      ops::Input(ops::Output(scalar_non_const)),
      ops::Input(ops::Const<int64>(root, 1LL << 40)),
  };
  auto pack = ops::Pack(root, inputs);
  TF_ASSERT_OK(root.status());

  Node* result;
  TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt64")
                   .Input(pack.node())
                   .Finalize(root.graph(), &result));

  ShapeRefiner m(OpRegistry::Global());
  for (const auto& input : inputs) {
    TF_ASSERT_OK(m.AddNode(input.node()));
  }
  TF_ASSERT_OK(m.AddNode(pack.node()));
  TF_ASSERT_OK(m.AddNode(result));

  shape_inference::InferenceContext* ctx = m.GetContext(result);
  EXPECT_EQ("[10,20,?,1099511627776]", ctx->DebugString(ctx->output(0)));
}

TEST(ShapeRefinerTest, ConstantValueAsShape_PackUnknownDim) {
  Scope root = Scope::NewRootScope();

  ops::InputList inputs{
      ops::Input(ops::Const<int64>(root, 10LL)),
      ops::Input(ops::Const<int64>(root, -1LL)),
  };
  auto pack = ops::Pack(root, inputs);
  TF_ASSERT_OK(root.status());

  Node* result;
  TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt64")
                   .Input(pack.node())
                   .Finalize(root.graph(), &result));

  ShapeRefiner m(OpRegistry::Global());
  for (const auto& input : inputs) {
    TF_ASSERT_OK(m.AddNode(input.node()));
  }
  TF_ASSERT_OK(m.AddNode(pack.node()));
  TF_ASSERT_OK(m.AddNode(result));

  shape_inference::InferenceContext* ctx = m.GetContext(result);
  EXPECT_EQ("[10,?]", ctx->DebugString(ctx->output(0)));
}

TEST(ShapeRefinerTest, ConstantValueAsShape_PackInvalidInput) {
  Scope root = Scope::NewRootScope();

  // Inputs are length 2 vectors instead of scalars.
  ops::InputList inputs{
      ops::Input(ops::Const<int64>(root, {10LL, 20LL})),
      ops::Input(ops::Const<int64>(root, {10LL, 21LL})),
  };
  auto pack = ops::Pack(root, inputs);
  TF_ASSERT_OK(root.status());

  Node* result;
  TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt64")
                   .Input(pack.node())
                   .Finalize(root.graph(), &result));

  ShapeRefiner m(OpRegistry::Global());
  for (const auto& input : inputs) {
    TF_ASSERT_OK(m.AddNode(input.node()));
  }
  TF_ASSERT_OK(m.AddNode(pack.node()));
  EXPECT_TRUE(
      StringPiece(m.AddNode(result).error_message()).contains("but is rank 2"));
}

TEST(ShapeRefinerTest, ConstantValueAsShape_Concat) {
  Scope root = Scope::NewRootScope();
  Graph* g = root.graph();
  Node* partial_1;
  Node* partial_2;
  TF_ASSERT_OK(NodeBuilder("in", "WithPartialShape").Finalize(g, &partial_1));
  TF_ASSERT_OK(NodeBuilder("in", "WithPartialShape2").Finalize(g, &partial_2));
  auto const_input = ops::Const(root, {9, 10, 11});
  ops::OutputList concat_inputs{
      ops::Shape(root, ops::Output(partial_1)),
      ops::Shape(root, ops::Output(partial_2)), const_input,
  };
  auto concat_dim = ops::Const(root, 0);
  auto concat = ops::Concat(root, concat_dim, concat_inputs);
  TF_ASSERT_OK(root.status());

  Node* result;
  TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt32")
                   .Input(concat.node())
                   .Finalize(g, &result));

  ShapeRefiner m(OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(partial_1));
  TF_ASSERT_OK(m.AddNode(partial_2));
  for (const auto& o : concat_inputs) {
    TF_ASSERT_OK(m.AddNode(o.node()));
  }
  TF_ASSERT_OK(m.AddNode(concat_dim.node()));
  TF_ASSERT_OK(m.AddNode(concat.node()));
  TF_ASSERT_OK(m.AddNode(result));

  shape_inference::InferenceContext* ctx = m.GetContext(result);
  EXPECT_EQ("[1,?,3,?,5,6,?,8,9,10,11]", ctx->DebugString(ctx->output(0)));
}

TEST(ShapeRefinerTest, ConstantValueAsShape_ConcatWithUnknown) {
  Scope root = Scope::NewRootScope();
  Graph* g = root.graph();
  Node* scalar_non_const;
  TF_ASSERT_OK(NodeBuilder("in", "NonConstScalarInt32")
                   .Finalize(root.graph(), &scalar_non_const));

  Node* partial_1;
  Node* partial_2;
  Node* unknown;
  TF_ASSERT_OK(NodeBuilder("in", "WithPartialShape").Finalize(g, &partial_1));
  TF_ASSERT_OK(NodeBuilder("in", "WithPartialShape2").Finalize(g, &partial_2));
  TF_ASSERT_OK(NodeBuilder("in", "WithUnknownShape").Finalize(g, &unknown));
  ops::OutputList concat_inputs{
      ops::Shape(root, ops::Output(partial_1)),
      ops::Shape(root, ops::Output(partial_2)),
      ops::Shape(root, ops::Output(unknown)),
  };
  auto concat_dim = ops::Const(root, 0);
  auto concat = ops::Concat(root, concat_dim, concat_inputs);
  TF_ASSERT_OK(root.status());

  Node* result;
  TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt32")
                   .Input(concat.node())
                   .Finalize(g, &result));

  ShapeRefiner m(OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(partial_1));
  TF_ASSERT_OK(m.AddNode(partial_2));
  TF_ASSERT_OK(m.AddNode(unknown));
  for (const auto& o : concat_inputs) {
    TF_ASSERT_OK(m.AddNode(o.node()));
  }
  TF_ASSERT_OK(m.AddNode(concat_dim.node()));
  TF_ASSERT_OK(m.AddNode(concat.node()));
  TF_ASSERT_OK(m.AddNode(result));

  shape_inference::InferenceContext* ctx = m.GetContext(result);
  EXPECT_EQ("?", ctx->DebugString(ctx->output(0)));
}

TEST(ShapeRefinerTest, ConstantValueAsShape_ConcatInvalidDimValue) {
  Scope root = Scope::NewRootScope();
  Graph* g = root.graph();
  Node* scalar_non_const;
  TF_ASSERT_OK(NodeBuilder("in", "NonConstScalarInt32")
                   .Finalize(root.graph(), &scalar_non_const));

  Node* partial_1;
  Node* partial_2;
  TF_ASSERT_OK(NodeBuilder("in", "WithPartialShape").Finalize(g, &partial_1));
  TF_ASSERT_OK(NodeBuilder("in", "WithPartialShape2").Finalize(g, &partial_2));
  auto const_input = ops::Const(root, {9, -2, 11});
  ops::OutputList concat_inputs{
      ops::Shape(root, ops::Output(partial_1)),
      ops::Shape(root, ops::Output(partial_2)),  //
      const_input,
  };
  auto concat_dim = ops::Const(root, 0);
  auto concat = ops::Concat(root, concat_dim, concat_inputs);
  TF_ASSERT_OK(root.status());

  Node* result;
  TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt32")
                   .Input(concat.node())
                   .Finalize(g, &result));

  ShapeRefiner m(OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(partial_1));
  TF_ASSERT_OK(m.AddNode(partial_2));
  for (const auto& o : concat_inputs) {
    TF_ASSERT_OK(m.AddNode(o.node()));
  }
  TF_ASSERT_OK(m.AddNode(concat_dim.node()));
  TF_ASSERT_OK(m.AddNode(concat.node()));
  EXPECT_EQ("Invalid value in tensor used for shape: -2",
            m.AddNode(result).error_message());
}

}  // namespace
}  // namespace tensorflow
