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

#include "tensorflow/core/graph/shape_refiner.h"

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
  ShapeRefiner m;
  TF_ASSERT_OK(m.AddNode(c.node()));

  EXPECT_SHAPE("[]", m, c, 0);
}

TEST(ShapeRefinerTest, MatMul) {
  ShapeRefiner m;

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
  ShapeRefiner m;
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
  ShapeRefiner m;
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
  ASSERT_EQ("Dimensions must be equal, but are 1 and 2", s.error_message());
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
    ShapeRefiner m;
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
    ShapeRefiner m;
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
    ShapeRefiner m;
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
  ShapeRefiner m;
  Graph graph(OpRegistry::Global());
  Node* node;

  Tensor a(DT_FLOAT, TensorShape({}));
  a.scalar<float>()() = 0.0;

  Tensor b(DT_FLOAT, TensorShape({}));
  b.scalar<float>()() = 0.0;

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

  ShapeRefiner m;
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

  ShapeRefiner m;
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

  ShapeRefiner m;
  TF_ASSERT_OK(m.AddNode(input.node()));
  TF_ASSERT_OK(m.AddNode(rank.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));

  shape_inference::InferenceContext* ctx = m.GetContext(shape_data);
  EXPECT_EQ("[2]", ctx->DebugString(ctx->output(0)));
}

}  // namespace
}  // namespace tensorflow
