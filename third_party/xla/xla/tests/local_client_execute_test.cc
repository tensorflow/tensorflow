/* Copyright 2017 The OpenXLA Authors.

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

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/hlo/builder/sharding_builder.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/service/platform_util.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/local_client_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/tests/test_utils.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/test_benchmark.h"

namespace xla {
namespace {

using ::testing::ContainsRegex;

class LocalClientExecuteTest : public LocalClientTestBase {
 protected:
  ErrorSpec error_spec_{0.0001};
};

XLA_TEST_F(LocalClientExecuteTest, Constant) {
  XlaBuilder builder(TestName());
  ConstantR0<float>(&builder, 123.0f);

  ScopedShapedBuffer result = ExecuteLocallyOrDie(builder.Build().value(), {});
  LiteralTestUtil::ExpectR0Near<float>(123.f, ShapedBufferToLiteral(result),
                                       error_spec_);
}

XLA_TEST_F(LocalClientExecuteTest, AddScalars) {
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "x");
  auto y = ConstantR0<float>(&builder, 123.0f);
  Add(x, y);

  auto x_value = LiteralToShapedBuffer(LiteralUtil::CreateR0<float>(42.0f));
  ScopedShapedBuffer result =
      ExecuteLocallyOrDie(builder.Build().value(), {&x_value});
  LiteralTestUtil::ExpectR0Near<float>(165.f, ShapedBufferToLiteral(result),
                                       error_spec_);
}

XLA_TEST_F(LocalClientExecuteTest, AddZeroElementVectors) {
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {0}), "x");
  auto y = ConstantR1<float>(&builder, {});
  Add(x, y);

  auto x_array = LiteralToShapedBuffer(LiteralUtil::CreateR1<float>({}));
  ScopedShapedBuffer result =
      ExecuteLocallyOrDie(builder.Build().value(), {&x_array});
  LiteralTestUtil::ExpectR1Near<float>({}, ShapedBufferToLiteral(result),
                                       error_spec_);
}

XLA_TEST_F(LocalClientExecuteTest, AddVectors) {
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {3}), "x");
  auto y = ConstantR1<float>(&builder, {2.0f, 3.0f, 4.0f});
  Add(x, y);

  auto x_array =
      LiteralToShapedBuffer(LiteralUtil::CreateR1<float>({0.0f, 1.0f, 2.0f}));
  ScopedShapedBuffer result =
      ExecuteLocallyOrDie(builder.Build().value(), {&x_array});
  LiteralTestUtil::ExpectR1Near<float>(
      {2.0f, 4.0f, 6.0f}, ShapedBufferToLiteral(result), error_spec_);
}

XLA_TEST_F(LocalClientExecuteTest, AddVectorsWithProfile) {
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {3}), "x");
  auto y = ConstantR1<float>(&builder, {2.0f, 3.0f, 4.0f});
  Add(x, y);

  auto x_array =
      LiteralToShapedBuffer(LiteralUtil::CreateR1<float>({0.0f, 1.0f, 2.0f}));
  ExecutionProfile profile;
  ScopedShapedBuffer result = ExecuteLocallyOrDie(
      builder.Build().value(), {&x_array}, DefaultExecutableBuildOptions(),
      DefaultExecutableRunOptions().set_execution_profile(&profile));

  LiteralTestUtil::ExpectR1Near<float>(
      {2.0f, 4.0f, 6.0f}, ShapedBufferToLiteral(result), error_spec_);
}

XLA_TEST_F(LocalClientExecuteTest, AddArraysWithDifferentInputLayouts) {
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {2, 2}), "x");
  auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {2, 2}), "y");
  Add(x, y);
  auto computation = builder.Build().value();

  // Create x as a col-major array.
  auto x_array = LiteralToShapedBuffer(LiteralUtil::CreateR2WithLayout(
      {{1.0f, 2.0f}, {3.0f, 4.0f}}, LayoutUtil::MakeLayout({0, 1})));
  EXPECT_TRUE(Layout::Equal().MinorToMajorOnly()(
      x_array.on_device_shape().layout(), LayoutUtil::MakeLayout({0, 1})));

  // Create y as a row-major array.
  auto y_array = LiteralToShapedBuffer(LiteralUtil::CreateR2WithLayout(
      {{10.0f, 20.0f}, {30.0f, 40.0f}}, LayoutUtil::MakeLayout({1, 0})));
  EXPECT_TRUE(Layout::Equal().MinorToMajorOnly()(
      y_array.on_device_shape().layout(), LayoutUtil::MakeLayout({1, 0})));

  ScopedShapedBuffer result_colmaj =
      ExecuteLocallyOrDie(computation, {&x_array, &y_array});
  LiteralTestUtil::ExpectR2Near<float>({{11.0f, 22.0f}, {33.0f, 44.0f}},
                                       ShapedBufferToLiteral(result_colmaj),
                                       error_spec_);

  // Run with the parameter values in a different order.
  ScopedShapedBuffer result_param_swap =
      ExecuteLocallyOrDie(computation, {&y_array, &x_array});
  LiteralTestUtil::ExpectR2Near<float>({{11.0f, 22.0f}, {33.0f, 44.0f}},
                                       ShapedBufferToLiteral(result_param_swap),
                                       error_spec_);
}

XLA_TEST_F(LocalClientExecuteTest, AddArraysWithDifferentOutputLayouts) {
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {2, 2}), "x");
  auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {2, 2}), "y");
  Add(x, y);
  auto computation = builder.Build().value();

  auto x_array = LiteralToShapedBuffer(
      LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}}));
  auto y_array = LiteralToShapedBuffer(
      LiteralUtil::CreateR2<float>({{10.0f, 20.0f}, {30.0f, 40.0f}}));

  // Run with col-major result layout.
  ScopedShapedBuffer result_colmaj =
      ExecuteLocallyOrDie(computation, {&x_array, &y_array},
                          DefaultExecutableBuildOptions().set_result_layout(
                              ShapeUtil::MakeShapeWithDenseLayout(
                                  F32, /*dimensions=*/{2, 2}, {0, 1})),
                          DefaultExecutableRunOptions());
  EXPECT_TRUE(Layout::Equal().MinorToMajorOnly()(
      result_colmaj.on_device_shape().layout(),
      LayoutUtil::MakeLayout({0, 1})));
  LiteralTestUtil::ExpectR2Near<float>({{11.0f, 22.0f}, {33.0f, 44.0f}},
                                       ShapedBufferToLiteral(result_colmaj),
                                       error_spec_);

  // Run with row-major result layout.
  ScopedShapedBuffer result_rowmaj =
      ExecuteLocallyOrDie(computation, {&x_array, &y_array},
                          DefaultExecutableBuildOptions().set_result_layout(
                              ShapeUtil::MakeShapeWithDenseLayout(
                                  F32, /*dimensions=*/{2, 2}, {1, 0})),
                          DefaultExecutableRunOptions());
  EXPECT_TRUE(Layout::Equal().MinorToMajorOnly()(
      result_rowmaj.on_device_shape().layout(),
      LayoutUtil::MakeLayout({1, 0})));
  LiteralTestUtil::ExpectR2Near<float>({{11.0f, 22.0f}, {33.0f, 44.0f}},
                                       ShapedBufferToLiteral(result_rowmaj),
                                       error_spec_);
}

XLA_TEST_F(LocalClientExecuteTest, TupleResult) {
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {2, 2}), "x");
  auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {2, 2}), "y");
  Tuple(&builder, {x, y, x});
  auto computation = builder.Build().value();

  auto x_array = LiteralToShapedBuffer(
      LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}}));
  auto y_array = LiteralToShapedBuffer(
      LiteralUtil::CreateR2<float>({{10.0f, 20.0f}, {30.0f, 40.0f}}));

  ScopedShapedBuffer result =
      ExecuteLocallyOrDie(computation, {&x_array, &y_array});

  EXPECT_TRUE(result.on_host_shape().IsTuple());
  EXPECT_EQ(3, ShapeUtil::TupleElementCount(result.on_host_shape()));

  Literal result_literal = ShapedBufferToLiteral(result);
  LiteralTestUtil::ExpectR2Equal<float>({{1.0f, 2.0f}, {3.0f, 4.0f}},
                                        LiteralSlice(result_literal, {0}));
  LiteralTestUtil::ExpectR2Equal<float>({{10.0f, 20.0f}, {30.0f, 40.0f}},
                                        LiteralSlice(result_literal, {1}));
  LiteralTestUtil::ExpectR2Equal<float>({{1.0f, 2.0f}, {3.0f, 4.0f}},
                                        LiteralSlice(result_literal, {2}));
}

XLA_TEST_F(LocalClientExecuteTest, NestedTupleResult) {
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {2, 2}), "x");
  auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {2, 2}), "y");
  auto inner_tuple = Tuple(&builder, {x, y, x});
  Tuple(&builder, {inner_tuple, x});
  auto computation = builder.Build().value();

  auto x_array = LiteralToShapedBuffer(
      LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}}));
  auto y_array = LiteralToShapedBuffer(
      LiteralUtil::CreateR2<float>({{10.0f, 20.0f}, {30.0f, 40.0f}}));

  ScopedShapedBuffer result =
      ExecuteLocallyOrDie(computation, {&x_array, &y_array});

  EXPECT_TRUE(result.on_host_shape().IsTuple());
  EXPECT_EQ(2, ShapeUtil::TupleElementCount(result.on_host_shape()));

  Literal result_literal = ShapedBufferToLiteral(result);
  LiteralTestUtil::ExpectR2Equal<float>({{1.0f, 2.0f}, {3.0f, 4.0f}},
                                        LiteralSlice(result_literal, {1}));
  LiteralTestUtil::ExpectR2Equal<float>({{1.0f, 2.0f}, {3.0f, 4.0f}},
                                        LiteralSlice(result_literal, {0, 0}));
  LiteralTestUtil::ExpectR2Equal<float>({{10.0f, 20.0f}, {30.0f, 40.0f}},
                                        LiteralSlice(result_literal, {0, 1}));
  LiteralTestUtil::ExpectR2Equal<float>({{1.0f, 2.0f}, {3.0f, 4.0f}},
                                        LiteralSlice(result_literal, {0, 2}));
}

XLA_TEST_F(LocalClientExecuteTest, TupleResultWithLayout) {
  // Verify setting the result layout of a computation with a tuple output.
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {2, 2}), "x");
  auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {2, 2}), "y");
  Tuple(&builder, {x, y});

  auto array = LiteralToShapedBuffer(
      LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}}));

  ExecutableBuildOptions options = DefaultExecutableBuildOptions();
  Shape shape_with_layout = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShapeWithDenseLayout(F32, /*dimensions=*/{2, 2},
                                           /*minor_to_major=*/{0, 1}),
       ShapeUtil::MakeShapeWithDenseLayout(F32, /*dimensions=*/{2, 2},
                                           /*minor_to_major=*/{1, 0})});
  options.set_result_layout(shape_with_layout);
  ScopedShapedBuffer result =
      ExecuteLocallyOrDie(builder.Build().value(), {&array, &array}, options,
                          DefaultExecutableRunOptions());

  Literal result_literal = ShapedBufferToLiteral(result);
  LiteralTestUtil::ExpectR2Equal<float>({{1.0f, 2.0f}, {3.0f, 4.0f}},
                                        LiteralSlice(result_literal, {0}));
  LiteralTestUtil::ExpectR2Equal<float>({{1.0f, 2.0f}, {3.0f, 4.0f}},
                                        LiteralSlice(result_literal, {1}));
}

XLA_TEST_F(LocalClientExecuteTest, TupleArguments) {
  const Shape array_shape = ShapeUtil::MakeShape(F32, {2, 2});
  const Shape vector_shape = ShapeUtil::MakeShape(F32, {3});

  const Shape tuple_shape0 =
      ShapeUtil::MakeTupleShape({array_shape, vector_shape});
  const Shape tuple_shape1 =
      ShapeUtil::MakeTupleShape({vector_shape, array_shape});

  // Computation adds the respective array and vector elements from each tuple
  // argument and returns the results as a tuple.
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, tuple_shape0, "x");
  auto y = Parameter(&builder, 1, tuple_shape1, "y");
  auto x_0 = GetTupleElement(x, 0);
  auto x_1 = GetTupleElement(x, 1);
  auto y_0 = GetTupleElement(y, 0);
  auto y_1 = GetTupleElement(y, 1);
  auto array_sum = Add(x_0, y_1);
  auto vector_diff = Sub(x_1, y_0);
  Tuple(&builder, {array_sum, vector_diff});
  auto computation = builder.Build().value();

  auto x_literal = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}}),
       LiteralUtil::CreateR1<float>({42.0, 75.0, 123.0})});
  auto y_literal = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR1<float>({2.0, 4.0, 6.0}),
       LiteralUtil::CreateR2<float>({{55.0, 44.0}, {33.0, 22.0}})});

  auto x_buffer = LiteralToShapedBuffer(x_literal);
  auto y_buffer = LiteralToShapedBuffer(y_literal);

  ScopedShapedBuffer result =
      ExecuteLocallyOrDie(computation, {&x_buffer, &y_buffer});

  EXPECT_TRUE(result.on_host_shape().IsTuple());
  EXPECT_EQ(2, ShapeUtil::TupleElementCount(result.on_host_shape()));

  Literal result_literal = ShapedBufferToLiteral(result);
  LiteralTestUtil::ExpectR2Equal<float>({{56.0f, 46.0f}, {36.0f, 26.0f}},
                                        LiteralSlice(result_literal, {0}));
  LiteralTestUtil::ExpectR1Equal<float>({40.0f, 71.0f, 117.0f},
                                        LiteralSlice(result_literal, {1}));
}

XLA_TEST_F(LocalClientExecuteTest, NestedTupleArgument) {
  const Shape array_shape = ShapeUtil::MakeShape(F32, {2, 2});
  const Shape vector_shape = ShapeUtil::MakeShape(F32, {3});

  const Shape inner_tuple_shape =
      ShapeUtil::MakeTupleShape({array_shape, vector_shape});
  const Shape nested_tuple_shape =
      ShapeUtil::MakeTupleShape({inner_tuple_shape, vector_shape});

  // Computation negates the array element and sums the two vector elements in
  // the nested tuple. The resulting array and vector are returned as a tuple.
  XlaBuilder builder(TestName());
  auto param = Parameter(&builder, 0, nested_tuple_shape, "param");
  auto inner_tuple = GetTupleElement(param, 0);
  auto inner_array = GetTupleElement(inner_tuple, 0);
  auto inner_vector = GetTupleElement(inner_tuple, 1);
  auto outer_vector = GetTupleElement(param, 1);

  auto negate_array = Neg(inner_array);
  auto vector_sum = Add(inner_vector, outer_vector);
  Tuple(&builder, {negate_array, vector_sum});
  auto computation = builder.Build().value();

  auto arg_literal = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::MakeTupleFromSlices(
           {LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}}),
            LiteralUtil::CreateR1<float>({42.0, 75.0, 123.0})}),
       LiteralUtil::CreateR1<float>({222.0, -2.0, 10.0})});
  auto arg_buffer = LiteralToShapedBuffer(arg_literal);

  ScopedShapedBuffer result = ExecuteLocallyOrDie(computation, {&arg_buffer});

  Literal result_literal = ShapedBufferToLiteral(result);
  LiteralTestUtil::ExpectR2Equal<float>({{-1.0, -2.0}, {-3.0, -4}},
                                        LiteralSlice(result_literal, {0}));
  LiteralTestUtil::ExpectR1Equal<float>({264.0, 73.0, 133.0},
                                        LiteralSlice(result_literal, {1}));
}

XLA_TEST_F(LocalClientExecuteTest, PassingTupleResultBackIntoComputation) {
  // Construct a computation which takes and returns the same shape (a
  // tuple). Feed the result of the computation back into the input. This
  // provides additional verification that the returned tuple is properly
  // constructed.
  const Shape array_shape = ShapeUtil::MakeShape(F32, {2, 2});
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({array_shape, array_shape});

  XlaBuilder builder(TestName());
  auto param = Parameter(&builder, 0, tuple_shape, "param");
  auto element_0 = GetTupleElement(param, 0);
  auto element_1 = GetTupleElement(param, 1);
  Tuple(&builder, {Neg(element_0), Add(element_1, element_1)});
  auto computation = builder.Build().value();

  auto arg_literal = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}}),
       LiteralUtil::CreateR2<float>({{11.0, 3.0}, {4.0, 5.0}})});
  auto arg_buffer = LiteralToShapedBuffer(arg_literal);

  ScopedShapedBuffer result_0 = ExecuteLocallyOrDie(computation, {&arg_buffer});
  Literal result_0_literal = ShapedBufferToLiteral(result_0);
  LiteralTestUtil::ExpectR2Equal<float>({{-1.0, -2.0}, {-3.0, -4.0}},
                                        LiteralSlice(result_0_literal, {0}));
  LiteralTestUtil::ExpectR2Equal<float>({{22.0, 6.0}, {8.0, 10}},
                                        LiteralSlice(result_0_literal, {1}));

  ScopedShapedBuffer result_1 = ExecuteLocallyOrDie(computation, {&result_0});
  Literal result_1_literal = ShapedBufferToLiteral(result_1);
  LiteralTestUtil::ExpectR2Equal<float>({{1.0, 2.0}, {3.0, 4.0}},
                                        LiteralSlice(result_1_literal, {0}));
  LiteralTestUtil::ExpectR2Equal<float>({{44.0, 12.0}, {16.0, 20}},
                                        LiteralSlice(result_1_literal, {1}));
}

XLA_TEST_F(LocalClientExecuteTest, LargeTuple) {
  // Construct a computation which takes a tuple parameter with a very large
  // number of elements.
  const int kElementCount = 1000;

  // Each element is a 2-element vector.
  const Shape element_shape = ShapeUtil::MakeShape(F32, {2});
  std::vector<Shape> element_shapes(kElementCount, element_shape);
  const Shape tuple_shape = ShapeUtil::MakeTupleShape(element_shapes);

  XlaBuilder builder(TestName());
  auto param = Parameter(&builder, 0, tuple_shape, "param");

  // Add each element's tuple index value to every element.
  std::vector<XlaOp> result_elements;
  result_elements.reserve(kElementCount);
  for (int i = 0; i < kElementCount; ++i) {
    auto element = GetTupleElement(param, i);
    result_elements.push_back(Add(element, ConstantR0<float>(&builder, i)));
  }
  Tuple(&builder, result_elements);
  auto computation = builder.Build().value();

  // Feed in a tuple where each two-element vector element is {tuple_index,
  // -tuple_index}.
  std::vector<Literal> arg_elements;
  arg_elements.reserve(kElementCount);
  for (int i = 0; i < kElementCount; ++i) {
    arg_elements.push_back(LiteralUtil::CreateR1<float>({1.0f * i, -1.0f * i}));
  }
  Literal arg_literal = LiteralUtil::MakeTupleOwned(std::move(arg_elements));
  auto arg_buffer = LiteralToShapedBuffer(arg_literal);

  ScopedShapedBuffer result = ExecuteLocallyOrDie(computation, {&arg_buffer});
  Literal result_literal = ShapedBufferToLiteral(result);

  for (int i = 0; i < kElementCount; ++i) {
    LiteralTestUtil::ExpectR1Near<float>(
        {2.0f * i, 0.0f}, LiteralSlice(result_literal, {i}), error_spec_);
  }
}

XLA_TEST_F(LocalClientExecuteTest, LargeNestedTuple) {
  // Construct and run a computation which takes a two-level nested tuple
  // parameter with a large fanout.
  const int kFanout = 40;

  // Tuple shape is full two-level tree with the given fanout.
  const Shape element_shape = ShapeUtil::MakeShape(F32, {});
  std::vector<Shape> element_shapes(kFanout, element_shape);
  const Shape inner_tuple_shape = ShapeUtil::MakeTupleShape(element_shapes);
  std::vector<Shape> inner_tuple_shapes(kFanout, inner_tuple_shape);
  const Shape tuple_shape = ShapeUtil::MakeTupleShape(inner_tuple_shapes);

  XlaBuilder builder(TestName());
  auto param = Parameter(&builder, 0, tuple_shape, "param");

  // The computation increments each leaf value by an amount equal to the leaf's
  // ordinal position in a traversal of the tuple.
  std::vector<XlaOp> result_elements;
  result_elements.reserve(kFanout);
  for (int i = 0; i < kFanout; ++i) {
    auto outer_element = GetTupleElement(param, i);
    std::vector<XlaOp> inner_result_elements;
    inner_result_elements.reserve(kFanout);
    for (int j = 0; j < kFanout; ++j) {
      auto inner_element = GetTupleElement(outer_element, j);
      inner_result_elements.push_back(
          Add(inner_element, ConstantR0<float>(&builder, i * kFanout + j)));
    }
    result_elements.push_back(Tuple(&builder, inner_result_elements));
  }
  Tuple(&builder, result_elements);
  auto computation = builder.Build().value();

  // Construct the argument to pass to the computation.
  std::vector<Literal> outer_tuple_elements;
  outer_tuple_elements.reserve(kFanout);
  for (int i = 0; i < kFanout; ++i) {
    std::vector<Literal> inner_tuple_elements;
    inner_tuple_elements.reserve(kFanout);
    for (int j = 0; j < kFanout; ++j) {
      inner_tuple_elements.push_back(LiteralUtil::CreateR0<float>(i + j));
    }
    outer_tuple_elements.push_back(
        LiteralUtil::MakeTupleOwned(std::move(inner_tuple_elements)));
  }
  auto arg_literal =
      LiteralUtil::MakeTupleOwned(std::move(outer_tuple_elements));
  auto arg_buffer = LiteralToShapedBuffer(arg_literal);

  ScopedShapedBuffer result = ExecuteLocallyOrDie(computation, {&arg_buffer});
  Literal result_literal = ShapedBufferToLiteral(result);

  for (int i = 0; i < kFanout; ++i) {
    for (int j = 0; j < kFanout; ++j) {
      LiteralTestUtil::ExpectR0Near<float>(i + j + i * kFanout + j,
                                           LiteralSlice(result_literal, {i, j}),
                                           error_spec_);
    }
  }
}

XLA_TEST_F(LocalClientExecuteTest, DeepTuple) {
  // Construct and run a computation which takes a very deep tuple. The tuple
  // has no fan out and a single scalar element at the bottom.
  const int kTupleDepth = 100;

  // Tuple shape is full two-level tree with the given fanout.
  Shape shape = ShapeUtil::MakeShape(F32, {});
  for (int i = 0; i < kTupleDepth; ++i) {
    shape = ShapeUtil::MakeTupleShape({shape});
  }

  XlaBuilder builder(TestName());
  auto element = Parameter(&builder, 0, shape, "param");
  for (int i = 0; i < kTupleDepth; ++i) {
    element = GetTupleElement(element, 0);
  }

  auto output = Add(element, ConstantR0<float>(&builder, 42.0));
  for (int i = 0; i < kTupleDepth; ++i) {
    output = Tuple(&builder, {output});
  }
  auto computation = builder.Build().value();

  // Construct the argument to pass to the computation.
  Literal arg_literal = LiteralUtil::CreateR0<float>(123.0);
  for (int i = 0; i < kTupleDepth; ++i) {
    std::vector<Literal> arg_vector;
    arg_vector.push_back(std::move(arg_literal));
    arg_literal = LiteralUtil::MakeTupleOwned(std::move(arg_vector));
  }
  auto arg_buffer = LiteralToShapedBuffer(arg_literal);

  ScopedShapedBuffer result = ExecuteLocallyOrDie(computation, {&arg_buffer});
  Literal result_literal = ShapedBufferToLiteral(result);

  ShapeIndex index;
  for (int i = 0; i < kTupleDepth; ++i) {
    index.push_back(0);
  }
  LiteralTestUtil::ExpectR0Equal<float>(165.0,
                                        LiteralSlice(result_literal, index));
}

XLA_TEST_F(LocalClientExecuteTest, InvalidNumberOfArguments) {
  // Test passing in an invalid number of arguments.
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {3}), "x");
  auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {3}), "y");
  Add(x, y);

  auto x_array =
      LiteralToShapedBuffer(LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f}));
  auto execute_status = ExecuteLocally(builder.Build().value(), {&x_array});

  EXPECT_FALSE(execute_status.ok());
  EXPECT_THAT(execute_status.status().message(),
              ContainsRegex("Invalid number of arguments"));
}

XLA_TEST_F(LocalClientExecuteTest, IncorrectArgumentShape) {
  // Test passing in an argument with the wrong shape.
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {3}), "x");
  Neg(x);

  auto x_array = LiteralToShapedBuffer(
      LiteralUtil::CreateR2<float>({{0.0f, 1.0f}, {2.0f, 3.0f}}));
  auto execute_status = ExecuteLocally(builder.Build().value(), {&x_array});

  EXPECT_FALSE(execute_status.ok());
  EXPECT_THAT(execute_status.status().message(),
              ContainsRegex("Invalid argument shape"))
      << execute_status.status();
}

XLA_TEST_F(LocalClientExecuteTest, InvalidResultLayout) {
  // Test passing in an invalid result layout parameter.
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {2, 2}), "x");
  Neg(x);

  auto x_array = LiteralToShapedBuffer(
      LiteralUtil::CreateR2<float>({{0.0f, 1.0f}, {2.0f, 3.0f}}));
  auto execute_status = ExecuteLocally(
      builder.Build().value(), {&x_array},
      DefaultExecutableBuildOptions().set_result_layout(
          ShapeUtil::MakeShapeWithDenseLayout(F32,
                                              /*dimensions=*/{1, 2, 3, 4},
                                              /*minor_to_major=*/{0, 1, 2, 3})),
      DefaultExecutableRunOptions());

  EXPECT_FALSE(execute_status.ok());
  EXPECT_THAT(execute_status.status().message(),
              ContainsRegex("not compatible with result shape"))
      << execute_status.status();
}

XLA_TEST_F(LocalClientExecuteTest, RunOnAllDeviceOrdinals) {
  // Try to run a trivial computation on every device on the system. If a
  // specific device is not supported, check that the right error is returned.
  XlaBuilder builder(TestName());
  ConstantR0<float>(&builder, 42.0f);
  auto computation = builder.Build().value();
  for (int d = 0; d < local_client_->device_count(); ++d) {
    if (!local_client_->device_ordinal_supported(d)) {
      auto execute_status =
          ExecuteLocally(computation, {},
                         DefaultExecutableBuildOptions().set_device_ordinal(d),
                         DefaultExecutableRunOptions().set_device_ordinal(d));
      EXPECT_FALSE(execute_status.ok());
      EXPECT_THAT(execute_status.status().message(),
                  ContainsRegex("device .* not supported"));
    } else {
      auto result = ExecuteLocallyOrDie(
          computation, {},
          DefaultExecutableBuildOptions().set_device_ordinal(d),
          DefaultExecutableRunOptions().set_device_ordinal(d));
      EXPECT_EQ(d, result.device_ordinal());
      LiteralTestUtil::ExpectR0Equal<float>(42.0f,
                                            ShapedBufferToLiteral(result));
    }
  }
}

XLA_TEST_F(LocalClientExecuteTest, InvalidDeviceOrdinalValues) {
  // Try running computations on devices with device ordinal values which do not
  // exist.
  XlaBuilder builder(TestName());
  ConstantR0<float>(&builder, 42.0f);
  auto computation = builder.Build().value();

  auto execute_status =
      ExecuteLocally(computation, {},
                     DefaultExecutableBuildOptions().set_device_ordinal(
                         local_client_->device_count()),
                     DefaultExecutableRunOptions().set_device_ordinal(
                         local_client_->device_count()));
  EXPECT_FALSE(execute_status.ok());
  EXPECT_THAT(execute_status.status().message(),
              ContainsRegex("Invalid device ordinal value"));
}

XLA_TEST_F(LocalClientExecuteTest, RunOnStream) {
  // Run a computation on a specific stream on each device on the system.
  XlaBuilder builder(TestName());
  ConstantR0<float>(&builder, 42.0f);
  auto computation = builder.Build().value();

  for (int d = 0; d < local_client_->device_count(); ++d) {
    if (!local_client_->device_ordinal_supported(d)) {
      continue;
    }
    se::StreamExecutor* executor =
        local_client_->platform()->ExecutorForDevice(d).value();
    TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

    auto result = ExecuteLocallyOrDie(
        computation, {}, DefaultExecutableBuildOptions(),
        DefaultExecutableRunOptions().set_stream(stream.get()));
    // As a check to verify that the computation ran of the device associated
    // with the stream. This is a weak check, but stronger verification is hard.
    EXPECT_EQ(d, result.device_ordinal());
    LiteralTestUtil::ExpectR0Equal<float>(42.0f, ShapedBufferToLiteral(result));
  }
}

// Disable this test on CPU because we're using the CPU as the platform
// which does not match the service platform.
XLA_TEST_F(LocalClientExecuteTest,
           DISABLED_ON_CPU(RunOnStreamForWrongPlatform)) {
  // Try to run a computation on a stream for a platform (CPU) which does not
  // match the platform of the service (!= CPU).
  se::Platform* wrong_platform =
      se::PlatformManager::PlatformWithId(se::host::kHostPlatformId).value();
  TF_ASSERT_OK_AND_ASSIGN(
      auto wrong_stream,
      wrong_platform->ExecutorForDevice(0).value()->CreateStream());

  XlaBuilder builder(TestName());
  ConstantR0<float>(&builder, 42.0f);
  auto execute_status = ExecuteLocally(
      builder.Build().value(), {}, DefaultExecutableBuildOptions(),
      DefaultExecutableRunOptions().set_stream(wrong_stream.get()));
  EXPECT_FALSE(execute_status.ok());
  EXPECT_THAT(execute_status.status().message(),
              ContainsRegex("stream is for platform .*, but service targets"));
}

XLA_TEST_F(LocalClientExecuteTest,
           DISABLED_ON_CPU(AllocatorDoesNotMatchPlatform)) {
  se::Platform* wrong_platform =
      se::PlatformManager::PlatformWithId(se::host::kHostPlatformId).value();
  TestAllocator allocator(wrong_platform);

  XlaBuilder builder(TestName());
  ConstantR0<float>(&builder, 123.0f);

  auto execute_status = ExecuteLocally(
      builder.Build().value(), {}, DefaultExecutableBuildOptions(),
      DefaultExecutableRunOptions().set_allocator(&allocator));
  EXPECT_FALSE(execute_status.ok());
  EXPECT_THAT(execute_status.status().message(),
              ContainsRegex("allocator platform .* does not match service"));
}

XLA_TEST_F(LocalClientExecuteTest, CompileExecutable) {
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {3}), "x");
  auto y = ConstantR1<float>(&builder, {2.0f, 3.0f, 4.0f});
  Add(x, y);

  Shape argument_layout =
      local_client_->backend().compiler()->DefaultDeviceShapeRepresentation(
          ShapeUtil::MakeShapeWithDenseLayout(F32, /*dimensions=*/{3}, {0}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto executables,
      local_client_->Compile(builder.Build().value(), {&argument_layout},
                             ExecutableBuildOptions()));
  EXPECT_EQ(1, executables.size());

  auto x_array =
      LiteralToShapedBuffer(LiteralUtil::CreateR1<float>({0.0f, 1.0f, 2.0f}));
  ScopedShapedBuffer result =
      executables[0]->Run({&x_array}, DefaultExecutableRunOptions()).value();
  ASSERT_IS_OK(local_client_->mutable_backend()
                   ->BorrowStream(0)
                   .value()
                   ->BlockHostUntilDone());

  LiteralTestUtil::ExpectR1Near<float>(
      {2.0f, 4.0f, 6.0f}, ShapedBufferToLiteral(result), error_spec_);
}

XLA_TEST_F(LocalClientExecuteTest,
           DISABLED_ON_TPU(CompilePartitionedExecutable)) {
  if (local_client_->device_count() < 2) {
    GTEST_SKIP_("requires two devices");
  }

  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {3}), "x");
  auto y = ConstantR1<float>(&builder, {2.0f, 3.0f, 4.0f});
  auto z = ConstantR1<float>(&builder, {5.0f, 6.0f, 7.0f});
  auto r = Add(x, y);
  builder.SetSharding(sharding_builder::AssignDevice(1));
  Add(r, z);
  builder.ClearSharding();

  Shape argument_layout =
      ShapeUtil::MakeShapeWithDenseLayout(F32, /*dimensions=*/{3}, {0});
  ExecutableBuildOptions build_options;
  build_options.set_num_partitions(2);
  TF_ASSERT_OK_AND_ASSIGN(
      auto executables,
      local_client_->Compile(builder.Build().value(), {&argument_layout},
                             build_options));
  EXPECT_EQ(2, executables.size());
}

XLA_TEST_F(LocalClientExecuteTest, DISABLED_ON_CPU(DISABLED_ON_INTERPRETER(
                                       SizeOfGeneratedCodeInBytes))) {
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "x");
  constexpr int size = 100000;
  TF_ASSERT_OK_AND_ASSIGN(auto literal,
                          LiteralUtil::CreateRandomLiteral<F32>(
                              ShapeUtil::MakeShape(F32, {size}), 0.0, 1.0));
  auto y = ConstantLiteral(&builder, literal);
  Add(x, y);

  Shape argument_layout =
      ShapeUtil::MakeShapeWithDenseLayout(F32, /*dimensions=*/{}, {});
  TF_ASSERT_OK_AND_ASSIGN(
      auto executables,
      local_client_->Compile(builder.Build().value(), {&argument_layout},
                             ExecutableBuildOptions()));
  EXPECT_EQ(1, executables.size());
  // The executable should be at least as large as the constant it contains.
  EXPECT_GT(executables.front()->executable()->SizeOfGeneratedCodeInBytes(),
            int64_t{sizeof(float) * size});
}

XLA_TEST_F(LocalClientExecuteTest, ShapeBufferToLiteralConversion) {
  // Test copying Literals to the device as ShapedBuffers, then copying them
  // back again to Literals.
  auto test_to_device_and_back = [this](const Literal& literal) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto shaped_buffer,
        local_client_->LiteralToShapedBuffer(
            literal, local_client_->default_device_ordinal(), allocator_));
    TF_ASSERT_OK_AND_ASSIGN(
        auto transferred_literal,
        local_client_->ShapedBufferToLiteral(shaped_buffer));
    EXPECT_EQ(literal, transferred_literal);
  };

  // Array shapes.
  test_to_device_and_back(LiteralUtil::CreateR0<float>(42.0));
  test_to_device_and_back(LiteralUtil::CreateR0<bool>(true));
  test_to_device_and_back(LiteralUtil::CreateR1<float>({1.0, 42.0, 744.4}));
  test_to_device_and_back(
      LiteralUtil::CreateR2<float>({{1.0, 2.0, 3.0}, {44.0, 0.1, -3}}));
  test_to_device_and_back(LiteralUtil::CreateR2<int32_t>({{2, 1}, {4444, 56}}));

  // Null shape (empty tuple).
  test_to_device_and_back(LiteralUtil::MakeTuple({}));

  // Non-nested tuples.
  test_to_device_and_back(LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<float>(12223.0)}));
  test_to_device_and_back(LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR1<float>({1.0, -42.0}),
       LiteralUtil::CreateR0<float>(123456.0)}));

  // Nested tuple.
  test_to_device_and_back(LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::MakeTupleFromSlices(
           {LiteralUtil::CreateR1<float>({1.0, -42.0}),
            LiteralUtil::CreateR0<float>(123456.0)}),
       LiteralUtil::CreateR0<bool>(false)}));
}

XLA_TEST_F(LocalClientExecuteTest, ShapeBufferToLiteralConversion64bit) {
  // Test copying Literals to the device as ShapedBuffers, then copying them
  // back again to Literals for 64-bit values.
  auto test_to_device_and_back = [this](const Literal& literal) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto shaped_buffer,
        local_client_->LiteralToShapedBuffer(
            literal, local_client_->default_device_ordinal(), allocator_));
    TF_ASSERT_OK_AND_ASSIGN(
        auto transferred_literal,
        local_client_->ShapedBufferToLiteral(shaped_buffer));
    EXPECT_EQ(literal, transferred_literal);
  };

  test_to_device_and_back(LiteralUtil::CreateR2<double>(
      {{1.0, 2.0, 3.0}, {44.0, 0.099999999999999978, -3}}));
  test_to_device_and_back(LiteralUtil::CreateR2<int64_t>({{2, 1}, {4444, 56}}));
  test_to_device_and_back(
      LiteralUtil::CreateR2<uint64_t>({{20000000000ULL, 1}, {4444, 56}}));
  test_to_device_and_back(LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR1<double>({1.0, -42.0}),
       LiteralUtil::CreateR0<int64_t>(123456789000LL)}));
}

// Disabled on interpreter backend since infeed HLO is unsupported.
XLA_TEST_F(LocalClientExecuteTest, DISABLED_ON_INTERPRETER(InfeedTest)) {
  XlaBuilder builder(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {3});
  auto in = Infeed(&builder, shape);
  auto constant = ConstantR1<float>(&builder, {1.0f, 2.0f, 3.0f});
  Add(in, constant);

  Literal result;
  std::unique_ptr<tsl::Thread> thread(tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "execute_thread", [&] {
        result = ShapedBufferToLiteral(
            ExecuteLocallyOrDie(builder.Build().value(), /*arguments=*/{}));
      }));

  ASSERT_IS_OK(local_client_->TransferToInfeedLocal(
      LiteralUtil::CreateR1<float>({-5.0, 123.0, 42.0}),
      local_client_->default_device_ordinal()));

  // Join the thread.
  thread.reset();

  LiteralTestUtil::ExpectR1Equal<float>({-4.0, 125.0, 45.0}, result);
}

// Disabled on interpreter backend since infeed/outfeed HLOs are unsupported.
XLA_TEST_F(LocalClientExecuteTest, DISABLED_ON_INTERPRETER(InfeedOutfeedTest)) {
  XlaBuilder builder(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {3});
  auto in = Infeed(&builder, shape);
  auto constant = ConstantR1<float>(&builder, {1.0f, 2.0f, 3.0f});
  auto sum = Add(in, constant);
  Outfeed(sum, shape, /*outfeed_config=*/"");

  std::unique_ptr<tsl::Thread> thread(tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "execute_thread",
      [&] { ExecuteLocallyOrDie(builder.Build().value(), {}); }));

  ASSERT_IS_OK(local_client_->TransferToInfeedLocal(
      LiteralUtil::CreateR1<float>({-5.0, 123.0, 42.0}),
      local_client_->default_device_ordinal()));

  Literal result(shape);
  ASSERT_IS_OK(local_client_->TransferFromOutfeedLocal(
      local_client_->default_device_ordinal(), &result));

  LiteralTestUtil::ExpectR1Equal<float>({-4.0, 125.0, 45.0}, result);
}

// Benchmark that measures the overhead of the LocalClient API when running a
// trivial computation
void BM_LocalClientOverhead(::testing::benchmark::State& state) {
  se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
  auto executors = PlatformUtil::GetStreamExecutors(platform).value();
  se::StreamExecutorMemoryAllocator allocator(platform, executors);
  LocalClient* client = ClientLibrary::GetOrCreateLocalClient(platform).value();
  auto* transfer_manager = TransferManager::GetForPlatform(platform).value();
  int device_ordinal = client->default_device_ordinal();

  // Use a tiny add operation as the computation.
  XlaBuilder builder("Add");
  auto shape = ShapeUtil::MakeShape(F32, {2, 3});
  auto x = Parameter(&builder, 0, shape, "x");
  Add(x, x);
  auto computation = builder.Build().value();

  auto buffer =
      transfer_manager
          ->AllocateScopedShapedBuffer(shape, &allocator, /*device_ordinal=*/0)
          .value();
  auto literal = LiteralUtil::CreateR2<float>({{0, 0, 0}, {0, 0, 0}});
  auto stream = client->mutable_backend()->BorrowStream(device_ordinal).value();
  ASSERT_IS_OK(
      transfer_manager->TransferLiteralToDevice(stream.get(), literal, buffer));

  const int kWarmups = 2;

  TF_ASSERT_OK_AND_ASSIGN(
      auto executables, client->Compile(computation, {&buffer.on_host_shape()},
                                        ExecutableBuildOptions()));
  std::unique_ptr<LocalExecutable> executable = std::move(executables[0]);

  ExecutableRunOptions run_options;
  run_options.set_allocator(&allocator).set_stream(stream.get());

  for (int i = 0; i < kWarmups; ++i) {
    auto result = executable->Run({&buffer}, run_options);
    ASSERT_IS_OK(result);
  }

  for (auto s : state) {
    auto result = executable->Run({&buffer}, run_options);
    ASSERT_IS_OK(result);
  }
}

XLA_TEST_F(LocalClientExecuteTest, ValidateFDOProfile) {
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {3}), "x");
  auto y = ConstantR1<float>(&builder, {2.0f, 3.0f, 4.0f});
  Add(x, y);

  Shape argument_layout =
      local_client_->backend().compiler()->DefaultDeviceShapeRepresentation(
          ShapeUtil::MakeShapeWithDenseLayout(F32, /*dimensions=*/{3}, {0}));
  ExecutableBuildOptions build_options;
  const char kFdoProfile[] = "Testing";
  *build_options.mutable_fdo_profile() = kFdoProfile;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executables,
      local_client_->Compile(builder.Build().value(), {&argument_layout},
                             build_options));
  EXPECT_EQ(1, executables.size());
  const HloModule& compiled_module =
      executables.front()->executable()->module();
  EXPECT_EQ(compiled_module.config().fdo_profile(), kFdoProfile);
  auto proto = compiled_module.ToProtoWithConfig();
  EXPECT_EQ(proto.config().fdo_profile(), kFdoProfile);
}

XLA_TEST_F(LocalClientExecuteTest, ValidateDeviceMemorySize) {
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {3}), "x");
  auto y = ConstantR1<float>(&builder, {2.0f, 3.0f, 4.0f});
  Add(x, y);

  Shape argument_layout =
      local_client_->backend().compiler()->DefaultDeviceShapeRepresentation(
          ShapeUtil::MakeShapeWithDenseLayout(F32, /*dimensions=*/{3}, {0}));
  ExecutableBuildOptions build_options;
  constexpr int64_t kDeviceMemorySize = 1024 * 1024 * 1024;
  build_options.set_device_memory_size(kDeviceMemorySize);
  TF_ASSERT_OK_AND_ASSIGN(
      auto executables,
      local_client_->Compile(builder.Build().value(), {&argument_layout},
                             build_options));
  EXPECT_EQ(1, executables.size());
  const HloModule& compiled_module =
      executables.front()->executable()->module();
  EXPECT_EQ(compiled_module.config().device_memory_size(), kDeviceMemorySize);
  auto proto = compiled_module.ToProtoWithConfig();
  EXPECT_EQ(proto.config().device_memory_size(), kDeviceMemorySize);
}

XLA_TEST_F(LocalClientExecuteTest, ValidateUseShardyPartitioner) {
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {3}), "x");
  auto y = ConstantR1<float>(&builder, {2.0f, 3.0f, 4.0f});
  Add(x, y);
  Shape argument_layout =
      local_client_->backend().compiler()->DefaultDeviceShapeRepresentation(
          ShapeUtil::MakeShapeWithDenseLayout(F32, /*dimensions=*/{3}, {0}));

  ExecutableBuildOptions build_options;
  build_options.set_use_shardy_partitioner(true);
  TF_ASSERT_OK_AND_ASSIGN(
      auto executables,
      local_client_->Compile(builder.Build().value(), {&argument_layout},
                             build_options));
  EXPECT_EQ(1, executables.size());
  const HloModule& compiled_module =
      executables.front()->executable()->module();
  EXPECT_EQ(compiled_module.config().use_shardy_partitioner(), true);
  auto proto = compiled_module.ToProtoWithConfig();
  EXPECT_EQ(proto.config().use_shardy_partitioner(), true);
}

XLA_TEST_F(LocalClientExecuteTest, ValidateExecTimeOptimizationEffort) {
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {3}), "x");
  auto y = ConstantR1<float>(&builder, {2.0f, 3.0f, 4.0f});
  Add(x, y);
  Shape argument_layout =
      local_client_->backend().compiler()->DefaultDeviceShapeRepresentation(
          ShapeUtil::MakeShapeWithDenseLayout(F32, /*dimensions=*/{3}, {0}));

  ExecutableBuildOptions build_options;
  build_options.set_exec_time_optimization_effort(-1.5f);
  TF_ASSERT_OK_AND_ASSIGN(
      auto executables,
      local_client_->Compile(builder.Build().value(), {&argument_layout},
                             build_options));
  EXPECT_EQ(1, executables.size());
  const HloModule& compiled_module =
      executables.front()->executable()->module();
  EXPECT_FLOAT_EQ(compiled_module.config().exec_time_optimization_effort(),
                  -1.5f);
  auto proto = compiled_module.ToProtoWithConfig();
  EXPECT_FLOAT_EQ(proto.config().exec_time_optimization_effort(), -1.5f);
}


XLA_TEST_F(LocalClientExecuteTest, ValidateOptimizationLevel) {
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {3}), "x");
  auto y = ConstantR1<float>(&builder, {2.0f, 3.0f, 4.0f});
  Add(x, y);
  Shape argument_layout =
      local_client_->backend().compiler()->DefaultDeviceShapeRepresentation(
          ShapeUtil::MakeShapeWithDenseLayout(F32, /*dimensions=*/{3}, {0}));

  ExecutableBuildOptions build_options;
  build_options.set_optimization_level(ExecutionOptions::EFFORT_O1);
  TF_ASSERT_OK_AND_ASSIGN(
      auto executables,
      local_client_->Compile(builder.Build().value(), {&argument_layout},
                             build_options));
  EXPECT_EQ(1, executables.size());
  const HloModule& compiled_module =
      executables.front()->executable()->module();
  EXPECT_EQ(compiled_module.config().optimization_level(),
            ExecutionOptions::EFFORT_O1);
  auto proto = compiled_module.ToProtoWithConfig();
  EXPECT_EQ(proto.config().optimization_level(), ExecutionOptions::EFFORT_O1);
}

XLA_TEST_F(LocalClientExecuteTest, ValidateMemoryFittingLevel) {
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {3}), "x");
  auto y = ConstantR1<float>(&builder, {2.0f, 3.0f, 4.0f});
  Add(x, y);
  Shape argument_layout =
      local_client_->backend().compiler()->DefaultDeviceShapeRepresentation(
          ShapeUtil::MakeShapeWithDenseLayout(F32, /*dimensions=*/{3}, {0}));

  ExecutableBuildOptions build_options;
  build_options.set_memory_fitting_level(ExecutionOptions::EFFORT_O3);
  TF_ASSERT_OK_AND_ASSIGN(
      auto executables,
      local_client_->Compile(builder.Build().value(), {&argument_layout},
                             build_options));
  EXPECT_EQ(1, executables.size());
  const HloModule& compiled_module =
      executables.front()->executable()->module();
  EXPECT_EQ(compiled_module.config().memory_fitting_level(),
            ExecutionOptions::EFFORT_O3);
  auto proto = compiled_module.ToProtoWithConfig();
  EXPECT_EQ(proto.config().memory_fitting_level(), ExecutionOptions::EFFORT_O3);
}

BENCHMARK(BM_LocalClientOverhead);

}  // namespace
}  // namespace xla
