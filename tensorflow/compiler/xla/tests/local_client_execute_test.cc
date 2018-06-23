/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <initializer_list>
#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/local_service.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/local_client_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace xla {
namespace {

using ::testing::ContainsRegex;

class LocalClientExecuteTest : public LocalClientTestBase {
 protected:
  ErrorSpec error_spec_{0.0001};
};

XLA_TEST_F(LocalClientExecuteTest, Constant) {
  XlaBuilder builder(TestName());
  auto y = builder.ConstantR0<float>(123.0f);

  ScopedShapedBuffer result =
      ExecuteLocallyOrDie(builder.Build().ValueOrDie(), {});
  LiteralTestUtil::ExpectR0Near<float>(123.f, *ShapedBufferToLiteral(result),
                                       error_spec_);
}

XLA_TEST_F(LocalClientExecuteTest, AddScalars) {
  XlaBuilder builder(TestName());
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
  auto y = builder.ConstantR0<float>(123.0f);
  builder.Add(x, y);

  auto x_value = LiteralToShapedBuffer(*Literal::CreateR0<float>(42.0f));
  ScopedShapedBuffer result =
      ExecuteLocallyOrDie(builder.Build().ValueOrDie(), {&x_value});
  LiteralTestUtil::ExpectR0Near<float>(165.f, *ShapedBufferToLiteral(result),
                                       error_spec_);
}

XLA_TEST_F(LocalClientExecuteTest, AddZeroElementVectors) {
  XlaBuilder builder(TestName());
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {0}), "x");
  auto y = builder.ConstantR1<float>({});
  builder.Add(x, y);

  auto x_array = LiteralToShapedBuffer(*Literal::CreateR1<float>({}));
  ScopedShapedBuffer result =
      ExecuteLocallyOrDie(builder.Build().ValueOrDie(), {&x_array});
  LiteralTestUtil::ExpectR1Near<float>({}, *ShapedBufferToLiteral(result),
                                       error_spec_);
}

XLA_TEST_F(LocalClientExecuteTest, AddVectors) {
  XlaBuilder builder(TestName());
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {3}), "x");
  auto y = builder.ConstantR1<float>({2.0f, 3.0f, 4.0f});
  builder.Add(x, y);

  auto x_array =
      LiteralToShapedBuffer(*Literal::CreateR1<float>({0.0f, 1.0f, 2.0f}));
  ScopedShapedBuffer result =
      ExecuteLocallyOrDie(builder.Build().ValueOrDie(), {&x_array});
  LiteralTestUtil::ExpectR1Near<float>(
      {2.0f, 4.0f, 6.0f}, *ShapedBufferToLiteral(result), error_spec_);
}

XLA_TEST_F(LocalClientExecuteTest, AddVectorsWithProfile) {
  XlaBuilder builder(TestName());
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {3}), "x");
  auto y = builder.ConstantR1<float>({2.0f, 3.0f, 4.0f});
  builder.Add(x, y);

  auto x_array =
      LiteralToShapedBuffer(*Literal::CreateR1<float>({0.0f, 1.0f, 2.0f}));
  ExecutionProfile profile;
  ScopedShapedBuffer result = ExecuteLocallyOrDie(
      builder.Build().ValueOrDie(), {&x_array}, DefaultExecutableBuildOptions(),
      DefaultExecutableRunOptions().set_execution_profile(&profile));

  LiteralTestUtil::ExpectR1Near<float>(
      {2.0f, 4.0f, 6.0f}, *ShapedBufferToLiteral(result), error_spec_);
  EXPECT_GT(profile.compute_and_transfer_time_ns(), 0);
}

XLA_TEST_F(LocalClientExecuteTest, AddArraysWithDifferentInputLayouts) {
  XlaBuilder builder(TestName());
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {2, 2}), "x");
  auto y = builder.Parameter(1, ShapeUtil::MakeShape(F32, {2, 2}), "y");
  builder.Add(x, y);
  auto computation = builder.Build().ConsumeValueOrDie();

  // Create x as a col-major array.
  auto x_array = LiteralToShapedBuffer(*Literal::CreateR2WithLayout(
      {{1.0f, 2.0f}, {3.0f, 4.0f}}, LayoutUtil::MakeLayout({0, 1})));
  EXPECT_TRUE(LayoutUtil::Equal(x_array.on_device_shape().layout(),
                                LayoutUtil::MakeLayout({0, 1})));

  // Create y as a row-major array.
  auto y_array = LiteralToShapedBuffer(*Literal::CreateR2WithLayout(
      {{10.0f, 20.0f}, {30.0f, 40.0f}}, LayoutUtil::MakeLayout({1, 0})));
  EXPECT_TRUE(LayoutUtil::Equal(y_array.on_device_shape().layout(),
                                LayoutUtil::MakeLayout({1, 0})));

  ScopedShapedBuffer result_colmaj =
      ExecuteLocallyOrDie(computation, {&x_array, &y_array});
  LiteralTestUtil::ExpectR2Near<float>({{11.0f, 22.0f}, {33.0f, 44.0f}},
                                       *ShapedBufferToLiteral(result_colmaj),
                                       error_spec_);

  // Run with the parameter values in a different order.
  ScopedShapedBuffer result_param_swap =
      ExecuteLocallyOrDie(computation, {&y_array, &x_array});
  LiteralTestUtil::ExpectR2Near<float>(
      {{11.0f, 22.0f}, {33.0f, 44.0f}},
      *ShapedBufferToLiteral(result_param_swap), error_spec_);
}

XLA_TEST_F(LocalClientExecuteTest, AddArraysWithDifferentOutputLayouts) {
  XlaBuilder builder(TestName());
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {2, 2}), "x");
  auto y = builder.Parameter(1, ShapeUtil::MakeShape(F32, {2, 2}), "y");
  builder.Add(x, y);
  auto computation = builder.Build().ConsumeValueOrDie();

  auto x_array = LiteralToShapedBuffer(
      *Literal::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}}));
  auto y_array = LiteralToShapedBuffer(
      *Literal::CreateR2<float>({{10.0f, 20.0f}, {30.0f, 40.0f}}));

  // Run with col-major result layout.
  ScopedShapedBuffer result_colmaj = ExecuteLocallyOrDie(
      computation, {&x_array, &y_array},
      DefaultExecutableBuildOptions().set_result_layout(
          ShapeUtil::MakeShapeWithLayout(F32, /*dimensions=*/{2, 2}, {0, 1})),
      DefaultExecutableRunOptions());
  EXPECT_TRUE(LayoutUtil::Equal(result_colmaj.on_device_shape().layout(),
                                LayoutUtil::MakeLayout({0, 1})));
  LiteralTestUtil::ExpectR2Near<float>({{11.0f, 22.0f}, {33.0f, 44.0f}},
                                       *ShapedBufferToLiteral(result_colmaj),
                                       error_spec_);

  // Run with row-major result layout.
  ScopedShapedBuffer result_rowmaj = ExecuteLocallyOrDie(
      computation, {&x_array, &y_array},
      DefaultExecutableBuildOptions().set_result_layout(
          ShapeUtil::MakeShapeWithLayout(F32, /*dimensions=*/{2, 2}, {1, 0})),
      DefaultExecutableRunOptions());
  EXPECT_TRUE(LayoutUtil::Equal(result_rowmaj.on_device_shape().layout(),
                                LayoutUtil::MakeLayout({1, 0})));
  LiteralTestUtil::ExpectR2Near<float>({{11.0f, 22.0f}, {33.0f, 44.0f}},
                                       *ShapedBufferToLiteral(result_rowmaj),
                                       error_spec_);
}

XLA_TEST_F(LocalClientExecuteTest, TupleResult) {
  XlaBuilder builder(TestName());
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {2, 2}), "x");
  auto y = builder.Parameter(1, ShapeUtil::MakeShape(F32, {2, 2}), "y");
  builder.Tuple({x, y, x});
  auto computation = builder.Build().ConsumeValueOrDie();

  auto x_array = LiteralToShapedBuffer(
      *Literal::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}}));
  auto y_array = LiteralToShapedBuffer(
      *Literal::CreateR2<float>({{10.0f, 20.0f}, {30.0f, 40.0f}}));

  ScopedShapedBuffer result =
      ExecuteLocallyOrDie(computation, {&x_array, &y_array});

  EXPECT_TRUE(ShapeUtil::IsTuple(result.on_host_shape()));
  EXPECT_EQ(3, ShapeUtil::TupleElementCount(result.on_host_shape()));

  std::unique_ptr<Literal> result_literal = ShapedBufferToLiteral(result);
  LiteralTestUtil::ExpectR2Equal<float>({{1.0f, 2.0f}, {3.0f, 4.0f}},
                                        LiteralSlice(*result_literal, {0}));
  LiteralTestUtil::ExpectR2Equal<float>({{10.0f, 20.0f}, {30.0f, 40.0f}},
                                        LiteralSlice(*result_literal, {1}));
  LiteralTestUtil::ExpectR2Equal<float>({{1.0f, 2.0f}, {3.0f, 4.0f}},
                                        LiteralSlice(*result_literal, {2}));
}

XLA_TEST_F(LocalClientExecuteTest, NestedTupleResult) {
  XlaBuilder builder(TestName());
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {2, 2}), "x");
  auto y = builder.Parameter(1, ShapeUtil::MakeShape(F32, {2, 2}), "y");
  auto inner_tuple = builder.Tuple({x, y, x});
  builder.Tuple({inner_tuple, x});
  auto computation = builder.Build().ConsumeValueOrDie();

  auto x_array = LiteralToShapedBuffer(
      *Literal::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}}));
  auto y_array = LiteralToShapedBuffer(
      *Literal::CreateR2<float>({{10.0f, 20.0f}, {30.0f, 40.0f}}));

  ScopedShapedBuffer result =
      ExecuteLocallyOrDie(computation, {&x_array, &y_array});

  EXPECT_TRUE(ShapeUtil::IsTuple(result.on_host_shape()));
  EXPECT_EQ(2, ShapeUtil::TupleElementCount(result.on_host_shape()));

  std::unique_ptr<Literal> result_literal = ShapedBufferToLiteral(result);
  LiteralTestUtil::ExpectR2Equal<float>({{1.0f, 2.0f}, {3.0f, 4.0f}},
                                        LiteralSlice(*result_literal, {1}));
  LiteralTestUtil::ExpectR2Equal<float>({{1.0f, 2.0f}, {3.0f, 4.0f}},
                                        LiteralSlice(*result_literal, {0, 0}));
  LiteralTestUtil::ExpectR2Equal<float>({{10.0f, 20.0f}, {30.0f, 40.0f}},
                                        LiteralSlice(*result_literal, {0, 1}));
  LiteralTestUtil::ExpectR2Equal<float>({{1.0f, 2.0f}, {3.0f, 4.0f}},
                                        LiteralSlice(*result_literal, {0, 2}));
}

XLA_TEST_F(LocalClientExecuteTest, TupleResultWithLayout) {
  // Verify setting the result layout of a computation with a tuple output.
  XlaBuilder builder(TestName());
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {2, 2}), "x");
  auto y = builder.Parameter(1, ShapeUtil::MakeShape(F32, {2, 2}), "y");
  builder.Tuple({x, y});

  auto array = LiteralToShapedBuffer(
      *Literal::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}}));

  ExecutableBuildOptions options = DefaultExecutableBuildOptions();
  Shape shape_with_layout = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShapeWithLayout(F32, /*dimensions=*/{2, 2},
                                      /*minor_to_major=*/{0, 1}),
       ShapeUtil::MakeShapeWithLayout(F32, /*dimensions=*/{2, 2},
                                      /*minor_to_major=*/{1, 0})});
  options.set_result_layout(shape_with_layout);
  ScopedShapedBuffer result =
      ExecuteLocallyOrDie(builder.Build().ValueOrDie(), {&array, &array},
                          options, DefaultExecutableRunOptions());

  std::unique_ptr<Literal> result_literal = ShapedBufferToLiteral(result);
  LiteralTestUtil::ExpectR2Equal<float>({{1.0f, 2.0f}, {3.0f, 4.0f}},
                                        LiteralSlice(*result_literal, {0}));
  LiteralTestUtil::ExpectR2Equal<float>({{1.0f, 2.0f}, {3.0f, 4.0f}},
                                        LiteralSlice(*result_literal, {1}));
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
  auto x = builder.Parameter(0, tuple_shape0, "x");
  auto y = builder.Parameter(1, tuple_shape1, "y");
  auto x_0 = builder.GetTupleElement(x, 0);
  auto x_1 = builder.GetTupleElement(x, 1);
  auto y_0 = builder.GetTupleElement(y, 0);
  auto y_1 = builder.GetTupleElement(y, 1);
  auto array_sum = builder.Add(x_0, y_1);
  auto vector_diff = builder.Sub(x_1, y_0);
  builder.Tuple({array_sum, vector_diff});
  auto computation = builder.Build().ConsumeValueOrDie();

  auto x_literal = Literal::MakeTuple(
      {Literal::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}}).get(),
       Literal::CreateR1<float>({42.0, 75.0, 123.0}).get()});
  auto y_literal = Literal::MakeTuple(
      {Literal::CreateR1<float>({2.0, 4.0, 6.0}).get(),
       Literal::CreateR2<float>({{55.0, 44.0}, {33.0, 22.0}}).get()});

  auto x_buffer = LiteralToShapedBuffer(*x_literal);
  auto y_buffer = LiteralToShapedBuffer(*y_literal);

  ScopedShapedBuffer result =
      ExecuteLocallyOrDie(computation, {&x_buffer, &y_buffer});

  EXPECT_TRUE(ShapeUtil::IsTuple(result.on_host_shape()));
  EXPECT_EQ(2, ShapeUtil::TupleElementCount(result.on_host_shape()));

  std::unique_ptr<Literal> result_literal = ShapedBufferToLiteral(result);
  LiteralTestUtil::ExpectR2Equal<float>({{56.0f, 46.0f}, {36.0f, 26.0f}},
                                        LiteralSlice(*result_literal, {0}));
  LiteralTestUtil::ExpectR1Equal<float>({40.0f, 71.0f, 117.0f},
                                        LiteralSlice(*result_literal, {1}));
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
  auto param = builder.Parameter(0, nested_tuple_shape, "param");
  auto inner_tuple = builder.GetTupleElement(param, 0);
  auto inner_array = builder.GetTupleElement(inner_tuple, 0);
  auto inner_vector = builder.GetTupleElement(inner_tuple, 1);
  auto outer_vector = builder.GetTupleElement(param, 1);

  auto negate_array = builder.Neg(inner_array);
  auto vector_sum = builder.Add(inner_vector, outer_vector);
  builder.Tuple({negate_array, vector_sum});
  auto computation = builder.Build().ConsumeValueOrDie();

  auto arg_literal = Literal::MakeTuple(
      {Literal::MakeTuple(
           {Literal::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}}).get(),
            Literal::CreateR1<float>({42.0, 75.0, 123.0}).get()})
           .get(),
       Literal::CreateR1<float>({222.0, -2.0, 10.0}).get()});
  auto arg_buffer = LiteralToShapedBuffer(*arg_literal);

  ScopedShapedBuffer result = ExecuteLocallyOrDie(computation, {&arg_buffer});

  std::unique_ptr<Literal> result_literal = ShapedBufferToLiteral(result);
  LiteralTestUtil::ExpectR2Equal<float>({{-1.0, -2.0}, {-3.0, -4}},
                                        LiteralSlice(*result_literal, {0}));
  LiteralTestUtil::ExpectR1Equal<float>({264.0, 73.0, 133.0},
                                        LiteralSlice(*result_literal, {1}));
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
  auto param = builder.Parameter(0, tuple_shape, "param");
  auto element_0 = builder.GetTupleElement(param, 0);
  auto element_1 = builder.GetTupleElement(param, 1);
  builder.Tuple({builder.Neg(element_0), builder.Add(element_1, element_1)});
  auto computation = builder.Build().ConsumeValueOrDie();

  auto arg_literal = Literal::MakeTuple(
      {Literal::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}}).get(),
       Literal::CreateR2<float>({{11.0, 3.0}, {4.0, 5.0}}).get()});
  auto arg_buffer = LiteralToShapedBuffer(*arg_literal);

  ScopedShapedBuffer result_0 = ExecuteLocallyOrDie(computation, {&arg_buffer});
  std::unique_ptr<Literal> result_0_literal = ShapedBufferToLiteral(result_0);
  LiteralTestUtil::ExpectR2Equal<float>({{-1.0, -2.0}, {-3.0, -4.0}},
                                        LiteralSlice(*result_0_literal, {0}));
  LiteralTestUtil::ExpectR2Equal<float>({{22.0, 6.0}, {8.0, 10}},
                                        LiteralSlice(*result_0_literal, {1}));

  ScopedShapedBuffer result_1 = ExecuteLocallyOrDie(computation, {&result_0});
  std::unique_ptr<Literal> result_1_literal = ShapedBufferToLiteral(result_1);
  LiteralTestUtil::ExpectR2Equal<float>({{1.0, 2.0}, {3.0, 4.0}},
                                        LiteralSlice(*result_1_literal, {0}));
  LiteralTestUtil::ExpectR2Equal<float>({{44.0, 12.0}, {16.0, 20}},
                                        LiteralSlice(*result_1_literal, {1}));
}

XLA_TEST_F(LocalClientExecuteTest, LargeTuple) {
  // Construct a computation which takes a tuple parameter with a very large
  // number of elements.

  // A larger number of elements would make for a better, more strenuous test,
  // but:
  // TODO(b/66959878): On cpu a large number of elements results in long
  //   compilation time.
  // TODO(b/66954197): On gpu a large number of elements OOMs.
  const int kElementCount = 100;

  // Each element is a 2-element vector.
  const Shape element_shape = ShapeUtil::MakeShape(F32, {2});
  std::vector<Shape> element_shapes(kElementCount, element_shape);
  const Shape tuple_shape = ShapeUtil::MakeTupleShape(element_shapes);

  XlaBuilder builder(TestName());
  auto param = builder.Parameter(0, tuple_shape, "param");

  // Add each element's tuple index value to every element.
  std::vector<XlaOp> result_elements;
  for (int i = 0; i < kElementCount; ++i) {
    auto element = builder.GetTupleElement(param, i);
    result_elements.push_back(
        builder.Add(element, builder.ConstantR0<float>(i)));
  }
  builder.Tuple(result_elements);
  auto computation = builder.Build().ConsumeValueOrDie();

  // Feed in a tuple where each two-element vector element is {tuple_index,
  // -tuple_index}.
  std::vector<std::unique_ptr<Literal>> arg_elements;
  for (int i = 0; i < kElementCount; ++i) {
    arg_elements.push_back(Literal::CreateR1<float>({1.0f * i, -1.0f * i}));
  }
  std::unique_ptr<Literal> arg_literal =
      Literal::MakeTupleOwned(std::move(arg_elements));
  auto arg_buffer = LiteralToShapedBuffer(*arg_literal);

  ScopedShapedBuffer result = ExecuteLocallyOrDie(computation, {&arg_buffer});
  std::unique_ptr<Literal> result_literal = ShapedBufferToLiteral(result);

  for (int i = 0; i < kElementCount; ++i) {
    LiteralTestUtil::ExpectR1Near<float>(
        {2.0f * i, 0.0f}, LiteralSlice(*result_literal, {i}), error_spec_);
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
  auto param = builder.Parameter(0, tuple_shape, "param");

  // The computation increments each leaf value by an amount equal to the leaf's
  // ordinal position in a traversal of the tuple.
  std::vector<XlaOp> result_elements;
  for (int i = 0; i < kFanout; ++i) {
    auto outer_element = builder.GetTupleElement(param, i);
    std::vector<XlaOp> inner_result_elements;
    for (int j = 0; j < kFanout; ++j) {
      auto inner_element = builder.GetTupleElement(outer_element, j);
      inner_result_elements.push_back(builder.Add(
          inner_element, builder.ConstantR0<float>(i * kFanout + j)));
    }
    result_elements.push_back(builder.Tuple(inner_result_elements));
  }
  builder.Tuple(result_elements);
  auto computation = builder.Build().ConsumeValueOrDie();

  // Construct the argument to pass to the computation.
  std::vector<std::unique_ptr<Literal>> outer_tuple_elements;
  for (int i = 0; i < kFanout; ++i) {
    std::vector<std::unique_ptr<Literal>> inner_tuple_elements;
    for (int j = 0; j < kFanout; ++j) {
      inner_tuple_elements.push_back(Literal::CreateR0<float>(i + j));
    }
    outer_tuple_elements.push_back(
        Literal::MakeTupleOwned(std::move(inner_tuple_elements)));
  }
  auto arg_literal = Literal::MakeTupleOwned(std::move(outer_tuple_elements));
  auto arg_buffer = LiteralToShapedBuffer(*arg_literal);

  ScopedShapedBuffer result = ExecuteLocallyOrDie(computation, {&arg_buffer});
  std::unique_ptr<Literal> result_literal = ShapedBufferToLiteral(result);

  for (int i = 0; i < kFanout; ++i) {
    for (int j = 0; j < kFanout; ++j) {
      LiteralTestUtil::ExpectR0Near<float>(
          i + j + i * kFanout + j, LiteralSlice(*result_literal, {i, j}),
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
  auto element = builder.Parameter(0, shape, "param");
  for (int i = 0; i < kTupleDepth; ++i) {
    element = builder.GetTupleElement(element, 0);
  }

  auto output = builder.Add(element, builder.ConstantR0<float>(42.0));
  for (int i = 0; i < kTupleDepth; ++i) {
    output = builder.Tuple({output});
  }
  auto computation = builder.Build().ConsumeValueOrDie();

  // Construct the argument to pass to the computation.
  std::unique_ptr<Literal> arg_literal = Literal::CreateR0<float>(123.0);
  for (int i = 0; i < kTupleDepth; ++i) {
    std::vector<std::unique_ptr<Literal>> arg_vector;
    arg_vector.push_back(std::move(arg_literal));
    arg_literal = Literal::MakeTupleOwned(std::move(arg_vector));
  }
  auto arg_buffer = LiteralToShapedBuffer(*arg_literal);

  ScopedShapedBuffer result = ExecuteLocallyOrDie(computation, {&arg_buffer});
  std::unique_ptr<Literal> result_literal = ShapedBufferToLiteral(result);

  ShapeIndex index;
  for (int i = 0; i < kTupleDepth; ++i) {
    index.push_back(0);
  }
  LiteralTestUtil::ExpectR0Equal<float>(165.0,
                                        LiteralSlice(*result_literal, index));
}

XLA_TEST_F(LocalClientExecuteTest, InvalidNumberOfArguments) {
  // Test passing in an invalid number of arguments.
  XlaBuilder builder(TestName());
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {3}), "x");
  auto y = builder.Parameter(1, ShapeUtil::MakeShape(F32, {3}), "y");
  builder.Add(x, y);

  auto x_array =
      LiteralToShapedBuffer(*Literal::CreateR1<float>({1.0f, 2.0f, 3.0f}));
  auto execute_status =
      ExecuteLocally(builder.Build().ValueOrDie(), {&x_array});

  EXPECT_FALSE(execute_status.ok());
  EXPECT_THAT(execute_status.status().error_message(),
              ContainsRegex("Invalid number of arguments"));
}

XLA_TEST_F(LocalClientExecuteTest, IncorrectArgumentShape) {
  // Test passing in an argument with the wrong shape.
  XlaBuilder builder(TestName());
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {3}), "x");
  builder.Neg(x);

  auto x_array = LiteralToShapedBuffer(
      *Literal::CreateR2<float>({{0.0f, 1.0f}, {2.0f, 3.0f}}));
  auto execute_status =
      ExecuteLocally(builder.Build().ValueOrDie(), {&x_array});

  EXPECT_FALSE(execute_status.ok());
  EXPECT_THAT(execute_status.status().error_message(),
              ContainsRegex("Invalid argument shape"))
      << execute_status.status();
}

XLA_TEST_F(LocalClientExecuteTest, InvalidResultLayout) {
  // Test passing in an invalid result layout parameter.
  XlaBuilder builder(TestName());
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {2, 2}), "x");
  builder.Neg(x);

  auto x_array = LiteralToShapedBuffer(
      *Literal::CreateR2<float>({{0.0f, 1.0f}, {2.0f, 3.0f}}));
  auto execute_status = ExecuteLocally(
      builder.Build().ValueOrDie(), {&x_array},
      DefaultExecutableBuildOptions().set_result_layout(
          ShapeUtil::MakeShapeWithLayout(F32,
                                         /*dimensions=*/{1, 2, 3, 4},
                                         /*minor_to_major=*/{0, 1, 2, 3})),
      DefaultExecutableRunOptions());

  EXPECT_FALSE(execute_status.ok());
  EXPECT_THAT(execute_status.status().error_message(),
              ContainsRegex("not compatible with result shape"))
      << execute_status.status();
}

XLA_TEST_F(LocalClientExecuteTest, RunOnAllDeviceOrdinals) {
  // Try to run a trivial computation on every device on the system. If a
  // specific device is not supported, check that the right error is returned.
  XlaBuilder builder(TestName());
  builder.ConstantR0<float>(42.0f);
  auto computation = builder.Build().ConsumeValueOrDie();
  for (int d = 0; d < local_client_->device_count(); ++d) {
    if (!local_client_->device_ordinal_supported(d)) {
      auto execute_status =
          ExecuteLocally(computation, {},
                         DefaultExecutableBuildOptions().set_device_ordinal(d),
                         DefaultExecutableRunOptions().set_device_ordinal(d));
      EXPECT_FALSE(execute_status.ok());
      EXPECT_THAT(execute_status.status().error_message(),
                  ContainsRegex("device .* not supported"));
    } else {
      auto result = ExecuteLocallyOrDie(
          computation, {},
          DefaultExecutableBuildOptions().set_device_ordinal(d),
          DefaultExecutableRunOptions().set_device_ordinal(d));
      EXPECT_EQ(d, result.device_ordinal());
      LiteralTestUtil::ExpectR0Equal<float>(42.0f,
                                            *ShapedBufferToLiteral(result));
    }
  }
}

XLA_TEST_F(LocalClientExecuteTest, InvalidDeviceOrdinalValues) {
  // Try running computations on devices with device ordinal values which do not
  // exist.
  XlaBuilder builder(TestName());
  builder.ConstantR0<float>(42.0f);
  auto computation = builder.Build().ConsumeValueOrDie();

  auto execute_status =
      ExecuteLocally(computation, {},
                     DefaultExecutableBuildOptions().set_device_ordinal(
                         local_client_->device_count()),
                     DefaultExecutableRunOptions().set_device_ordinal(
                         local_client_->device_count()));
  EXPECT_FALSE(execute_status.ok());
  EXPECT_THAT(execute_status.status().error_message(),
              ContainsRegex("Invalid device ordinal value"));
}

XLA_TEST_F(LocalClientExecuteTest, RunOnStream) {
  // Run a computation on a specific stream on each device on the system.
  XlaBuilder builder(TestName());
  builder.ConstantR0<float>(42.0f);
  auto computation = builder.Build().ConsumeValueOrDie();

  for (int d = 0; d < local_client_->device_count(); ++d) {
    if (!local_client_->device_ordinal_supported(d)) {
      continue;
    }
    se::StreamExecutor* executor =
        local_client_->platform()->ExecutorForDevice(d).ValueOrDie();
    se::Stream stream(executor);
    stream.Init();

    auto result =
        ExecuteLocallyOrDie(computation, {}, DefaultExecutableBuildOptions(),
                            DefaultExecutableRunOptions().set_stream(&stream));
    // As a check to verify that the computation ran of the device associated
    // with the stream. This is a weak check, but stronger verification is hard.
    EXPECT_EQ(d, result.device_ordinal());
    LiteralTestUtil::ExpectR0Equal<float>(42.0f,
                                          *ShapedBufferToLiteral(result));
  }
}

// Disable this test on CPU because we're using the CPU as the platform
// which does not match the service platform.
XLA_TEST_F(LocalClientExecuteTest,
           DISABLED_ON_CPU(RunOnStreamForWrongPlatform)) {
  // Try to run a computation on a stream for a platform (CPU) which does not
  // match the platform of the service (!= CPU).
  se::Platform* wrong_platform =
      se::MultiPlatformManager::PlatformWithId(se::host::kHostPlatformId)
          .ValueOrDie();
  se::Stream wrong_stream(wrong_platform->ExecutorForDevice(0).ValueOrDie());
  wrong_stream.Init();

  XlaBuilder builder(TestName());
  builder.ConstantR0<float>(42.0f);
  auto execute_status = ExecuteLocally(
      builder.Build().ValueOrDie(), {}, DefaultExecutableBuildOptions(),
      DefaultExecutableRunOptions().set_stream(&wrong_stream));
  EXPECT_FALSE(execute_status.ok());
  EXPECT_THAT(execute_status.status().error_message(),
              ContainsRegex("stream is for platform .*, but service targets"));
}

XLA_TEST_F(LocalClientExecuteTest,
           DISABLED_ON_CPU(AllocatorDoesNotMatchPlatform)) {
  se::Platform* wrong_platform =
      se::MultiPlatformManager::PlatformWithId(se::host::kHostPlatformId)
          .ValueOrDie();
  TestAllocator allocator(wrong_platform);

  XlaBuilder builder(TestName());
  auto y = builder.ConstantR0<float>(123.0f);

  auto execute_status = ExecuteLocally(
      builder.Build().ValueOrDie(), {}, DefaultExecutableBuildOptions(),
      DefaultExecutableRunOptions().set_allocator(&allocator));
  EXPECT_FALSE(execute_status.ok());
  EXPECT_THAT(execute_status.status().error_message(),
              ContainsRegex("allocator platform .* does not match service"));
}

XLA_TEST_F(LocalClientExecuteTest, RunOnUninitializedStream) {
  // Try to run a computation on a stream that has not been initialized.
  XlaBuilder builder(TestName());
  builder.ConstantR0<float>(42.0f);

  LOG(INFO) << "default device = " << local_client_->default_device_ordinal();
  se::StreamExecutor* executor =
      local_client_->platform()
          ->ExecutorForDevice(local_client_->default_device_ordinal())
          .ValueOrDie();
  se::Stream stream(executor);
  // Don't call stream.Init().

  auto execute_status = ExecuteLocally(
      builder.Build().ValueOrDie(), {}, DefaultExecutableBuildOptions(),
      DefaultExecutableRunOptions().set_stream(&stream));
  EXPECT_FALSE(execute_status.ok());
  EXPECT_THAT(execute_status.status().error_message(),
              ContainsRegex("stream is uninitialized or in an error state"));
}

XLA_TEST_F(LocalClientExecuteTest, SelectBetweenTuples) {
  XlaBuilder builder(TestName());

  std::initializer_list<float> vec1 = {1.f, 2.f, 3.f};
  std::initializer_list<float> vec2 = {2.f, 4.f, 6.f};
  auto tuple12 = builder.Tuple(
      {builder.ConstantR1<float>(vec1), builder.ConstantR1<float>(vec2)});
  auto tuple21 = builder.Tuple(
      {builder.ConstantR1<float>(vec2), builder.ConstantR1<float>(vec1)});
  builder.Select(builder.ConstantR0<bool>(false), tuple12, tuple21);

  ScopedShapedBuffer result =
      ExecuteLocallyOrDie(builder.Build().ValueOrDie(), {});
  std::unique_ptr<Literal> tuple_literal = ShapedBufferToLiteral(result);
  LiteralTestUtil::ExpectR1Equal<float>({2.0f, 4.0f, 6.0f},
                                        LiteralSlice(*tuple_literal, {0}));
  LiteralTestUtil::ExpectR1Equal<float>({1.0f, 2.0f, 3.0f},
                                        LiteralSlice(*tuple_literal, {1}));
}

XLA_TEST_F(LocalClientExecuteTest, CompileExecutable) {
  XlaBuilder builder(TestName());
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {3}), "x");
  auto y = builder.ConstantR1<float>({2.0f, 3.0f, 4.0f});
  builder.Add(x, y);

  Shape argument_layout =
      ShapeUtil::MakeShapeWithLayout(F32, /*dimensions=*/{3}, {0});
  auto executable_status =
      local_client_->Compile(builder.Build().ValueOrDie(), {&argument_layout},
                             ExecutableBuildOptions());
  ASSERT_IS_OK(executable_status);
  std::unique_ptr<LocalExecutable> executable =
      executable_status.ConsumeValueOrDie();

  auto x_array =
      LiteralToShapedBuffer(*Literal::CreateR1<float>({0.0f, 1.0f, 2.0f}));
  ScopedShapedBuffer result =
      executable->Run({&x_array}, DefaultExecutableRunOptions())
          .ConsumeValueOrDie();

  LiteralTestUtil::ExpectR1Near<float>(
      {2.0f, 4.0f, 6.0f}, *ShapedBufferToLiteral(result), error_spec_);
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
    EXPECT_EQ(literal, *transferred_literal);
  };

  // Array shapes.
  test_to_device_and_back(*Literal::CreateR0<float>(42.0));
  test_to_device_and_back(*Literal::CreateR0<bool>(true));
  test_to_device_and_back(*Literal::CreateR1<float>({1.0, 42.0, 744.4}));
  test_to_device_and_back(
      *Literal::CreateR2<float>({{1.0, 2.0, 3.0}, {44.0, 0.1, -3}}));
  test_to_device_and_back(*Literal::CreateR2<int32>({{2, 1}, {4444, 56}}));

  // Null shape (empty tuple).
  test_to_device_and_back(*Literal::MakeTuple({}));

  // Non-nested tuples.
  test_to_device_and_back(
      *Literal::MakeTuple({Literal::CreateR0<float>(12223.0).get()}));
  test_to_device_and_back(
      *Literal::MakeTuple({Literal::CreateR1<float>({1.0, -42.0}).get(),
                           Literal::CreateR0<float>(123456.0).get()}));

  // Nested tuple.
  test_to_device_and_back(*Literal::MakeTuple(
      {Literal::MakeTuple({Literal::CreateR1<float>({1.0, -42.0}).get(),
                           Literal::CreateR0<float>(123456.0).get()})
           .get(),
       Literal::CreateR0<bool>(false).get()}));
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
    EXPECT_EQ(literal, *transferred_literal);
  };

  test_to_device_and_back(
      *Literal::CreateR2<double>({{1.0, 2.0, 3.0}, {44.0, 0.1, -3}}));
  test_to_device_and_back(*Literal::CreateR2<int64>({{2, 1}, {4444, 56}}));
  test_to_device_and_back(
      *Literal::CreateR2<uint64>({{20000000000ULL, 1}, {4444, 56}}));
  test_to_device_and_back(
      *Literal::MakeTuple({Literal::CreateR1<double>({1.0, -42.0}).get(),
                           Literal::CreateR0<int64>(123456789000LL).get()}));
}

// TODO(b/34359662): Support infeed/outfeed on GPU and CPU parallel.
// 2017-10-18.
XLA_TEST_F(LocalClientExecuteTest, DISABLED_ON_GPU(InfeedOutfeedTest)) {
  XlaBuilder builder(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {3});
  auto in = builder.Infeed(shape);
  auto constant = builder.ConstantR1<float>({1.0f, 2.0f, 3.0f});
  auto sum = builder.Add(in, constant);
  builder.Outfeed(sum, shape, /*outfeed_config=*/"");

  std::unique_ptr<tensorflow::Thread> thread(
      tensorflow::Env::Default()->StartThread(
          tensorflow::ThreadOptions(), "execute_thread",
          [&] { ExecuteLocallyOrDie(builder.Build().ValueOrDie(), {}); }));

  ASSERT_IS_OK(local_client_->TransferToInfeedLocal(
      *Literal::CreateR1<float>({-5.0, 123.0, 42.0}),
      local_client_->default_device_ordinal()));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Literal> result,
                          local_client_->TransferFromOutfeedLocal(
                              shape, local_client_->default_device_ordinal()));

  LiteralTestUtil::ExpectR1Equal<float>({-4.0, 125.0, 45.0}, *result);
}

// Benchmark that measures the overhead of the LocalClient API when running a
// trivial computation
void BM_LocalClientOverhead(int num_iters) {
  tensorflow::testing::StopTiming();

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().ValueOrDie();
  auto executors = PlatformUtil::GetStreamExecutors(platform).ValueOrDie();
  StreamExecutorMemoryAllocator allocator(platform, executors);
  LocalClient* client =
      ClientLibrary::GetOrCreateLocalClient(platform).ValueOrDie();
  auto* transfer_manager =
      TransferManager::GetForPlatform(platform).ValueOrDie();
  int device_ordinal = client->default_device_ordinal();

  // Use a tiny add operation as the computation.
  XlaBuilder builder("Add");
  auto shape = ShapeUtil::MakeShape(F32, {2, 3});
  auto x = builder.Parameter(0, shape, "x");
  builder.Add(x, x);
  auto computation = builder.Build().ConsumeValueOrDie();

  auto buffer =
      transfer_manager
          ->AllocateScopedShapedBuffer(shape, &allocator, /*device_ordinal=*/0)
          .ConsumeValueOrDie();
  auto literal = Literal::CreateR2<float>({{0, 0, 0}, {0, 0, 0}});
  auto stream =
      client->mutable_backend()->BorrowStream(device_ordinal).ValueOrDie();
  ASSERT_IS_OK(transfer_manager->TransferLiteralToDevice(stream.get(), *literal,
                                                         buffer));

  const int kWarmups = 2;

  auto executable_status = client->Compile(
      computation, {&buffer.on_host_shape()}, ExecutableBuildOptions());
  ASSERT_IS_OK(executable_status);
  std::unique_ptr<LocalExecutable> executable =
      executable_status.ConsumeValueOrDie();

  ExecutableRunOptions run_options;
  run_options.set_allocator(&allocator).set_stream(stream.get());

  for (int i = 0; i < kWarmups; ++i) {
    auto result = executable->Run({&buffer}, run_options);
    ASSERT_IS_OK(result);
  }

  tensorflow::testing::StartTiming();
  for (int i = 0; i < num_iters; ++i) {
    auto result = executable->Run({&buffer}, run_options);
    ASSERT_IS_OK(result);
  }
}

BENCHMARK(BM_LocalClientOverhead);

}  // namespace
}  // namespace xla
