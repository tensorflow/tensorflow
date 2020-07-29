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

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class WhileTest : public ClientLibraryTestBase {};

// Tests a while node when the result type T is S32.
//
// int32 result = 0;
// while (result < 5) {
//   result = result + 1;
// }
XLA_TEST_F(WhileTest, WhileWithScalarS32Result) {
  auto result_shape = ShapeUtil::MakeShape(S32, {});

  // Create a computation for the condition: repeat for 5 iterations.
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    Gt(ConstantR0<int32>(&builder, 5), prev);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body: add 1 to the result variable.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto input = ConstantR0<int32>(&builder, 1);
    Add(input, prev);
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  XlaBuilder builder(TestName());
  auto init = ConstantR0<int32>(&builder, 0);
  While(condition, body, init);

  ComputeAndCompareR0<int32>(&builder, 5, {});
}

// Tests a while node when the result type T is S64.
//
// int32 result = 0;
// while (result < 5) {
//   result = result + 1;
// }
XLA_TEST_F(WhileTest, WhileWithScalarS64Result) {
  auto result_shape = ShapeUtil::MakeShape(S64, {});

  // Create a computation for the condition: repeat for 5 iterations.
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    Gt(ConstantR0<int64>(&builder, 5), prev);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body: add 1 to the result variable.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto input = ConstantR0<int64>(&builder, 1);
    Add(input, prev);
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  XlaBuilder builder(TestName());
  auto init = ConstantR0<int64>(&builder, 0);
  While(condition, body, init);

  ComputeAndCompareR0<int64>(&builder, 5, {});
}

XLA_TEST_F(WhileTest, WhileWithScalarResultNonConstInit) {
  auto result_shape = ShapeUtil::MakeShape(S32, {});
  auto orig_shape = ShapeUtil::MakeShape(S32, {2});

  // Create a computation for the condition: repeat for 5 iterations.
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    Gt(ConstantR0<int32>(&builder, 5), prev);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body: add 1 to the result variable.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto input = ConstantR0<int32>(&builder, 1);
    Add(input, prev);
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  XlaBuilder builder(TestName());
  auto init =
      Reduce(ConstantR1<int32>(&builder, 2, 1), ConstantR0<int32>(&builder, 0),
             CreateScalarAddComputation(S32, &builder), {0});
  While(condition, body, init);

  ComputeAndCompareR0<int32>(&builder, 5, {});
}

XLA_TEST_F(WhileTest, WhileWithPredicateResult) {
  auto result_shape = ShapeUtil::MakeShape(PRED, {});

  // Create a computation for the condition: run until condition is true.
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    Ne(ConstantR0<bool>(&builder, true), prev);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body: or condition with true.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    Or(prev, ConstantR0<bool>(&builder, true));
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  XlaBuilder builder(TestName());
  auto init =
      Ne(ConstantR0<bool>(&builder, false), ConstantR0<bool>(&builder, true));
  While(condition, body, init);

  ComputeAndCompareR0<bool>(&builder, true, {});
}

// Tests a while node when the result type T is a vector.
//
// All constants are chosen to produce exact results.
// vector<float> result(0);
// while (result.sum() < 15.5f) {
//   result = result + vector<float>(0);
// }
XLA_TEST_F(WhileTest, DISABLED_ON_INTERPRETER(WhileWithEmptyVectorResult)) {
  Shape result_shape = ShapeUtil::MakeShape(F32, {0});

  // Create a computation for the reduction.
  XlaComputation add;
  {
    XlaBuilder builder("add");
    auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "x");
    auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {}), "y");
    Add(x, y);
    add = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the condition.
  // Repeat until the sum of the result vector is less than 15.5f.
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto sum = Reduce(prev, ConstantR0<float>(&builder, 0.0f), add,
                      /*dimensions_to_reduce=*/{0});
    Gt(ConstantR0<float>(&builder, 15.5f), sum);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body.
  // Add a constant vector of 1.f to the result vector.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto input = ConstantR1<float>(&builder, {});
    Add(input, prev);
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  XlaBuilder builder("while");
  auto init = ConstantR1<float>(&builder, {});
  auto result = While(condition, body, init);
  VLOG(2) << "while = "
          << ShapeUtil::HumanString(
                 builder.GetShape(result).ConsumeValueOrDie());

  ComputeAndCompareR1<float>(&builder, {}, {}, ErrorSpec(0.0001));
}

// Tests a while node when the result type T is a vector.
//
// All constants are chosen to produce exact results.
// vector<float> result(8, 0.0f);
// while (result.sum() < 15.5f) {
//   result = result + vector<float>(8, 0.125f);
// }
XLA_TEST_F(WhileTest, WhileWithVectorResult) {
  Shape result_shape = ShapeUtil::MakeShape(F32, {8});

  // Create a computation for the reduction.
  XlaComputation add;
  {
    XlaBuilder builder("add");
    auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "x");
    auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {}), "y");
    Add(x, y);
    add = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the condition.
  // Repeat until the sum of the result vector is less than 5.5f.
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto sum = Reduce(prev, ConstantR0<float>(&builder, 0.0f), add,
                      /*dimensions_to_reduce=*/{0});
    Gt(ConstantR0<float>(&builder, 15.5f), sum);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body.
  // Add a constant vector of 1.f to the result vector.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto input = ConstantR1<float>(&builder, 8, 0.125f);
    Add(input, prev);
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  XlaBuilder builder("while");
  auto init = ConstantR1<float>(&builder, 8, 0.f);
  auto result = While(condition, body, init);
  VLOG(2) << "while = "
          << ShapeUtil::HumanString(
                 builder.GetShape(result).ConsumeValueOrDie());

  // Individual elements with increase by 1/8 each time through the loop, so
  // the sum will increase by 1.0.  It will first be >15.5 when the elements
  // have all reached 2.0.
  std::vector<float> expected = {2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

// Tests a while node when the result type is a vector which is part
// of the result tuple.
//
// All constants are chosen to produce exact results.
// vector<float> result(8, 0.0f);
// while (result.sum() < 15.5f) {
//   result = result + vector<float>(8, 0.125f);
// }
// tuple = tuple { while }
XLA_TEST_F(WhileTest, WhileWithVectorResultIntoTuple) {
  Shape result_shape = ShapeUtil::MakeShape(F32, {8});

  // Create a computation for the reduction.
  XlaComputation add;
  {
    XlaBuilder builder("add");
    auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "x");
    auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {}), "y");
    Add(x, y);
    add = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the condition.
  // Repeat until the sum of the result vector is less than 5.5f.
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto sum = Reduce(prev, ConstantR0<float>(&builder, 0.0f), add,
                      /*dimensions_to_reduce=*/{0});
    Gt(ConstantR0<float>(&builder, 15.5f), sum);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body.
  // Add a constant vector of 1.f to the result vector.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto input = ConstantR1<float>(&builder, 8, 0.125f);
    Add(input, prev);
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  XlaBuilder builder("while");
  auto init = ConstantR1<float>(&builder, 8, 0.f);
  auto result = While(condition, body, init);
  VLOG(2) << "while = "
          << ShapeUtil::HumanString(
                 builder.GetShape(result).ConsumeValueOrDie());
  Tuple(&builder, {result});

  // Individual elements with increase by 1/8 each time through the loop, so
  // the sum will increase by 1.0.  It will first be >15.5 when the elements
  // have all reached 2.0.
  auto expected_data =
      LiteralUtil::CreateR1<float>({2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f});
  auto expected = LiteralUtil::MakeTuple({&expected_data});
  VLOG(2) << "expected = " << ShapeUtil::HumanString(expected.shape());
  ComputeAndCompareTuple(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(WhileTest, WhileWithPermutationAndTupleResult) {
  std::vector<Shape> shape_elements = {
      ShapeUtil::MakeShape(S32, {}), ShapeUtil::MakeShape(F32, {3}),
      ShapeUtil::MakeShape(F32, {3}), ShapeUtil::MakeShape(F32, {3})};
  Shape result_shape = ShapeUtil::MakeTupleShape(shape_elements);

  // Create a computation for the condition.
  // Repeat for N iterations.
  const int N = 2;
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    Gt(ConstantR0<int32>(&builder, N), iteration);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body.
  // Add 1 to the iteration variable and permute the weights.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    auto w1 = GetTupleElement(prev, 1);
    auto w2 = GetTupleElement(prev, 2);
    auto w3 = GetTupleElement(prev, 3);
    Tuple(&builder,
          {Add(iteration, ConstantR0<int32>(&builder, 1)), w3, w1, w2});
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  XlaBuilder builder("while");
  auto init = Tuple(&builder, {ConstantR0<int32>(&builder, 0),
                               ConstantR1<float>(&builder, 3, 1.f),
                               ConstantR1<float>(&builder, 3, 2.f),
                               ConstantR1<float>(&builder, 3, 3.f)});
  auto result = While(condition, body, init);
  VLOG(2) << "result = "
          << ShapeUtil::HumanString(
                 builder.GetShape(result).ConsumeValueOrDie());

  auto expected_counter = LiteralUtil::CreateR0<int32>(N);
  auto expected_w1 = LiteralUtil::CreateR1<float>({1.0f, 1.0f, 1.0f});
  auto expected_w2 = LiteralUtil::CreateR1<float>({2.0f, 2.0f, 2.0f});
  auto expected_w3 = LiteralUtil::CreateR1<float>({3.0f, 3.0f, 3.0f});
  auto expected = LiteralUtil::MakeTuple(
      {&expected_counter, &expected_w2, &expected_w3, &expected_w1});
  VLOG(2) << "expected = " << ShapeUtil::HumanString(expected.shape());
  ComputeAndCompareTuple(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(WhileTest, WhileWithPermutationAndVectorResult) {
  std::vector<Shape> shape_elements = {
      ShapeUtil::MakeShape(S32, {}), ShapeUtil::MakeShape(F32, {3}),
      ShapeUtil::MakeShape(F32, {3}), ShapeUtil::MakeShape(F32, {3})};
  Shape result_shape = ShapeUtil::MakeTupleShape(shape_elements);

  // Create a computation for the condition.
  // Repeat for N iterations.
  const int N = 2;
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    Gt(ConstantR0<int32>(&builder, N), iteration);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body.
  // Add 1 to the iteration variable permute the weights.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    auto w1 = GetTupleElement(prev, 1);
    auto w2 = GetTupleElement(prev, 2);
    auto w3 = GetTupleElement(prev, 3);
    Tuple(&builder,
          {Add(iteration, ConstantR0<int32>(&builder, 1)), w3, w1, w2});
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  XlaBuilder builder("while");
  auto init = Tuple(&builder, {ConstantR0<int32>(&builder, 0),
                               ConstantR1<float>(&builder, 3, 1.f),
                               ConstantR1<float>(&builder, 3, 2.f),
                               ConstantR1<float>(&builder, 3, 3.f)});
  auto xla_while = While(condition, body, init);

  auto add12 =
      Add(GetTupleElement(xla_while, 1), GetTupleElement(xla_while, 2));
  auto result = Add(add12, GetTupleElement(xla_while, 3));
  VLOG(2) << "result = "
          << ShapeUtil::HumanString(
                 builder.GetShape(result).ConsumeValueOrDie());
  std::vector<float> expected = {6.f, 6.f, 6.f};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

// Tests a while node when the result type T is a Tuple.
//
// tuple<int32, vector<float>> result(0, vector<float>(10, 0.0f));
// while (get<0>(result) < 5) {
//   get<0>(result) = get<0>(result) + 1;
//   get<1>(result) = get<1>(result) + vector<float>(10, 1.0f);
// }
XLA_TEST_F(WhileTest, WhileWithTupleResult) {
  std::vector<Shape> shape_elements = {ShapeUtil::MakeShape(S32, {}),
                                       ShapeUtil::MakeShape(F32, {10})};
  Shape result_shape = ShapeUtil::MakeTupleShape(shape_elements);

  // Create a computation for the condition.
  // Repeat for 5 iterations.
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    Gt(ConstantR0<int32>(&builder, 5), iteration);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body.
  // Add 1 to the iteration variable and add a constant vector of 1.0f to
  // the weight variable, both of which are tuple elements.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    auto weights = GetTupleElement(prev, 1);
    auto input = ConstantR1<float>(&builder, 10, 1.f);
    auto new_weights = Add(weights, input);
    Tuple(&builder,
          {Add(iteration, ConstantR0<int32>(&builder, 1)), new_weights});
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  XlaBuilder builder("while");
  auto init = Tuple(&builder, {ConstantR0<int32>(&builder, 0),
                               ConstantR1<float>(&builder, 10, 0.f)});
  auto result = While(condition, body, init);
  VLOG(2) << "while = "
          << ShapeUtil::HumanString(
                 builder.GetShape(result).ConsumeValueOrDie());

  auto expected_counter = LiteralUtil::CreateR0<int32>(5);
  auto expected_data = LiteralUtil::CreateR1<float>(
      {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f});
  auto expected = LiteralUtil::MakeTuple({&expected_counter, &expected_data});
  VLOG(2) << "expected = " << ShapeUtil::HumanString(expected.shape());
  ComputeAndCompareTuple(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(WhileTest, WhileWithPredicateTupleResult) {
  std::vector<Shape> shape_elements = {ShapeUtil::MakeShape(S32, {}),
                                       ShapeUtil::MakeShape(PRED, {})};
  Shape result_shape = ShapeUtil::MakeTupleShape(shape_elements);

  // Create a computation for the condition.
  // Repeat for 5 iterations.
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    Gt(ConstantR0<int32>(&builder, 5), iteration);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body.
  // Add 1 to the iteration variable and or the predicate with true
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    auto pred = GetTupleElement(prev, 1);
    auto new_pred = Or(pred, ConstantR0<bool>(&builder, true));
    Tuple(&builder, {Add(iteration, ConstantR0<int32>(&builder, 1)), new_pred});
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  XlaBuilder builder("while");
  auto init = Tuple(&builder, {ConstantR0<int32>(&builder, 0),
                               Ne(ConstantR0<bool>(&builder, false),
                                  ConstantR0<bool>(&builder, true))});
  auto result = While(condition, body, init);
  VLOG(2) << "while = "
          << ShapeUtil::HumanString(
                 builder.GetShape(result).ConsumeValueOrDie());

  auto expected_counter = LiteralUtil::CreateR0<int32>(5);
  auto expected_predicate = LiteralUtil::CreateR0<bool>(true);
  auto expected =
      LiteralUtil::MakeTuple({&expected_counter, &expected_predicate});
  ComputeAndCompareTuple(&builder, expected, {}, ErrorSpec(0));
}

XLA_TEST_F(WhileTest, WhileWithTupleConstantScalarResult) {
  std::vector<Shape> shape_elements = {ShapeUtil::MakeShape(S32, {}),
                                       ShapeUtil::MakeShape(S32, {})};
  Shape result_shape = ShapeUtil::MakeTupleShape(shape_elements);

  // Create a computation for the condition.
  // Repeat for 5 iterations.
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    Gt(ConstantR0<int32>(&builder, 5), iteration);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body.
  // Add 1 to the iteration variable and set the other tuple element to a
  // constant.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    Tuple(&builder, {Add(iteration, ConstantR0<int32>(&builder, 1)),
                     ConstantR0<int32>(&builder, 7)});
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  XlaBuilder builder("while");
  auto init = Tuple(&builder, {ConstantR0<int32>(&builder, 0),
                               ConstantR0<int32>(&builder, 7)});
  auto result = While(condition, body, init);
  VLOG(2) << "while = "
          << ShapeUtil::HumanString(
                 builder.GetShape(result).ConsumeValueOrDie());

  auto expected_counter = LiteralUtil::CreateR0<int32>(5);
  auto expected_data = LiteralUtil::CreateR0<int32>(7);
  auto expected = LiteralUtil::MakeTuple({&expected_counter, &expected_data});
  VLOG(2) << "expected = " << ShapeUtil::HumanString(expected.shape());
  ComputeAndCompareTuple(&builder, expected, {}, ErrorSpec(0.0001));
}

// Tests two while nodes when the result type T is a Tuple and the second
// while node uses the result of the first while node which is used in two
// nodes.
// tuple<int32, vector<float>> w0(0, vector<float>(10, 0.0f));
// w0 = while (get<0>(w0) < c1) {
//        get<0>(w0) = get<0>(w0) + 1;
//        get<1>(w0) = get<1>(w0) + vector<float>(10, 1.0f);
//      }
// tuple<int32, vector<float>> w1(get<0>(w0), get<1>(w0));
// w1 = while (get<0>(w1) < c2) {
//        get<0>(w1) = get<0>(w1) + 1;
//        get<1>(w1) = get<1>(w1) + vector<float>(10, 1.0f);
//      }
// result = get<1>(w0) + get<1>(w1)
XLA_TEST_F(WhileTest, TwoWhileWithTupleResult) {
  std::vector<Shape> shape_elements = {ShapeUtil::MakeShape(S32, {}),
                                       ShapeUtil::MakeShape(F32, {10})};
  Shape result_shape = ShapeUtil::MakeTupleShape(shape_elements);

  // Create a computation for the condition.
  // Repeat for 5 iterations.
  XlaComputation condition;
  const int c1 = 5;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    Lt(iteration, ConstantR0<int32>(&builder, c1));
    TF_ASSERT_OK_AND_ASSIGN(condition, builder.Build());
  }

  XlaComputation condition2;
  const int c2 = 7;
  {
    XlaBuilder builder("condition2");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    Lt(iteration, ConstantR0<int32>(&builder, c2));
    TF_ASSERT_OK_AND_ASSIGN(condition2, builder.Build());
  }

  // Create a computation for the body.
  // Add 1 to the iteration variable and add a constant vector of 1.0f to
  // the weight variable, both of which are tuple elements.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    auto weights = GetTupleElement(prev, 1);
    auto input = ConstantR1<float>(&builder, 10, 1.f);
    auto new_weights = Add(weights, input);
    Tuple(&builder,
          {Add(iteration, ConstantR0<int32>(&builder, 1)), new_weights});
    TF_ASSERT_OK_AND_ASSIGN(body, builder.Build());
  }

  XlaComputation body2;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    auto weights = GetTupleElement(prev, 1);
    auto input = ConstantR1<float>(&builder, 10, 1.f);
    auto new_weights = Add(weights, input);
    Tuple(&builder,
          {Add(iteration, ConstantR0<int32>(&builder, 1)), new_weights});
    TF_ASSERT_OK_AND_ASSIGN(body2, builder.Build());
  }

  // Create a While node with computations for the condition and the body.
  XlaBuilder builder("while");
  auto init = Tuple(&builder, {ConstantR0<int32>(&builder, 0),
                               ConstantR1<float>(&builder, 10, 0.f)});
  auto while1 = While(condition, body, init);

  auto while2 = While(condition2, body2, while1);

  auto while_result1 = GetTupleElement(while1, 1);
  auto while_result2 = GetTupleElement(while2, 1);
  VLOG(2) << "while_result2 = "
          << ShapeUtil::HumanString(
                 builder.GetShape(while_result2).ConsumeValueOrDie());
  auto result = Add(while_result1, while_result2);
  VLOG(2) << "result = "
          << ShapeUtil::HumanString(
                 builder.GetShape(result).ConsumeValueOrDie());
  const float sum = c1 + c2;
  std::vector<float> expected(10, sum);
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

// Test while nodes that share the while body computation.
XLA_TEST_F(WhileTest, TwoWhileLoopsAndSharedBody) {
  std::vector<Shape> shape_elements = {ShapeUtil::MakeShape(S32, {}),
                                       ShapeUtil::MakeShape(F32, {10})};
  Shape result_shape = ShapeUtil::MakeTupleShape(shape_elements);

  // Create a computation for the condition.
  // Repeat for 5 iterations.
  XlaComputation condition;
  const int c1 = 5;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    Lt(iteration, ConstantR0<int32>(&builder, c1));
    TF_ASSERT_OK_AND_ASSIGN(condition, builder.Build());
  }

  XlaComputation condition2;
  const int c2 = 7;
  {
    XlaBuilder builder("condition2");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    Lt(iteration, ConstantR0<int32>(&builder, c2));
    TF_ASSERT_OK_AND_ASSIGN(condition2, builder.Build());
  }

  // Create a computation for the body.
  // Add 1 to the iteration variable and add a constant vector of 1.0f to
  // the weight variable, both of which are tuple elements.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    auto weights = GetTupleElement(prev, 1);
    auto input = ConstantR1<float>(&builder, 10, 1.f);
    auto new_weights = Add(weights, input);
    Tuple(&builder,
          {Add(iteration, ConstantR0<int32>(&builder, 1)), new_weights});
    TF_ASSERT_OK_AND_ASSIGN(body, builder.Build());
  }

  // Create a While node with computations for the condition and the body.
  XlaBuilder builder("while");
  auto init = Tuple(&builder, {ConstantR0<int32>(&builder, 0),
                               ConstantR1<float>(&builder, 10, 0.f)});
  auto while1 = While(condition, body, init);

  auto while2 = While(condition2, body, while1);

  auto while_result1 = GetTupleElement(while1, 1);
  auto while_result2 = GetTupleElement(while2, 1);
  VLOG(2) << "while_result2 = "
          << ShapeUtil::HumanString(
                 builder.GetShape(while_result2).ConsumeValueOrDie());
  auto result = Add(while_result1, while_result2);
  VLOG(2) << "result = "
          << ShapeUtil::HumanString(
                 builder.GetShape(result).ConsumeValueOrDie());
  const float sum = c1 + c2;
  std::vector<float> expected(10, sum);
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(WhileTest, WhileLoopsWithSharedBodyAndInit) {
  std::vector<Shape> shape_elements = {ShapeUtil::MakeShape(S32, {}),
                                       ShapeUtil::MakeShape(F32, {10})};
  Shape result_shape = ShapeUtil::MakeTupleShape(shape_elements);

  // Create a computation for the condition.
  // Repeat for 5 iterations.
  XlaComputation condition;
  const int c1 = 5;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    Lt(iteration, ConstantR0<int32>(&builder, c1));
    TF_ASSERT_OK_AND_ASSIGN(condition, builder.Build());
  }

  XlaComputation condition2;
  const int c2 = 7;
  {
    XlaBuilder builder("condition2");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    Lt(iteration, ConstantR0<int32>(&builder, c2));
    TF_ASSERT_OK_AND_ASSIGN(condition2, builder.Build());
  }

  // Create a computation for the body.
  // Add 1 to the iteration variable and add a constant vector of 1.0f to
  // the weight variable, both of which are tuple elements.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    auto weights = GetTupleElement(prev, 1);
    auto input = ConstantR1<float>(&builder, 10, 1.f);
    auto new_weights = Add(weights, input);
    Tuple(&builder,
          {Add(iteration, ConstantR0<int32>(&builder, 1)), new_weights});
    TF_ASSERT_OK_AND_ASSIGN(body, builder.Build());
  }

  // Create a While node with computations for the condition and the body.
  XlaBuilder builder("while");
  auto init = Tuple(&builder, {ConstantR0<int32>(&builder, 0),
                               ConstantR1<float>(&builder, 10, 0.f)});
  auto while1 = While(condition, body, init);
  auto while2 = While(condition2, body, init);

  auto while_result1 = GetTupleElement(while1, 1);
  auto while_result2 = GetTupleElement(while2, 1);
  VLOG(2) << "while_result2 = "
          << ShapeUtil::HumanString(
                 builder.GetShape(while_result2).ConsumeValueOrDie());
  auto result = Add(while_result1, while_result2);
  VLOG(2) << "result = "
          << ShapeUtil::HumanString(
                 builder.GetShape(result).ConsumeValueOrDie());
  const float sum = c1 + c2;
  std::vector<float> expected(10, sum);
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

// WhileTest that uses DynamicUpdateSlice instruction in body computation.
// Loop state tuple element 1 has as its single user operand(0) of
// DynamicUpdateSlice, which will trigger in-place dynamic slice update on GPU.
XLA_TEST_F(WhileTest, WhileWithDynamicUpdateSlice) {
  std::vector<Shape> shape_elements = {ShapeUtil::MakeShape(S32, {}),
                                       ShapeUtil::MakeShape(F32, {10})};
  Shape result_shape = ShapeUtil::MakeTupleShape(shape_elements);

  // Create a computation for the condition.
  // Repeat for 5 iterations.
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    Gt(ConstantR0<int32>(&builder, 5), iteration);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body.
  // Add 1 to the iteration variable and add a constant vector of 1.0f to
  // the weight variable, both of which are tuple elements.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    // TupleElement 0
    auto iteration = GetTupleElement(prev, 0);
    auto out0 = Add(iteration, ConstantR0<int32>(&builder, 1));
    // TupleElement 1
    auto input = GetTupleElement(prev, 1);
    // Update.
    auto update = ConvertElementType(Broadcast(out0, {2}), F32);
    // Starts = iteration * 2;
    auto starts = Mul(iteration, ConstantR0<int32>(&builder, 2));
    // UpdateSlice.
    auto out1 = DynamicUpdateSlice(input, update, {starts});

    Tuple(&builder, {out0, out1});
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  XlaBuilder builder("while");
  auto init = Tuple(&builder, {ConstantR0<int32>(&builder, 0),
                               ConstantR1<float>(&builder, 10, 0.f)});
  auto result = While(condition, body, init);
  VLOG(2) << "while = "
          << ShapeUtil::HumanString(
                 builder.GetShape(result).ConsumeValueOrDie());

  auto expected_counter = LiteralUtil::CreateR0<int32>(5);
  auto expected_data = LiteralUtil::CreateR1<float>(
      {1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f});
  auto expected = LiteralUtil::MakeTuple({&expected_counter, &expected_data});
  VLOG(2) << "expected = " << ShapeUtil::HumanString(expected.shape());
  ComputeAndCompareTuple(&builder, expected, {}, ErrorSpec(0.0001));
}

// Tests a while node when the result type T is a vector of S32.
//
// int32 result = (0, 0, 0, 0, 0, 0);
// while (result[0] < count) {
//   result += (1, U[0, 100], U[0, 100], U[0, 100], U[0, 100], U[0, 100]);
// }
//
// This test misuses a vector WhileTest.WhileLoopsWithSharedBodyto represent a
// pair:
//   ((iteration, (random vector))).
//
// Note: this test currently only tests generating random values within a loop.
// Per backend the values generated can be different as the different backends
// use different random number generators.
// TODO(b/32240857): Extend test to verify outputs.
XLA_TEST_F(WhileTest, WhileWithPrngScalarResult) {
  auto v6s32 = ShapeUtil::MakeShape(S32, {6});

  // Create a computation for the condition: repeat for count iterations.
  auto build_condition = [this, v6s32](int count) {
    XlaBuilder builder(TestName());
    auto prev = Reshape(
        Slice(Parameter(&builder, 0, v6s32, "prev"), {0}, {1}, {1}), {0}, {});
    Gt(ConstantR0<int32>(&builder, count), prev);
    return builder.Build().ConsumeValueOrDie();
  };

  // Create a computation for the body: add 1 to the result variable.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, v6s32, "prev");
    auto inc = ConcatInDim(&builder,
                           {ConstantR1<int32>(&builder, {1}),
                            RngUniform(ConstantR0<int32>(&builder, 0),
                                       ConstantR0<int32>(&builder, 100),
                                       ShapeUtil::MakeShape(S32, {5}))},
                           0);
    Add(inc, prev);
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  auto while_loop = [this, &body, build_condition](int count) {
    XlaBuilder builder(TestName());
    auto init = ConstantR1<int32>(&builder, {0, 0, 0, 0, 0, 0});
    While(build_condition(count), body, init);
    return builder.Build();
  };

  for (int i = 1; i < 4; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(auto computation, while_loop(i));

    ExecutionOptions execution_options = execution_options_;
    execution_options.set_seed(65);
    TF_ASSERT_OK_AND_ASSIGN(
        auto result,
        client_->ExecuteAndTransfer(computation, {}, &execution_options));
  }
}

XLA_TEST_F(WhileTest, WhileThatSwapsParameterWithTupleElement) {
  auto element_shape = ShapeUtil::MakeShape(F32, {2});

  XlaBuilder outer("outer");
  auto p = Parameter(&outer, 0, element_shape, "param");
  auto t = Tuple(&outer, {p, ConstantR1<float>(&outer, {1, 1})});

  TF_ASSERT_OK_AND_ASSIGN(Shape tuple_shape, outer.GetShape(t));

  XlaBuilder cond("cond");
  auto cond_t = Parameter(&cond, 0, tuple_shape, "t");
  Any(Eq(GetTupleElement(cond_t, 0), ConstantR1<float>(&cond, {42, 42})));

  XlaBuilder body("body");
  auto body_t = Parameter(&body, 0, tuple_shape, "t");
  auto e = GetTupleElement(body_t, 1);
  Tuple(&body, {e, e});

  TF_ASSERT_OK_AND_ASSIGN(auto cond_computation, cond.Build());
  TF_ASSERT_OK_AND_ASSIGN(auto body_computation, body.Build());
  While(cond_computation, body_computation, t);

  auto expected_element = LiteralUtil::CreateR1<float>({1, 1});
  auto expected =
      LiteralUtil::MakeTuple({&expected_element, &expected_element});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GlobalData> parameter_data,
      client_->TransferToServer(LiteralUtil::CreateR1<float>({42, 42})));
  ComputeAndCompareTuple(&outer, expected, {parameter_data.get()},
                         ErrorSpec(1e-6));
}

XLA_TEST_F(WhileTest, WhileThatSwapsParameterWithBroadcast) {
  auto element_shape = ShapeUtil::MakeShape(F32, {2});

  XlaBuilder outer("outer");
  auto p = Parameter(&outer, 0, element_shape, "param");

  XlaBuilder cond("cond");
  auto cond_t = Parameter(&cond, 0, element_shape, "t");
  Any(Eq(cond_t, ConstantR1<float>(&cond, {42, 42})));

  XlaBuilder body("body");
  Parameter(&body, 0, element_shape, "t");
  Broadcast(ConstantR0<float>(&body, 1.0), {2});

  TF_ASSERT_OK_AND_ASSIGN(auto cond_computation, cond.Build());
  TF_ASSERT_OK_AND_ASSIGN(auto body_computation, body.Build());
  While(cond_computation, body_computation, p);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GlobalData> parameter_data,
      client_->TransferToServer(LiteralUtil::CreateR1<float>({42, 42})));
  ComputeAndCompareR1<float>(&outer, {1.0f, 1.0f}, {parameter_data.get()},
                             ErrorSpec(1e-6));
}

XLA_TEST_F(WhileTest, WhileThatTurnsScalarParameterToTupleElement) {
  auto element_shape = ShapeUtil::MakeShape(F32, {});

  XlaBuilder outer("outer");
  auto p = Parameter(&outer, 0, element_shape, "param");

  XlaBuilder cond("cond");
  auto cond_t = Parameter(&cond, 0, element_shape, "t");
  Eq(cond_t, ConstantR0<float>(&cond, 42));

  XlaBuilder body("body");
  auto body_t = Parameter(&body, 0, element_shape, "t");
  auto tuple = Tuple(&body, {body_t, Add(body_t, ConstantR0<float>(&body, 1))});
  GetTupleElement(tuple, 1);

  TF_ASSERT_OK_AND_ASSIGN(auto cond_computation, cond.Build());
  TF_ASSERT_OK_AND_ASSIGN(auto body_computation, body.Build());
  While(cond_computation, body_computation, p);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GlobalData> parameter_data,
      client_->TransferToServer(LiteralUtil::CreateR0<float>(42)));
  ComputeAndCompareR0<float>(&outer, 43.0f, {parameter_data.get()},
                             ErrorSpec(1e-6));
}

// Tests loop where the init value comes from two sources (constant and
// parameter).
//
// int32 result = (0, 1);
// while (result[0] + result[1] < 30) {
//   result[0] = result[0] + 1;
//   result[1] = result[1] + 1;
// }
XLA_TEST_F(WhileTest, WhileWithMixedTupleElements) {
  auto result_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(S32, {}), ShapeUtil::MakeShape(S32, {})});

  XlaBuilder outer("outer");
  auto p =
      Tuple(&outer, {ConstantR0<int32>(&outer, 0),
                     Parameter(&outer, 0, ShapeUtil::MakeShape(S32, {}), "t")});

  XlaBuilder cond("cond");
  auto params = Parameter(&cond, 0, result_shape, "prev");
  auto cond_t = Add(GetTupleElement(params, 1), GetTupleElement(params, 0));
  Lt(cond_t, ConstantR0<int32>(&cond, 30));

  XlaBuilder body("body");
  auto body_t = Parameter(&body, 0, result_shape, "t");

  Tuple(&body, {Add(GetTupleElement(body_t, 0), ConstantR0<int32>(&body, 1)),
                Add(GetTupleElement(body_t, 1), ConstantR0<int32>(&body, 1))});

  TF_ASSERT_OK_AND_ASSIGN(auto cond_computation, cond.Build());
  TF_ASSERT_OK_AND_ASSIGN(auto body_computation, body.Build());
  While(cond_computation, body_computation, p);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GlobalData> parameter_data,
      client_->TransferToServer(LiteralUtil::CreateR0<int32>(1)));

  auto add1 = LiteralUtil::CreateR0<int32>(15);
  auto add2 = LiteralUtil::CreateR0<int32>(16);
  auto expected = LiteralUtil::MakeTuple({&add1, &add2});
  ComputeAndCompareTuple(&outer, expected, {parameter_data.get()},
                         ErrorSpec(1e-6));
}

// Tests nested while loops.
//
// int32 result = 0;
// while (result < 30) {
//   int i = 0;
//   while (i < 7) {
//     result = result + 2;
//     i = i + 1;
//   }
// }
XLA_TEST_F(WhileTest, NestedWhileWithScalarResult) {
  auto outer_result_shape = ShapeUtil::MakeShape(S32, {});
  auto inner_result_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(S32, {}), ShapeUtil::MakeShape(S32, {})});

  XlaComputation inner_condition;
  {
    XlaBuilder builder("inner_condition");
    auto params = Parameter(&builder, 0, inner_result_shape, "prev");
    auto i = GetTupleElement(params, 0);
    Lt(i, ConstantR0<int32>(&builder, 7));
    inner_condition = builder.Build().ConsumeValueOrDie();
  }

  // Creates a computation for the outer loop condition:
  // repeat while result < 30.
  XlaComputation outer_condition;
  {
    XlaBuilder builder("outer_condition");
    auto prev = Parameter(&builder, 0, outer_result_shape, "prev");
    Lt(prev, ConstantR0<int32>(&builder, 30));
    outer_condition = builder.Build().ConsumeValueOrDie();
  }

  // Creates a computation for the inner loop body: add 1 to `i`, and add 2 to
  // `result`.
  XlaComputation inner_body;
  {
    XlaBuilder builder("inner_body");
    auto params = Parameter(&builder, 0, inner_result_shape, "prev");
    auto i = GetTupleElement(params, 0);
    auto result = GetTupleElement(params, 1);
    i = Add(ConstantR0<int32>(&builder, 1), i);
    result = Add(ConstantR0<int32>(&builder, 2), result);
    Tuple(&builder, {i, result});
    inner_body = builder.Build().ConsumeValueOrDie();
  }

  // Creates a computation for the outer loop: run the inner loop with i = 0.
  XlaComputation outer_body;
  {
    XlaBuilder builder("outer_body");
    auto prev = Parameter(&builder, 0, outer_result_shape, "prev");
    auto init = Tuple(&builder, {ConstantR0<int32>(&builder, 0), prev});
    auto result = While(inner_condition, inner_body, init);
    GetTupleElement(result, 1);
    outer_body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  XlaBuilder builder(TestName());
  auto init = ConstantR0<int32>(&builder, 0);
  While(outer_condition, outer_body, init);

  ComputeAndCompareR0<int32>(&builder, 42, {});
}

// Tests a while node when the result type T is S32.
// f = lambda result: tuple({result < 5})
// int32 result = 0;
// while (f(result).get<0>()) {
//   result = result + 1;
// }
XLA_TEST_F(WhileTest, WhileWithCallInsideCondition) {
  auto result_shape = ShapeUtil::MakeShape(S32, {});

  // Create a computation for the condition: repeat for 5 iterations.
  XlaComputation condition_callee;
  {
    XlaBuilder builder("condition_callee");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    Tuple(&builder, {Gt(ConstantR0<int32>(&builder, 5), prev)});

    condition_callee = builder.Build().ConsumeValueOrDie();
  }

  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto result = Call(&builder, condition_callee, {prev});
    GetTupleElement(result, 0);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body: add 1 to the result variable.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto input = ConstantR0<int32>(&builder, 1);
    Add(input, prev);
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  XlaBuilder builder(TestName());
  auto init = ConstantR0<int32>(&builder, 0);
  While(condition, body, init);

  ComputeAndCompareR0<int32>(&builder, 5, {});
}

XLA_TEST_F(WhileTest, WhileWithLoopInvariantOperation) {
  auto matrix_shape = ShapeUtil::MakeShape(F32, {2, 2});
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  auto while_shape = ShapeUtil::MakeTupleShape(
      {scalar_s32, matrix_shape, matrix_shape, matrix_shape});

  // Create a computation for the condition: repeat for 5 iterations.
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto state = Parameter(&builder, 0, while_shape, "state");
    Gt(ConstantR0<int32>(&builder, 5), GetTupleElement(state, 0));
    TF_ASSERT_OK_AND_ASSIGN(condition, builder.Build());
  }

  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto state = Parameter(&builder, 0, while_shape, "state");
    auto indvar = GetTupleElement(state, 0);
    auto input_0 = GetTupleElement(state, 1);
    auto input_1 = GetTupleElement(state, 2);
    auto output = Tanh(Dot(input_0, input_1));
    auto indvar_next = Add(indvar, ConstantR0<int32>(&builder, 1));
    Tuple(&builder, {indvar_next, input_0, input_1, output});
    TF_ASSERT_OK_AND_ASSIGN(body, builder.Build());
  }

  XlaBuilder builder(TestName());
  auto matrix_input = Parameter(&builder, 0, matrix_shape, "matrix");
  auto init = Tuple(&builder, {ConstantR0<int32>(&builder, 0), matrix_input,
                               matrix_input, matrix_input});
  auto while_instruction = While(condition, body, init);
  GetTupleElement(while_instruction, 3);

  TF_ASSERT_OK_AND_ASSIGN(
      auto param_value, client_->TransferToServer(LiteralUtil::CreateR2<float>(
                            {{1.0, 2.0}, {-1.0, -2.0}})));

  ComputeAndCompareR2<float>(
      &builder, {{-0.76159416, -0.96402758}, {0.76159416, 0.96402758}},
      {param_value.get()}, ErrorSpec(4e-5));
}

XLA_TEST_F(WhileTest, DISABLED_ON_INTERPRETER(WhileInfeedCondition)) {
  auto while_shape = ShapeUtil::MakeShape(S32, {});

  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    Parameter(&builder, 0, while_shape, "state");
    Infeed(&builder, ShapeUtil::MakeShape(PRED, {}));
    TF_ASSERT_OK_AND_ASSIGN(condition, builder.Build());
  }

  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto indvar = Parameter(&builder, 0, while_shape, "state");
    Add(indvar, ConstantR0<int32>(&builder, 1));
    TF_ASSERT_OK_AND_ASSIGN(body, builder.Build());
  }

  XlaBuilder builder(TestName());
  While(condition, body, ConstantR0<int32>(&builder, 0));

  TF_ASSERT_OK(client_->TransferToInfeed(LiteralUtil::CreateR0<bool>(true)));
  TF_ASSERT_OK(client_->TransferToInfeed(LiteralUtil::CreateR0<bool>(true)));
  TF_ASSERT_OK(client_->TransferToInfeed(LiteralUtil::CreateR0<bool>(false)));

  ComputeAndCompareR0<int32>(&builder, 2, {});
}

void BM_WhileLoop(int num_iters) {
  // Benchmark a simple kernel to measure while loop overheads.
  tensorflow::testing::StopTiming();

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().ValueOrDie();
  auto executors = PlatformUtil::GetStreamExecutors(platform).ValueOrDie();
  se::StreamExecutorMemoryAllocator allocator(platform, executors);
  LocalClient* client =
      ClientLibrary::GetOrCreateLocalClient(platform).ValueOrDie();

  const int64 seq_len = 100;
  Shape loop_state_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(S32, {}),
       ShapeUtil::MakeShape(F32, {seq_len, 1024, 1024})});

  // Create while condition computation with 'loop_limit'.
  const int32 loop_limit = 100;
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, loop_state_shape, "prev");
    auto iteration = GetTupleElement(prev, 0);
    Lt(iteration, ConstantR0<int32>(&builder, loop_limit));
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create while body computation with unit loop increment.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, loop_state_shape, "prev");
    // TupleElement 0
    auto iteration = GetTupleElement(prev, 0);
    auto out0 = Add(iteration, ConstantR0<int32>(&builder, 1));
    // TupleElement 1
    auto input = GetTupleElement(prev, 1);
    // Update.
    auto one = ConstantR0<float>(&builder, 1.0);
    auto update = Broadcast(one, {1, 1024, 1024});
    // Starts = iteration * 2;
    auto zero = ConstantR0<int32>(&builder, 0);
    // UpdateSlice.
    auto out1 = DynamicUpdateSlice(input, update, {zero, zero, zero});
    Tuple(&builder, {out0, out1});
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While instruction.
  XlaBuilder builder("while");
  auto zero = ConstantR0<float>(&builder, 0.0);
  auto input = Broadcast(zero, {seq_len, 1024, 1024});
  auto init = Tuple(&builder, {ConstantR0<int32>(&builder, 0), input});
  While(condition, body, init);
  auto computation = builder.Build().ConsumeValueOrDie();

  TF_ASSERT_OK_AND_ASSIGN(
      auto executables,
      client->Compile(computation, {}, ExecutableBuildOptions()));
  auto executable = std::move(executables[0]);

  // Run some warm-up executions.
  ExecutableRunOptions options;
  options.set_allocator(&allocator);
  const int kWarmups = 2;
  for (int i = 0; i < kWarmups; ++i) {
    auto result =
        executable->Run(absl::Span<const ShapedBuffer* const>(), options);
    ASSERT_TRUE(result.ok());
  }

  // Run benchmark.
  tensorflow::testing::StartTiming();
  for (int i = 0; i < num_iters; ++i) {
    auto result =
        executable->Run(absl::Span<const ShapedBuffer* const>(), options);
    ASSERT_TRUE(result.ok());
  }
}

BENCHMARK(BM_WhileLoop);
}  // namespace
}  // namespace xla
