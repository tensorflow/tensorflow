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

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class TupleTest : public ClientLibraryTestBase {
 public:
  ErrorSpec error_spec_{0.0001};
};

// Tests a tuple-shaped constant.
XLA_TEST_F(TupleTest, TupleConstant) {
  ComputationBuilder builder(client_, TestName());

  const float constant_scalar = 7.3f;
  std::initializer_list<float> constant_vector = {1.1f, 2.0f, 3.3f};
  std::initializer_list<std::initializer_list<float>> constant_matrix = {
      {1.1f, 2.2f, 3.5f},  // row 0
      {4.8f, 5.0f, 6.7f},  // row 1
  };
  auto value =
      Literal::MakeTuple({Literal::CreateR0<float>(constant_scalar).get(),
                          Literal::CreateR1<float>(constant_vector).get(),
                          Literal::CreateR2<float>(constant_matrix).get()});

  auto result = builder.ConstantLiteral(*value);
  ComputeAndCompareTuple(&builder, *value, {}, error_spec_);
}

// Tests the creation of tuple data.
XLA_TEST_F(TupleTest, TupleCreate) {
  ComputationBuilder builder(client_, TestName());

  const float constant_scalar = 7.3f;
  std::initializer_list<float> constant_vector = {1.1f, 2.0f, 3.3f};
  std::initializer_list<std::initializer_list<float>> constant_matrix = {
      {1.1f, 2.2f, 3.5f},  // row 0
      {4.8f, 5.0f, 6.7f},  // row 1
  };
  auto result = builder.Tuple({builder.ConstantR0<float>(constant_scalar),
                               builder.ConstantR1<float>(constant_vector),
                               builder.ConstantR2<float>(constant_matrix)});

  auto expected =
      Literal::MakeTuple({Literal::CreateR0<float>(constant_scalar).get(),
                          Literal::CreateR1<float>(constant_vector).get(),
                          Literal::CreateR2<float>(constant_matrix).get()});
  ComputeAndCompareTuple(&builder, *expected, {}, error_spec_);
}

// Tests the creation of tuple data.
XLA_TEST_F(TupleTest, TupleCreateWithZeroElementEntry) {
  ComputationBuilder builder(client_, TestName());

  auto result = builder.Tuple(
      {builder.ConstantR0<float>(7.0), builder.ConstantR1<float>({})});

  auto expected = Literal::MakeTuple({Literal::CreateR0<float>(7.0).get(),
                                      Literal::CreateR1<float>({}).get()});
  ComputeAndCompareTuple(&builder, *expected, {}, error_spec_);
}

// Tests the creation of an empty tuple.
XLA_TEST_F(TupleTest, EmptyTupleCreate) {
  ComputationBuilder builder(client_, TestName());
  auto result = builder.Tuple({});
  auto expected = Literal::MakeTuple({});
  ComputeAndCompareTuple(&builder, *expected, {}, error_spec_);
}

// Trivial test for extracting a tuple element with GetTupleElement.
XLA_TEST_F(TupleTest, GetTupleElement) {
  ComputationBuilder builder(client_, TestName());
  std::initializer_list<float> constant_vector = {1.f, 2.f, 3.f};
  std::initializer_list<std::initializer_list<float>> constant_matrix = {
      {1.f, 2.f, 3.f},  // row 0
      {4.f, 5.f, 6.f},  // row 1
  };
  auto tuple_data = builder.Tuple({builder.ConstantR1<float>(constant_vector),
                                   builder.ConstantR2<float>(constant_matrix)});
  auto matrix_element = builder.GetTupleElement(tuple_data, 1);
  ComputeAndCompareR2<float>(&builder, Array2D<float>(constant_matrix), {},
                             error_spec_);
}

// Trivial test for extracting a tuple element with GetTupleElement.
XLA_TEST_F(TupleTest, GetTupleElementWithZeroElements) {
  ComputationBuilder builder(client_, TestName());
  auto tuple_data = builder.Tuple(
      {builder.ConstantR1<float>({}),
       builder.ConstantR2FromArray2D<float>(Array2D<float>(0, 101))});
  auto matrix_element = builder.GetTupleElement(tuple_data, 1);
  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 101), {}, error_spec_);
}

XLA_TEST_F(TupleTest, GetTupleElementOfNonTupleFailsGracefully) {
  ComputationBuilder builder(client_, TestName());
  auto value = builder.ConstantR1<float>({4.5f});
  builder.GetTupleElement(value, 1);
  auto result_status = builder.Build();
  EXPECT_FALSE(result_status.ok());
  EXPECT_THAT(
      result_status.status().error_message(),
      ::testing::HasSubstr("Operand to GetTupleElement() is not a tuple"));
}

// Extracts both elements from a tuple with GetTupleElement and then adds them
// together.
XLA_TEST_F(TupleTest, AddTupleElements) {
  ComputationBuilder builder(client_, TestName());
  std::initializer_list<float> constant_vector = {1.f, 2.f, 3.f};
  std::initializer_list<std::initializer_list<float>> constant_matrix = {
      {1.f, 2.f, 3.f},  // row 0
      {4.f, 5.f, 6.f},  // row 1
  };
  auto tuple_data = builder.Tuple({builder.ConstantR1<float>(constant_vector),
                                   builder.ConstantR2<float>(constant_matrix)});
  auto vector_element = builder.GetTupleElement(tuple_data, 0);
  auto matrix_element = builder.GetTupleElement(tuple_data, 1);
  auto vector_shape = builder.GetShape(vector_element).ConsumeValueOrDie();
  auto matrix_shape = builder.GetShape(matrix_element).ConsumeValueOrDie();
  auto result = builder.Add(matrix_element, vector_element,
                            /*broadcast_dimensions=*/{1});

  Array2D<float> expected({
      {2.f, 4.f, 6.f},  // row 0
      {5.f, 7.f, 9.f},  // row 1
  });
  ASSERT_TRUE(ShapeUtil::ShapeIs(*vector_shape, F32, {3}));
  ASSERT_TRUE(ShapeUtil::ShapeIs(*matrix_shape, F32, {/*y=*/2, /*x=*/3}));
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

// Extracts both elements from a tuple and then puts them into a new tuple in
// the opposite order.
XLA_TEST_F(TupleTest, TupleGTEToTuple) {
  ComputationBuilder builder(client_, TestName());
  std::initializer_list<float> constant_vector = {1.f, 2.f, 3.f};
  std::initializer_list<std::initializer_list<float>> constant_matrix = {
      {1.f, 2.f, 3.f},  // row 0
      {4.f, 5.f, 6.f},  // row 1
  };
  auto tuple_data = builder.Tuple({builder.ConstantR1<float>(constant_vector),
                                   builder.ConstantR2<float>(constant_matrix)});
  auto new_tuple = builder.Tuple({builder.GetTupleElement(tuple_data, 1),
                                  builder.GetTupleElement(tuple_data, 0)});
  auto expected =
      Literal::MakeTuple({Literal::CreateR2<float>(constant_matrix).get(),
                          Literal::CreateR1<float>(constant_vector).get()});
  ComputeAndCompareTuple(&builder, *expected, {}, error_spec_);
}

XLA_TEST_F(TupleTest, SelectBetweenPredTuples) {
  ComputationBuilder b(client_, TestName());
  ComputationDataHandle v1, v2;

  for (bool direction : {false, true}) {
    std::unique_ptr<GlobalData> v1_data =
        CreateR0Parameter<float>(0.0f, /*parameter_number=*/0, /*name=*/"v1",
                                 /*builder=*/&b, /*data_handle=*/&v1);
    std::unique_ptr<GlobalData> v2_data =
        CreateR0Parameter<float>(1.0f, /*parameter_number=*/1, /*name=*/"v2",
                                 /*builder=*/&b, /*data_handle=*/&v2);
    auto v1_gt = b.Gt(v1, v2);             // false
    auto v2_gt = b.Gt(v2, v1);             // true
    auto v1_v2 = b.Tuple({v1_gt, v2_gt});  // {false, true}
    auto v2_v1 = b.Tuple({v2_gt, v1_gt});  // {true, false}
    auto select = b.Select(direction ? v1_gt : v2_gt, v1_v2, v2_v1);
    auto expected =
        Literal::MakeTuple({Literal::CreateR0<bool>(direction).get(),
                            Literal::CreateR0<bool>(!direction).get()});

    ComputeAndCompareTuple(&b, *expected, {v1_data.get(), v2_data.get()},
                           error_spec_);
  }
}

// Builds two new tuples from an existing tuple (by means of GetTupleElement),
// then adds up the components of the new tuples.
XLA_TEST_F(TupleTest, TupleGTEToTupleToGTEAdd) {
  //
  // v------           --(GTE 0)--             --(GTE 0)----------
  //        \         /           \           /                   \
  //         (tuple)--             (tuple01)--                     \
  //        /   |     \           /           \                     \
  // m------    |      --(GTE 1)--             --(GTE 1)------------ \
  //            |                                                   \ \
  //            |                                                    (add)
  //            |                                                   / /
  //            |--------(GTE 1)--             --(GTE 0)------------ /
  //             \                \           /                     /
  //              \                (tuple10)--                     /
  //               \              /           \                   /
  //                -----(GTE 0)--             --(GTE 1)----------
  ComputationBuilder builder(client_, TestName());
  std::initializer_list<float> constant_vector = {1.f, 2.f, 3.f};
  std::initializer_list<std::initializer_list<float>> constant_matrix = {
      {1.f, 2.f, 3.f},  // row 0
      {4.f, 5.f, 6.f},  // row 1
  };
  auto tuple_data = builder.Tuple({builder.ConstantR1<float>(constant_vector),
                                   builder.ConstantR2<float>(constant_matrix)});
  auto new_tuple01 = builder.Tuple({builder.GetTupleElement(tuple_data, 0),
                                    builder.GetTupleElement(tuple_data, 1)});
  auto new_tuple10 = builder.Tuple({builder.GetTupleElement(tuple_data, 1),
                                    builder.GetTupleElement(tuple_data, 0)});
  auto vector_from_01 = builder.GetTupleElement(new_tuple01, 0);
  auto vector_from_10 = builder.GetTupleElement(new_tuple10, 1);
  auto matrix_from_01 = builder.GetTupleElement(new_tuple01, 1);
  auto matrix_from_10 = builder.GetTupleElement(new_tuple10, 0);

  auto addvectors = builder.Add(vector_from_01, vector_from_10);
  auto addmatrices = builder.Add(matrix_from_01, matrix_from_10);

  auto result = builder.Add(addmatrices, addvectors,
                            /*broadcast_dimensions=*/{1});

  Array2D<float> expected({
      {4.f, 8.f, 12.f},    // row 0
      {10.f, 14.f, 18.f},  // row 1
  });
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(TupleTest, DISABLED_ON_CPU_PARALLEL(SelectBetweenTuplesOnFalse)) {
  // Tests a selection between tuples with "false" path taken.
  ComputationBuilder builder(client_, TestName());

  std::initializer_list<float> vec1 = {1.f, 2.f, 3.f};
  std::initializer_list<float> vec2 = {2.f, 4.f, 6.f};
  auto tuple12 = builder.Tuple(
      {builder.ConstantR1<float>(vec1), builder.ConstantR1<float>(vec2)});
  auto tuple21 = builder.Tuple(
      {builder.ConstantR1<float>(vec2), builder.ConstantR1<float>(vec1)});

  auto select =
      builder.Select(builder.ConstantR0<bool>(false), tuple12, tuple21);
  auto expected = Literal::MakeTuple({Literal::CreateR1<float>(vec2).get(),
                                      Literal::CreateR1<float>(vec1).get()});
  ComputeAndCompareTuple(&builder, *expected, {}, error_spec_);
}

XLA_TEST_F(TupleTest, TuplesInAMap) {
  Computation tuple_computation;
  {
    // tuple_computation(x) = 100 * min(x, x^2) + max(x, x^2) using tuples.
    //
    // Need to put a select in there to prevent HLO-level optimizations from
    // optimizing out the tuples.
    ComputationBuilder b(client_, "sort_square");
    auto x = b.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
    auto x2 = b.Mul(x, x);
    auto x_smaller_tuple = b.Tuple({x, x2});
    auto x2_smaller_tuple = b.Tuple({x2, x});
    auto sorted = b.Select(b.Lt(x, x2), x_smaller_tuple, x2_smaller_tuple);
    auto smaller = b.GetTupleElement(sorted, 0);
    auto greater = b.GetTupleElement(sorted, 1);
    b.Add(greater, b.Mul(b.ConstantR0<float>(100.0f), smaller));
    auto computation_status = b.Build();
    ASSERT_IS_OK(computation_status.status());
    tuple_computation = computation_status.ConsumeValueOrDie();
  }

  ComputationBuilder b(client_, TestName());
  auto input = b.ConstantR1<float>({-1.0f, 1.0f, 2.1f});
  b.Map({input}, tuple_computation, {0});
  ComputeAndCompareR1<float>(&b, {-99.0f, 101.0f, 214.41f}, {}, error_spec_);
}

XLA_TEST_F(TupleTest, DISABLED_ON_CPU_PARALLEL(SelectBetweenTuplesOnTrue)) {
  // Tests a selection between tuples with "true" path taken.
  ComputationBuilder builder(client_, TestName());

  std::initializer_list<float> vec1 = {1.f, 2.f, 3.f};
  std::initializer_list<float> vec2 = {2.f, 4.f, 6.f};
  auto tuple12 = builder.Tuple(
      {builder.ConstantR1<float>(vec1), builder.ConstantR1<float>(vec2)});
  auto tuple21 = builder.Tuple(
      {builder.ConstantR1<float>(vec2), builder.ConstantR1<float>(vec1)});

  auto select =
      builder.Select(builder.ConstantR0<bool>(true), tuple12, tuple21);
  auto expected = Literal::MakeTuple({Literal::CreateR1<float>(vec1).get(),
                                      Literal::CreateR1<float>(vec2).get()});
  ComputeAndCompareTuple(&builder, *expected, {}, error_spec_);
}

XLA_TEST_F(TupleTest, SelectBetweenTuplesElementResult) {
  // Tests a selection between tuples but the final result is an element of the
  // tuple, not the whole tuple.
  ComputationBuilder builder(client_, TestName());

  std::initializer_list<float> vec1 = {1.f, 2.f, 3.f};
  std::initializer_list<float> vec2 = {2.f, 4.f, 6.f};
  auto tuple12 = builder.Tuple(
      {builder.ConstantR1<float>(vec1), builder.ConstantR1<float>(vec2)});
  auto tuple21 = builder.Tuple(
      {builder.ConstantR1<float>(vec2), builder.ConstantR1<float>(vec1)});

  auto select =
      builder.Select(builder.ConstantR0<bool>(false), tuple12, tuple21);
  auto element = builder.GetTupleElement(select, 0);

  ComputeAndCompareR1<float>(&builder, vec2, {}, error_spec_);
}

// Cascaded selects between tuple types.
XLA_TEST_F(TupleTest, DISABLED_ON_CPU_PARALLEL(SelectBetweenTuplesCascaded)) {
  //
  //                       vec1     vec2   vec2     vec1
  //                        |        |      |        |
  //                        |        |      |        |
  //                        (tuple 12)      (tuple 21)
  //                               \            /
  //                                \          /
  //                                 \        /
  //  true  --            --(GTE 0)--(select 1)
  //          \          /             |
  //       (pred tuple)--              |          --(GTE 0)--
  //          /          \             V         /           \
  //  false --            --(GTE 1)--(select 2)--             --(add)
  //                                 /           \           /
  //                                /             --(GTE 1)--
  //                               /
  //                          (tuple 21)
  ComputationBuilder builder(client_, TestName());

  std::initializer_list<float> vec1 = {1.f, 2.f, 3.f};
  std::initializer_list<float> vec2 = {2.f, 4.f, 6.f};

  auto pred_tuple = builder.Tuple(
      {builder.ConstantR0<bool>(true), builder.ConstantR0<bool>(false)});
  auto tuple12 = builder.Tuple(
      {builder.ConstantR1<float>(vec1), builder.ConstantR1<float>(vec2)});
  auto tuple21 = builder.Tuple(
      {builder.ConstantR1<float>(vec2), builder.ConstantR1<float>(vec1)});

  auto select1 =
      builder.Select(builder.GetTupleElement(pred_tuple, 0), tuple12, tuple21);
  auto select2 =
      builder.Select(builder.GetTupleElement(pred_tuple, 1), tuple21, select1);
  auto result = builder.Add(builder.GetTupleElement(select2, 0),
                            builder.GetTupleElement(select2, 1));

  ComputeAndCompareR1<float>(&builder, {3.f, 6.f, 9.f}, {}, error_spec_);
}

XLA_TEST_F(TupleTest,
           DISABLED_ON_CPU_PARALLEL(SelectBetweenTuplesReuseConstants)) {
  // Similar to SelectBetweenTuples, but the constants are shared between the
  // input tuples.
  ComputationBuilder builder(client_, TestName());

  std::initializer_list<float> vec1 = {1.f, 2.f, 3.f};
  std::initializer_list<float> vec2 = {2.f, 4.f, 6.f};
  auto c1 = builder.ConstantR1<float>(vec1);
  auto c2 = builder.ConstantR1<float>(vec2);
  auto tuple12 = builder.Tuple({c1, c2});
  auto tuple21 = builder.Tuple({c2, c1});

  auto select =
      builder.Select(builder.ConstantR0<bool>(false), tuple12, tuple21);
  auto expected = Literal::MakeTuple({Literal::CreateR1<float>(vec2).get(),
                                      Literal::CreateR1<float>(vec1).get()});
  ComputeAndCompareTuple(&builder, *expected, {}, error_spec_);
}

XLA_TEST_F(TupleTest, NestedTuples) {
  ComputationBuilder builder(client_, TestName());
  auto inner_tuple = builder.Tuple(
      {builder.ConstantR1<float>({1.0, 2.0}), builder.ConstantR0<float>(42.0)});
  auto outer_tuple =
      builder.Tuple({inner_tuple, builder.ConstantR1<float>({22.0, 44.0})});

  auto expected_v1 = Literal::CreateR1<float>({1.0, 2.0});
  auto expected_s = Literal::CreateR0<float>(42.0);
  auto expected_inner_tuple =
      Literal::MakeTuple({expected_v1.get(), expected_s.get()});
  auto expected_v2 = Literal::CreateR1<float>({22.0, 44.0});
  auto expected =
      Literal::MakeTuple({expected_inner_tuple.get(), expected_v2.get()});

  ComputeAndCompareTuple(&builder, *expected, {}, error_spec_);
}

XLA_TEST_F(TupleTest, GetTupleElementOfNestedTuple) {
  ComputationBuilder builder(client_, TestName());

  Shape data_shape = ShapeUtil::MakeShape(F32, {3});
  Shape inner_tuple_shape = ShapeUtil::MakeTupleShape({data_shape, data_shape});
  Shape outer_tuple_shape =
      ShapeUtil::MakeTupleShape({inner_tuple_shape, data_shape});

  auto input = builder.Parameter(0, outer_tuple_shape, "input");
  auto gte0 = builder.GetTupleElement(input, 0);
  auto gte1 = builder.GetTupleElement(gte0, 1);
  builder.Add(gte1, builder.ConstantR1<float>({10.0, 11.0, 12.0}));

  std::unique_ptr<GlobalData> data =
      client_
          ->TransferToServer(*Literal::MakeTuple({
              Literal::MakeTuple(
                  {
                      Literal::CreateR1<float>({1.0, 2.0, 3.0}).get(),
                      Literal::CreateR1<float>({4.0, 5.0, 6.0}).get(),
                  })
                  .get(),
              Literal::CreateR1<float>({7.0, 8.0, 9.0}).get(),
          }))
          .ConsumeValueOrDie();

  std::vector<GlobalData*> arguments = {data.get()};
  const std::vector<float> expected = {4.0 + 10.0, 5.0 + 11.0, 6.0 + 12.0};
  ComputeAndCompareR1<float>(&builder, expected, arguments, ErrorSpec(1e-5));
}

}  // namespace
}  // namespace xla
