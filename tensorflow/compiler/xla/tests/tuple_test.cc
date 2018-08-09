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
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class TupleTest : public ClientLibraryTestBase {
 public:
  ErrorSpec error_spec_{0.0001};
};

// Tests a tuple-shaped constant.
XLA_TEST_F(TupleTest, TupleConstant) {
  XlaBuilder builder(TestName());

  const float constant_scalar = 7.3f;
  std::initializer_list<float> constant_vector = {1.1f, 2.0f, 3.3f};
  std::initializer_list<std::initializer_list<float>> constant_matrix = {
      {1.1f, 2.2f, 3.5f},  // row 0
      {4.8f, 5.0f, 6.7f},  // row 1
  };
  auto value = LiteralUtil::MakeTuple(
      {LiteralUtil::CreateR0<float>(constant_scalar).get(),
       LiteralUtil::CreateR1<float>(constant_vector).get(),
       LiteralUtil::CreateR2<float>(constant_matrix).get()});

  ConstantLiteral(&builder, *value);
  ComputeAndCompareTuple(&builder, *value, {}, error_spec_);
}

// Tests a tuple made of scalar constants.
XLA_TEST_F(TupleTest, TupleScalarConstant) {
  XlaBuilder builder(TestName());

  const float constant_scalar1 = 7.3f;
  const float constant_scalar2 = 1.2f;
  auto value = LiteralUtil::MakeTuple(
      {LiteralUtil::CreateR0<float>(constant_scalar1).get(),
       LiteralUtil::CreateR0<float>(constant_scalar2).get()});

  ConstantLiteral(&builder, *value);
  ComputeAndCompareTuple(&builder, *value, {}, error_spec_);
}

// Tests the creation of tuple data.
XLA_TEST_F(TupleTest, TupleCreate) {
  XlaBuilder builder(TestName());

  const float constant_scalar = 7.3f;
  std::initializer_list<float> constant_vector = {1.1f, 2.0f, 3.3f};
  std::initializer_list<std::initializer_list<float>> constant_matrix = {
      {1.1f, 2.2f, 3.5f},  // row 0
      {4.8f, 5.0f, 6.7f},  // row 1
  };
  Tuple(&builder, {ConstantR0<float>(&builder, constant_scalar),
                   ConstantR1<float>(&builder, constant_vector),
                   ConstantR2<float>(&builder, constant_matrix)});

  auto expected = LiteralUtil::MakeTuple(
      {LiteralUtil::CreateR0<float>(constant_scalar).get(),
       LiteralUtil::CreateR1<float>(constant_vector).get(),
       LiteralUtil::CreateR2<float>(constant_matrix).get()});
  ComputeAndCompareTuple(&builder, *expected, {}, error_spec_);
}

// Tests the creation of tuple data.
XLA_TEST_F(TupleTest, TupleCreateWithZeroElementEntry) {
  XlaBuilder builder(TestName());

  Tuple(&builder,
        {ConstantR0<float>(&builder, 7.0), ConstantR1<float>(&builder, {})});

  auto expected =
      LiteralUtil::MakeTuple({LiteralUtil::CreateR0<float>(7.0).get(),
                              LiteralUtil::CreateR1<float>({}).get()});
  ComputeAndCompareTuple(&builder, *expected, {}, error_spec_);
}

// Tests the creation of an empty tuple.
XLA_TEST_F(TupleTest, EmptyTupleCreate) {
  XlaBuilder builder(TestName());
  Tuple(&builder, {});
  auto expected = LiteralUtil::MakeTuple({});
  ComputeAndCompareTuple(&builder, *expected, {}, error_spec_);
}

// Trivial test for extracting a tuple element with GetTupleElement.
XLA_TEST_F(TupleTest, GetTupleElement) {
  XlaBuilder builder(TestName());
  std::initializer_list<float> constant_vector = {1.f, 2.f, 3.f};
  std::initializer_list<std::initializer_list<float>> constant_matrix = {
      {1.f, 2.f, 3.f},  // row 0
      {4.f, 5.f, 6.f},  // row 1
  };
  auto tuple_data =
      Tuple(&builder, {ConstantR1<float>(&builder, constant_vector),
                       ConstantR2<float>(&builder, constant_matrix)});
  GetTupleElement(tuple_data, 1);
  ComputeAndCompareR2<float>(&builder, Array2D<float>(constant_matrix), {},
                             error_spec_);
}

// Trivial test for extracting a tuple element with GetTupleElement.
XLA_TEST_F(TupleTest, GetTupleElementWithZeroElements) {
  XlaBuilder builder(TestName());
  auto tuple_data =
      Tuple(&builder,
            {ConstantR1<float>(&builder, {}),
             ConstantR2FromArray2D<float>(&builder, Array2D<float>(0, 101))});
  GetTupleElement(tuple_data, 1);
  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 101), {}, error_spec_);
}

XLA_TEST_F(TupleTest, GetTupleElementOfNonTupleFailsGracefully) {
  XlaBuilder builder(TestName());
  auto value = ConstantR1<float>(&builder, {4.5f});
  GetTupleElement(value, 1);
  auto result_status = builder.Build();
  EXPECT_FALSE(result_status.ok());
  EXPECT_THAT(
      result_status.status().error_message(),
      ::testing::HasSubstr("Operand to GetTupleElement() is not a tuple"));
}

// Extracts both elements from a tuple with GetTupleElement and then adds them
// together.
XLA_TEST_F(TupleTest, AddTupleElements) {
  XlaBuilder builder(TestName());
  std::initializer_list<float> constant_vector = {1.f, 2.f, 3.f};
  std::initializer_list<std::initializer_list<float>> constant_matrix = {
      {1.f, 2.f, 3.f},  // row 0
      {4.f, 5.f, 6.f},  // row 1
  };
  auto tuple_data =
      Tuple(&builder, {ConstantR1<float>(&builder, constant_vector),
                       ConstantR2<float>(&builder, constant_matrix)});
  auto vector_element = GetTupleElement(tuple_data, 0);
  auto matrix_element = GetTupleElement(tuple_data, 1);
  auto vector_shape = builder.GetShape(vector_element).ConsumeValueOrDie();
  auto matrix_shape = builder.GetShape(matrix_element).ConsumeValueOrDie();
  Add(matrix_element, vector_element,
      /*broadcast_dimensions=*/{1});

  Array2D<float> expected({
      {2.f, 4.f, 6.f},  // row 0
      {5.f, 7.f, 9.f},  // row 1
  });
  ASSERT_TRUE(ShapeUtil::ShapeIs(vector_shape, F32, {3}));
  ASSERT_TRUE(ShapeUtil::ShapeIs(matrix_shape, F32, {/*y=*/2, /*x=*/3}));
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

// Extracts both elements from a tuple and then puts them into a new tuple in
// the opposite order.
XLA_TEST_F(TupleTest, TupleGTEToTuple) {
  XlaBuilder builder(TestName());
  std::initializer_list<float> constant_vector = {1.f, 2.f, 3.f};
  std::initializer_list<std::initializer_list<float>> constant_matrix = {
      {1.f, 2.f, 3.f},  // row 0
      {4.f, 5.f, 6.f},  // row 1
  };
  auto tuple_data =
      Tuple(&builder, {ConstantR1<float>(&builder, constant_vector),
                       ConstantR2<float>(&builder, constant_matrix)});
  Tuple(&builder,
        {GetTupleElement(tuple_data, 1), GetTupleElement(tuple_data, 0)});
  auto expected = LiteralUtil::MakeTuple(
      {LiteralUtil::CreateR2<float>(constant_matrix).get(),
       LiteralUtil::CreateR1<float>(constant_vector).get()});
  ComputeAndCompareTuple(&builder, *expected, {}, error_spec_);
}

XLA_TEST_F(TupleTest, SelectBetweenPredTuples) {
  XlaBuilder b(TestName());
  XlaOp v1, v2;

  for (bool direction : {false, true}) {
    std::unique_ptr<GlobalData> v1_data =
        CreateR0Parameter<float>(0.0f, /*parameter_number=*/0, /*name=*/"v1",
                                 /*builder=*/&b, /*data_handle=*/&v1);
    std::unique_ptr<GlobalData> v2_data =
        CreateR0Parameter<float>(1.0f, /*parameter_number=*/1, /*name=*/"v2",
                                 /*builder=*/&b, /*data_handle=*/&v2);
    auto v1_gt = Gt(v1, v2);                 // false
    auto v2_gt = Gt(v2, v1);                 // true
    auto v1_v2 = Tuple(&b, {v1_gt, v2_gt});  // {false, true}
    auto v2_v1 = Tuple(&b, {v2_gt, v1_gt});  // {true, false}
    Select(direction ? v1_gt : v2_gt, v1_v2, v2_v1);
    auto expected =
        LiteralUtil::MakeTuple({LiteralUtil::CreateR0<bool>(direction).get(),
                                LiteralUtil::CreateR0<bool>(!direction).get()});

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
  XlaBuilder builder(TestName());
  std::initializer_list<float> constant_vector = {1.f, 2.f, 3.f};
  std::initializer_list<std::initializer_list<float>> constant_matrix = {
      {1.f, 2.f, 3.f},  // row 0
      {4.f, 5.f, 6.f},  // row 1
  };
  auto tuple_data =
      Tuple(&builder, {ConstantR1<float>(&builder, constant_vector),
                       ConstantR2<float>(&builder, constant_matrix)});
  auto new_tuple01 = Tuple(&builder, {GetTupleElement(tuple_data, 0),
                                      GetTupleElement(tuple_data, 1)});
  auto new_tuple10 = Tuple(&builder, {GetTupleElement(tuple_data, 1),
                                      GetTupleElement(tuple_data, 0)});
  auto vector_from_01 = GetTupleElement(new_tuple01, 0);
  auto vector_from_10 = GetTupleElement(new_tuple10, 1);
  auto matrix_from_01 = GetTupleElement(new_tuple01, 1);
  auto matrix_from_10 = GetTupleElement(new_tuple10, 0);

  auto addvectors = Add(vector_from_01, vector_from_10);
  auto addmatrices = Add(matrix_from_01, matrix_from_10);

  Add(addmatrices, addvectors,
      /*broadcast_dimensions=*/{1});

  Array2D<float> expected({
      {4.f, 8.f, 12.f},    // row 0
      {10.f, 14.f, 18.f},  // row 1
  });
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(TupleTest, SelectBetweenTuplesOnFalse) {
  // Tests a selection between tuples with "false" path taken.
  XlaBuilder builder(TestName());

  std::initializer_list<float> vec1 = {1.f, 2.f, 3.f};
  std::initializer_list<float> vec2 = {2.f, 4.f, 6.f};
  auto tuple12 = Tuple(&builder, {ConstantR1<float>(&builder, vec1),
                                  ConstantR1<float>(&builder, vec2)});
  auto tuple21 = Tuple(&builder, {ConstantR1<float>(&builder, vec2),
                                  ConstantR1<float>(&builder, vec1)});

  Select(ConstantR0<bool>(&builder, false), tuple12, tuple21);
  auto expected =
      LiteralUtil::MakeTuple({LiteralUtil::CreateR1<float>(vec2).get(),
                              LiteralUtil::CreateR1<float>(vec1).get()});
  ComputeAndCompareTuple(&builder, *expected, {}, error_spec_);
}

XLA_TEST_F(TupleTest, TuplesInAMap) {
  XlaComputation tuple_computation;
  {
    // tuple_computation(x) = 100 * min(x, x^2) + max(x, x^2) using tuples.
    //
    // Need to put a select in there to prevent HLO-level optimizations from
    // optimizing out the tuples.
    XlaBuilder b("sort_square");
    auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {}), "x");
    auto x2 = Mul(x, x);
    auto x_smaller_tuple = Tuple(&b, {x, x2});
    auto x2_smaller_tuple = Tuple(&b, {x2, x});
    auto sorted = Select(Lt(x, x2), x_smaller_tuple, x2_smaller_tuple);
    auto smaller = GetTupleElement(sorted, 0);
    auto greater = GetTupleElement(sorted, 1);
    Add(greater, Mul(ConstantR0<float>(&b, 100.0f), smaller));
    auto computation_status = b.Build();
    ASSERT_IS_OK(computation_status.status());
    tuple_computation = computation_status.ConsumeValueOrDie();
  }

  XlaBuilder b(TestName());
  auto input = ConstantR1<float>(&b, {-1.0f, 1.0f, 2.1f});
  Map(&b, {input}, tuple_computation, {0});
  ComputeAndCompareR1<float>(&b, {-99.0f, 101.0f, 214.41f}, {}, error_spec_);
}

XLA_TEST_F(TupleTest, SelectBetweenTuplesOnTrue) {
  // Tests a selection between tuples with "true" path taken.
  XlaBuilder builder(TestName());

  std::initializer_list<float> vec1 = {1.f, 2.f, 3.f};
  std::initializer_list<float> vec2 = {2.f, 4.f, 6.f};
  auto tuple12 = Tuple(&builder, {ConstantR1<float>(&builder, vec1),
                                  ConstantR1<float>(&builder, vec2)});
  auto tuple21 = Tuple(&builder, {ConstantR1<float>(&builder, vec2),
                                  ConstantR1<float>(&builder, vec1)});

  Select(ConstantR0<bool>(&builder, true), tuple12, tuple21);
  auto expected =
      LiteralUtil::MakeTuple({LiteralUtil::CreateR1<float>(vec1).get(),
                              LiteralUtil::CreateR1<float>(vec2).get()});
  ComputeAndCompareTuple(&builder, *expected, {}, error_spec_);
}

XLA_TEST_F(TupleTest, SelectBetweenTuplesElementResult) {
  // Tests a selection between tuples but the final result is an element of the
  // tuple, not the whole tuple.
  XlaBuilder builder(TestName());

  std::initializer_list<float> vec1 = {1.f, 2.f, 3.f};
  std::initializer_list<float> vec2 = {2.f, 4.f, 6.f};
  auto tuple12 = Tuple(&builder, {ConstantR1<float>(&builder, vec1),
                                  ConstantR1<float>(&builder, vec2)});
  auto tuple21 = Tuple(&builder, {ConstantR1<float>(&builder, vec2),
                                  ConstantR1<float>(&builder, vec1)});

  auto select = Select(ConstantR0<bool>(&builder, false), tuple12, tuple21);
  GetTupleElement(select, 0);

  ComputeAndCompareR1<float>(&builder, vec2, {}, error_spec_);
}

// Cascaded selects between tuple types.
XLA_TEST_F(TupleTest, SelectBetweenTuplesCascaded) {
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
  XlaBuilder builder(TestName());

  std::initializer_list<float> vec1 = {1.f, 2.f, 3.f};
  std::initializer_list<float> vec2 = {2.f, 4.f, 6.f};

  auto pred_tuple = Tuple(&builder, {ConstantR0<bool>(&builder, true),
                                     ConstantR0<bool>(&builder, false)});
  auto tuple12 = Tuple(&builder, {ConstantR1<float>(&builder, vec1),
                                  ConstantR1<float>(&builder, vec2)});
  auto tuple21 = Tuple(&builder, {ConstantR1<float>(&builder, vec2),
                                  ConstantR1<float>(&builder, vec1)});

  auto select1 = Select(GetTupleElement(pred_tuple, 0), tuple12, tuple21);
  auto select2 = Select(GetTupleElement(pred_tuple, 1), tuple21, select1);
  Add(GetTupleElement(select2, 0), GetTupleElement(select2, 1));

  ComputeAndCompareR1<float>(&builder, {3.f, 6.f, 9.f}, {}, error_spec_);
}

XLA_TEST_F(TupleTest, SelectBetweenTuplesReuseConstants) {
  // Similar to SelectBetweenTuples, but the constants are shared between the
  // input tuples.
  XlaBuilder builder(TestName());

  std::initializer_list<float> vec1 = {1.f, 2.f, 3.f};
  std::initializer_list<float> vec2 = {2.f, 4.f, 6.f};
  auto c1 = ConstantR1<float>(&builder, vec1);
  auto c2 = ConstantR1<float>(&builder, vec2);
  auto tuple12 = Tuple(&builder, {c1, c2});
  auto tuple21 = Tuple(&builder, {c2, c1});

  Select(ConstantR0<bool>(&builder, false), tuple12, tuple21);

  auto expected =
      LiteralUtil::MakeTuple({LiteralUtil::CreateR1<float>(vec2).get(),
                              LiteralUtil::CreateR1<float>(vec1).get()});
  ComputeAndCompareTuple(&builder, *expected, {}, error_spec_);
}

XLA_TEST_F(TupleTest, NestedTuples) {
  XlaBuilder builder(TestName());
  auto inner_tuple = Tuple(&builder, {ConstantR1<float>(&builder, {1.0, 2.0}),
                                      ConstantR0<float>(&builder, 42.0)});
  Tuple(&builder, {inner_tuple, ConstantR1<float>(&builder, {22.0, 44.0})});

  auto expected_v1 = LiteralUtil::CreateR1<float>({1.0, 2.0});
  auto expected_s = LiteralUtil::CreateR0<float>(42.0);
  auto expected_inner_tuple =
      LiteralUtil::MakeTuple({expected_v1.get(), expected_s.get()});
  auto expected_v2 = LiteralUtil::CreateR1<float>({22.0, 44.0});
  auto expected =
      LiteralUtil::MakeTuple({expected_inner_tuple.get(), expected_v2.get()});

  ComputeAndCompareTuple(&builder, *expected, {}, error_spec_);
}

XLA_TEST_F(TupleTest, GetTupleElementOfNestedTuple) {
  XlaBuilder builder(TestName());

  Shape data_shape = ShapeUtil::MakeShape(F32, {3});
  Shape inner_tuple_shape = ShapeUtil::MakeTupleShape({data_shape, data_shape});
  Shape outer_tuple_shape =
      ShapeUtil::MakeTupleShape({inner_tuple_shape, data_shape});

  auto input = Parameter(&builder, 0, outer_tuple_shape, "input");
  auto gte0 = GetTupleElement(input, 0);
  auto gte1 = GetTupleElement(gte0, 1);
  Add(gte1, ConstantR1<float>(&builder, {10.0, 11.0, 12.0}));

  std::unique_ptr<GlobalData> data =
      client_
          ->TransferToServer(*LiteralUtil::MakeTuple({
              LiteralUtil::MakeTuple(
                  {
                      LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0}).get(),
                      LiteralUtil::CreateR1<float>({4.0, 5.0, 6.0}).get(),
                  })
                  .get(),
              LiteralUtil::CreateR1<float>({7.0, 8.0, 9.0}).get(),
          }))
          .ConsumeValueOrDie();

  std::vector<GlobalData*> arguments = {data.get()};
  const std::vector<float> expected = {4.0 + 10.0, 5.0 + 11.0, 6.0 + 12.0};
  ComputeAndCompareR1<float>(&builder, expected, arguments, ErrorSpec(1e-5));
}

XLA_TEST_F(TupleTest, ComplexTuples) {
  XlaBuilder builder(TestName());
  {
    Shape c64r0 = ShapeUtil::MakeShape(C64, {});
    Shape c64r1 = ShapeUtil::MakeShape(C64, {2});
    Shape c64r2 = ShapeUtil::MakeShape(C64, {3, 2});
    Shape arg0_shape = ShapeUtil::MakeTupleShape(
        {c64r0, ShapeUtil::MakeTupleShape({c64r1, c64r2})});
    auto input0 = Parameter(&builder, 0, arg0_shape, "input0");
    auto t0 = GetTupleElement(input0, 0);
    auto t1 = GetTupleElement(input0, 1);
    auto t10 = GetTupleElement(t1, 0);
    auto t11 = GetTupleElement(t1, 1);
    auto sum = Add(Add(t10, t11, {1}), t0);
    auto input1 = Parameter(&builder, 1, c64r1, "input1");
    auto prod = Mul(input1, sum, {1});
    Tuple(&builder, {Tuple(&builder, {prod, sum}),
                     ConstantR0<complex64>(&builder, {123, 456})});
  }

  std::unique_ptr<GlobalData> arg0 =
      client_
          ->TransferToServer(*LiteralUtil::MakeTuple(
              {LiteralUtil::CreateR0<complex64>({1, 2}).get(),
               LiteralUtil::MakeTuple(
                   {LiteralUtil::CreateR1<complex64>({{10, 20}, {30, 40}})
                        .get(),
                    LiteralUtil::CreateR2<complex64>(
                        {{{100, 200}, {300, 400}},
                         {{1000, 2000}, {3000, 4000}},
                         {{10000, 20000}, {30000, 40000}}})
                        .get()})
                   .get()}))
          .ConsumeValueOrDie();
  std::unique_ptr<GlobalData> arg1 =
      client_
          ->TransferToServer(
              *LiteralUtil::CreateR1<complex64>({{1, 2}, {1, -2}}))
          .ConsumeValueOrDie();
  auto sum =
      LiteralUtil::CreateR2<complex64>({{{111, 222}, {331, 442}},
                                        {{1011, 2022}, {3031, 4042}},
                                        {{10011, 20022}, {30031, 40042}}});
  auto prod = MakeUnique<Literal>(sum->shape());
  ASSERT_TRUE(prod->Populate<complex64>(
                      [&sum](tensorflow::gtl::ArraySlice<int64> indexes) {
                        return sum->Get<complex64>(indexes) *
                               (indexes[indexes.size() - 1] == 0
                                    ? complex64(1, 2)
                                    : complex64(1, -2));
                      })
                  .ok());
  auto expected = LiteralUtil::MakeTuple(
      {LiteralUtil::MakeTuple({prod.get(), sum.get()}).get(),
       LiteralUtil::CreateR0<complex64>({123, 456}).get()});
  ComputeAndCompareTuple(&builder, *expected, {arg0.get(), arg1.get()},
                         error_spec_);
}

class TupleHloTest : public HloTestBase {};

// Disabled on the interpreter because bitcast doesn't exist on the interpreter.
XLA_TEST_F(TupleHloTest, DISABLED_ON_INTERPRETER(BitcastAfterGTE)) {
  const char* testcase = R"(
    HloModule m

    ENTRY test {
      name.1 = (f32[3]{0}) parameter(0)
      get-tuple-element.1 = f32[3]{0} get-tuple-element(name.1), index=0
      bitcast = f32[1,3]{1,0} bitcast(get-tuple-element.1)
      copy = f32[1,3]{1,0} copy(bitcast)
      ROOT tuple.4 = (f32[1,3]{1,0}) tuple(copy)
    }
  )";
  auto module =
      HloRunner::CreateModuleFromString(testcase, GetDebugOptionsForTest())
          .ValueOrDie();
  auto param =
      LiteralUtil::MakeTupleOwned(LiteralUtil::CreateR1<float>({1, 2, 3}));
  auto result = ExecuteNoHloPasses(std::move(module), {param.get()});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      *LiteralUtil::MakeTupleOwned(LiteralUtil::CreateR2<float>({{1, 2, 3}})),
      *result));
}

// Disabled on interpreter due to lack of outfeed.
XLA_TEST_F(TupleHloTest,
           DISABLED_ON_INTERPRETER(NonAmbiguousTopLevelAllocation)) {
  const char* testcase = R"(
    HloModule tuple

    ENTRY main {
      a = f32[2] parameter(0)
      b = f32[2] parameter(1)
      c = f32[2] parameter(2)
      d = f32[2] parameter(3)
      cond = pred[] parameter(4)

      tup0 = (f32[2],f32[2]) tuple(a, b)
      tup1 = (f32[2],f32[2]) tuple(c, d)

      s = (f32[2],f32[2]) tuple-select(cond, tup0, tup1)
      gte = f32[2] get-tuple-element(s), index=0
      tuple = (f32[2]) tuple(gte)
      token = token[] after-all()
      ROOT outfeed = token[] outfeed(tuple, token)
    }
  )";
  auto module =
      HloRunner::CreateModuleFromString(testcase, GetDebugOptionsForTest())
          .ValueOrDie();
  auto param0 = LiteralUtil::CreateR1<float>({1, 2});
  auto param1 = LiteralUtil::CreateR1<float>({2, 3});
  auto param4 = LiteralUtil::CreateR0<bool>(false);
  // Put execution on a separate thread so we can block on outfeed.
  std::unique_ptr<tensorflow::Thread> thread(
      tensorflow::Env::Default()->StartThread(
          tensorflow::ThreadOptions(), "execute_thread", [&] {
            TF_EXPECT_OK(Execute(std::move(module),
                                 {param0.get(), param1.get(), param1.get(),
                                  param0.get(), param4.get()})
                             .status());
          }));
  auto expected =
      LiteralUtil::MakeTupleOwned(LiteralUtil::CreateR1<float>({2, 3}));
  auto literal = Literal::CreateFromShape(expected->shape());
  TF_EXPECT_OK(backend().transfer_manager()->TransferLiteralFromOutfeed(
      backend().default_stream_executor(), expected->shape(), *literal));
  EXPECT_TRUE(LiteralTestUtil::Equal(*expected, *literal));
}

}  // namespace
}  // namespace xla
