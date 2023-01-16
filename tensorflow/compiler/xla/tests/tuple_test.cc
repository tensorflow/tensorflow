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
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/test.h"

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
  auto value = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<float>(constant_scalar),
       LiteralUtil::CreateR1<float>(constant_vector),
       LiteralUtil::CreateR2<float>(constant_matrix)});

  ConstantLiteral(&builder, value);
  ComputeAndCompareTuple(&builder, value, {}, error_spec_);
}

// Tests a tuple made of scalar constants.
XLA_TEST_F(TupleTest, TupleScalarConstant) {
  XlaBuilder builder(TestName());

  const float constant_scalar1 = 7.3f;
  const float constant_scalar2 = 1.2f;
  auto value = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<float>(constant_scalar1),
       LiteralUtil::CreateR0<float>(constant_scalar2)});

  ConstantLiteral(&builder, value);
  ComputeAndCompareTuple(&builder, value, {}, error_spec_);
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

  auto expected = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<float>(constant_scalar),
       LiteralUtil::CreateR1<float>(constant_vector),
       LiteralUtil::CreateR2<float>(constant_matrix)});
  ComputeAndCompareTuple(&builder, expected, {}, error_spec_);
}

// Tests the creation of tuple data.
XLA_TEST_F(TupleTest, TupleCreateWithZeroElementEntry) {
  XlaBuilder builder(TestName());

  Tuple(&builder,
        {ConstantR0<float>(&builder, 7.0), ConstantR1<float>(&builder, {})});

  auto expected = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<float>(7.0), LiteralUtil::CreateR1<float>({})});
  ComputeAndCompareTuple(&builder, expected, {}, error_spec_);
}

// Tests the creation of an empty tuple.
XLA_TEST_F(TupleTest, EmptyTupleCreate) {
  XlaBuilder builder(TestName());
  Tuple(&builder, {});
  auto expected = LiteralUtil::MakeTuple({});
  ComputeAndCompareTuple(&builder, expected, {}, error_spec_);
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
  auto vector_shape = builder.GetShape(vector_element).value();
  auto matrix_shape = builder.GetShape(matrix_element).value();
  Add(matrix_element, vector_element,
      /*broadcast_dimensions=*/{1});

  Array2D<float> expected({
      {2.f, 4.f, 6.f},  // row 0
      {5.f, 7.f, 9.f},  // row 1
  });
  ASSERT_TRUE(ShapeUtil::Equal(vector_shape, ShapeUtil::MakeShape(F32, {3})));
  ASSERT_TRUE(ShapeUtil::Equal(matrix_shape,
                               ShapeUtil::MakeShape(F32, {/*y=*/2, /*x=*/3})));
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
  auto expected = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR2<float>(constant_matrix),
       LiteralUtil::CreateR1<float>(constant_vector)});
  ComputeAndCompareTuple(&builder, expected, {}, error_spec_);
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

XLA_TEST_F(TupleTest, NestedTuples) {
  if (IsMlirLoweringEnabled()) {
    GTEST_SKIP() << "Nested tuples are not supported by the MLIR pipeline";
  }

  XlaBuilder builder(TestName());
  auto inner_tuple = Tuple(&builder, {ConstantR1<float>(&builder, {1.0, 2.0}),
                                      ConstantR0<float>(&builder, 42.0)});
  Tuple(&builder, {inner_tuple, ConstantR1<float>(&builder, {22.0, 44.0})});

  auto expected_v1 = LiteralUtil::CreateR1<float>({1.0, 2.0});
  auto expected_s = LiteralUtil::CreateR0<float>(42.0);
  auto expected_inner_tuple =
      LiteralUtil::MakeTuple({&expected_v1, &expected_s});
  auto expected_v2 = LiteralUtil::CreateR1<float>({22.0, 44.0});
  auto expected = LiteralUtil::MakeTuple({&expected_inner_tuple, &expected_v2});

  ComputeAndCompareTuple(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(TupleTest, GetTupleElementOfNestedTuple) {
  if (IsMlirLoweringEnabled()) {
    GTEST_SKIP() << "Nested tuples are not supported by the MLIR pipeline";
  }

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
          ->TransferToServer(LiteralUtil::MakeTupleFromSlices({
              LiteralUtil::MakeTupleFromSlices({
                  LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0}),
                  LiteralUtil::CreateR1<float>({4.0, 5.0, 6.0}),
              }),
              LiteralUtil::CreateR1<float>({7.0, 8.0, 9.0}),
          }))
          .value();

  std::vector<GlobalData*> arguments = {data.get()};
  const std::vector<float> expected = {4.0 + 10.0, 5.0 + 11.0, 6.0 + 12.0};
  ComputeAndCompareR1<float>(&builder, expected, arguments, ErrorSpec(1e-5));
}

XLA_TEST_F(TupleTest, ComplexTuples) {
  if (IsMlirLoweringEnabled()) {
    GTEST_SKIP() << "Nested tuples are not supported by the MLIR pipeline";
  }

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
          ->TransferToServer(LiteralUtil::MakeTupleFromSlices(
              {LiteralUtil::CreateR0<complex64>({1, 2}),
               LiteralUtil::MakeTupleFromSlices(
                   {LiteralUtil::CreateR1<complex64>({{10, 20}, {30, 40}}),
                    LiteralUtil::CreateR2<complex64>(
                        {{{100, 200}, {300, 400}},
                         {{1000, 2000}, {3000, 4000}},
                         {{10000, 20000}, {30000, 40000}}})})}))
          .value();
  std::unique_ptr<GlobalData> arg1 =
      client_
          ->TransferToServer(
              LiteralUtil::CreateR1<complex64>({{1, 2}, {1, -2}}))
          .value();
  auto sum =
      LiteralUtil::CreateR2<complex64>({{{111, 222}, {331, 442}},
                                        {{1011, 2022}, {3031, 4042}},
                                        {{10011, 20022}, {30031, 40042}}});
  Literal prod(sum.shape());
  ASSERT_TRUE(prod.Populate<complex64>([&sum](
                                           absl::Span<const int64_t> indexes) {
                    return sum.Get<complex64>(indexes) *
                           (indexes[indexes.size() - 1] == 0
                                ? complex64(1, 2)
                                : complex64(1, -2));
                  })
                  .ok());
  auto expected = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::MakeTupleFromSlices({prod, sum}),
       LiteralUtil::CreateR0<complex64>({123, 456})});
  ComputeAndCompareTuple(&builder, expected, {arg0.get(), arg1.get()},
                         error_spec_);
}

class TupleHloTest : public HloTestBase {};

XLA_TEST_F(TupleHloTest, BadTupleShapeFailsGracefully) {
  const char* testcase = R"(
    HloModule m, is_scheduled=true

    ENTRY test {
      parameter = f32[3]{0} parameter(0)
      ROOT tuple = (f32[3]{0}, f32[3]{0}) tuple(parameter)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(testcase));
  auto status = verifier().Run(module.get()).status();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.error_message(),
      ::testing::HasSubstr("Expected instruction to have shape equal to"));
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("actual shape is"));
}

XLA_TEST_F(TupleHloTest, BitcastAfterGTE) {
  if (IsMlirLoweringEnabled()) {
    // Bitcasts are not generated by frontends directly and are not supported by
    // the MLIR pipeline.
    GTEST_SKIP() << "Bitcast is not supported by the MLIR pipeline";
  }

  const char* testcase = R"(
    HloModule m, is_scheduled=true

    ENTRY test {
      name.1 = (f32[3]{0}) parameter(0)
      get-tuple-element.1 = f32[3]{0} get-tuple-element(name.1), index=0
      bitcast = f32[1,3]{1,0} bitcast(get-tuple-element.1)
      copy = f32[1,3]{1,0} copy(bitcast)
      ROOT tuple.4 = (f32[1,3]{1,0}) tuple(copy)
    }
  )";
  auto module = ParseAndReturnVerifiedModule(testcase).value();
  auto param =
      LiteralUtil::MakeTupleOwned(LiteralUtil::CreateR1<float>({1, 2, 3}));
  auto result = ExecuteNoHloPasses(std::move(module), {&param});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::MakeTupleOwned(LiteralUtil::CreateR2<float>({{1, 2, 3}})),
      result));
}

}  // namespace
}  // namespace xla
