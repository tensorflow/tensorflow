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

#include <cmath>
#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class VecOpsSimpleTest : public ClientLibraryTestBase {
 public:
  explicit VecOpsSimpleTest(perftools::gputools::Platform* platform = nullptr)
      : ClientLibraryTestBase(platform) {
    mutable_debug_options()->add_xla_disable_hlo_passes("algsimp");
    mutable_debug_options()->add_xla_disable_hlo_passes("inline");
  }

  ErrorSpec error_spec_{0.0001};
};

XLA_TEST_F(VecOpsSimpleTest, ExpTenValues) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<float>(
      {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  auto exp = builder.Exp(x);

  std::vector<float> expected = {8.1662,     7.4274e-02, 13.4637,    1.8316e-02,
                                 8.1662,     9.9742,     6.7379e-03, 4.0657e-01,
                                 9.0718e-02, 4.9530};

  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, ExpManyValues) {
  for (int count : {63, 64, 65, 127, 128, 129, 17 * 4096}) {
    ComputationBuilder builder(client_, TestName());
    std::vector<float> exponents;
    exponents.reserve(count);
    for (int i = 0; i < count; ++i) {
      exponents.push_back(i / static_cast<float>(count));
    }
    auto x = builder.ConstantR1<float>(exponents);
    auto exp = builder.Exp(x);

    std::vector<float> expected;
    expected.reserve(exponents.size());
    for (float exponent : exponents) {
      expected.push_back(std::exp(exponent));
    }

    ComputeAndCompareR1<float>(&builder, expected, {},
                               ErrorSpec(/*aabs=*/1e-2, /*arel=*/1e-3));
  }
}

XLA_TEST_F(VecOpsSimpleTest, ExpIn4D) {
  ComputationBuilder builder(client_, TestName());
  Array4D<float> exponents(2, 2, 2, 2);

  std::vector<float> exponents_vector;
  std::vector<float> expected_vector;
  for (int i = 0; i < exponents.num_elements(); ++i) {
    exponents_vector.push_back(static_cast<float>(i) /
                               exponents.num_elements());
    expected_vector.push_back(std::exp(exponents_vector.back()));
  }
  exponents.SetValues(exponents_vector);

  Array4D<float> expected(2, 2, 2, 2, expected_vector);

  auto x = builder.ConstantR4FromArray4D<float>(exponents);
  auto exp = builder.Exp(x);

  ComputeAndCompareR4<float>(&builder, expected, {},
                             ErrorSpec(/*aabs=*/1e-2, /*arel=*/1e-3));
}

XLA_TEST_F(VecOpsSimpleTest, NegateTenFloatValues) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<float>(
      {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  builder.Neg(x);

  std::vector<float> expected = {-2.1, 2.6, -2.6, 4.0, -2.1,
                                 -2.3, 5.0, 0.9,  2.4, -1.6};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, NegateTenInt32Values) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<int32>({2, -2, 12, -4, 5, 20, -15, 0, -2, 1});
  builder.Neg(x);

  std::vector<int> expected = {-2, 2, -12, 4, -5, -20, 15, 0, 2, -1};
  ComputeAndCompareR1<int32>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, NegateUint32Values) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<uint32>(
      {0, 1, 42, static_cast<uint32>(-1), static_cast<uint32>(-12)});
  builder.Neg(x);
  std::vector<uint32> expected = {0, static_cast<uint32>(-1),
                                  static_cast<uint32>(-42), 1, 12};
  ComputeAndCompareR1<uint32>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, SquareTenValues) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<float>(
      {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  builder.SquareF32(x);

  std::vector<float> expected = {4.41, 6.76, 6.76, 16.,  4.41,
                                 5.29, 25.,  0.81, 5.76, 2.56};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, ReciprocalTenValues) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<float>(
      {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  builder.ReciprocalF32(x);

  std::vector<float> expected = {
      0.47619048, -0.38461538, 0.38461538,  -0.25,       0.47619048,
      0.43478261, -0.2,        -1.11111111, -0.41666667, 0.625};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, SqrtZeroes) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<float>({0.0, -0.0});
  auto exp = builder.SqrtF32(x);

  ComputeAndCompareR1<float>(&builder, {0, 0}, {}, error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, SqrtSixValues) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<float>({16.0, 1.0, 1024.0, 0.16, 0.2, 12345});
  auto exp = builder.SqrtF32(x);

  std::vector<float> expected = {4, 1, 32, 0.4, 0.4472, 111.1080};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, InvSqrtSevenValues) {
  ComputationBuilder builder(client_, TestName());
  auto x =
      builder.ConstantR1<float>({16.0, 1.0, 1024.0, 0.16, 0.2, 12345, 1.2345});
  auto exp = builder.Pow(x, builder.ConstantR0<float>(-.5f));

  std::vector<float> expected = {.25,     1,       .03125, 2.5,
                                 2.23607, .009000, .900025};

  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, AddTenValuesViaMap) {
  ComputationBuilder builder(client_, TestName());
  auto add = CreateScalarAddComputation(F32, &builder);

  auto x = builder.ConstantR1<float>(
      {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  auto y = builder.ConstantR1<float>(
      {-0.4, -0.6, -3.0, 0.2, 3.8, -2.2, -1.8, 4.9, 1.4, 0.6});
  auto max = builder.Map({x, y}, add, {0});

  std::vector<float> expected = {1.7, -3.2, -0.4, -3.8, 5.9,
                                 0.1, -6.8, 4.,   -1.,  2.2};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, MaxTenValues) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<float>(
      {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  auto y = builder.ConstantR1<float>(
      {-0.4, -0.6, -3.0, 0.2, 3.8, -2.2, -1.8, 4.9, 1.4, 0.6});
  auto max = builder.Max(x, y);

  std::vector<float> expected = {2.1, -0.6, 2.6, 0.2, 3.8,
                                 2.3, -1.8, 4.9, 1.4, 1.6};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, MaxTenValuesFromParams) {
  // Similar to MaxTenValues, except that the inputs come from params rather
  // than constants.
  ComputationBuilder builder(client_, TestName());
  ComputationDataHandle v1, v2;
  std::unique_ptr<GlobalData> param0_data = CreateR1Parameter<float>(
      {41.0f, 2.0f, 3.0f, 84.0f}, /*parameter_number=*/0, /*name=*/"v1",
      /*builder=*/&builder, /*data_handle=*/&v1);
  std::unique_ptr<GlobalData> param1_data = CreateR1Parameter<float>(
      {21.0f, 22.0f, 23.0f, 24.0f}, /*parameter_number=*/1, /*name=*/"v2",
      /*builder=*/&builder, /*data_handle=*/&v2);

  auto max = builder.Max(v1, v2);
  ComputeAndCompareR1<float>(&builder, {41.0f, 22.0f, 23.0f, 84.0f},
                             {param0_data.get(), param1_data.get()},
                             error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, Max15000ValuesFromParams) {
  // Similar to MaxTenValuesFromParams, except that the data size passed in and
  // out is large.
  ComputationBuilder builder(client_, TestName());

  // Number of floats in the data passed into and out of the computation.
  constexpr int datalen = 15 * 1000;

  // The inputs are initialized with a special pattern where in the first third
  // of the data v1[i] > v2[i] and elsewhere it's vice versa.
  std::vector<float> v1vec;
  std::vector<float> v2vec;
  std::vector<float> expected_vec;
  for (int i = 0; i < datalen; ++i) {
    float smaller = i;
    float larger = i * 2;
    if (i < datalen / 3) {
      v1vec.push_back(larger);
      v2vec.push_back(smaller);
    } else {
      v1vec.push_back(smaller);
      v2vec.push_back(larger);
    }
    expected_vec.push_back(larger);
  }

  ComputationDataHandle v1, v2;
  std::unique_ptr<GlobalData> param0_data =
      CreateR1Parameter<float>(v1vec, /*parameter_number=*/0, /*name=*/"v1",
                               /*builder=*/&builder, /*data_handle=*/&v1);
  std::unique_ptr<GlobalData> param1_data =
      CreateR1Parameter<float>(v2vec, /*parameter_number=*/1, /*name=*/"v2",
                               /*builder=*/&builder, /*data_handle=*/&v2);

  auto max = builder.Max(v1, v2);
  ComputeAndCompareR1<float>(&builder, expected_vec,
                             {param0_data.get(), param1_data.get()},
                             error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, MaxTenValuesWithScalar) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<float>(
      {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  auto y = builder.ConstantR0<float>(0);
  auto max = builder.Max(x, y);

  std::vector<float> expected = {2.1, 0.0, 2.6, 0.0, 2.1,
                                 2.3, 0.0, 0.0, 0.0, 1.6};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, MinTenValues) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<float>(
      {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  auto y = builder.ConstantR1<float>(
      {-0.4, -0.6, -3.0, 0.2, 3.8, -2.2, -1.8, 4.9, 1.4, 0.6});
  auto min = builder.Min(x, y);

  std::vector<float> expected = {-0.4, -2.6, -3.0, -4.0, 2.1,
                                 -2.2, -5.0, -0.9, -2.4, 0.6};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, MinMaxTenValues) {
  ComputationBuilder builder(client_, TestName());
  auto zero = builder.ConstantR0<float>(0);
  auto one = builder.ConstantR0<float>(1);
  auto x = builder.ConstantR1<float>(
      {2.1, -2.6, 2.6, 0.3, 3.1, 0.9, -5.0, 0.1, -2.4, 0.6});
  auto clamp = builder.Min(builder.Max(x, zero), one);

  std::vector<float> expected = {1.0, 0.0, 1.0, 0.3, 1.0,
                                 0.9, 0.0, 0.1, 0.0, 0.6};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, ClampTenValuesConstant) {
  ComputationBuilder builder(client_, TestName());
  auto zero = builder.ConstantR0<float>(0);
  auto one = builder.ConstantR0<float>(1);
  auto x = builder.ConstantR1<float>(
      {2.1, -2.6, 2.6, 0.3, 3.1, 0.9, -5.0, 0.1, -2.4, 0.6});
  auto clamp = builder.Clamp(zero, x, one);

  std::vector<float> expected = {1.0, 0.0, 1.0, 0.3, 1.0,
                                 0.9, 0.0, 0.1, 0.0, 0.6};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, ClampTwoValuesConstant) {
  ComputationBuilder builder(client_, TestName());
  auto zero = builder.ConstantR1<float>({0.0f, 0.0f});
  auto one = builder.ConstantR1<float>({1.0f, 1.0f});
  auto x = builder.ConstantR1<float>({2.1, -2.6});
  auto clamp = builder.Clamp(zero, x, one);

  std::vector<float> expected = {1.0, 0.0};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, ClampTenValuesConstantNonzeroLower) {
  ComputationBuilder builder(client_, TestName());
  auto one = builder.ConstantR0<float>(1);
  auto two = builder.ConstantR0<float>(2);
  auto x = builder.ConstantR1<float>(
      {2.1, -2.6, 2.6, 0.3, 3.1, 0.9, -5.0, 0.1, -2.4, 0.6});
  auto clamp = builder.Clamp(one, x, two);

  std::vector<float> expected = {2.0, 1.0, 2.0, 1.0, 2.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, MapTenValues) {
  Computation add_half;
  {
    // add_half(x) = x + 0.5
    ComputationBuilder builder(client_, "add_half");
    auto x_value =
        builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x_value");
    auto half = builder.ConstantR0<float>(0.5);
    builder.Add(x_value, half);
    auto computation_status = builder.Build();
    ASSERT_IS_OK(computation_status.status());
    add_half = computation_status.ConsumeValueOrDie();
  }

  Computation clamp;
  {
    // clamp(y) = clamp<0,5>(y)
    ComputationBuilder builder(client_, "clamp");
    auto y_value =
        builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "y_value");
    auto zero = builder.ConstantR0<float>(0.0);
    auto clamped = builder.Clamp(zero, y_value, builder.ConstantR0<float>(5));
    auto computation_status = builder.Build();
    ASSERT_IS_OK(computation_status.status());
    clamp = computation_status.ConsumeValueOrDie();
  }

  Computation mult_relu_add;
  {
    // mult_relu_add(z) = clamp(add_half(2 * max(z, 0)))
    ComputationBuilder builder(client_, "mult_relu_add");
    auto z_value =
        builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "z_value");
    auto zero = builder.ConstantR0<float>(0.0);
    auto two = builder.ConstantR0<float>(2.0);
    auto max = builder.Max(z_value, zero);
    auto mult = builder.Mul(two, max);
    auto inner = builder.Map({mult}, add_half, {});
    builder.Map({inner}, clamp, {});
    auto computation_status = builder.Build();
    ASSERT_IS_OK(computation_status.status());
    mult_relu_add = computation_status.ConsumeValueOrDie();
  }

  ComputationBuilder builder(client_, "map10");
  {
    auto x = builder.ConstantR1<float>(
        {2.1, -21.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
    auto activations = builder.Map({x}, mult_relu_add, {0});
  }

  std::vector<float> expected = {4.7, 0.5, 5.0, 0.5, 4.7,
                                 5.0, 0.5, 0.5, 0.5, 3.7};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, RemainderTenValuesS32) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<int32>({-5, -4, -3, -2, -1, 0, 1, 2, 3, 4});
  auto y = builder.ConstantR0<int32>(3);
  builder.Rem(x, y);

  std::vector<int32> expected = {-2, -1, 0, -2, -1, 0, 1, 2, 0, 1};
  ComputeAndCompareR1<int32>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, VectorPredicateEqual) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<bool>({false, true});
  auto y = builder.ConstantR1<bool>({true, false});
  builder.Eq(x, y);

  std::array<bool, 2> expected = {{false, false}};
  ComputeAndCompareR1<bool>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, VectorPredicateNotEqual) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<bool>({false, true});
  auto y = builder.ConstantR1<bool>({true, false});
  builder.Ne(x, y);

  std::array<bool, 2> expected = {{true, true}};
  ComputeAndCompareR1<bool>(&builder, expected, {});
}

}  // namespace
}  // namespace xla
