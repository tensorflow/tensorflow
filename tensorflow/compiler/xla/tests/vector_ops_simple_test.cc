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
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class VecOpsSimpleTest : public ClientLibraryTestBase {
 public:
  explicit VecOpsSimpleTest(se::Platform* platform = nullptr)
      : ClientLibraryTestBase(platform) {
    mutable_debug_options()->add_xla_disable_hlo_passes("algsimp");
    mutable_debug_options()->add_xla_disable_hlo_passes("inline");
  }

  ErrorSpec error_spec_{0.0001};
};

XLA_TEST_F(VecOpsSimpleTest, ExpTenValues) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  Exp(x);

  std::vector<float> expected = {8.1662,     7.4274e-02, 13.4637,    1.8316e-02,
                                 8.1662,     9.9742,     6.7379e-03, 4.0657e-01,
                                 9.0718e-02, 4.9530};

  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, ExpManyValues) {
  for (int count : {63, 64, 65, 127, 128, 129, 17 * 4096}) {
    XlaBuilder builder(TestName());
    std::vector<float> exponents;
    exponents.reserve(count);
    for (int i = 0; i < count; ++i) {
      exponents.push_back(i / static_cast<float>(count));
    }
    auto x = ConstantR1<float>(&builder, exponents);
    Exp(x);

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
  XlaBuilder builder(TestName());
  Array4D<float> exponents(2, 2, 2, 2);

  std::vector<float> exponents_vector;
  std::vector<float> expected_vector;
  const auto num_elements = exponents.num_elements();
  exponents_vector.reserve(num_elements);
  expected_vector.reserve(num_elements);
  for (int i = 0; i < exponents.num_elements(); ++i) {
    exponents_vector.push_back(static_cast<float>(i) /
                               exponents.num_elements());
    expected_vector.push_back(std::exp(exponents_vector.back()));
  }
  exponents.SetValues(exponents_vector);

  Array4D<float> expected(2, 2, 2, 2, expected_vector);

  auto x = ConstantR4FromArray4D<float>(&builder, exponents);
  Exp(x);

  ComputeAndCompareR4<float>(&builder, expected, {},
                             ErrorSpec(/*aabs=*/1e-2, /*arel=*/1e-3));
}

XLA_TEST_F(VecOpsSimpleTest, NegateTenFloatValues) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  Neg(x);

  std::vector<float> expected = {-2.1, 2.6, -2.6, 4.0, -2.1,
                                 -2.3, 5.0, 0.9,  2.4, -1.6};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, NegateTenInt32Values) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<int32_t>(&builder, {2, -2, 12, -4, 5, 20, -15, 0, -2, 1});
  Neg(x);

  std::vector<int> expected = {-2, 2, -12, 4, -5, -20, 15, 0, 2, -1};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, NegateUint32Values) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<uint32_t>(&builder, {0, 1, 42, static_cast<uint32_t>(-1),
                                           static_cast<uint32_t>(-12)});
  Neg(x);
  std::vector<uint32_t> expected = {0, static_cast<uint32_t>(-1),
                                    static_cast<uint32_t>(-42), 1, 12};
  ComputeAndCompareR1<uint32_t>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, InvSqrtSevenValues) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(&builder,
                             {16.0, 1.0, 1024.0, 0.16, 0.2, 12345, 1.2345});
  Pow(x, ConstantR0<float>(&builder, -.5f));

  std::vector<float> expected = {.25,     1,       .03125, 2.5,
                                 2.23607, .009000, .900025};

  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, AddTenValuesViaMap) {
  XlaBuilder builder(TestName());
  auto add = CreateScalarAddComputation(F32, &builder);

  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  auto y = ConstantR1<float>(
      &builder, {-0.4, -0.6, -3.0, 0.2, 3.8, -2.2, -1.8, 4.9, 1.4, 0.6});
  Map(&builder, {x, y}, add, {0});

  std::vector<float> expected = {1.7, -3.2, -0.4, -3.8, 5.9,
                                 0.1, -6.8, 4.,   -1.,  2.2};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, MaxTenValues) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  auto y = ConstantR1<float>(
      &builder, {-0.4, -0.6, -3.0, 0.2, 3.8, -2.2, -1.8, 4.9, 1.4, 0.6});
  Max(x, y);

  std::vector<float> expected = {2.1, -0.6, 2.6, 0.2, 3.8,
                                 2.3, -1.8, 4.9, 1.4, 1.6};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, MaxTenValuesFromParams) {
  // Similar to MaxTenValues, except that the inputs come from params rather
  // than constants.
  XlaBuilder builder(TestName());
  XlaOp v1, v2;
  std::unique_ptr<GlobalData> param0_data = CreateR1Parameter<float>(
      {41.0f, 2.0f, 3.0f, 84.0f}, /*parameter_number=*/0, /*name=*/"v1",
      /*builder=*/&builder, /*data_handle=*/&v1);
  std::unique_ptr<GlobalData> param1_data = CreateR1Parameter<float>(
      {21.0f, 22.0f, 23.0f, 24.0f}, /*parameter_number=*/1, /*name=*/"v2",
      /*builder=*/&builder, /*data_handle=*/&v2);

  Max(v1, v2);
  ComputeAndCompareR1<float>(&builder, {41.0f, 22.0f, 23.0f, 84.0f},
                             {param0_data.get(), param1_data.get()},
                             error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, Max15000ValuesFromParams) {
  // Similar to MaxTenValuesFromParams, except that the data size passed in and
  // out is large.
  XlaBuilder builder(TestName());

  // Number of floats in the data passed into and out of the computation.
  constexpr int datalen = 15 * 1000;

  // The inputs are initialized with a special pattern where in the first third
  // of the data v1[i] > v2[i] and elsewhere it's vice versa.
  std::vector<float> v1vec;
  std::vector<float> v2vec;
  std::vector<float> expected_vec;
  v1vec.reserve(datalen);
  v2vec.reserve(datalen);
  expected_vec.reserve(datalen);
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

  XlaOp v1, v2;
  std::unique_ptr<GlobalData> param0_data =
      CreateR1Parameter<float>(v1vec, /*parameter_number=*/0, /*name=*/"v1",
                               /*builder=*/&builder, /*data_handle=*/&v1);
  std::unique_ptr<GlobalData> param1_data =
      CreateR1Parameter<float>(v2vec, /*parameter_number=*/1, /*name=*/"v2",
                               /*builder=*/&builder, /*data_handle=*/&v2);

  Max(v1, v2);
  ComputeAndCompareR1<float>(&builder, expected_vec,
                             {param0_data.get(), param1_data.get()},
                             error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, MaxTenValuesWithScalar) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  auto y = ConstantR0<float>(&builder, 0);
  Max(x, y);

  std::vector<float> expected = {2.1, 0.0, 2.6, 0.0, 2.1,
                                 2.3, 0.0, 0.0, 0.0, 1.6};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, MinTenValues) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  auto y = ConstantR1<float>(
      &builder, {-0.4, -0.6, -3.0, 0.2, 3.8, -2.2, -1.8, 4.9, 1.4, 0.6});
  Min(x, y);

  std::vector<float> expected = {-0.4, -2.6, -3.0, -4.0, 2.1,
                                 -2.2, -5.0, -0.9, -2.4, 0.6};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, MinMaxTenValues) {
  XlaBuilder builder(TestName());
  auto zero = ConstantR0<float>(&builder, 0);
  auto one = ConstantR0<float>(&builder, 1);
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, 0.3, 3.1, 0.9, -5.0, 0.1, -2.4, 0.6});
  Min(Max(x, zero), one);

  std::vector<float> expected = {1.0, 0.0, 1.0, 0.3, 1.0,
                                 0.9, 0.0, 0.1, 0.0, 0.6};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, ClampTenValuesConstant) {
  XlaBuilder builder(TestName());
  auto zero = ConstantR0<float>(&builder, 0);
  auto one = ConstantR0<float>(&builder, 1);
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, 0.3, 3.1, 0.9, -5.0, 0.1, -2.4, 0.6});
  Clamp(zero, x, one);

  std::vector<float> expected = {1.0, 0.0, 1.0, 0.3, 1.0,
                                 0.9, 0.0, 0.1, 0.0, 0.6};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, ClampTwoValuesConstant) {
  XlaBuilder builder(TestName());
  auto zero = ConstantR1<float>(&builder, {0.0f, 0.0f});
  auto one = ConstantR1<float>(&builder, {1.0f, 1.0f});
  auto x = ConstantR1<float>(&builder, {2.1, -2.6});
  Clamp(zero, x, one);

  std::vector<float> expected = {1.0, 0.0};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, ClampTenValuesConstantNonzeroLower) {
  XlaBuilder builder(TestName());
  auto one = ConstantR0<float>(&builder, 1);
  auto two = ConstantR0<float>(&builder, 2);
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, 0.3, 3.1, 0.9, -5.0, 0.1, -2.4, 0.6});
  Clamp(one, x, two);

  std::vector<float> expected = {2.0, 1.0, 2.0, 1.0, 2.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, ClampFloatEdgeCases) {
  XlaBuilder builder(TestName());
  SetFastMathDisabled(true);
  auto low = ConstantR1<float>(&builder, {NAN, 1, 1});
  auto high = ConstantR1<float>(&builder, {3, NAN, 3});
  auto x = ConstantR1<float>(&builder, {2, 2, NAN});
  Clamp(low, x, high);

  std::vector<float> expected = {NAN, NAN, NAN};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, ClampValuesConstantS64) {
  XlaBuilder builder(TestName());
  auto zero = ConstantR0<int64_t>(&builder, 0);
  auto one = ConstantR0<int64_t>(&builder, 10);
  auto x = ConstantR1<int64_t>(&builder, {-3, 3, 9, 13});
  Clamp(zero, x, one);

  std::vector<int64_t> expected = {0, 3, 9, 10};
  ComputeAndCompareR1<int64_t>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, MapTenValues) {
  XlaComputation add_half;
  {
    // add_half(x) = x + 0.5
    XlaBuilder builder("add_half");
    auto x_value =
        Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "x_value");
    auto half = ConstantR0<float>(&builder, 0.5);
    Add(x_value, half);
    auto computation_status = builder.Build();
    ASSERT_IS_OK(computation_status.status());
    add_half = std::move(computation_status).value();
  }

  XlaComputation clamp;
  {
    // clamp(y) = clamp<0,5>(y)
    XlaBuilder builder("clamp");
    auto y_value =
        Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "y_value");
    auto zero = ConstantR0<float>(&builder, 0.0);
    Clamp(zero, y_value, ConstantR0<float>(&builder, 5));
    auto computation_status = builder.Build();
    ASSERT_IS_OK(computation_status.status());
    clamp = std::move(computation_status).value();
  }

  XlaComputation mult_relu_add;
  {
    // mult_relu_add(z) = clamp(add_half(2 * max(z, 0)))
    XlaBuilder builder("mult_relu_add");
    auto z_value =
        Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "z_value");
    auto zero = ConstantR0<float>(&builder, 0.0);
    auto two = ConstantR0<float>(&builder, 2.0);
    auto max = Max(z_value, zero);
    auto mult = Mul(two, max);
    auto inner = Map(&builder, {mult}, add_half, {});
    Map(&builder, {inner}, clamp, {});
    auto computation_status = builder.Build();
    ASSERT_IS_OK(computation_status.status());
    mult_relu_add = std::move(computation_status).value();
  }

  XlaBuilder builder("map10");
  {
    auto x = ConstantR1<float>(
        &builder, {2.1, -21.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
    Map(&builder, {x}, mult_relu_add, {0});
  }

  std::vector<float> expected = {4.7, 0.5, 5.0, 0.5, 4.7,
                                 5.0, 0.5, 0.5, 0.5, 3.7};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, RemainderTenValuesS32) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<int32_t>(&builder, {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4});
  auto y = ConstantR0<int32_t>(&builder, 3);
  Rem(x, y);

  std::vector<int32_t> expected = {-2, -1, 0, -2, -1, 0, 1, 2, 0, 1};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, VectorPredicateEqual) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<bool>(&builder, {false, true});
  auto y = ConstantR1<bool>(&builder, {true, false});
  Eq(x, y);

  std::array<bool, 2> expected = {{false, false}};
  ComputeAndCompareR1<bool>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, VectorPredicateNotEqual) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<bool>(&builder, {false, true});
  auto y = ConstantR1<bool>(&builder, {true, false});
  Ne(x, y);

  std::array<bool, 2> expected = {{true, true}};
  ComputeAndCompareR1<bool>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, CbrtSevenValues) {
  XlaBuilder builder(TestName());
  std::vector<float> expected = {16.0, 1888.0, -102.0, 0.16, 0.2, 0., 1.23};
  std::vector<float> cube = {4096.0, 6729859072., -1061208, .004096,
                             0.008,  0.,          1.860867};
  auto x = ConstantR1<float>(&builder, cube);
  Cbrt(x);
  ComputeAndCompareR1<float>(&builder, expected, {},
                             ErrorSpec(/*aabs=*/1e-7, /*arel=*/3e-7));
}

}  // namespace
}  // namespace xla
