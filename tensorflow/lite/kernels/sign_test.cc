// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;

template <typename T>
TensorType GetTTEnum();

template <>
TensorType GetTTEnum<float>() {
  return TensorType_FLOAT32;
}

template <>
TensorType GetTTEnum<double>() {
  return TensorType_FLOAT64;
}

template <>
TensorType GetTTEnum<int32_t>() {
  return TensorType_INT32;
}

class SignModel : public SingleOpModel {
 public:
  SignModel(const TensorData& x, const TensorData& output) {
    x_ = AddInput(x);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_SIGN, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(x_)});
  }

  template <typename T>
  std::vector<T> RunAndGetOutput(const std::vector<T>& x) {
    PopulateTensor<T>(x_, x);
    Invoke();
    return ExtractVector<T>(output_);
  }

 private:
  int x_;
  int output_;
};

template <typename Float>
class SignTestFloat : public testing::Test {
 public:
  using FloatType = Float;
};

using TestTypes = testing::Types<float, double>;

TYPED_TEST_SUITE(SignTestFloat, TestTypes);

TYPED_TEST(SignTestFloat, TestScalarFloat) {
  using Float = typename TestFixture::FloatType;
  TensorData x = {GetTTEnum<Float>(), {}};
  TensorData output = {GetTTEnum<Float>(), {}};
  SignModel m(x, output);
  std::vector<Float> got = m.RunAndGetOutput<Float>({Float(0.0)});
  ASSERT_EQ(got.size(), 1);
  EXPECT_FLOAT_EQ(got[0], Float(0.0));

  std::vector<Float> got_pos = m.RunAndGetOutput<Float>({Float(5.0)});
  ASSERT_EQ(got_pos.size(), 1);
  EXPECT_FLOAT_EQ(got_pos[0], Float(1.0));

  std::vector<Float> got_neg = m.RunAndGetOutput<Float>({Float(-3.0)});
  ASSERT_EQ(got_neg.size(), 1);
  EXPECT_FLOAT_EQ(got_neg[0], Float(-1.0));
}

TYPED_TEST(SignTestFloat, TestNaNFloat) {
  using Float = typename TestFixture::FloatType;
  TensorData x = {GetTTEnum<Float>(), {}};
  TensorData output = {GetTTEnum<Float>(), {}};
  SignModel m(x, output);
  std::vector<Float> got =
      m.RunAndGetOutput<Float>({std::numeric_limits<Float>::quiet_NaN()});
  ASSERT_EQ(got.size(), 1);
  EXPECT_TRUE(std::isnan(got[0]));
}

TYPED_TEST(SignTestFloat, TestBatchFloat) {
  using Float = typename TestFixture::FloatType;
  TensorData x = {GetTTEnum<Float>(), {4, 2, 1}};
  TensorData output = {GetTTEnum<Float>(), {4, 2, 1}};
  SignModel m(x, output);

  std::vector<Float> x_data = {Float(0.8), Float(-0.7), Float(0.6), Float(-0.5),
                               Float(0.4), Float(-0.3), Float(0.2), Float(0.0)};

  std::vector<Float> got = m.RunAndGetOutput<Float>(x_data);

  EXPECT_THAT(got,
              ElementsAre(Float(1.0), Float(-1.0), Float(1.0), Float(-1.0),
                          Float(1.0), Float(-1.0), Float(1.0), Float(0.0)));
}

template <typename Int>
class SignTestInt : public testing::Test {
 public:
  using IntType = Int;
};
using TestTypesInt = testing::Types<int32_t>;

TYPED_TEST_SUITE(SignTestInt, TestTypesInt);

TYPED_TEST(SignTestInt, TestScalarInt) {
  using Int = typename TestFixture::IntType;
  TensorData x = {GetTTEnum<Int>(), {}};
  TensorData output = {GetTTEnum<Int>(), {}};
  SignModel m(x, output);
  std::vector<Int> got = m.RunAndGetOutput<Int>({Int(0)});
  ASSERT_EQ(got.size(), 1);
  EXPECT_EQ(got[0], Int(0));

  std::vector<Int> got_pos = m.RunAndGetOutput<Int>({Int(5)});
  ASSERT_EQ(got_pos.size(), 1);
  EXPECT_EQ(got_pos[0], Int(1));

  std::vector<Int> got_neg = m.RunAndGetOutput<Int>({Int(-3)});
  ASSERT_EQ(got_neg.size(), 1);
  EXPECT_EQ(got_neg[0], Int(-1));
}

TYPED_TEST(SignTestInt, TestBatchInt) {
  using Int = typename TestFixture::IntType;
  TensorData x = {GetTTEnum<Int>(), {4, 2, 1}};
  TensorData output = {GetTTEnum<Int>(), {4, 2, 1}};
  SignModel m(x, output);

  std::vector<Int> got = m.RunAndGetOutput<Int>(
      {Int(0), Int(-7), Int(6), Int(-5), Int(4), Int(-3), Int(2), Int(1)});

  EXPECT_THAT(got, ElementsAre(Int(0), Int(-1), Int(1), Int(-1), Int(1),
                               Int(-1), Int(1), Int(1)));
}

}  // namespace
}  // namespace tflite
