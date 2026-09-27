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
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/custom_ops_register.h"
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

class SignModel : public SingleOpModel {
 public:
  SignModel(const TensorData& x, const TensorData& output) {
    x_ = AddInput(x);
    output_ = AddOutput(output);
    SetCustomOp("Sign", {}, ops::custom::Register_SIGN);
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
class SignCustomTest : public testing::Test {
 public:
  using FloatType = Float;
};

using TestTypes = testing::Types<float, double>;

TYPED_TEST_SUITE(SignCustomTest, TestTypes);

TYPED_TEST(SignCustomTest, TestScalar) {
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

TYPED_TEST(SignCustomTest, TestNaN) {
  using Float = typename TestFixture::FloatType;
  TensorData x = {GetTTEnum<Float>(), {}};
  TensorData output = {GetTTEnum<Float>(), {}};
  SignModel m(x, output);
  std::vector<Float> got =
      m.RunAndGetOutput<Float>({std::numeric_limits<Float>::quiet_NaN()});
  ASSERT_EQ(got.size(), 1);
  EXPECT_TRUE(std::isnan(got[0]));
}

TYPED_TEST(SignCustomTest, TestBatch) {
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

}  // namespace
}  // namespace tflite
