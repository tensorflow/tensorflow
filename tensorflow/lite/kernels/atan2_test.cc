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

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

template <typename T>
tflite::TensorType GetTTEnum();
template <>
tflite::TensorType GetTTEnum<float>() {
  return tflite::TensorType_FLOAT32;
}

template <>
tflite::TensorType GetTTEnum<double>() {
  return tflite::TensorType_FLOAT64;
}

template <>
tflite::TensorType GetTTEnum<Eigen::half>() {
  return tflite::TensorType_FLOAT16;
}

template <>
tflite::TensorType GetTTEnum<Eigen::bfloat16>() {
  return tflite::TensorType_BFLOAT16;
}

class Atan2Model : public tflite::SingleOpModel {
 public:
  Atan2Model(tflite::TensorData y, tflite::TensorData x,
             tflite::TensorData output) {
    y_ = AddInput(y);
    x_ = AddInput(x);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ATAN2, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(y_), GetShape(x_)});
  }

  template <typename T>
  std::vector<T> GetOutput(const std::vector<T>& y, const std::vector<T>& x) {
    PopulateTensor<T>(y_, y);
    PopulateTensor<T>(x_, x);
    Invoke();
    return ExtractVector<T>(output_);
  }

 private:
  int y_;
  int x_;
  int output_;
};

template <typename Float>
class Atan2Test : public ::testing::Test {
 public:
  using FloatType = Float;
};

using TestTypes = ::testing::Types<float, double, Eigen::half, Eigen::bfloat16>;

TYPED_TEST_SUITE(Atan2Test, TestTypes);

TYPED_TEST(Atan2Test, TestScalar) {
  using Float = typename TestFixture::FloatType;
  tflite::TensorData y = {GetTTEnum<Float>(), {}};
  tflite::TensorData x = {GetTTEnum<Float>(), {}};
  tflite::TensorData output = {GetTTEnum<Float>(), {}};
  Atan2Model m(y, x, output);

  auto got = m.GetOutput<Float>({Float(0.0)}, {Float(0.0)});
  ASSERT_EQ(got.size(), 1);
  EXPECT_FLOAT_EQ(got[0], 0.0);
  ASSERT_FLOAT_EQ(m.GetOutput<Float>({Float(1.0)}, {Float(0.0)})[0],
                  Float(M_PI / 2));
  ASSERT_FLOAT_EQ(m.GetOutput<Float>({Float(0.0)}, {Float(1.0)})[0],
                  Float(0.0));
  ASSERT_FLOAT_EQ(m.GetOutput<Float>({Float(-1.0)}, {Float(0.0)})[0],
                  Float(-M_PI / 2));
}

TYPED_TEST(Atan2Test, TestBatch) {
  using Float = typename TestFixture::FloatType;
  tflite::TensorData y = {GetTTEnum<Float>(), {4, 2, 1}};
  tflite::TensorData x = {GetTTEnum<Float>(), {4, 2, 1}};
  tflite::TensorData output = {GetTTEnum<Float>(), {4, 2, 1}};
  Atan2Model m(y, x, output);
  std::vector<Float> y_data = {Float(0.1), Float(0.2), Float(0.3), Float(0.4),
                               Float(0.5), Float(0.6), Float(0.7), Float(0.8)};
  std::vector<Float> x_data = {Float(0.8), Float(0.7), Float(0.6), Float(0.5),
                               Float(0.4), Float(0.3), Float(0.2), Float(0.1)};
  auto got = m.GetOutput<Float>(y_data, x_data);
  ASSERT_EQ(got.size(), 8);
  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(got[i], Float(std::atan2(y_data[i], x_data[i])));
  }
}

}  // namespace
}  // namespace tflite
