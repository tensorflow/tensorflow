/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <math.h>

#include <initializer_list>
#include <limits>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseExpOpModel : public SingleOpModel {
 public:
  BaseExpOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_EXP, BuiltinOptions_ExpOptions,
                 CreateExpOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int output_;
};

class FloatExpOpModel : public BaseExpOpModel {
 public:
  using BaseExpOpModel::BaseExpOpModel;

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class QuantizedExpOpModel : public BaseExpOpModel {
 public:
  using BaseExpOpModel::BaseExpOpModel;

  template <class T>
  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_dtype>(ExtractVector<integer_dtype>(output_),
                                     GetScale(output_), GetZeroPoint(output_));
  }
};

// A LUT of 256 values is used in the int8 case and 512 values (+1 for
// interpolation) for the int16 case
template <typename T>
inline float GetTolerance(float min, float max) {
  float kQuantizedTolerance = (max - min) / (std::numeric_limits<T>::max() -
                                             std::numeric_limits<T>::min());
  if (std::is_same<T, int8_t>::value) {
    kQuantizedTolerance += (max - min) / 256.0f;
  } else if (std::is_same<T, int16_t>::value) {
    kQuantizedTolerance += (max - min) / 512.0f;
  }

  return kQuantizedTolerance;
}

TEST(ExpOpTest, ExpFloat) {
  std::initializer_list<float> data = {0.0f,    1.0f,  -1.0f, 100.0f,
                                       -100.0f, 0.01f, -0.01f};
  FloatExpOpModel m({TensorType_FLOAT32, {1, 1, 7}}, {TensorType_FLOAT32, {}});
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 7}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {std::exp(0.0f), std::exp(1.0f), std::exp(-1.0f), std::exp(100.0f),
           std::exp(-100.0f), std::exp(0.01f), std::exp(-0.01f)})));
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedExpSymmetricTest() {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<integer_dtype>::max() /
      static_cast<float>(std::numeric_limits<integer_dtype>::max() + 1);
  const float kQuantizedTolerance = GetTolerance<integer_dtype>(-3.1, 3.1);
  QuantizedExpOpModel m({tensor_type, {1, 2, 2, 2}, 1.3f * kMin, 1.3f * kMax},
                        {tensor_type, {}, 3.01f * kMin, 3.01f * kMax});
  m.SetInput<integer_dtype>({-1.3, -1.0, -0.3, 0, 0.1, 0.5, 1.0, 1.1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 2, 2}));
  EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear(
                  {0.2725, 0.3679, 0.7408, 1.0, 1.1052, 1.6487, 2.7183, 3.0042},
                  kQuantizedTolerance)));
}

TEST(ExpOpTest, ExpSymmetricInt8) {
  QuantizedExpSymmetricTest<TensorType_INT8, int8_t>();
}

TEST(ExpOpTest, ExpSymmetricInt16) {
  QuantizedExpSymmetricTest<TensorType_INT16, int16_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedExpAsymmetricTest() {
  const float kQuantizedTolerance = GetTolerance<integer_dtype>(-1.3, 3.01);
  QuantizedExpOpModel m({tensor_type, {1, 2, 2, 2}, -1.3, 1.1},
                        {tensor_type, {}, 0.0, 3.01});
  m.SetInput<integer_dtype>({-1.3, -1.0, -0.3, 0, 0.1, 0.5, 1.0, 1.1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 2, 2}));
  EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear(
                  {0.2725, 0.3679, 0.7408, 1.0, 1.1052, 1.6487, 2.7183, 3.0042},
                  kQuantizedTolerance)));
}

TEST(ExpOpTest, ExpAsymmetricInt8) {
  QuantizedExpAsymmetricTest<TensorType_INT8, int8_t>();
}

}  // namespace
}  // namespace tflite
