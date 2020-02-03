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
#include <gtest/gtest.h>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class NegOpModel : public SingleOpModel {
 public:
  NegOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_NEG, BuiltinOptions_NegOptions,
                 CreateNegOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  template <class T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }

  template <class T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  int input_;
  int output_;
};

TEST(NegOpModel, NegFloat) {
  NegOpModel m({TensorType_FLOAT32, {2, 3}}, {TensorType_FLOAT32, {2, 3}});
  m.SetInput<float>({-2.0f, -1.0f, 0.f, 1.0f, 2.0f, 3.0f});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({2.0f, 1.0f, 0.f, -1.0f, -2.0f, -3.0f}));
}

TEST(NegOpModel, NegInt32) {
  NegOpModel m({TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 3}});
  m.SetInput<int32_t>({-2, -1, 0, 1, 2, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({2, 1, 0, -1, -2, -3}));
}

TEST(NegOpModel, NegInt64) {
  NegOpModel m({TensorType_INT64, {2, 3}}, {TensorType_INT64, {2, 3}});
  m.SetInput<int64_t>({-2, -1, 0, 1, 2, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int64_t>(), ElementsAreArray({2, 1, 0, -1, -2, -3}));
}

class QuantizedNegOpModel : public NegOpModel {
 public:
  using NegOpModel::NegOpModel;

  int input() { return input_; }

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_dtype>(ExtractVector<integer_dtype>(output_),
                                     GetScale(output_), GetZeroPoint(output_));
  }
};

constexpr float GetToleranceInt8(float min, float max) {
  float kQuantizedStep = (max - min) / 255.0;
  return kQuantizedStep;
}

constexpr float GetToleranceInt16(float min, float max) {
  float kQuantizedStep = (max - min) / std::numeric_limits<int16_t>::max();
  return kQuantizedStep;
}

// input_quantization_buffer: buffer used for the quantization data
TEST(QuantizedNegOpModel, NegQuantizedInt8) {
  constexpr auto min = -6.0f;
  constexpr auto max = 6.0f;
  constexpr auto quantized_tolerance = GetToleranceInt8(min, max);

  const auto expected_output =
      std::vector<float>{3.5f, 2.0f, 1.0f, 0.0f, -1.0f, -2.0f, -3.0f, -3.5f};

  QuantizedNegOpModel model{{TensorType_INT8, {1, 2, 2, 2, 1}, min, max},
                            {TensorType_INT8, {1, 2, 2, 2, 1}, min, max}};
  model.QuantizeAndPopulate<int8_t>(
      model.input(), {-3.5f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 3.5f});
  model.Invoke();

  EXPECT_THAT(
      model.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear(expected_output, quantized_tolerance)));
}

}  // namespace
}  // namespace tflite
