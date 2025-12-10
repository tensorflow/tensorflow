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
#include <stdint.h>

#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

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

  int input() const { return input_; }
  int output() const { return output_; }

 protected:
  int input_;
  int output_;
};

TEST(NegOpModel, NegFloat32) {
  NegOpModel m({TensorType_FLOAT32, {2, 3}}, {TensorType_FLOAT32, {2, 3}});
  m.SetInput<float>({-2.0f, -1.0f, 0.f, 1.0f, 2.0f, 3.0f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<float>(),
      Pointwise(FloatingPointEq(), {2.0f, 1.0f, 0.f, -1.0f, -2.0f, -3.0f}));
}

TEST(NegOpModel, NegFloat16) {
  NegOpModel m({TensorType_FLOAT16, {6}}, {TensorType_FLOAT16, {6}});
  m.SetInput<Eigen::half>({Eigen::half(-2.0f), Eigen::half(-1.0f),
                           Eigen::half(0.f), Eigen::half(1.0f),
                           Eigen::half(2.0f), Eigen::half(3.0f)});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<Eigen::half>(),
              ElementsAreArray({Eigen::half(2.0f), Eigen::half(1.0f),
                                Eigen::half(0.f), Eigen::half(-1.0f),
                                Eigen::half(-2.0f), Eigen::half(-3.0f)}));
}

TEST(NegOpModel, NegBfloat16) {
  NegOpModel m({TensorType_BFLOAT16, {6}}, {TensorType_BFLOAT16, {6}});
  m.SetInput<Eigen::bfloat16>({Eigen::bfloat16(-2.0f), Eigen::bfloat16(-1.0f),
                               Eigen::bfloat16(0.f), Eigen::bfloat16(1.0f),
                               Eigen::bfloat16(2.0f), Eigen::bfloat16(3.0f)});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<Eigen::bfloat16>(),
      ElementsAreArray({Eigen::bfloat16(2.0f), Eigen::bfloat16(1.0f),
                        Eigen::bfloat16(0.f), Eigen::bfloat16(-1.0f),
                        Eigen::bfloat16(-2.0f), Eigen::bfloat16(-3.0f)}));
}

TEST(NegOpModel, NegInt32) {
  NegOpModel m({TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 3}});
  m.SetInput<int32_t>({-2, -1, 0, 1, 2, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({2, 1, 0, -1, -2, -3}));
}

TEST(NegOpModel, NegInt64) {
  NegOpModel m({TensorType_INT64, {2, 3}}, {TensorType_INT64, {2, 3}});
  m.SetInput<int64_t>({-2, -1, 0, 1, 2, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int64_t>(), ElementsAreArray({2, 1, 0, -1, -2, -3}));
}

class NegOpQuantizedModel : public NegOpModel {
 public:
  NegOpQuantizedModel(const TensorData& input, const TensorData& output)
      : NegOpModel(SymmetricInt16Scaling(std::move(input)),
                   SymmetricInt16Scaling(std::move(output))) {}

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_dtype>(ExtractVector<integer_dtype>(output_),
                                     GetScale(output_), GetZeroPoint(output_));
  }

 private:
  TensorData SymmetricInt16Scaling(TensorData tensor) {
    if (tensor.type == TensorType_INT16) {
      CHECK_EQ(std::abs(tensor.min), tensor.max);
      tensor.scale = tensor.max / std::numeric_limits<int16_t>::max();
      tensor.zero_point = 0;
      tensor.min = 0;
      tensor.max = 0;
    }
    return tensor;
  }
};

template <typename T>
float GetTolerance(float min, float max) {
  const float kQuantizedStep =
      2.0 * (max - min) /
      (std::numeric_limits<T>::max() - std::numeric_limits<T>::min());
  return kQuantizedStep;
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedTests() {
  const float kQuantizedTolerance = GetTolerance<integer_dtype>(-128.0, 128.0);
  const std::vector<float> input = {-128.0f, -9, 0, 8, 127};
  const std::vector<float> result = {128.0f, 9, 0, -8, -127};

  NegOpQuantizedModel m({tensor_type, {5}, -128.0, 128.0},
                        {tensor_type, {5}, -128.0, 128.0});

  m.QuantizeAndPopulate<integer_dtype>(m.input(), input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear(result, kQuantizedTolerance)));
}

TEST(NegOpQuantizedModel, NegInt8) {
  QuantizedTests<TensorType_INT8, int8_t>();
}

TEST(NegOpQuantizedModel, NegInt16) {
  QuantizedTests<TensorType_INT16, int16_t>();
}

}  // namespace
}  // namespace tflite
