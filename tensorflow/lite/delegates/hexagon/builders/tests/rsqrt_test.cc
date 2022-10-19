/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"

namespace tflite {
using testing::ElementsAreArray;

class RsqrtOpModel : public SingleOpModelWithHexagon {
 public:
  RsqrtOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_RSQRT, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

  int input() const { return input_; }

 private:
  int input_;
  int output_;
};

TEST(RsqrtOpTest, Int8) {
  std::vector<float> data = {15., 46., 78., 142., 1., 17., 49., 113.};
  std::vector<float> rsqrt_data(data.size());
  for (int i = 0; i < rsqrt_data.size(); i++) {
    rsqrt_data[i] = 1.f / std::sqrt(data[i]);
  }
  const float kInputScale = 142.0 / 255.0;
  const float kOutputScale = 1.0 / 255.0;
  int32_t zero_point = -128;
  RsqrtOpModel m({TensorType_INT8,
                  {1, 8},
                  0,
                  142.0,
                  kInputScale,
                  zero_point,
                  true,
                  {kInputScale},
                  {zero_point}},
                 {TensorType_INT8,
                  {1, 8},
                  0,
                  1.0,
                  kOutputScale,
                  zero_point,
                  true,
                  {kOutputScale},
                  {zero_point}});
  m.QuantizeAndPopulate<int8_t>(m.input(), data);
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(rsqrt_data, kInputScale)));
}

TEST(RsqrtOpTest, Int8_2D) {
  std::vector<float> data = {15., 46., 78., 142., 1., 17., 49., 113.};
  std::vector<float> rsqrt_data(data.size());
  for (int i = 0; i < rsqrt_data.size(); i++) {
    rsqrt_data[i] = 1.f / std::sqrt(data[i]);
  }
  const float kInputScale = 142.0 / 255.0;
  const float kOutputScale = 1.0 / 255.0;
  int32_t zero_point = -128;
  RsqrtOpModel m({TensorType_INT8,
                  {2, 4},
                  0,
                  142.0,
                  kInputScale,
                  zero_point,
                  true,
                  {kInputScale},
                  {zero_point}},
                 {TensorType_INT8,
                  {2, 4},
                  0,
                  1.0,
                  kOutputScale,
                  zero_point,
                  true,
                  {kOutputScale},
                  {zero_point}});
  m.QuantizeAndPopulate<int8_t>(m.input(), data);
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(rsqrt_data, kInputScale)));
}

}  // namespace tflite
