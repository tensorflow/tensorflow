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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
using ::testing::ElementsAreArray;

class SquaredDifferenceOpModel : public SingleOpModelWithHexagon {
 public:
  SquaredDifferenceOpModel(const TensorData& input1, const TensorData& input2,
                           const TensorData& output) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_SQUARED_DIFFERENCE,
                 BuiltinOptions_SquaredDifferenceOptions,
                 CreateSquaredDifferenceOptions(builder_).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<int8_t>(ExtractVector<int8_t>(output_), GetScale(output_),
                              GetZeroPoint(output_));
  }

 protected:
  int input1_;
  int input2_;
  int output_;
};

float GetTolerance(int min, int max) {
  float kQuantizedStep = (max - min) / 255.0;
  return kQuantizedStep;
}

TEST(QuantizedSquaredDifferenceOpTest, Quantized_SameShape) {
  float kQuantizedTolerance = GetTolerance(0, 1);
  SquaredDifferenceOpModel m({TensorType_INT8, {1, 2, 2, 1}, -1.2, 0.8},
                             {TensorType_INT8, {1, 2, 2, 1}, -1.5, 0.5},
                             {TensorType_INT8, {}, 0.0, 0.5});
  m.QuantizeAndPopulate<int8_t>(m.input1(), {-0.2, 0.2, -1.2, 0.8});
  m.QuantizeAndPopulate<int8_t>(m.input2(), {0.5, 0.2, -1.5, 0.5});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({0.49, 0.0, 0.09, 0.09},
                                              kQuantizedTolerance)));
}

TEST(QuantizedSquaredDifferenceOpTest, Quantized_VariousInputShapes) {
  // NOTE: the min/max are 0 and 9. We use larger threshold for accuracy
  // issue in Hexagon.
  float kQuantizedTolerance = GetTolerance(0, 10);
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    SquaredDifferenceOpModel m({TensorType_INT8, test_shapes[i], -2.0, 1.7},
                               {TensorType_INT8, test_shapes[i], -1.0, 1.0},
                               {TensorType_INT8, {}, 0.0, 9.0});
    m.QuantizeAndPopulate<int8_t>(m.input1(), {-2.0, 0.2, 0.3, 0.8, 1.1, -2.0});
    m.QuantizeAndPopulate<int8_t>(m.input2(), {1.0, 0.2, 0.6, 0.4, -1.0, -0.0});
    m.ApplyDelegateAndInvoke();
    EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
                ElementsAreArray(ArrayFloatNear(
                    {9.0, 0.0, 0.09, 0.16, 4.41, 4.0}, kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedSquaredDifferenceOpTest, Quantized_WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  float kQuantizedTolerance = GetTolerance(0, 1);
  for (int i = 0; i < test_shapes.size(); ++i) {
    SquaredDifferenceOpModel m({TensorType_INT8, test_shapes[i], -0.2, 1.1},
                               {TensorType_INT8, {}, 0.0, 0.1},
                               {TensorType_INT8, {}, 0.0, 1.0});
    m.QuantizeAndPopulate<int8_t>(m.input1(), {-0.2, 0.2, 0.5, 0.8, 0.11, 1.1});
    m.QuantizeAndPopulate<int8_t>(m.input2(), {0.1});
    m.ApplyDelegateAndInvoke();
    EXPECT_THAT(
        m.GetDequantizedOutput<int8_t>(),
        ElementsAreArray(ArrayFloatNear({0.09, 0.01, 0.16, 0.49, 0.0001, 1.0},
                                        kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

}  // namespace tflite
