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
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class L2NormOpModel : public SingleOpModel {
 public:
  L2NormOpModel(const std::initializer_list<int> input_shape,
                const TensorType tensor_type,
                const ActivationFunctionType activation_type) {
    TensorData data = TensorData{tensor_type};
    if (tensor_type != TensorType_FLOAT32) {
      data.min = -2.0;
      data.max = 2.0;
      data.scale = 2.0;
      data.zero_point = 128;
    }
    input_ = AddInput(data);
    if (tensor_type != TensorType_FLOAT32) {
      data.min = -1.0;
      data.max = 127.0 / 128.0;
    }
    output_ = AddOutput(data);
    SetBuiltinOp(BuiltinOperator_L2_NORMALIZATION, BuiltinOptions_L2NormOptions,
                 CreateL2NormOptions(builder_, activation_type).Union());
    BuildInterpreter({input_shape});
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
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

TEST(L2NormOpTest, SimpleFloatTest) {
  L2NormOpModel m({1, 1, 1, 6}, TensorType_FLOAT32,
                  ActivationFunctionType_NONE);
  m.SetInput({-1.1, 0.6, 0.7, 1.2, -0.7, 0.1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({-0.55, 0.3, 0.35, 0.6, -0.35, 0.05}));
}

TEST(L2NormOpTest, ZerosVectorFloatTest) {
  L2NormOpModel m({1, 1, 1, 6}, TensorType_FLOAT32,
                  ActivationFunctionType_NONE);
  m.SetInput({0, 0, 0, 0, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({0, 0, 0, 0, 0, 0})));
}

TEST(L2NormOpTest, SimpleFloatWithRankLessThanFourTest) {
  L2NormOpModel m({1, 6}, TensorType_FLOAT32, ActivationFunctionType_NONE);
  m.SetInput({-1.1, 0.6, 0.7, 1.2, -0.7, 0.1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({-0.55, 0.3, 0.35, 0.6, -0.35, 0.05}));
}

TEST(L2NormOpTest, MultipleBatchFloatTest) {
  L2NormOpModel m({3, 1, 1, 6}, TensorType_FLOAT32,
                  ActivationFunctionType_NONE);
  m.SetInput({
      -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 1
      -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 2
      -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 3
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({
                  -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 1
                  -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 2
                  -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 3
              }));
}

TEST(L2NormOpTest, ZerosVectorUint8Test) {
  L2NormOpModel m({1, 1, 1, 6}, TensorType_UINT8, ActivationFunctionType_NONE);

  m.QuantizeAndPopulate<uint8_t>(m.input(), {0, 0, 0, 0, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({128, 128, 128, 128, 128, 128}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({0, 0, 0, 0, 0, 0}, 0.1)));
}

TEST(L2NormOpTest, SimpleUint8Test) {
  L2NormOpModel m({1, 1, 1, 6}, TensorType_UINT8, ActivationFunctionType_NONE);

  m.QuantizeAndPopulate<uint8_t>(m.input(), {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({58, 166, 173, 205, 83, 134}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({-0.55, 0.3, 0.35, 0.6, -0.35, 0.05}, 0.1)));
}

TEST(L2NormOpTest, SimpleInt8Test) {
  L2NormOpModel m({1, 1, 1, 6}, TensorType_INT8, ActivationFunctionType_NONE);

  m.QuantizeAndPopulate<int8_t>(m.input(), {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({-70, 38, 45, 77, -45, 6}));

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({-0.55, 0.3, 0.35, 0.6, -0.35, 0.05}, 0.1)));
}

TEST(L2NormOpTest, ZerosVectorInt8Test) {
  L2NormOpModel m({1, 1, 1, 6}, TensorType_INT8, ActivationFunctionType_NONE);

  m.QuantizeAndPopulate<int8_t>(m.input(), {0, 0, 0, 0, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({0, 0, 0, 0, 0, 0}));

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({0, 0, 0, 0, 0, 0}, 0.1)));
}

TEST(L2NormOpTest, MultipleBatchUint8Test) {
  L2NormOpModel m({3, 1, 1, 6}, TensorType_UINT8, ActivationFunctionType_NONE);

  m.QuantizeAndPopulate<uint8_t>(m.input(),
                                 {
                                     -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 1
                                     -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 2
                                     -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 3
                                 });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({
                  58, 166, 173, 205, 83, 134,  // batch 1
                  58, 166, 173, 205, 83, 134,  // batch 2
                  58, 166, 173, 205, 83, 134,  // batch 3
              }));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 1
                      -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 2
                      -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 3
                  },
                  0.1)));
}

TEST(L2NormOpTest, MultipleBatchInt8Test) {
  L2NormOpModel m({3, 1, 1, 6}, TensorType_INT8, ActivationFunctionType_NONE);

  m.QuantizeAndPopulate<int8_t>(m.input(),
                                {
                                    -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 1
                                    -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 2
                                    -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 3
                                });
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({
                                         -70, 38, 45, 77, -45, 6,  // batch 1
                                         -70, 38, 45, 77, -45, 6,  // batch 2
                                         -70, 38, 45, 77, -45, 6,  // batch 3
                                     }));
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 1
                      -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 2
                      -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 3
                  },
                  0.1)));
}

}  // namespace
}  // namespace tflite
