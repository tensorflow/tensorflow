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

class BaseAbsOpModel : public SingleOpModel {
 public:
  BaseAbsOpModel(const TensorData& input) {
    input_ = AddInput(input);
    output_ = AddOutput(input);
    SetBuiltinOp(BuiltinOperator_ABS, BuiltinOptions_AbsOptions,
                 CreateAbsOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }
  int output() { return output_; }

 protected:
  int input_;
  int output_;
};

template <typename T>
class AbsOpModel : public BaseAbsOpModel {
 public:
  using BaseAbsOpModel::BaseAbsOpModel;

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
};

class QuantizedAbsOpModel : public BaseAbsOpModel {
 public:
  using BaseAbsOpModel::BaseAbsOpModel;

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_dtype>(ExtractVector<integer_dtype>(output_),
                                     GetScale(output_), GetZeroPoint(output_));
  }
};

template <TensorType tensor_type, typename integer_dtype>
void TestQuantizeAbsOp() {
  const int kQmin = -1.0;
  const int kQmax = 1.0;
  const float kQuantizedTolerance = GetQuantizeTolerance(kQmin, kQmax);
  QuantizedAbsOpModel m({tensor_type, {1, 2, 2, 1}, kQmin, kQmax});
  m.QuantizeAndPopulate<integer_dtype>(m.input(), {-0.8, 0.2, -0.9, 0.7});
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(
                  ArrayFloatNear({0.8, 0.2, 0.9, 0.7}, kQuantizedTolerance)));
}

TEST(AbsOpTest, FloatTest) {
  AbsOpModel<float> m({TensorType_FLOAT32, {1, 2, 4, 1}});
  m.PopulateTensor<float>(m.input(), {
                                         0.f,
                                         -6.2f,
                                         2.f,
                                         4.f,
                                         3.f,
                                         -2.f,
                                         10.f,
                                         1.f,
                                     });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0.f,
                                 6.2f,
                                 2.f,
                                 4.f,
                                 3.f,
                                 2.f,
                                 10.f,
                                 1.f,
                             }));
}

TEST(AbsOpTest, Int32Test) {
  AbsOpModel<int32_t> m({TensorType_INT32, {1, 2, 4, 1}});
  m.PopulateTensor<int32_t>(m.input(), {
                                           100,
                                           -7,
                                           -1024,
                                           23,
                                           -38,
                                           902,
                                           -849,
                                           3924,
                                       });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 100,
                                 7,
                                 1024,
                                 23,
                                 38,
                                 902,
                                 849,
                                 3924,
                             }));
}

TEST(AbsOpTest, QuantizedUInt8Test) {
  TestQuantizeAbsOp<TensorType_UINT8, uint8_t>();
}

TEST(AbsOpTest, QuantizedInt8Test) {
  TestQuantizeAbsOp<TensorType_INT8, int8_t>();
}

}  // namespace
}  // namespace tflite
