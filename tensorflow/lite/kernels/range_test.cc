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

using ::testing::ElementsAre;

template <typename T>
class RangeOpModel : public SingleOpModel {
 public:
  explicit RangeOpModel(const TensorType& dtype) {
    start_ = AddInput(dtype);
    limit_ = AddInput(dtype);
    delta_ = AddInput(dtype);
    output_ = AddOutput(dtype);
    SetBuiltinOp(BuiltinOperator_RANGE, BuiltinOptions_RangeOptions,
                 CreateRangeOptions(builder_).Union());
    BuildInterpreter({GetShape(start_), GetShape(limit_), GetShape(delta_)});
  }

  int start() { return start_; }
  int limit() { return limit_; }
  int delta() { return delta_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int start_;
  int limit_;
  int delta_;
  int output_;
};

TEST(RangeOpModel, Simple) {
  RangeOpModel<int32_t> model(TensorType_INT32);
  model.PopulateTensor<int32_t>(model.start(), {0});
  model.PopulateTensor<int32_t>(model.limit(), {4});
  model.PopulateTensor<int32_t>(model.delta(), {1});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(RangeOpModel, DeltaGreaterThanOne) {
  RangeOpModel<int32_t> model(TensorType_INT32);
  model.PopulateTensor<int32_t>(model.start(), {2});
  model.PopulateTensor<int32_t>(model.limit(), {9});
  model.PopulateTensor<int32_t>(model.delta(), {2});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, NegativeDelta) {
  RangeOpModel<int32_t> model(TensorType_INT32);
  model.PopulateTensor<int32_t>(model.start(), {10});
  model.PopulateTensor<int32_t>(model.limit(), {3});
  model.PopulateTensor<int32_t>(model.delta(), {-3});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3));
  EXPECT_THAT(model.GetOutput(), ElementsAre(10, 7, 4));
}

TEST(RangeOpModel, FloatSimple) {
  RangeOpModel<float> model(TensorType_FLOAT32);
  model.PopulateTensor<float>(model.start(), {0});
  model.PopulateTensor<float>(model.limit(), {4});
  model.PopulateTensor<float>(model.delta(), {1});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(RangeOpModel, FloatDeltaGreaterThanOne) {
  RangeOpModel<float> model(TensorType_FLOAT32);
  model.PopulateTensor<float>(model.start(), {2});
  model.PopulateTensor<float>(model.limit(), {9});
  model.PopulateTensor<float>(model.delta(), {2});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, FloatNegativeDelta) {
  RangeOpModel<float> model(TensorType_FLOAT32);
  model.PopulateTensor<float>(model.start(), {10});
  model.PopulateTensor<float>(model.limit(), {3});
  model.PopulateTensor<float>(model.delta(), {-3});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3));
  EXPECT_THAT(model.GetOutput(), ElementsAre(10, 7, 4));
}

const float kQuantizedTolerance = 2 * (1. / 256);
class QuantizedRangeOpModel : public SingleOpModel {
 public:
  explicit QuantizedRangeOpModel(TensorData start, TensorData limit,
                                 TensorData delta) {
    start_ = AddInput(start);
    limit_ = AddInput(limit);
    delta_ = AddInput(delta);
    output_ = AddOutput({start.type, {}, start.min, start.max});
    SetBuiltinOp(BuiltinOperator_RANGE, BuiltinOptions_RangeOptions,
                 CreateRangeOptions(builder_).Union());
    BuildInterpreter({GetShape(start_), GetShape(limit_), GetShape(delta_)});
  }

  template <typename T>
  void SetStart(std::initializer_list<float> data) {
    QuantizeAndPopulate<T>(start_, data);
  }

  template <typename T>
  void SetDelta(std::initializer_list<float> data) {
    QuantizeAndPopulate<T>(delta_, data);
  }

  template <typename T>
  void SetLimit(std::initializer_list<float> data) {
    QuantizeAndPopulate<T>(limit_, data);
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

 private:
  int start_;
  int limit_;
  int delta_;
  int output_;
};

TEST(QuantizedRangeOpModel, RangeUInt8) {
  float kMin = std::numeric_limits<uint8_t>::min();
  float kMax = std::numeric_limits<uint8_t>::max();

  QuantizedRangeOpModel m({TensorType_UINT8, {}, kMin, kMax},
                          {TensorType_UINT8, {}, kMin, kMax},
                          {TensorType_UINT8, {}, kMin, kMax});

  m.SetStart<uint8_t>({0});
  m.SetDelta<uint8_t>({1});
  m.SetLimit<uint8_t>({4});
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0,
                      1,
                      2,
                      3,
                  },
                  kQuantizedTolerance)));
}

TEST(QuantizedRangeOpModel, QUIntDeltaGreaterThanOne) {
  float kMin = std::numeric_limits<uint8_t>::min();
  float kMax = std::numeric_limits<uint8_t>::max();

  QuantizedRangeOpModel m({TensorType_UINT8, {}, kMin, kMax},
                          {TensorType_UINT8, {}, kMin, kMax},
                          {TensorType_UINT8, {}, kMin, kMax});

  m.SetStart<uint8_t>({2});
  m.SetDelta<uint8_t>({2});
  m.SetLimit<uint8_t>({9});
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      2,
                      4,
                      6,
                      8,
                  },
                  kQuantizedTolerance)));
}

TEST(QuantizedRangeOpModel, RangeInt8) {
  float kMin = std::numeric_limits<int8_t>::min();
  float kMax = std::numeric_limits<int8_t>::max();

  QuantizedRangeOpModel m({TensorType_INT8, {}, kMin, kMax},
                          {TensorType_INT8, {}, kMin, kMax},
                          {TensorType_INT8, {}, kMin, kMax});

  m.SetStart<int8_t>({0});
  m.SetDelta<int8_t>({1});
  m.SetLimit<int8_t>({4});
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(), ElementsAreArray(ArrayFloatNear(
                                                    {
                                                        0,
                                                        1,
                                                        2,
                                                        3,
                                                    },
                                                    kQuantizedTolerance)));
}

TEST(QuantizedRangeOpModel, QIntDeltaGreaterThanOne) {
  float kMin = std::numeric_limits<int8_t>::min();
  float kMax = std::numeric_limits<int8_t>::max();
  QuantizedRangeOpModel m({TensorType_INT8, {}, kMin, kMax},
                          {TensorType_INT8, {}, kMin, kMax},
                          {TensorType_INT8, {}, kMin, kMax});

  m.SetStart<int8_t>({2});
  m.SetDelta<int8_t>({2});
  m.SetLimit<int8_t>({9});
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(), ElementsAreArray(ArrayFloatNear(
                                                    {
                                                        2,
                                                        4,
                                                        6,
                                                        8,
                                                    },
                                                    kQuantizedTolerance)));
}

TEST(QuantizedRangeOpModel, QIntNegativeDelta) {
  float kMin = std::numeric_limits<int8_t>::min();
  float kMax = std::numeric_limits<int8_t>::max();
  QuantizedRangeOpModel m({TensorType_INT8, {}, kMin, kMax},
                          {TensorType_INT8, {}, kMin, kMax},
                          {TensorType_INT8, {}, kMin, kMax});

  m.SetStart<int8_t>({10});
  m.SetDelta<int8_t>({-3});
  m.SetLimit<int8_t>({3});
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(), ElementsAreArray(ArrayFloatNear(
                                                    {
                                                        10,
                                                        7,
                                                        4,
                                                    },
                                                    kQuantizedTolerance)));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
