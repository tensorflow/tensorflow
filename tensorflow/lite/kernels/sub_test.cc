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
#include <stdint.h>

#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseSubOpModel : public SingleOpModel {
 public:
  BaseSubOpModel(const TensorData& input1, const TensorData& input2,
                 const TensorData& output,
                 ActivationFunctionType activation_type) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_SUB, BuiltinOptions_SubOptions,
                 CreateSubOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input1_;
  int input2_;
  int output_;
};

class FloatSubOpModel : public BaseSubOpModel {
 public:
  using BaseSubOpModel::BaseSubOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class IntegerSubOpModel : public BaseSubOpModel {
 public:
  using BaseSubOpModel::BaseSubOpModel;

  std::vector<int32_t> GetOutput() { return ExtractVector<int32_t>(output_); }
};

class Int64SubOpModel : public BaseSubOpModel {
 public:
  using BaseSubOpModel::BaseSubOpModel;

  std::vector<int64_t> GetOutput() { return ExtractVector<int64_t>(output_); }
};

class QuantizedSubOpModel : public BaseSubOpModel {
 public:
  QuantizedSubOpModel(TensorData input1, TensorData input2, TensorData output,
                      ActivationFunctionType activation_type)
      : BaseSubOpModel(SymmetricInt16Scaling(std::move(input1)),
                       SymmetricInt16Scaling(std::move(input2)),
                       SymmetricInt16Scaling(std::move(output)),
                       activation_type) {}

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_dtype>(ExtractVector<integer_dtype>(output_),
                                     GetScale(output_), GetZeroPoint(output_));
  }

 private:
  TensorData SymmetricInt16Scaling(TensorData tensor) {
    // Symmetric range and null zero-point is required for INT16 tensors. As
    // SingleOpModel::QuantizationParams calculates the scale on an asymmetric
    // base [int_type::min, int_type::max], manually calculate the scale on a
    // symmetric range [int_type::min+1, int_type::max] to ensure a null
    // zero-point.
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

// for quantized Sub, the error shouldn't exceed step
template <typename T>
float GetTolerance(float min, float max) {
  float kQuantizedStep = (max - min) / (std::numeric_limits<T>::max() -
                                        std::numeric_limits<T>::min());
  return 2.0 * kQuantizedStep;
}

TEST(FloatSubOpModel, FirstInputZero) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  FloatSubOpModel m({TensorType_FLOAT32, {0}}, {TensorType_FLOAT32, {}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input2(), {0.1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray<int>({0}));
}

TEST(FloatSubOpModel, SecondInputZero) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  FloatSubOpModel m({TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {0}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {0.1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray<int>({0}));
}

TEST(FloatSubOpModel, NoActivation) {
  FloatSubOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.8});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-2.1, 0.0, 1.4, -0.3})));
}

TEST(FloatSubOpModel, ActivationRELU_N1_TO_1) {
  FloatSubOpModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.8});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-1.0, 0.0, 1.0, -0.3})));
}

TEST(FloatSubOpModel, VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatSubOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5, -1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.8, -1.1, 0.1});
    m.Invoke();
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-2.1, 0.0, 1.4, -0.3, 0.0, 1.9})))
        << "With shape number " << i;
  }
}

TEST(FloatSubOpModel, WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatSubOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5, -1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.5});
    m.Invoke();
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-2.5, -0.3, 1.2, 0.0, -1.6, 1.5})))
        << "With shape number " << i;
  }
}

TEST(FloatSubOpModel, WithBroadcast5D) {
  std::vector<std::vector<int>> test_shapes = {{1, 3, 1, 2, 1}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatSubOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5, -1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.5});
    m.Invoke();
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-2.5, -0.3, 1.2, 0.0, -1.6, 1.5})))
        << "With shape number " << i;
  }
}

TEST(IntegerSubOpModel, NoActivation) {
  IntegerSubOpModel m({TensorType_INT32, {1, 2, 2, 1}},
                      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {}},
                      ActivationFunctionType_NONE);
  m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 7, 8});
  m.PopulateTensor<int32_t>(m.input2(), {1, 2, 3, 5});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-21, 0, 4, 3}));
}

TEST(IntegerSubOpModel, ActivationRELU_N1_TO_1) {
  IntegerSubOpModel m({TensorType_INT32, {1, 2, 2, 1}},
                      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {}},
                      ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 7, 8});
  m.PopulateTensor<int32_t>(m.input2(), {1, 2, 3, 5});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1, 0, 1, 1}));
}

TEST(IntegerSubOpModel, VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerSubOpModel m({TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 7, 8, 11, 20});
    m.PopulateTensor<int32_t>(m.input2(), {1, 2, 3, 5, 11, 1});
    m.Invoke();
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({-21, 0, 4, 3, 0, 19}))
        << "With shape number " << i;
  }
}

TEST(IntegerSubOpModel, WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}, {1, 3, 1, 2, 1}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerSubOpModel m({TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, {}},  // always a scalar
                        {TensorType_INT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 7, 8, 11, 20});
    m.PopulateTensor<int32_t>(m.input2(), {1});
    m.Invoke();
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray(ArrayFloatNear({-21, 1, 6, 7, 10, 19})))
        << "With shape number " << i;
  }
}

TEST(Int64SubOpModel, NoActivation) {
  Int64SubOpModel m({TensorType_INT64, {1, 2, 2, 1}},
                    {TensorType_INT64, {1, 2, 2, 1}}, {TensorType_INT64, {}},
                    ActivationFunctionType_NONE);
  m.PopulateTensor<int64_t>(m.input1(), {-20, 2, 7, 8});
  m.PopulateTensor<int64_t>(m.input2(), {1, 2, 3, 5});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-21, 0, 4, 3}));
}

TEST(Int64SubOpModel, ActivationRELU_N1_TO_1) {
  Int64SubOpModel m({TensorType_INT64, {1, 2, 2, 1}},
                    {TensorType_INT64, {1, 2, 2, 1}}, {TensorType_INT64, {}},
                    ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<int64_t>(m.input1(), {-20, 2, 7, 8});
  m.PopulateTensor<int64_t>(m.input2(), {1, 2, 3, 5});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1, 0, 1, 1}));
}

TEST(Int64SubOpModel, VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    Int64SubOpModel m({TensorType_INT64, test_shapes[i]},
                      {TensorType_INT64, test_shapes[i]},
                      {TensorType_INT64, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int64_t>(m.input1(), {-20, 2, 7, 8, 11, 20});
    m.PopulateTensor<int64_t>(m.input2(), {1, 2, 3, 5, 11, 1});
    m.Invoke();
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({-21, 0, 4, 3, 0, 19}))
        << "With shape number " << i;
  }
}

TEST(Int64SubOpModel, WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}, {1, 3, 1, 2, 1}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    Int64SubOpModel m({TensorType_INT64, test_shapes[i]},
                      {TensorType_INT64, {}},  // always a scalar
                      {TensorType_INT64, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int64_t>(m.input1(), {-20, 2, 7, 8, 11, 20});
    m.PopulateTensor<int64_t>(m.input2(), {1});
    m.Invoke();
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray(ArrayFloatNear({-21, 1, 6, 7, 10, 19})))
        << "With shape number " << i;
  }
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedTestsNoActivation() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-0.9, 0.9);
  std::vector<std::vector<float>> inputs1 = {
      {0.1, 0.2, 0.3, 0.4}, {-0.2, 0.2, 0.4, 0.7}, {-0.01, 0.2, 0.7, 0.3}};
  std::vector<std::vector<float>> inputs2 = {
      {0.6, 0.4, 0.3, 0.1}, {0.6, 0.4, 0.5, -0.2}, {0.6, 0.4, -0.18, 0.5}};
  std::vector<std::vector<float>> results = {{-0.5, -0.2, 0.0, 0.3},
                                             {-0.8, -0.2, -0.1, 0.9},
                                             {-0.61, -0.2, 0.88, -0.2}};
  for (int i = 0; i < inputs1.size(); ++i) {
    QuantizedSubOpModel m({tensor_type, {1, 2, 2, 1}, -0.7, 0.7},
                          {tensor_type, {1, 2, 2, 1}, -0.6, 0.6},
                          {tensor_type, {}, -0.9, 0.9},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<integer_dtype>(m.input1(), inputs1[i]);
    m.QuantizeAndPopulate<integer_dtype>(m.input2(), inputs2[i]);
    m.Invoke();
    EXPECT_THAT(
        m.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear(results[i], kQuantizedTolerance)))
        << "With test number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedTestsNoActivationUInt8) {
  QuantizedTestsNoActivation<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedSubOpModel, QuantizedTestsNoActivationInt8) {
  QuantizedTestsNoActivation<TensorType_INT8, int8_t>();
}

TEST(QuantizedSubOpModel, QuantizedTestsNoActivationGenericInt16) {
  QuantizedTestsNoActivation<TensorType_INT16, int16_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedTestsActivationRELU_N1_TO_1() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-1.0, 1.0);
  std::vector<std::vector<float>> inputs1 = {{-0.8, 0.2, 0.9, 0.7},
                                             {-0.8, 0.2, 0.7, 0.5}};
  std::vector<std::vector<float>> inputs2 = {{0.6, 0.4, 0.9, -0.8},
                                             {0.6, 0.4, -0.8, 0.3}};
  std::vector<std::vector<float>> results = {{-1.0, -0.2, 0.0, 1.0},
                                             {-1.0, -0.2, 1.0, 0.2}};
  for (int i = 0; i < inputs1.size(); ++i) {
    QuantizedSubOpModel m({tensor_type, {1, 2, 2, 1}, -0.9, 0.9},
                          {tensor_type, {1, 2, 2, 1}, -0.9, 0.9},
                          {tensor_type, {}, -1.0, 1.0},
                          ActivationFunctionType_RELU_N1_TO_1);
    m.QuantizeAndPopulate<integer_dtype>(m.input1(), inputs1[i]);
    m.QuantizeAndPopulate<integer_dtype>(m.input2(), inputs2[i]);
    m.Invoke();
    EXPECT_THAT(
        m.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear(results[i], kQuantizedTolerance)))
        << "With test number " << i;
  }
}
TEST(QuantizedSubOpModel, QuantizedTestsActivationRELUN1TO1UInt8) {
  QuantizedTestsActivationRELU_N1_TO_1<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedSubOpModel, QuantizedTestsActivationRELUN1TO1Int8) {
  QuantizedTestsActivationRELU_N1_TO_1<TensorType_INT8, int8_t>();
}

TEST(QuantizedSubOpModel, QuantizedTestsActivationRELUN1TO1Int16) {
  QuantizedTestsActivationRELU_N1_TO_1<TensorType_INT16, int16_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedVariousInputShapes() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-2.1, 2.1);
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedSubOpModel m({tensor_type, test_shapes[i], -2.0, 2.0},
                          {tensor_type, test_shapes[i], -1.1, 1.1},
                          {tensor_type, {}, -2.1, 2.1},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<integer_dtype>(m.input1(),
                                         {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.QuantizeAndPopulate<integer_dtype>(m.input2(),
                                         {0.1, 0.3, 0.3, 0.5, 1.1, 0.1});
    m.Invoke();
    EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
                ElementsAreArray(ArrayFloatNear(
                    {-2.1, -0.1, 0.4, 0.3, 0.0, 1.9}, kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedVariousInputShapesUInt8) {
  QuantizedVariousInputShapes<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedSubOpModel, QuantizedVariousInputShapesInt8) {
  QuantizedVariousInputShapes<TensorType_INT8, int8_t>();
}

TEST(QuantizedSubOpModel, QuantizedVariousInputShapesInt16) {
  QuantizedVariousInputShapes<TensorType_INT16, int16_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedWithBroadcast() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-2.7, 2.7);
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedSubOpModel m(
        {tensor_type, test_shapes[i], -2.0, 2.0}, {tensor_type, {}, -0.7, 0.7},
        {tensor_type, {}, -2.7, 2.7}, ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<integer_dtype>(m.input1(),
                                         {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.QuantizeAndPopulate<integer_dtype>(m.input2(), {0.7});
    m.Invoke();
    EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
                ElementsAreArray(ArrayFloatNear(
                    {-2.7, -0.5, 0.0, 0.1, 0.4, 1.3}, kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedWithBroadcastUInt8) {
  QuantizedWithBroadcast<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedSubOpModel, QuantizedWithBroadcastInt8) {
  QuantizedWithBroadcast<TensorType_INT8, int8_t>();
}

TEST(QuantizedSubOpModel, QuantizedWithBroadcastInt16) {
  QuantizedWithBroadcast<TensorType_INT16, int16_t>();
}

TEST(QuantizedSubOpModel, QuantizedTestsNoActivationInt16) {
  float kQuantizedTolerance = GetTolerance<int16_t>(-1.1, 1.1);
  std::vector<std::vector<float>> inputs1 = {
      {0.7, 0.6, 0.6, 0.5}, {-0.2, 0.6, 0.9, -0.1}, {-0.2, 0.6, -0.3, 0.8}};
  std::vector<std::vector<float>> inputs2 = {
      {0.6, 0.4, 0.3, 0.1}, {0.6, 0.4, 0.5, -0.8}, {0.6, 0.4, 0.8, 0.5}};
  std::vector<std::vector<float>> results = {
      {0.1, 0.2, 0.3, 0.4}, {-0.8, 0.2, 0.4, 0.7}, {-0.8, 0.2, -1.1, 0.3}};
  for (int i = 0; i < inputs1.size(); ++i) {
    QuantizedSubOpModel m({TensorType_INT16, {1, 2, 2, 1}, -0.9, 0.9},
                          {TensorType_INT16, {1, 2, 2, 1}, -0.8, 0.8},
                          {TensorType_INT16, {}, -1.1, 1.1},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<int16_t>(m.input1(), inputs1[i]);
    m.QuantizeAndPopulate<int16_t>(m.input2(), inputs2[i]);
    m.Invoke();
    EXPECT_THAT(
        m.GetDequantizedOutput<int16_t>(),
        ElementsAreArray(ArrayFloatNear(results[i], kQuantizedTolerance)))
        << "With test number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedTestsReluActivationInt16) {
  float kQuantizedTolerance = GetTolerance<int16_t>(-1.0, 1.0);
  std::vector<std::vector<float>> inputs1 = {{-0.8, 0.2, 0.9, 0.7},
                                             {-0.8, 0.2, 0.7, 0.5}};
  std::vector<std::vector<float>> inputs2 = {{0.6, 0.4, 0.9, -0.8},
                                             {0.6, 0.4, -0.8, 0.3}};
  std::vector<std::vector<float>> results = {{-1.0, -0.2, 0.0, 1.0},
                                             {-1.0, -0.2, 1.0, 0.2}};
  for (int i = 0; i < inputs1.size(); ++i) {
    QuantizedSubOpModel m({TensorType_INT16, {1, 2, 2, 1}, -0.9, 0.9},
                          {TensorType_INT16, {1, 2, 2, 1}, -0.9, 0.9},
                          {TensorType_INT16, {}, -1.0, 1.0},
                          ActivationFunctionType_RELU_N1_TO_1);
    m.QuantizeAndPopulate<int16_t>(m.input1(), inputs1[i]);
    m.QuantizeAndPopulate<int16_t>(m.input2(), inputs2[i]);
    m.Invoke();
    EXPECT_THAT(
        m.GetDequantizedOutput<int16_t>(),
        ElementsAreArray(ArrayFloatNear(results[i], kQuantizedTolerance)))
        << "With test number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedTestsNoActivationBroadcastInt16) {
  float kQuantizedTolerance = GetTolerance<int16_t>(-1.1, 1.1);
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}, {1, 3, 1, 2, 1}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedSubOpModel m({TensorType_INT16, test_shapes[i], -0.9, 0.9},
                          {TensorType_INT16, {}, -0.2, 0.2},
                          {TensorType_INT16, {}, -1.1, 1.1},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<int16_t>(m.input1(),
                                   {-0.9, -0.7, -0.3, 0.0, 0.3, 0.5});
    m.QuantizeAndPopulate<int16_t>(m.input2(), {0.2});
    m.Invoke();
    EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
                ElementsAreArray(ArrayFloatNear(
                    {-1.1, -0.9, -0.5, -0.2, 0.1, 0.3}, kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedTestsReluActivationBroadcastInt16) {
  float kQuantizedTolerance = GetTolerance<int16_t>(-1.0, 1.0);
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}, {1, 3, 1, 2, 1}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedSubOpModel m({TensorType_INT16, test_shapes[i], -0.9, 0.9},
                          {TensorType_INT16, {}, -0.2, 0.2},
                          {TensorType_INT16, {}, -1.0, 1.0},
                          ActivationFunctionType_RELU_N1_TO_1);
    m.QuantizeAndPopulate<int16_t>(m.input1(),
                                   {-0.9, -0.7, -0.3, 0.0, 0.3, 0.5});
    m.QuantizeAndPopulate<int16_t>(m.input2(), {0.2});
    m.Invoke();
    EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
                ElementsAreArray(ArrayFloatNear(
                    {-1.0, -0.9, -0.5, -0.2, 0.1, 0.3}, kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

}  // namespace
}  // namespace tflite
