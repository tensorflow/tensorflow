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

#include <initializer_list>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
class TileOpBaseModel : public SingleOpModel {
 public:
  int Input() { return input_; }
  int Multiplier() { return multipliers_; }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

 protected:
  int input_;
  int multipliers_;
  int output_;
};

template <typename MultipliersType>
class TileOpConstModel : public TileOpBaseModel {
 public:
  TileOpConstModel(const TensorData& input, const TensorData& multiplier,
                   std::initializer_list<MultipliersType> multipliers_data,
                   const TensorData& output) {
    input_ = AddInput(input);
    multipliers_ = AddConstInput(multiplier, multipliers_data);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_TILE, BuiltinOptions_TileOptions, 0);
    BuildInterpreter({GetShape(input_)});
  }
};

class TileOpDynamicModel : public TileOpBaseModel {
 public:
  TileOpDynamicModel(const TensorData& input, const TensorData& multiplier,
                     const TensorData& output) {
    input_ = AddInput(input);
    multipliers_ = AddInput(multiplier);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_TILE, BuiltinOptions_TileOptions, 0);
    BuildInterpreter({GetShape(input_)});
  }
};

enum class TestType {
  kConst = 0,
  kDynamic = 1,
};

template <typename InputType, typename MultipliersType = int32_t>
void CheckQuantized(const TensorData& input,
                    std::initializer_list<float> input_data,
                    const TensorData& multiplier,
                    std::initializer_list<MultipliersType> multipliers_data,
                    const TensorData& output,
                    std::initializer_list<int> exp_output_shape,
                    std::initializer_list<float> exp_output_data,
                    float tolerance, TestType test_type) {
  switch (test_type) {
    case TestType::kConst: {
      TileOpConstModel<MultipliersType> m(input, multiplier, multipliers_data,
                                          output);
      m.template QuantizeAndPopulate<InputType>(m.Input(), input_data);
      m.Invoke();
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(exp_output_shape));
      EXPECT_THAT(m.template GetDequantizedOutput<InputType>(),
                  ElementsAreArray(ArrayFloatNear(exp_output_data, tolerance)));
      return;
    }
    case TestType::kDynamic: {
      TileOpDynamicModel m(input, multiplier, output);
      m.QuantizeAndPopulate<InputType>(m.Input(), input_data);
      m.PopulateTensor<MultipliersType>(m.Multiplier(), multipliers_data);
      m.Invoke();
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(exp_output_shape));
      EXPECT_THAT(m.GetDequantizedOutput<InputType>(),
                  ElementsAreArray(ArrayFloatNear(exp_output_data, tolerance)));
      return;
    }
  }
}

template <typename InputType, typename MultipliersType = int32_t>
void Check(const TensorData& input, std::initializer_list<InputType> input_data,
           const TensorData& multiplier,
           std::initializer_list<MultipliersType> multipliers_data,
           const TensorData& output,
           std::initializer_list<int> exp_output_shape,
           std::initializer_list<InputType> exp_output_data,
           TestType test_type) {
  switch (test_type) {
    case TestType::kConst: {
      if (SingleOpModel::GetForceUseNnapi() &&
          !std::is_same<InputType, std::string>::value) {
        // NNAPI does not support graphs with all constant inputs.
        return;
      }
      TileOpConstModel<MultipliersType> m(input, multiplier, multipliers_data,
                                          output);
      m.template PopulateTensor<InputType>(m.Input(), input_data);
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(exp_output_shape));
      EXPECT_THAT(m.template GetOutput<InputType>(),
                  ElementsAreArray(exp_output_data));
      return;
    }
    case TestType::kDynamic: {
      TileOpDynamicModel m(input, multiplier, output);
      m.PopulateTensor<InputType>(m.Input(), input_data);
      m.PopulateTensor<MultipliersType>(m.Multiplier(), multipliers_data);
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(exp_output_shape));
      EXPECT_THAT(m.template GetOutput<InputType>(),
                  ElementsAreArray(exp_output_data));
      return;
    }
  }
}

class TileTest : public ::testing::TestWithParam<TestType> {};

template <typename T>
float GetTolerance(float min, float max) {
  float kQuantizedStep = (max - min) / (std::numeric_limits<T>::max() -
                                        std::numeric_limits<T>::min());
  return kQuantizedStep;
}

TEST_P(TileTest, Float32Vector) {
  Check<float>(
      /*input=*/{TensorType_FLOAT32, {3}},
      /*input_data=*/{1.0, 2.0, 3.0},
      /*multiplier=*/{TensorType_INT32, {1}},
      /*multipliers_data=*/{2},
      /*output=*/{TensorType_FLOAT32, {}},
      /*exp_output_shape=*/{6},
      /*exp_output_data=*/
      {1.0, 2.0, 3.0, 1.0, 2.0, 3.0},
      /*test_type=*/GetParam());
}

TEST_P(TileTest, Float32Matrix) {
  Check<float>(
      /*input=*/{TensorType_FLOAT32, {2, 3}},
      /*input_data=*/{11.f, 12.f, 13.f, 21.f, 22.f, 23.f},
      /*multiplier=*/{TensorType_INT32, {2}},
      /*multipliers_data=*/{2, 1},
      /*output=*/{TensorType_FLOAT32, {}},
      /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/
      {11.f, 12.f, 13.f, 21.f, 22.f, 23.f, 11.f, 12.f, 13.f, 21.f, 22.f, 23.f},
      /*test_type=*/GetParam());
}

TEST_P(TileTest, Float32HighDimension) {
  Check<float>(
      /*input=*/{TensorType_FLOAT32, {1, 2, 3}},
      /*input_data=*/{11.f, 12.f, 13.f, 21.f, 22.f, 23.f},
      /*multiplier=*/{TensorType_INT32, {3}},
      /*multipliers_data=*/{2, 3, 1},
      /*output=*/{TensorType_FLOAT32, {}},
      /*exp_output_shape=*/{2, 6, 3},
      /*exp_output_data=*/
      {11.f, 12.f, 13.f, 21.f, 22.f, 23.f, 11.f, 12.f, 13.f, 21.f, 22.f, 23.f,
       11.f, 12.f, 13.f, 21.f, 22.f, 23.f, 11.f, 12.f, 13.f, 21.f, 22.f, 23.f,
       11.f, 12.f, 13.f, 21.f, 22.f, 23.f, 11.f, 12.f, 13.f, 21.f, 22.f, 23.f},
      /*test_type=*/GetParam());
}

TEST_P(TileTest, Uint8Matrix) {
  Check<uint8_t>({/*input=*/TensorType_UINT8, {2, 3}},
                 /*input_data=*/{11, 12, 13, 21, 22, 23},
                 /*multiplier=*/{TensorType_INT32, {2}},
                 /*multipliers_data=*/{2, 1},
                 /*output=*/{TensorType_UINT8, {}},
                 /*exp_output_shape=*/{4, 3},
                 /*exp_output_data=*/
                 {11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23},
                 /*test_type=*/GetParam());
}

TEST_P(TileTest, Int8Matrix) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  const float kQuantizedTolerance = GetTolerance<int8_t>(0.0f, 23.0f);
  CheckQuantized<int8_t>(
      /*input=*/{TensorType_INT8, {2, 3}, 0.0f, 23.0f},
      /*input_data=*/{11.f, 12.f, 13.f, 21.f, 22.f, 23.f},
      /*multiplier=*/{TensorType_INT32, {2}},
      /*multipliers_data=*/{2, 1},
      /*output=*/{TensorType_INT8, {}, 0.0f, 23.0f},
      /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/
      {11.f, 12.f, 13.f, 21.f, 22.f, 23.f, 11.f, 12.f, 13.f, 21.f, 22.f, 23.f},
      /*tolerance=*/kQuantizedTolerance,
      /*test_type=*/GetParam());
}

TEST_P(TileTest, Int16Matrix) {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<int16_t>::max() /
      static_cast<float>(std::numeric_limits<int16_t>::max() + 1);
  const float kQuantizedTolerance = GetTolerance<int16_t>(-23.0, 23.0);
  CheckQuantized<int16_t>(
      /*input=*/{TensorType_INT16, {2, 3}, 23.0f * kMin, 23.0f * kMax},
      /*input_data=*/{11.f, 12.f, 13.f, 21.f, 22.f, 23.f},
      /*multiplier=*/{TensorType_INT32, {2}},
      /*multipliers_data=*/{2, 1},
      /*output=*/{TensorType_INT16, {}, 23.0f * kMin, 23.0f * kMax},
      /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/
      {11.f, 12.f, 13.f, 21.f, 22.f, 23.f, 11.f, 12.f, 13.f, 21.f, 22.f, 23.f},
      /*tolerance=*/kQuantizedTolerance,
      /*test_type=*/GetParam());
}

TEST_P(TileTest, Int32Matrix) {
  Check<int32_t>(
      /*input=*/{TensorType_INT32, {2, 3}},
      /*input_data=*/{11, 12, 13, 21, 22, 23},
      /*multiplier=*/{TensorType_INT32, {2}},
      /*multipliers_data=*/{2, 1},
      /*output=*/{TensorType_INT32, {}},
      /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/
      {11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23},
      /*test_type=*/GetParam());
}

TEST_P(TileTest, Int64Matrix) {
  Check<int64_t>(
      /*input=*/{TensorType_INT64, {2, 3}},
      /*input_data=*/{11, 12, 13, 21, 22, 23},
      /*multiplier=*/{TensorType_INT32, {2}},
      /*multipliers_data=*/{2, 1},
      /*output=*/{TensorType_INT64, {}},
      /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/
      {11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23},
      /*test_type=*/GetParam());
}

TEST_P(TileTest, BooleanMatrix) {
  Check<bool>(
      /*input=*/{TensorType_BOOL, {2, 3}},
      /*input_data=*/{true, false, false, true, true, false},
      /*multiplier=*/{TensorType_INT32, {2}},
      /*multipliers_data=*/{2, 1},
      /*output=*/{TensorType_BOOL, {}},
      /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/
      {true, false, false, true, true, false, true, false, false, true, true,
       false},
      /*test_type=*/GetParam());
}

TEST_P(TileTest, Int64Matrix64Multipliers) {
  Check<int64_t, int64_t>(
      /*input=*/{TensorType_INT64, {2, 3}},
      /*input_data=*/{11, 12, 13, 21, 22, 23},
      /*multiplier=*/{TensorType_INT64, {2}},
      /*multipliers_data=*/{2, 1},
      /*output=*/{TensorType_INT64, {}},
      /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/
      {11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23},
      /*test_type=*/GetParam());
}

TEST_P(TileTest, StringMatrix) {
  Check<std::string>(
      /*input=*/{TensorType_STRING, {2, 3}},
      /*input_data=*/{"AA", "AB", "AC", "BA", "BB", "BC"},
      /*multiplier=*/{TensorType_INT32, {2}},
      /*multipliers_data=*/{1, 2},
      /*output=*/{TensorType_STRING, {}},
      /*exp_output_shape=*/{2, 6},
      /*exp_output_data=*/
      {"AA", "AB", "AC", "AA", "AB", "AC", "BA", "BB", "BC", "BA", "BB", "BC"},
      /*test_type=*/GetParam());
}

TEST_P(TileTest, StringMatrix64Multipliers) {
  Check<std::string, int64_t>(
      /*input=*/{TensorType_STRING, {2, 3}},
      /*input_data=*/{"AA", "AB", "AC", "BA", "BB", "BC"},
      /*multiplier=*/{TensorType_INT64, {2}},
      /*multipliers_data=*/{2, 1},
      /*output=*/{TensorType_STRING, {}},
      /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/
      {"AA", "AB", "AC", "BA", "BB", "BC", "AA", "AB", "AC", "BA", "BB", "BC"},
      /*test_type=*/GetParam());
}

TEST_P(TileTest, StringMatrix2) {
  Check<std::string>(
      /*input=*/{TensorType_STRING, {3, 2, 1}},
      /*input_data=*/{"AA", "AB", "AC", "BA", "BB", "BC"},
      /*multiplier=*/{TensorType_INT32, {3}},
      /*multipliers_data=*/{2, 2, 2},
      /*output=*/{TensorType_STRING, {}},
      /*exp_output_shape=*/{6, 4, 2},
      /*exp_output_data=*/
      {"AA", "AA", "AB", "AB", "AA", "AA", "AB", "AB", "AC", "AC", "BA", "BA",
       "AC", "AC", "BA", "BA", "BB", "BB", "BC", "BC", "BB", "BB", "BC", "BC",
       "AA", "AA", "AB", "AB", "AA", "AA", "AB", "AB", "AC", "AC", "BA", "BA",
       "AC", "AC", "BA", "BA", "BB", "BB", "BC", "BC", "BB", "BB", "BC", "BC"},
      /*test_type=*/GetParam());
}

TEST_P(TileTest, StringMatrixEmptyInputElements) {
  Check<std::string>(
      /*input=*/{TensorType_STRING, {0, 1, 1}},
      /*input_data=*/{},
      /*multiplier=*/{TensorType_INT32, {3}},
      /*multipliers_data=*/{2, 2, 2},
      /*output=*/{TensorType_STRING, {}},
      /*exp_output_shape=*/{0, 2, 2},
      /*exp_output_data=*/
      {},
      /*test_type=*/GetParam());
}

TEST(TileTest, TestEmptyInput) {
  TileOpDynamicModel m({TensorType_INT32, {2, 1, 3}}, {TensorType_INT32, {3}},
                       {TensorType_INT32, {}});
  m.PopulateTensor<int32_t>(m.Input(), {11, 12, 13, 21, 22, 23});
  m.PopulateTensor<int32_t>(m.Multiplier(), {2, 0, 2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 0, 6}));
}

INSTANTIATE_TEST_SUITE_P(TileTest, TileTest,
                         ::testing::Values(TestType::kConst,
                                           TestType::kDynamic));
}  // namespace
}  // namespace tflite
