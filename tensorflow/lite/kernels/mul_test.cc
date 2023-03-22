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
#include <stddef.h>
#include <stdint.h>

#include <complex>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename InputType, typename QuantizedType = InputType>
class BaseMulOpModel : public SingleOpModel {
 public:
  BaseMulOpModel(const TensorData& input1, const TensorData& input2,
                 const TensorData& output,
                 ActivationFunctionType activation_type,
                 const std::vector<InputType>& input1_data,
                 const std::vector<InputType>& input2_data,
                 bool constant_tensors) {
    if (constant_tensors) {
      input1_ = AddConstInput(input1, input1_data);
      input2_ = AddConstInput(input2, input2_data);
    } else {
      input1_ = AddInput(input1);
      input2_ = AddInput(input2);
    }
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_MUL, BuiltinOptions_MulOptions,
                 CreateMulOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
    if (!constant_tensors) {
      PopulateTensor<QuantizedType>(input1_, input1_data);
      PopulateTensor<QuantizedType>(input2_, input2_data);
    }
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

 protected:
  int input1_;
  int input2_;
  int output_;
};

class FloatMulOpModel : public BaseMulOpModel<float> {
 public:
  using BaseMulOpModel::BaseMulOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class ComplexMulOpModel : public BaseMulOpModel<std::complex<float>> {
 public:
  using BaseMulOpModel::BaseMulOpModel;

  std::vector<std::complex<float>> GetOutput() {
    return ExtractVector<std::complex<float>>(output_);
  }
};

class IntegerMulOpModel : public BaseMulOpModel<int32_t> {
 public:
  using BaseMulOpModel::BaseMulOpModel;

  std::vector<int32_t> GetOutput() { return ExtractVector<int32_t>(output_); }
};

// For quantized Mul, the error shouldn't exceed (2*step + step^2).
// The param min=-1.0 & max=1.0 is used in the following tests.
// The tolerance value is ~0.0157.
const float kQuantizedStep = 2.0 / 255.0;
const float kQuantizedTolerance =
    2.0 * kQuantizedStep + kQuantizedStep * kQuantizedStep;
const float kQuantizedStepInt16 = 2.0 / 32767.0;
const float kQuantizedToleranceInt16 =
    2.0 * kQuantizedStepInt16 + kQuantizedStepInt16 * kQuantizedStepInt16;

template <typename InputType, typename QuantizedType>
class QuantizedMulOpModel : public SingleOpModel {
 public:
  QuantizedMulOpModel(const TensorData& input1, const TensorData& input2,
                      const TensorData& output,
                      ActivationFunctionType activation_type,
                      const std::vector<float>& input1_data,
                      const std::vector<float>& input2_data,
                      bool constant_tensors) {
    if (constant_tensors) {
      std::vector<InputType> quantized_input1_data(input1_data.size());
      std::vector<InputType> quantized_input2_data(input2_data.size());
      std::pair<float, int32_t> input1_quantization_params =
          QuantizationParams<InputType>(input1.min, input1.max);
      std::pair<float, int32_t> input2_quantization_params =
          QuantizationParams<InputType>(input2.min, input2.max);
      quantized_input1_data =
          Quantize<InputType>(input1_data, input1_quantization_params.first,
                              input1_quantization_params.second);
      quantized_input2_data =
          Quantize<InputType>(input2_data, input2_quantization_params.first,
                              input2_quantization_params.second);
      input1_ = AddConstInput(input1, quantized_input1_data);
      input2_ = AddConstInput(input2, quantized_input2_data);
    } else {
      input1_ = AddInput(input1);
      input2_ = AddInput(input2);
    }
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_MUL, BuiltinOptions_MulOptions,
                 CreateMulOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
    if (!constant_tensors) {
      QuantizeAndPopulate<InputType>(input1_, input1_data);
      QuantizeAndPopulate<InputType>(input2_, input2_data);
    }
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<QuantizedType>(
        this->template ExtractVector<QuantizedType>(this->output_),
        GetScale(this->output_), GetZeroPoint(this->output_));
  }

 protected:
  int input1_;
  int input2_;
  int output_;
};

using MulOpTest = testing::TestWithParam<bool>;

TEST_P(MulOpTest, NoActivationFloat) {
  const bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  FloatMulOpModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE,
      {-2.0, 0.2, 0.7, 0.8}, {0.1, 0.2, 0.3, 0.5}, constant_tensors);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-0.2, 0.04, 0.21, 0.4})));
}

TEST_P(MulOpTest, FloatActivationRELU_N1_TO_1) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  FloatMulOpModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, ActivationFunctionType_RELU_N1_TO_1,
      {-2.0, 0.2, 0.7, 0.8}, {0.1, 0.2, 0.3, 5}, constant_tensors);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-0.2, 0.04, 0.21, 1.0})));
}

TEST_P(MulOpTest, FloatVariousInputShapes) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  const std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatMulOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE,
                      {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0},
                      {0.1, 0.2, 0.3, 0.5, 1.1, 0.1}, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-0.2, 0.04, 0.21, 0.4, 1.21, 0.2})))
        << "With shape number " << i;
  }
}

TEST_P(MulOpTest, FloatWithScalarBroadcast) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  const std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatMulOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE,
                      {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0}, {0.1}, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-0.2, 0.02, 0.07, 0.08, 0.11, 0.2})))
        << "With shape number " << i;
  }
}

TEST_P(MulOpTest, FloatWithBroadcast) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  const std::vector<std::vector<int>> test_shapes = {
      {2, 4}, {2, 1, 4}, {1, 2, 4}, {1, 2, 1, 4}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatMulOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {4}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE,
                      {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0, 1.1, 0.8},
                      {0.1, 0.2, 0.3, 0.4}, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray(ArrayFloatNear(
                    {-0.2, 0.04, 0.21, 0.32, 0.11, 0.4, 0.33, 0.32})))
        << "With shape number " << i;
  }
}

TEST_P(MulOpTest, FloatMixedBroadcast) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  const std::vector<int> base_shape = {2, 3, 1, 2};
  const std::vector<std::vector<int>> test_shapes = {
      {1, 1, 3, 2}, {1, 3, 1, 2}, {2, 1, 3, 1}, {2, 3, 1, 1}};
  const std::vector<std::vector<float>> test_outputs = {
      {-0.06f, 0.69f,  0.12f,  1.15f, -0.30f, 2.07f,  0.18f,  0.15f, -0.36f,
       0.25f,  0.90f,  0.45f,  0.16f, -0.33f, -0.32f, -0.55f, 0.80f, -0.99f,
       0.24f,  0.84f,  -0.48f, 1.40f, 1.20f,  2.52f,  -0.32f, 0.00f, 0.64f,
       0.00f,  -1.60f, 0.00f,  0.14f, -0.66f, -0.28f, -1.10f, 0.70f, -1.98f},
      {-0.06f, 0.69f, -0.36f, 0.25f, 0.80f, -0.99f, 0.24f, 0.84f, 0.64f, 0.00f,
       0.70f, -1.98f},
      {-0.06f, 0.46f,  -0.09f, 0.69f, 0.12f,  -0.92f, 0.18f,  0.10f,  0.27f,
       0.15f,  -0.36f, -0.20f, 0.16f, -0.22f, 0.24f,  -0.33f, -0.32f, 0.44f,
       0.60f,  1.40f,  1.20f,  2.80f, 1.08f,  2.52f,  -0.80f, 0.00f,  -1.60f,
       0.00f,  -1.44f, 0.00f,  0.35f, -1.10f, 0.70f,  -2.20f, 0.63f,  -1.98f},
      {-0.06f, 0.46f, 0.27f, 0.15f, -0.32f, 0.44f, 0.60f, 1.40f, -1.60f, 0.00f,
       0.63f, -1.98f}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    FloatMulOpModel model_fixture(
        {TensorType_FLOAT32, base_shape}, {TensorType_FLOAT32, test_shapes[i]},
        {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE,
        {-0.3f, 2.3f, 0.9f, 0.5f, 0.8f, -1.1f, 1.2f, 2.8f, -1.6f, 0.0f, 0.7f,
         -2.2f},
        {0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f}, constant_tensors);
    ASSERT_EQ(model_fixture.Invoke(), kTfLiteOk);

    EXPECT_THAT(model_fixture.GetOutput(),
                ElementsAreArray(ArrayFloatNear(test_outputs[i], 0.0001f)))
        << "With shape number " << i;
  }
  // Re-run with exchanged inputs.
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    FloatMulOpModel model_fixture(
        {TensorType_FLOAT32, test_shapes[i]}, {TensorType_FLOAT32, base_shape},
        {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE,
        {0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f},
        {-0.3f, 2.3f, 0.9f, 0.5f, 0.8f, -1.1f, 1.2f, 2.8f, -1.6f, 0.0f, 0.7f,
         -2.2f},
        constant_tensors);
    ASSERT_EQ(model_fixture.Invoke(), kTfLiteOk);
    EXPECT_THAT(model_fixture.GetOutput(),
                ElementsAreArray(ArrayFloatNear(test_outputs[i], 0.0001f)))
        << "With shape number " << i;
  }
}

TEST_P(MulOpTest, FloatWithBroadcast2Elements) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  const std::vector<std::vector<int>> test_shapes = {
      {2, 2}, {2, 1, 2}, {1, 2, 2}, {1, 2, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatMulOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {2}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE,
                      {-2.0, 0.2, 0.7, 0.8}, {0.1, 0.2}, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray(ArrayFloatNear({-0.2, 0.04, 0.07, 0.16})))
        << "With shape number " << i;
  }
}

TEST_P(MulOpTest, FloatScalarAndOneElement) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  FloatMulOpModel m({TensorType_FLOAT32, {1}}, {TensorType_FLOAT32, {}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE,
                    {0.8}, {0.5}, constant_tensors);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({0.4})));
}

TEST_P(MulOpTest, IntegerNoActivation) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  IntegerMulOpModel m({TensorType_INT32, {1, 2, 2, 1}},
                      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {}},
                      ActivationFunctionType_NONE, {-20, 2, 7, 8}, {1, 2, 3, 5},
                      constant_tensors);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-20, 4, 21, 40}));
}

TEST_P(MulOpTest, ComplexBaseTest) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  ComplexMulOpModel m({TensorType_COMPLEX64, {1, 2, 2, 1}},
                      {TensorType_COMPLEX64, {1, 2, 2, 1}},
                      {TensorType_COMPLEX64, {}}, ActivationFunctionType_NONE,
                      {-20, {2, 3}, {7, 2}, 8}, {1, {2, -3}, {3, -4}, 5},
                      constant_tensors);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::complex<float> expected_result[4] = {-20, 13, {29, -22}, 40};
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(expected_result));
}

TEST_P(MulOpTest, ComplexWithBroadcast) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  const std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    ComplexMulOpModel m({TensorType_COMPLEX64, test_shapes[i]},
                        {TensorType_COMPLEX64, {}}, {TensorType_COMPLEX64, {}},
                        ActivationFunctionType_NONE, {-20, 2, 7, 8, 11, 20},
                        {1}, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({-20, 2, 7, 8, 11, 20}))
        << "With shape number " << i;
  }
}

#if GTEST_HAS_DEATH_TEST
TEST(MulOpTest, IncompatibleActivation) {
  EXPECT_DEATH(ComplexMulOpModel({TensorType_COMPLEX64, {1, 2, 2, 1}},
                                 {TensorType_COMPLEX64, {1, 2, 2, 1}},
                                 {TensorType_COMPLEX64, {}},
                                 ActivationFunctionType_RELU_N1_TO_1,
                                 {1, 2, 3, 4}, {1, 2, 3, 4}, false),
               "Activation is not allowed for COMPLEX64 input");
}
#endif

TEST_P(MulOpTest, Int32ActivationRELU_N1_TO_1) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  IntegerMulOpModel m({TensorType_INT32, {1, 2, 2, 1}},
                      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {}},
                      ActivationFunctionType_RELU_N1_TO_1, {-20, 2, 7, 8},
                      {1, 2, 3, 5}, constant_tensors);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1, 1, 1, 1}));
}

TEST_P(MulOpTest, Int32VariousInputShapes) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  const std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerMulOpModel m(
        {TensorType_INT32, test_shapes[i]}, {TensorType_INT32, test_shapes[i]},
        {TensorType_INT32, {}}, ActivationFunctionType_NONE,
        {-20, 2, 7, 8, 11, 20}, {1, 2, 3, 5, 11, 1}, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({-20, 4, 21, 40, 121, 20}))
        << "With shape number " << i;
  }
}

TEST_P(MulOpTest, Int32WithBroadcast) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  const std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerMulOpModel m({TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, {}},  // always a scalar
                        {TensorType_INT32, {}}, ActivationFunctionType_NONE,
                        {-20, 2, 7, 8, 11, 20}, {1}, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray(ArrayFloatNear({-20, 2, 7, 8, 11, 20})))
        << "With shape number " << i;
  }
}

template <TensorType tensor_type, typename integer_dtype>
void NoActivation(bool constant_tensors) {
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  QuantizedMulOpModel<integer_dtype, integer_dtype> m(
      {tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
      {tensor_type, {1, 2, 2, 1}, -1.0, 1.0}, {tensor_type, {}, -1.0, 1.0},
      ActivationFunctionType_NONE, {-0.8, 0.2, 0.9, 0.7}, {0.6, 0.4, 0.9, 0.8},
      constant_tensors);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({-0.48, 0.08, 0.81, 0.56},
                                              kQuantizedTolerance)));
}

template <TensorType tensor_type, typename integer_dtype>
void NoActivationLargeMultiplier(bool constant_tensors) {
  // Intentionally pathological output range much narrower than needed
  // to represent input values to exercise the multiplier>1 case.
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  QuantizedMulOpModel<integer_dtype, integer_dtype> m(
      {tensor_type, {1, 2, 2, 1}, -100, 100},
      {tensor_type, {1, 2, 2, 1}, -100, 100}, {tensor_type, {}, -10, 10},
      ActivationFunctionType_NONE, {-4, 2, 3, 1}, {-1, -3, 4, 2},
      constant_tensors);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  // Note the large tolerance. This computation is inherently inaccurate.
  const float kTolerance = 1.4f;
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({4, -6, 10, 2}, kTolerance)));
}

TEST_P(MulOpTest, NoActivationUInt8) {
  bool constant_tensors = GetParam();
  NoActivation<TensorType_UINT8, uint8_t>(constant_tensors);
  NoActivationLargeMultiplier<TensorType_UINT8, uint8_t>(constant_tensors);
}

TEST_P(MulOpTest, NoActivationInt8) {
  bool constant_tensors = GetParam();
  NoActivation<TensorType_INT8, int8_t>(constant_tensors);
  NoActivationLargeMultiplier<TensorType_INT8, int8_t>(constant_tensors);
}

TEST_P(MulOpTest, NoActivationInt16) {
  bool constant_tensors = GetParam();
  const float kMin = -1.f;
  const float kMax = 32767.f / 32768.f;
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  QuantizedMulOpModel<int16_t, int16_t> m(
      {TensorType_INT16, {1, 2, 2, 1}, kMin, kMax},
      {TensorType_INT16, {1, 2, 2, 1}, kMin, kMax},
      {TensorType_INT16, {}, kMin, kMax}, ActivationFunctionType_NONE,
      {-0.8, 0.2, 0.9, 0.7}, {0.6, 0.4, 0.9, 0.8}, constant_tensors);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({-0.48, 0.08, 0.81, 0.56},
                                              kQuantizedToleranceInt16)));
}

TEST_P(MulOpTest, NoActivationInt16Scaled) {
  bool constant_tensors = GetParam();
  const float kMin = -2.f;
  const float kMax = 2.f * 32767.f / 32768.f;
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  QuantizedMulOpModel<int16_t, int16_t> m(
      {TensorType_INT16, {1, 2, 3, 1}, kMin, kMax},
      {TensorType_INT16, {1, 2, 3, 1}, 2 * kMin, 2 * kMax},
      {TensorType_INT16, {}, 8 * kMin, 8 * kMax}, ActivationFunctionType_NONE,
      {-1.8, 0.2, 0.9, 1.7, 0.1, -1.95}, {3.6, -3.4, 3.9, 0.8, -1.0, -3.95},
      constant_tensors);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const float kQuantizedToleranceInt16Scaled =
      6.0 * kQuantizedStepInt16 + kQuantizedStepInt16 * kQuantizedStepInt16;

  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({-6.48, -0.68, 3.51, 1.36, -0.1, 7.7025},
                                      kQuantizedToleranceInt16Scaled)));
}

template <TensorType tensor_type, typename integer_dtype>
void NoActivationInt16With8BitOutput(bool constant_tensors) {
  const float kMinInt16 = -1.f;
  const float kMaxInt16 = 32767.f / 32768.f;
  const float kMinUint8 = -1.f;
  const float kMaxUint8 = 127.f / 128.f;
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  QuantizedMulOpModel<int16_t, integer_dtype> m(
      {TensorType_INT16, {1, 2, 2, 1}, kMinInt16, kMaxInt16},
      {TensorType_INT16, {1, 2, 2, 1}, kMinInt16, kMaxInt16},
      {tensor_type, {}, kMinUint8, kMaxUint8}, ActivationFunctionType_NONE,
      {-0.8, 0.2, 0.9, 0.7}, {0.6, 0.4, 0.9, 0.8}, constant_tensors);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({-0.48, 0.08, 0.81, 0.56},
                                              kQuantizedTolerance)));
}

TEST_P(MulOpTest, NoActivationInt16WithUint8Output) {
  bool constant_tensors = GetParam();
  NoActivationInt16With8BitOutput<TensorType_UINT8, uint8_t>(constant_tensors);
}

TEST_P(MulOpTest, NoActivationInt16Withint8Output) {
  bool constant_tensors = GetParam();
  NoActivationInt16With8BitOutput<TensorType_INT8, int8_t>(constant_tensors);
}

// for quantized Mul, the error shouldn't exceed 2*step
float GetTolerance(int min, int max) {
  float kQuantizedStep = (max - min) / 255.0;
  float kQuantizedTolerance = 2.0 * kQuantizedStep;
  return kQuantizedTolerance;
}

template <TensorType tensor_type, typename integer_dtype>
void WithBroadcast(bool constant_tensors) {
  const float kQuantizedTolerance = GetTolerance(-3.0, 3.0);
  const std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  // Test with a smaller than 1 and greater than 1 quantization multiplier
  const std::vector<std::pair<float, float>> test_input_range = {{-3.0, 3.0},
                                                                 {-6.0, 6.0}};
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  for (int i = 0; i < test_shapes.size(); ++i) {
    for (int j = 0; j < test_input_range.size(); ++j) {
      const std::pair<float, float>& input_range = test_input_range[j];
      QuantizedMulOpModel<integer_dtype, integer_dtype> m(
          {tensor_type, test_shapes[i], input_range.first, input_range.second},
          {tensor_type, {}, input_range.first, input_range.second},
          {tensor_type, {}, -0.2, 0.2}, ActivationFunctionType_NONE,
          {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0}, {0.1}, constant_tensors);
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(
          m.GetDequantizedOutput(),
          ElementsAreArray(ArrayFloatNear({-0.2, 0.02, 0.07, 0.08, 0.11, 0.2},
                                          kQuantizedTolerance)))
          << "With shape number " << i << " and range number " << j;
    }
  }
}

template <enum TensorType tensor_type, typename integer_dtype>
void QuantizedWithMixedBroadcast(bool constant_tensors) {
  const float kQuantizedTolerance = GetTolerance(-3.f, 3.f);
  const std::vector<int> base_shape = {2, 3, 1, 2};
  const std::vector<std::vector<int>> test_shapes = {
      {1, 1, 3, 2}, {1, 3, 1, 2}, {2, 1, 3, 1}, {2, 3, 1, 1}};
  const std::vector<std::vector<float>> test_outputs = {
      {-0.06f, 0.69f,  0.12f,  1.15f, -0.30f, 2.07f,  0.18f,  0.15f, -0.36f,
       0.25f,  0.90f,  0.45f,  0.16f, -0.33f, -0.32f, -0.55f, 0.80f, -0.99f,
       0.24f,  0.84f,  -0.48f, 1.40f, 1.20f,  2.52f,  -0.32f, 0.00f, 0.64f,
       0.00f,  -1.60f, 0.00f,  0.14f, -0.66f, -0.28f, -1.10f, 0.70f, -1.98f},
      {-0.06f, 0.69f, -0.36f, 0.25f, 0.80f, -0.99f, 0.24f, 0.84f, 0.64f, 0.00f,
       0.70f, -1.98f},
      {-0.06f, 0.46f,  -0.09f, 0.69f, 0.12f,  -0.92f, 0.18f,  0.10f,  0.27f,
       0.15f,  -0.36f, -0.20f, 0.16f, -0.22f, 0.24f,  -0.33f, -0.32f, 0.44f,
       0.60f,  1.40f,  1.20f,  2.80f, 1.08f,  2.52f,  -0.80f, 0.00f,  -1.60f,
       0.00f,  -1.44f, 0.00f,  0.35f, -1.10f, 0.70f,  -2.20f, 0.63f,  -1.98f},
      {-0.06f, 0.46f, 0.27f, 0.15f, -0.32f, 0.44f, 0.60f, 1.40f, -1.60f, 0.00f,
       0.63f, -1.98f}};
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    QuantizedMulOpModel<integer_dtype, integer_dtype> model_fixture(
        {tensor_type, base_shape, -3.f, 3.f},
        {tensor_type, test_shapes[i], -3.f, 3.f}, {tensor_type, {}, -3.f, 3.f},
        ActivationFunctionType_NONE,
        {-0.3f, 2.3f, 0.9f, 0.5f, 0.8f, -1.1f, 1.2f, 2.8f, -1.6f, 0.0f, 0.7f,
         -2.2f},
        {0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f}, constant_tensors);
    ASSERT_EQ(model_fixture.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        model_fixture.GetDequantizedOutput(),
        ElementsAreArray(ArrayFloatNear(test_outputs[i], kQuantizedTolerance)))
        << "With shape number " << i;
  }
  // Re-run with exchanged inputs.
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    QuantizedMulOpModel<integer_dtype, integer_dtype> model_fixture(
        {tensor_type, test_shapes[i], -3.f, 3.f},
        {tensor_type, base_shape, -3.f, 3.f}, {tensor_type, {}, -3.f, 3.f},
        ActivationFunctionType_NONE, {0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f},
        {-0.3f, 2.3f, 0.9f, 0.5f, 0.8f, -1.1f, 1.2f, 2.8f, -1.6f, 0.0f, 0.7f,
         -2.2f},
        constant_tensors);
    ASSERT_EQ(model_fixture.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        model_fixture.GetDequantizedOutput(),
        ElementsAreArray(ArrayFloatNear(test_outputs[i], kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST_P(MulOpTest, WithBroadcastUInt8) {
  bool constant_tensors = GetParam();
  WithBroadcast<TensorType_UINT8, uint8_t>(constant_tensors);
}

TEST_P(MulOpTest, WithBroadcastInt8) {
  bool constant_tensors = GetParam();
  WithBroadcast<TensorType_INT8, int8_t>(constant_tensors);
}

TEST_P(MulOpTest, QuantizedWithMixedBroadcastUInt8) {
  bool constant_tensors = GetParam();
  QuantizedWithMixedBroadcast<TensorType_UINT8, uint8_t>(constant_tensors);
}

TEST_P(MulOpTest, QuantizedWithMixedBroadcastInt8) {
  bool constant_tensors = GetParam();
  QuantizedWithMixedBroadcast<TensorType_INT8, int8_t>(constant_tensors);
}

INSTANTIATE_TEST_SUITE_P(ConstantInputs, MulOpTest, testing::Bool());

}  // namespace
}  // namespace tflite
