/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <array>
#include <complex>
#include <functional>
#include <numeric>
#include <random>
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
    SetBypassDefaultDelegates();
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
    if (!constant_tensors) {
      PopulateTensor<QuantizedType>(input1_, input1_data);
      PopulateTensor<QuantizedType>(input2_, input2_data);
    }
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  std::vector<InputType> GetOutput() {
    return ExtractVector<InputType>(output_);
  }

 protected:
  int input1_;
  int input2_;
  int output_;
};

class FloatMulOpModel : public BaseMulOpModel<float> {
 public:
  using BaseMulOpModel::BaseMulOpModel;
};

class ComplexMulOpModel : public BaseMulOpModel<std::complex<float>> {
 public:
  using BaseMulOpModel::BaseMulOpModel;
};

template <typename InputType>
class IntegerMulOpModel : public BaseMulOpModel<InputType> {
 public:
  using BaseMulOpModel<InputType>::BaseMulOpModel;
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

TEST(MulOpTest, NoActivationFloatInplaceInput0) {
  FloatMulOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE,
                    {-2.0, 0.2, 0.7, 0.8}, {0.1, 0.2, 0.3, 0.5}, false);
  const int kInplaceInputTensorIdx = 0;
  const int kInplaceOutputTensorIdx = 0;
  const TfLiteTensor* input_tensor = m.GetInputTensor(kInplaceInputTensorIdx);
  TfLiteTensor* output_tensor = m.GetOutputTensor(kInplaceOutputTensorIdx);
  output_tensor->data.data = input_tensor->data.data;
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-0.2, 0.04, 0.21, 0.4})));
  EXPECT_EQ(output_tensor->data.data, input_tensor->data.data);
}

TEST(MulOpTest, NoActivationFloatInplaceInput1) {
  FloatMulOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE,
                    {-2.0, 0.2, 0.7, 0.8}, {0.1, 0.2, 0.3, 0.5}, false);
  const int kInplaceInputTensorIdx = 1;
  const int kInplaceOutputTensorIdx = 0;
  const TfLiteTensor* input_tensor = m.GetInputTensor(kInplaceInputTensorIdx);
  TfLiteTensor* output_tensor = m.GetOutputTensor(kInplaceOutputTensorIdx);
  output_tensor->data.data = input_tensor->data.data;
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-0.2, 0.04, 0.21, 0.4})));
  EXPECT_EQ(output_tensor->data.data, input_tensor->data.data);
}

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
  IntegerMulOpModel<int32_t> m(
      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {1, 2, 2, 1}},
      {TensorType_INT32, {}}, ActivationFunctionType_NONE, {-20, 2, 7, 8},
      {1, 2, 3, 5}, constant_tensors);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-20, 4, 21, 40}));
}

TEST_P(MulOpTest, Int16ActivationRELU_N1_TO_1) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  IntegerMulOpModel<int16_t> m(
      {TensorType_INT16, {1, 2, 2, 1}}, {TensorType_INT16, {1, 2, 2, 1}},
      {TensorType_INT16, {}}, ActivationFunctionType_RELU_N1_TO_1,
      {-20, 2, 7, 8}, {1, 2, 3, 5}, constant_tensors);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1, 1, 1, 1}));
}

TEST_P(MulOpTest, Int16VariousInputShapes) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  const std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerMulOpModel<int16_t> m(
        {TensorType_INT16, test_shapes[i]}, {TensorType_INT16, test_shapes[i]},
        {TensorType_INT16, {}}, ActivationFunctionType_NONE,
        {-20, 2, 7, 8, 11, 20}, {1, 2, 3, 5, 11, 1}, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({-20, 4, 21, 40, 121, 20}))
        << "With shape number " << i;
  }
}

TEST_P(MulOpTest, Int16WithBroadcast) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  const std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerMulOpModel<int16_t> m({TensorType_INT16, test_shapes[i]},
                                 {TensorType_INT16, {}},  // always a scalar
                                 {TensorType_INT16, {}},
                                 ActivationFunctionType_NONE,
                                 {-20, 2, 7, 8, 11, 20}, {1}, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({-20, 2, 7, 8, 11, 20}))
        << "With shape number " << i;
  }
}

TEST_P(MulOpTest, 16BitIntegerNoActivation) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  IntegerMulOpModel<int16_t> m({TensorType_INT16, {4}}, {TensorType_INT16, {4}},
                               {TensorType_INT16, {}},
                               ActivationFunctionType_NONE, {-20, 2, 7, 8},
                               {1, 2, 3, 5}, constant_tensors);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-20, 4, 21, 40}));
}

TEST_P(MulOpTest, 32BitUnsignedIntegerNoActivation) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  IntegerMulOpModel<uint32_t> m(
      {TensorType_UINT32, {1, 2, 2, 1}}, {TensorType_UINT32, {1, 2, 2, 1}},
      {TensorType_UINT32, {}}, ActivationFunctionType_NONE, {20, 2, 7, 8},
      {1, 2, 3, 5}, constant_tensors);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({20, 4, 21, 40}));
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
  IntegerMulOpModel<int32_t> m(
      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {1, 2, 2, 1}},
      {TensorType_INT32, {}}, ActivationFunctionType_RELU_N1_TO_1,
      {-20, 2, 7, 8}, {1, 2, 3, 5}, constant_tensors);
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
    IntegerMulOpModel<int32_t> m(
        {TensorType_INT32, test_shapes[i]}, {TensorType_INT32, test_shapes[i]},
        {TensorType_INT32, {}}, ActivationFunctionType_NONE,
        {-20, 2, 7, 8, 11, 20}, {1, 2, 3, 5, 11, 1}, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({-20, 4, 21, 40, 121, 20}))
        << "With shape number " << i;
  }
}

// Neon intrinsics are only dispatched when tensor has at least 16 elements.
TEST_P(MulOpTest, Int32LargeInputShapeNoActivation) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  const std::vector<int> test_shape = {4, 4, 4, 4};
  constexpr int kFlatSize = 4 * 4 * 4 * 4;

  std::vector<int> lhs_data(kFlatSize);
  std::iota(lhs_data.begin(), lhs_data.end(), 0);

  std::vector<int> rhs_data(kFlatSize);
  std::iota(rhs_data.begin(), rhs_data.end(), 0);

  IntegerMulOpModel<int32_t> m(
      {TensorType_INT32, test_shape}, {TensorType_INT32, test_shape},
      {TensorType_INT32, {}}, ActivationFunctionType_NONE, lhs_data, rhs_data,
      constant_tensors);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const std::vector<int> output = m.GetOutput();
  ASSERT_EQ(output.size(), kFlatSize);
  for (int i = 0; i < kFlatSize; ++i) {
    EXPECT_EQ(output[i], i * i);
  }
}

// Neon intrinsics are only dispatched when tensor has at least 16 elements.
TEST_P(MulOpTest, Int32LargeInputShapeRELU6) {
  bool constant_tensors = GetParam();
  if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
    // NNAPI does not support graphs with all constant inputs.
    return;
  }
  const std::vector<int> test_shape = {4, 4, 4, 4};
  constexpr int kFlatSize = 4 * 4 * 4 * 4;

  std::vector<int> lhs_data(kFlatSize);
  std::iota(lhs_data.begin(), lhs_data.end(), 0);

  std::vector<int> rhs_data(kFlatSize);
  std::iota(rhs_data.begin(), rhs_data.end(), 0);

  IntegerMulOpModel<int32_t> m(
      {TensorType_INT32, test_shape}, {TensorType_INT32, test_shape},
      {TensorType_INT32, {}}, ActivationFunctionType_RELU6, lhs_data, rhs_data,
      constant_tensors);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const std::vector<int> output = m.GetOutput();
  ASSERT_EQ(output.size(), kFlatSize);
  for (int i = 0; i < kFlatSize; ++i) {
    EXPECT_EQ(output[i], std::min(i * i, 6));
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
    IntegerMulOpModel<int32_t> m({TensorType_INT32, test_shapes[i]},
                                 {TensorType_INT32, {}},  // always a scalar
                                 {TensorType_INT32, {}},
                                 ActivationFunctionType_NONE,
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

constexpr int kDim1 = 2;
constexpr int kDim2 = 3;
constexpr int kDim3 = 4;
constexpr int kDim4 = 5;
constexpr int kDim5 = 6;
constexpr int kDim6 = 7;

constexpr int kMaxMulBroadcastDim = 6;

void TestFloatBroadcast(std::vector<int> input1_shape,
                        std::vector<int> input2_shape) {
  std::array<int, kMaxMulBroadcastDim> input1_dims;
  std::array<int, kMaxMulBroadcastDim> input2_dims;
  std::array<int, kMaxMulBroadcastDim> output_dims;
  std::array<int, kMaxMulBroadcastDim> input1_strides;
  std::array<int, kMaxMulBroadcastDim> input2_strides;
  std::array<int, kMaxMulBroadcastDim> output_strides;
  std::fill(input1_dims.begin(), input1_dims.end(), 1);
  std::fill(input2_dims.begin(), input2_dims.end(), 1);
  std::fill(output_dims.begin(), output_dims.end(), 1);
  std::copy(input1_shape.cbegin(), input1_shape.cend(),
            input1_dims.end() - input1_shape.size());
  std::copy(input2_shape.cbegin(), input2_shape.cend(),
            input2_dims.end() - input2_shape.size());

  for (size_t i = 0; i < kMaxMulBroadcastDim; i++) {
    if (input1_dims[i] != 1 && input2_dims[i] != 1) {
      ASSERT_EQ(input1_dims[i], input2_dims[i]);
    }
    output_dims[i] = std::max(input1_dims[i], input2_dims[i]);
  }
  // Compute generalized strides.
  size_t input1_stride = 1, input2_stride = 1, output_stride = 1;
  for (size_t i = kMaxMulBroadcastDim; i != 0; i--) {
    input1_strides[i - 1] = input1_dims[i - 1] == 1 ? 0 : input1_stride;
    input2_strides[i - 1] = input2_dims[i - 1] == 1 ? 0 : input2_stride;
    output_strides[i - 1] = output_stride;
    input1_stride *= input1_dims[i - 1];
    input2_stride *= input2_dims[i - 1];
    output_stride *= output_dims[i - 1];
  }
  const int num_input1_elements = std::accumulate(
      input1_dims.begin(), input1_dims.end(), 1, std::multiplies<int>());
  const int num_input2_elements = std::accumulate(
      input2_dims.begin(), input2_dims.end(), 1, std::multiplies<int>());
  const int num_output_elements = std::accumulate(
      output_dims.begin(), output_dims.end(), 1, std::multiplies<int>());
  std::vector<float> input1(num_input1_elements);
  std::vector<float> input2(num_input2_elements);
  std::vector<float> output_ref(num_output_elements);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_real_distribution<float> f32dist(0.01f, 1.0f);

  std::generate(input1.begin(), input1.end(), [&]() { return f32dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]() { return f32dist(rng); });

  // Compute reference results.
  for (size_t i = 0; i < output_dims[0]; i++) {
    for (size_t j = 0; j < output_dims[1]; j++) {
      for (size_t k = 0; k < output_dims[2]; k++) {
        for (size_t l = 0; l < output_dims[3]; l++) {
          for (size_t m = 0; m < output_dims[4]; m++) {
            for (size_t n = 0; n < output_dims[5]; n++) {
              output_ref[i * output_strides[0] + j * output_strides[1] +
                         k * output_strides[2] + l * output_strides[3] +
                         m * output_strides[4] + n * output_strides[5]] =
                  input1[i * input1_strides[0] + j * input1_strides[1] +
                         k * input1_strides[2] + l * input1_strides[3] +
                         m * input1_strides[4] + n * input1_strides[5]] *
                  input2[i * input2_strides[0] + j * input2_strides[1] +
                         k * input2_strides[2] + l * input2_strides[3] +
                         m * input2_strides[4] + n * input2_strides[5]];
            }
          }
        }
      }
    }
  }

  FloatMulOpModel m({TensorType_FLOAT32, input1_shape},
                    {TensorType_FLOAT32, input2_shape},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE,
                    input1, input2, /*constant_tensors=*/false);
  m.PopulateTensor<float>(m.input1(), input1);
  m.PopulateTensor<float>(m.input2(), input2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), testing::ContainerEq(output_ref));
}

template <typename IntegerType>
void TestIntegerBroadcast(std::vector<int> input1_shape,
                          std::vector<int> input2_shape) {
  std::array<int, kMaxMulBroadcastDim> input1_dims;
  std::array<int, kMaxMulBroadcastDim> input2_dims;
  std::array<int, kMaxMulBroadcastDim> output_dims;
  std::array<int, kMaxMulBroadcastDim> input1_strides;
  std::array<int, kMaxMulBroadcastDim> input2_strides;
  std::array<int, kMaxMulBroadcastDim> output_strides;
  std::fill(input1_dims.begin(), input1_dims.end(), 1);
  std::fill(input2_dims.begin(), input2_dims.end(), 1);
  std::fill(output_dims.begin(), output_dims.end(), 1);
  std::copy(input1_shape.cbegin(), input1_shape.cend(),
            input1_dims.end() - input1_shape.size());
  std::copy(input2_shape.cbegin(), input2_shape.cend(),
            input2_dims.end() - input2_shape.size());

  for (size_t i = 0; i < kMaxMulBroadcastDim; i++) {
    if (input1_dims[i] != 1 && input2_dims[i] != 1) {
      ASSERT_EQ(input1_dims[i], input2_dims[i]);
    }
    output_dims[i] = std::max(input1_dims[i], input2_dims[i]);
  }
  // Compute generalized strides.
  size_t input1_stride = 1, input2_stride = 1, output_stride = 1;
  for (size_t i = kMaxMulBroadcastDim; i != 0; i--) {
    input1_strides[i - 1] = input1_dims[i - 1] == 1 ? 0 : input1_stride;
    input2_strides[i - 1] = input2_dims[i - 1] == 1 ? 0 : input2_stride;
    output_strides[i - 1] = output_stride;
    input1_stride *= input1_dims[i - 1];
    input2_stride *= input2_dims[i - 1];
    output_stride *= output_dims[i - 1];
  }
  const int num_input1_elements = std::accumulate(
      input1_dims.begin(), input1_dims.end(), 1, std::multiplies<int>());
  const int num_input2_elements = std::accumulate(
      input2_dims.begin(), input2_dims.end(), 1, std::multiplies<int>());
  const int num_output_elements = std::accumulate(
      output_dims.begin(), output_dims.end(), 1, std::multiplies<int>());
  std::vector<IntegerType> input1(num_input1_elements);
  std::vector<IntegerType> input2(num_input2_elements);
  std::vector<IntegerType> output_ref(num_output_elements);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_int_distribution<IntegerType> dist(0, 256);

  std::generate(input1.begin(), input1.end(), [&]() { return dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]() { return dist(rng); });

  // Compute reference results.
  for (size_t i = 0; i < output_dims[0]; i++) {
    for (size_t j = 0; j < output_dims[1]; j++) {
      for (size_t k = 0; k < output_dims[2]; k++) {
        for (size_t l = 0; l < output_dims[3]; l++) {
          for (size_t m = 0; m < output_dims[4]; m++) {
            for (size_t n = 0; n < output_dims[5]; n++) {
              output_ref[i * output_strides[0] + j * output_strides[1] +
                         k * output_strides[2] + l * output_strides[3] +
                         m * output_strides[4] + n * output_strides[5]] =
                  input1[i * input1_strides[0] + j * input1_strides[1] +
                         k * input1_strides[2] + l * input1_strides[3] +
                         m * input1_strides[4] + n * input1_strides[5]] *
                  input2[i * input2_strides[0] + j * input2_strides[1] +
                         k * input2_strides[2] + l * input2_strides[3] +
                         m * input2_strides[4] + n * input2_strides[5]];
            }
          }
        }
      }
    }
  }

  IntegerMulOpModel<IntegerType> m({GetTensorType<IntegerType>(), input1_shape},
                                   {GetTensorType<IntegerType>(), input2_shape},
                                   {GetTensorType<IntegerType>(), {}},
                                   ActivationFunctionType_NONE, input1, input2,
                                   /*constant_tensors=*/false);
  m.template PopulateTensor<IntegerType>(m.input1(), input1);
  m.template PopulateTensor<IntegerType>(m.input2(), input2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), testing::ContainerEq(output_ref));
}

// To improve automatic test sharding (via shard_count in the BUILD file),
// we need to ensure that each individual test case runs in a reasonable time,
// otherwise we end up being limited by the performance of the longest shard.
// Since TestFloat32MultiDimBroadcast has 2^12 iterations, it takes a
// long time (over 30 seconds) to execute all iterations -- too long for a
// single shard.  So we split it into a few "subshards" and have a separate
// TYPED_TEST macro invocation for each subshard.

void TestFloat32MultiDimBroadcast(int selected_subshard, int subshard_count) {
  int iteration = 0;
  for (uint32_t bm1 = 0;
       bm1 < (static_cast<uint32_t>(1) << kMaxMulBroadcastDim); bm1++) {
    for (uint32_t bm2 = 0;
         bm2 < (static_cast<uint32_t>(1) << kMaxMulBroadcastDim); bm2++) {
      if (iteration++ % subshard_count != selected_subshard) {
        continue;  // This iteration of the loop is not part of this subshard.
      }
      const bool input1_broadcast_dim1 = bm1 & (static_cast<uint32_t>(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (static_cast<uint32_t>(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (static_cast<uint32_t>(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (static_cast<uint32_t>(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (static_cast<uint32_t>(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (static_cast<uint32_t>(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (static_cast<uint32_t>(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (static_cast<uint32_t>(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (static_cast<uint32_t>(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (static_cast<uint32_t>(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (static_cast<uint32_t>(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (static_cast<uint32_t>(1) << 5);
      const int input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const int input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const int input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const int input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const int input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const int input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const int input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const int input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const int input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const int input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const int input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const int input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      std::vector<int> input1_full_shape{input1_dim1, input1_dim2, input1_dim3,
                                         input1_dim4, input1_dim5, input1_dim6};
      std::vector<int> input2_full_shape{input2_dim1, input2_dim2, input2_dim3,
                                         input2_dim4, input2_dim5, input2_dim6};
      for (int input1_dims = 1; input1_dims <= kMaxMulBroadcastDim;
           ++input1_dims) {
        for (int input2_dims = 1; input2_dims <= kMaxMulBroadcastDim;
             ++input2_dims) {
          std::vector<int> input1_shape(input1_dims), input2_shape(input2_dims);
          std::copy(input1_full_shape.end() - input1_dims,
                    input1_full_shape.end(), input1_shape.data());
          std::copy(input2_full_shape.end() - input2_dims,
                    input2_full_shape.end(), input2_shape.data());
          TestFloatBroadcast(input1_shape, input2_shape);
        }
      }
    }
  }
}

// Should match the number of TEST or TYPED_TEST invoations for each of
// Float32MultiDimBroadcastSubshard*,
// IntegerMultiDimBroadcastSubshard*,
// Int8QuantizedMultiDimBroadcastSubshard*, and
// Uint8QuantizedMultiDimBroadcastSubshard* below.
constexpr int kMultiDimBroadcastSubshardCount = 10;

TEST(FloatMulOpModel, Float32MultiDimBroadcastSubshard0) {
  TestFloat32MultiDimBroadcast(0, kMultiDimBroadcastSubshardCount);
}
TEST(FloatMulOpModel, Float32MultiDimBroadcastSubshard1) {
  TestFloat32MultiDimBroadcast(1, kMultiDimBroadcastSubshardCount);
}
TEST(FloatMulOpModel, Float32MultiDimBroadcastSubshard2) {
  TestFloat32MultiDimBroadcast(2, kMultiDimBroadcastSubshardCount);
}
TEST(FloatMulOpModel, Float32MultiDimBroadcastSubshard3) {
  TestFloat32MultiDimBroadcast(3, kMultiDimBroadcastSubshardCount);
}
TEST(FloatMulOpModel, Float32MultiDimBroadcastSubshard4) {
  TestFloat32MultiDimBroadcast(4, kMultiDimBroadcastSubshardCount);
}
TEST(FloatMulOpModel, Float32MultiDimBroadcastSubshard5) {
  TestFloat32MultiDimBroadcast(5, kMultiDimBroadcastSubshardCount);
}
TEST(FloatMulOpModel, Float32MultiDimBroadcastSubshard6) {
  TestFloat32MultiDimBroadcast(6, kMultiDimBroadcastSubshardCount);
}
TEST(FloatMulOpModel, Float32MultiDimBroadcastSubshard7) {
  TestFloat32MultiDimBroadcast(7, kMultiDimBroadcastSubshardCount);
}
TEST(FloatMulOpModel, Float32MultiDimBroadcastSubshard8) {
  TestFloat32MultiDimBroadcast(8, kMultiDimBroadcastSubshardCount);
}
TEST(FloatMulOpModel, Float32MultiDimBroadcastSubshard9) {
  TestFloat32MultiDimBroadcast(9, kMultiDimBroadcastSubshardCount);
}

template <typename T>
class IntegerMulOpTest : public ::testing::Test {};

using Int16OrInt32Or64Types = ::testing::Types<int16_t, int32_t, int64_t>;
TYPED_TEST_SUITE(IntegerMulOpTest, Int16OrInt32Or64Types);

// To improve automatic test sharding (via shard_count in the BUILD file),
// we need to ensure that each individual test case runs in a reasonable time,
// otherwise we end up being limited by the performance of the longest shard.
// Since TestIntegerMultiDimBroadcast has 2^12 iterations, it takes a
// long time (over 30 seconds) to execute all iterations -- too long for a
// single shard.  So we split it into a few "subshards" and have a separate
// TYPED_TEST macro invocation for each subshard.

template <class TypeParam>
void TestIntegerMultiDimBroadcast(int selected_subshard, int subshard_count) {
  ASSERT_LT(selected_subshard, subshard_count);
  int iteration = 0;
  for (uint32_t bm1 = 0;
       bm1 < (static_cast<uint32_t>(1) << kMaxMulBroadcastDim); bm1++) {
    for (uint32_t bm2 = 0;
         bm2 < (static_cast<uint32_t>(1) << kMaxMulBroadcastDim); bm2++) {
      if (iteration++ % subshard_count != selected_subshard) {
        continue;  // This iteration of the loop is not part of this subshard.
      }
      const bool input1_broadcast_dim1 = bm1 & (static_cast<uint32_t>(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (static_cast<uint32_t>(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (static_cast<uint32_t>(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (static_cast<uint32_t>(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (static_cast<uint32_t>(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (static_cast<uint32_t>(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (static_cast<uint32_t>(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (static_cast<uint32_t>(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (static_cast<uint32_t>(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (static_cast<uint32_t>(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (static_cast<uint32_t>(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (static_cast<uint32_t>(1) << 5);
      const int input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const int input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const int input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const int input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const int input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const int input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const int input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const int input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const int input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const int input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const int input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const int input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      std::vector<int> input1_full_shape{input1_dim1, input1_dim2, input1_dim3,
                                         input1_dim4, input1_dim5, input1_dim6};
      std::vector<int> input2_full_shape{input2_dim1, input2_dim2, input2_dim3,
                                         input2_dim4, input2_dim5, input2_dim6};
      for (int input1_dims = 1; input1_dims <= kMaxMulBroadcastDim;
           ++input1_dims) {
        for (int input2_dims = 1; input2_dims <= kMaxMulBroadcastDim;
             ++input2_dims) {
          std::vector<int> input1_shape(input1_dims), input2_shape(input2_dims);
          std::copy(input1_full_shape.end() - input1_dims,
                    input1_full_shape.end(), input1_shape.data());
          std::copy(input2_full_shape.end() - input2_dims,
                    input2_full_shape.end(), input2_shape.data());
          TestIntegerBroadcast<TypeParam>(input1_shape, input2_shape);
        }
      }
    }
  }
}

TYPED_TEST(IntegerMulOpTest, IntegerMultiDimBroadcastSubshard0) {
  TestIntegerMultiDimBroadcast<TypeParam>(0, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerMulOpTest, IntegerMultiDimBroadcastSubshard1) {
  TestIntegerMultiDimBroadcast<TypeParam>(1, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerMulOpTest, IntegerMultiDimBroadcastSubshard2) {
  TestIntegerMultiDimBroadcast<TypeParam>(2, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerMulOpTest, IntegerMultiDimBroadcastSubshard3) {
  TestIntegerMultiDimBroadcast<TypeParam>(3, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerMulOpTest, IntegerMultiDimBroadcastSubshard4) {
  TestIntegerMultiDimBroadcast<TypeParam>(4, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerMulOpTest, IntegerMultiDimBroadcastSubshard5) {
  TestIntegerMultiDimBroadcast<TypeParam>(5, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerMulOpTest, IntegerMultiDimBroadcastSubshard6) {
  TestIntegerMultiDimBroadcast<TypeParam>(6, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerMulOpTest, IntegerMultiDimBroadcastSubshard7) {
  TestIntegerMultiDimBroadcast<TypeParam>(7, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerMulOpTest, IntegerMultiDimBroadcastSubshard8) {
  TestIntegerMultiDimBroadcast<TypeParam>(8, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerMulOpTest, IntegerMultiDimBroadcastSubshard9) {
  TestIntegerMultiDimBroadcast<TypeParam>(9, kMultiDimBroadcastSubshardCount);
}

template <typename QuantizedType>
void TestQuantizedBroadcast(std::vector<int> input1_shape,
                            std::vector<int> input2_shape) {
  std::array<int, kMaxMulBroadcastDim> input1_dims;
  std::array<int, kMaxMulBroadcastDim> input2_dims;
  std::array<int, kMaxMulBroadcastDim> output_dims;
  std::array<int, kMaxMulBroadcastDim> input1_strides;
  std::array<int, kMaxMulBroadcastDim> input2_strides;
  std::array<int, kMaxMulBroadcastDim> output_strides;
  std::fill(input1_dims.begin(), input1_dims.end(), 1);
  std::fill(input2_dims.begin(), input2_dims.end(), 1);
  std::fill(output_dims.begin(), output_dims.end(), 1);
  std::copy(input1_shape.cbegin(), input1_shape.cend(),
            input1_dims.end() - input1_shape.size());
  std::copy(input2_shape.cbegin(), input2_shape.cend(),
            input2_dims.end() - input2_shape.size());

  for (size_t i = 0; i < kMaxMulBroadcastDim; i++) {
    if (input1_dims[i] != 1 && input2_dims[i] != 1) {
      ASSERT_EQ(input1_dims[i], input2_dims[i]);
    }
    output_dims[i] = std::max(input1_dims[i], input2_dims[i]);
  }
  // Compute generalized strides.
  size_t input1_stride = 1, input2_stride = 1, output_stride = 1;
  for (size_t i = kMaxMulBroadcastDim; i != 0; i--) {
    input1_strides[i - 1] = input1_dims[i - 1] == 1 ? 0 : input1_stride;
    input2_strides[i - 1] = input2_dims[i - 1] == 1 ? 0 : input2_stride;
    output_strides[i - 1] = output_stride;
    input1_stride *= input1_dims[i - 1];
    input2_stride *= input2_dims[i - 1];
    output_stride *= output_dims[i - 1];
  }
  const int num_input1_elements = std::accumulate(
      input1_dims.begin(), input1_dims.end(), 1, std::multiplies<int>());
  const int num_input2_elements = std::accumulate(
      input2_dims.begin(), input2_dims.end(), 1, std::multiplies<int>());
  const int num_output_elements = std::accumulate(
      output_dims.begin(), output_dims.end(), 1, std::multiplies<int>());
  std::vector<float> input1(num_input1_elements);
  std::vector<float> input2(num_input2_elements);
  std::vector<float> output_ref(num_output_elements);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());

  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  std::generate(input1.begin(), input1.end(), [&]() { return dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]() { return dist(rng); });

  QuantizedMulOpModel<QuantizedType, QuantizedType> m(
      {GetTensorType<QuantizedType>(), input1_shape, -0.5f, 0.5f},
      {GetTensorType<QuantizedType>(), input2_shape, -0.5f, 0.5f},
      {GetTensorType<QuantizedType>(), {}, -1.f, 1.f},
      ActivationFunctionType_NONE, input1, input2, /*constant_tensors=*/false);
  // Compute reference results.
  for (size_t i = 0; i < output_dims[0]; i++) {
    for (size_t j = 0; j < output_dims[1]; j++) {
      for (size_t k = 0; k < output_dims[2]; k++) {
        for (size_t l = 0; l < output_dims[3]; l++) {
          for (size_t m = 0; m < output_dims[4]; m++) {
            for (size_t n = 0; n < output_dims[5]; n++) {
              float x = input1[i * input1_strides[0] + j * input1_strides[1] +
                               k * input1_strides[2] + l * input1_strides[3] +
                               m * input1_strides[4] + n * input1_strides[5]];
              float y = input2[i * input2_strides[0] + j * input2_strides[1] +
                               k * input2_strides[2] + l * input2_strides[3] +
                               m * input2_strides[4] + n * input2_strides[5]];
              output_ref[i * output_strides[0] + j * output_strides[1] +
                         k * output_strides[2] + l * output_strides[3] +
                         m * output_strides[4] + n * output_strides[5]] = x * y;
            }
          }
        }
      }
    }
  }

  for (float& output_value : output_ref) {
    output_value = std::max<float>(output_value, -1.0f);
    output_value = std::min<float>(output_value, 1.0f);
  }

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<float> output = m.GetDequantizedOutput();
  for (size_t i = 0; i < output_dims[0]; i++) {
    for (size_t j = 0; j < output_dims[1]; j++) {
      for (size_t k = 0; k < output_dims[2]; k++) {
        for (size_t l = 0; l < output_dims[3]; l++) {
          for (size_t m = 0; m < output_dims[4]; m++) {
            for (size_t n = 0; n < output_dims[5]; n++) {
              const size_t index =
                  i * output_strides[0] + j * output_strides[1] +
                  k * output_strides[2] + l * output_strides[3] +
                  m * output_strides[4] + n * output_strides[5];
              EXPECT_NEAR(output[index], output_ref[index], 0.6f)
                  << "(i, j, k, l, m, n) = (" << i << ", " << j << ", " << k
                  << ", " << l << ", " << m << ", " << n << ")";
            }
          }
        }
      }
    }
  }
}

// To improve automatic test sharding (via shard_count in the BUILD file),
// we need to ensure that each individual test case runs in a reasonable time,
// otherwise we end up being limited by the performance of the longest shard.
// Since TestQuantizedMultiDimBroadcast has 2^12 iterations, it takes a
// long time (over 30 seconds) to execute all iterations -- too long for a
// single shard.  So we split it into a few "subshards" and have a separate
// TEST macro invocation for each subshard.

template <class T>
void TestQuantizedMultiDimBroadcast(int selected_subshard, int subshard_count) {
  ASSERT_LT(selected_subshard, subshard_count);
  int iteration = 0;
  for (uint32_t bm1 = 0;
       bm1 < (static_cast<uint32_t>(1) << kMaxMulBroadcastDim); bm1++) {
    for (uint32_t bm2 = 0;
         bm2 < (static_cast<int32_t>(1) << kMaxMulBroadcastDim); bm2++) {
      if (iteration++ % subshard_count != selected_subshard) {
        continue;  // This iteration of the loop is not part of this subshard.
      }
      const bool input1_broadcast_dim1 = bm1 & (static_cast<uint32_t>(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (static_cast<uint32_t>(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (static_cast<uint32_t>(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (static_cast<uint32_t>(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (static_cast<uint32_t>(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (static_cast<uint32_t>(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (static_cast<uint32_t>(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (static_cast<uint32_t>(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (static_cast<uint32_t>(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (static_cast<uint32_t>(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (static_cast<uint32_t>(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (static_cast<uint32_t>(1) << 5);
      const int input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const int input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const int input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const int input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const int input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const int input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const int input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const int input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const int input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const int input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const int input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const int input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      std::vector<int> input1_full_shape{input1_dim1, input1_dim2, input1_dim3,
                                         input1_dim4, input1_dim5, input1_dim6};
      std::vector<int> input2_full_shape{input2_dim1, input2_dim2, input2_dim3,
                                         input2_dim4, input2_dim5, input2_dim6};
      for (int input1_dims = 1; input1_dims <= kMaxMulBroadcastDim;
           ++input1_dims) {
        for (int input2_dims = 1; input2_dims <= kMaxMulBroadcastDim;
             ++input2_dims) {
          std::vector<int> input1_shape(input1_dims), input2_shape(input2_dims);
          std::copy(input1_full_shape.end() - input1_dims,
                    input1_full_shape.end(), input1_shape.data());
          std::copy(input2_full_shape.end() - input2_dims,
                    input2_full_shape.end(), input2_shape.data());
          TestQuantizedBroadcast<T>(input1_shape, input2_shape);
        }
      }
    }
  }
}

TEST(QuantizedMulOpModel, Int8QuantizedMultiDimBroadcastSubshard0) {
  TestQuantizedMultiDimBroadcast<int8_t>(0, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedMulOpModel, Int8QuantizedMultiDimBroadcastSubshard1) {
  TestQuantizedMultiDimBroadcast<int8_t>(1, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedMulOpModel, Int8QuantizedMultiDimBroadcastSubshard2) {
  TestQuantizedMultiDimBroadcast<int8_t>(2, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedMulOpModel, Int8QuantizedMultiDimBroadcastSubshard3) {
  TestQuantizedMultiDimBroadcast<int8_t>(3, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedMulOpModel, Int8QuantizedMultiDimBroadcastSubshard4) {
  TestQuantizedMultiDimBroadcast<int8_t>(4, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedMulOpModel, Int8QuantizedMultiDimBroadcastSubshard5) {
  TestQuantizedMultiDimBroadcast<int8_t>(5, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedMulOpModel, Int8QuantizedMultiDimBroadcastSubshard6) {
  TestQuantizedMultiDimBroadcast<int8_t>(6, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedMulOpModel, Int8QuantizedMultiDimBroadcastSubshard7) {
  TestQuantizedMultiDimBroadcast<int8_t>(7, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedMulOpModel, Int8QuantizedMultiDimBroadcastSubshard8) {
  TestQuantizedMultiDimBroadcast<int8_t>(8, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedMulOpModel, Int8QuantizedMultiDimBroadcastSubshard9) {
  TestQuantizedMultiDimBroadcast<int8_t>(9, kMultiDimBroadcastSubshardCount);
}

TEST(QuantizedMulOpModel, Uint8QuantizedMultiDimBroadcastSubshard0) {
  TestQuantizedMultiDimBroadcast<uint8_t>(0, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedMulOpModel, Uint8QuantizedMultiDimBroadcastSubshard1) {
  TestQuantizedMultiDimBroadcast<uint8_t>(1, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedMulOpModel, Uint8QuantizedMultiDimBroadcastSubshard2) {
  TestQuantizedMultiDimBroadcast<uint8_t>(2, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedMulOpModel, Uint8QuantizedMultiDimBroadcastSubshard3) {
  TestQuantizedMultiDimBroadcast<uint8_t>(3, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedMulOpModel, Uint8QuantizedMultiDimBroadcastSubshard4) {
  TestQuantizedMultiDimBroadcast<uint8_t>(4, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedMulOpModel, Uint8QuantizedMultiDimBroadcastSubshard5) {
  TestQuantizedMultiDimBroadcast<uint8_t>(5, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedMulOpModel, Uint8QuantizedMultiDimBroadcastSubshard6) {
  TestQuantizedMultiDimBroadcast<uint8_t>(6, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedMulOpModel, Uint8QuantizedMultiDimBroadcastSubshard7) {
  TestQuantizedMultiDimBroadcast<uint8_t>(7, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedMulOpModel, Uint8QuantizedMultiDimBroadcastSubshard8) {
  TestQuantizedMultiDimBroadcast<uint8_t>(8, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedMulOpModel, Uint8QuantizedMultiDimBroadcastSubshard9) {
  TestQuantizedMultiDimBroadcast<uint8_t>(9, kMultiDimBroadcastSubshardCount);
}

}  // namespace
}  // namespace tflite
