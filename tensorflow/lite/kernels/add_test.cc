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

#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseAddOpModel : public SingleOpModel {
 public:
  BaseAddOpModel(const TensorData& input1, const TensorData& input2,
                 const TensorData& output,
                 ActivationFunctionType activation_type) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_, activation_type).Union());
    SetBypassDefaultDelegates();
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  BaseAddOpModel(TensorType type, const std::vector<int>& input1_shape,
                 const std::vector<int>& input2_shape,
                 ActivationFunctionType activation_type) {
    input1_ = AddInput(type);
    input2_ = AddInput(type);
    output_ = AddOutput(type);
    SetBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_, activation_type).Union());
    SetBypassDefaultDelegates();
    BuildInterpreter({input1_shape, input2_shape});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

 protected:
  int input1_;
  int input2_;
  int output_;
};

class FloatAddOpModel : public BaseAddOpModel {
 public:
  using BaseAddOpModel::BaseAddOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class IntegerAddOpModel : public BaseAddOpModel {
 public:
  using BaseAddOpModel::BaseAddOpModel;

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
};

class QuantizedAddOpModel : public BaseAddOpModel {
 public:
  QuantizedAddOpModel(TensorData input1, TensorData input2, TensorData output,
                      ActivationFunctionType activation_type)
      : BaseAddOpModel(SymmetricInt16Scaling(std::move(input1)),
                       SymmetricInt16Scaling(std::move(input2)),
                       SymmetricInt16Scaling(std::move(output)),
                       activation_type) {}

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_dtype>(ExtractVector<integer_dtype>(output_),
                                     GetScale(output_), GetZeroPoint(output_));
  }

  std::vector<float> GetDequantizedOutputInt16() {
    return Dequantize<int16_t>(ExtractVector<int16_t>(output_),
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

// for quantized Add, the error shouldn't exceed step
template <typename T>
float GetTolerance(float min, float max) {
  float kQuantizedStep =
      2.0 * (max - min) /
      (std::numeric_limits<T>::max() - std::numeric_limits<T>::min());
  return kQuantizedStep;
}

TEST(FloatAddOpModel, NoActivation) {
  FloatAddOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1.9, 0.4, 1.0, 1.3}));
}

TEST(FloatAddOpModel, ActivationRELU_N1_TO_1) {
  FloatAddOpModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1.0, 0.4, 1.0, 1.0}));
}

TEST(FloatAddOpModel, VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    FloatAddOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5, 1.1, 0.1});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray({-1.9, 0.4, 1.0, 1.3, 2.2, 2.1}))
        << "With shape number " << i;
  }
}

TEST(FloatAddOpModel, WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    FloatAddOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.1});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-1.9, 0.3, 0.8, 0.9, 1.2, 2.1})))
        << "With shape number " << i;
  }
}

TEST(FloatAddOpModel, WithBroadcastGeneric) {
  std::vector<int> test_shape1 = {1, 3, 1};
  std::vector<int> test_shape2 = {2, 1, 2};
  FloatAddOpModel m({TensorType_FLOAT32, test_shape1},
                    {TensorType_FLOAT32, test_shape2}, {TensorType_FLOAT32, {}},
                    ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {0.1, 0.2, 0.3});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({0.2, 0.3, 0.3, 0.4, 0.4, 0.5,
                                               0.4, 0.5, 0.5, 0.6, 0.6, 0.7})));
}

TEST(FloatAddOpModel, MixedBroadcast) {
  const std::vector<int> base_shape = {2, 3, 1, 2};
  std::vector<std::vector<int>> test_shapes = {
      {1, 1, 3, 2}, {1, 3, 1, 2}, {2, 1, 3, 1}, {2, 3, 1, 1}};
  std::vector<std::vector<float>> test_outputs = {
      {-0.1f, 2.6f,  -0.7f, 2.8f, 0.7f,  3.2f, 1.1f,  0.8f, 0.5f,
       1.0f,  1.9f,  1.4f,  1.0f, -0.8f, 0.4f, -0.6f, 1.8f, -0.2f,
       1.4f,  3.1f,  0.8f,  3.3f, 2.2f,  3.7f, -1.4f, 0.3f, -2.0f,
       0.5f,  -0.6f, 0.9f,  0.9f, -1.9f, 0.3f, -1.7f, 1.7f, -1.3f},
      {-0.1f, 2.6f, 0.5f, 1.0f, 1.8f, -0.2f, 1.4f, 3.1f, -2.0f, 0.5f, 1.7f,
       -1.3f},
      {-0.1f, 2.5f,  0.0f, 2.6f, -0.7f, 1.9f, 1.1f,  0.7f, 1.2f,
       0.8f,  0.5f,  0.1f, 1.0f, -0.9f, 1.1f, -0.8f, 0.4f, -1.5f,
       1.7f,  3.3f,  2.2f, 3.8f, 2.1f,  3.7f, -1.1f, 0.5f, -0.6f,
       1.0f,  -0.7f, 0.9f, 1.2f, -1.7f, 1.7f, -1.2f, 1.6f, -1.3f},
      {-0.1f, 2.5f, 1.2f, 0.8f, 0.4f, -1.5f, 1.7f, 3.3f, -0.6f, 1.0f, 1.6f,
       -1.3f}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    FloatAddOpModel model_fixture(
        {TensorType_FLOAT32, base_shape}, {TensorType_FLOAT32, test_shapes[i]},
        {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    model_fixture.PopulateTensor<float>(
        model_fixture.input1(), {-0.3f, 2.3f, 0.9f, 0.5f, 0.8f, -1.1f, 1.2f,
                                 2.8f, -1.6f, 0.0f, 0.7f, -2.2f});
    model_fixture.PopulateTensor<float>(model_fixture.input2(),
                                        {0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f});
    ASSERT_EQ(model_fixture.Invoke(), kTfLiteOk);
    EXPECT_THAT(model_fixture.GetOutput(),
                ElementsAreArray(ArrayFloatNear(test_outputs[i], 0.0001f)))
        << "With shape number " << i;
  }
  // Re-run with exchanged inputs.
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    FloatAddOpModel model_fixture(
        {TensorType_FLOAT32, test_shapes[i]}, {TensorType_FLOAT32, base_shape},
        {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    model_fixture.PopulateTensor<float>(model_fixture.input1(),
                                        {0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f});
    model_fixture.PopulateTensor<float>(
        model_fixture.input2(), {-0.3f, 2.3f, 0.9f, 0.5f, 0.8f, -1.1f, 1.2f,
                                 2.8f, -1.6f, 0.0f, 0.7f, -2.2f});
    ASSERT_EQ(model_fixture.Invoke(), kTfLiteOk);
    EXPECT_THAT(model_fixture.GetOutput(),
                ElementsAreArray(ArrayFloatNear(test_outputs[i], 0.0001f)))
        << "With shape number " << i;
  }
}

constexpr int kDim1 = 2;
constexpr int kDim2 = 3;
constexpr int kDim3 = 4;
constexpr int kDim4 = 5;
constexpr int kDim5 = 6;
constexpr int kDim6 = 7;

void TestFloatBroadcast(std::vector<int> input1_shape,
                        std::vector<int> input2_shape) {
  std::array<int, 6> input1_dims;
  std::array<int, 6> input2_dims;
  std::array<int, 6> output_dims;
  std::array<int, 6> input1_strides;
  std::array<int, 6> input2_strides;
  std::array<int, 6> output_strides;
  std::fill(input1_dims.begin(), input1_dims.end(), 1);
  std::fill(input2_dims.begin(), input2_dims.end(), 1);
  std::fill(output_dims.begin(), output_dims.end(), 1);
  std::copy(input1_shape.cbegin(), input1_shape.cend(),
            input1_dims.end() - input1_shape.size());
  std::copy(input2_shape.cbegin(), input2_shape.cend(),
            input2_dims.end() - input2_shape.size());

  for (size_t i = 0; i < 6; i++) {
    if (input1_dims[i] != 1 && input2_dims[i] != 1) {
      ASSERT_EQ(input1_dims[i], input2_dims[i]);
    }
    output_dims[i] = std::max(input1_dims[i], input2_dims[i]);
  }
  // Compute generalized strides.
  size_t input1_stride = 1, input2_stride = 1, output_stride = 1;
  for (size_t i = 6; i != 0; i--) {
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
                         m * input1_strides[4] + n * input1_strides[5]] +
                  input2[i * input2_strides[0] + j * input2_strides[1] +
                         k * input2_strides[2] + l * input2_strides[3] +
                         m * input2_strides[4] + n * input2_strides[5]];
            }
          }
        }
      }
    }
  }

  FloatAddOpModel m({TensorType_FLOAT32, input1_shape},
                    {TensorType_FLOAT32, input2_shape},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), input1);
  m.PopulateTensor<float>(m.input2(), input2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), testing::ContainerEq(output_ref));
}

template <typename IntegerType>
void TestIntegerBroadcast(std::vector<int> input1_shape,
                          std::vector<int> input2_shape) {
  std::array<int, 6> input1_dims;
  std::array<int, 6> input2_dims;
  std::array<int, 6> output_dims;
  std::array<int, 6> input1_strides;
  std::array<int, 6> input2_strides;
  std::array<int, 6> output_strides;
  std::fill(input1_dims.begin(), input1_dims.end(), 1);
  std::fill(input2_dims.begin(), input2_dims.end(), 1);
  std::fill(output_dims.begin(), output_dims.end(), 1);
  std::copy(input1_shape.cbegin(), input1_shape.cend(),
            input1_dims.end() - input1_shape.size());
  std::copy(input2_shape.cbegin(), input2_shape.cend(),
            input2_dims.end() - input2_shape.size());

  for (size_t i = 0; i < 6; i++) {
    if (input1_dims[i] != 1 && input2_dims[i] != 1) {
      ASSERT_EQ(input1_dims[i], input2_dims[i]);
    }
    output_dims[i] = std::max(input1_dims[i], input2_dims[i]);
  }
  // Compute generalized strides.
  size_t input1_stride = 1, input2_stride = 1, output_stride = 1;
  for (size_t i = 6; i != 0; i--) {
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
                         m * input1_strides[4] + n * input1_strides[5]] +
                  input2[i * input2_strides[0] + j * input2_strides[1] +
                         k * input2_strides[2] + l * input2_strides[3] +
                         m * input2_strides[4] + n * input2_strides[5]];
            }
          }
        }
      }
    }
  }

  IntegerAddOpModel m({GetTensorType<IntegerType>(), input1_shape},
                      {GetTensorType<IntegerType>(), input2_shape},
                      {GetTensorType<IntegerType>(), {}},
                      ActivationFunctionType_NONE);
  m.PopulateTensor<IntegerType>(m.input1(), input1);
  m.PopulateTensor<IntegerType>(m.input2(), input2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<IntegerType>(), testing::ContainerEq(output_ref));
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
  for (uint32_t bm1 = 0; bm1 < (static_cast<uint32_t>(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (static_cast<uint32_t>(1) << 6); bm2++) {
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
      for (int input1_dims = 1; input1_dims <= 6; ++input1_dims) {
        for (int input2_dims = 1; input2_dims <= 6; ++input2_dims) {
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

TEST(FloatAddOpModel, Float32MultiDimBroadcastSubshard0) {
  TestFloat32MultiDimBroadcast(0, kMultiDimBroadcastSubshardCount);
}
TEST(FloatAddOpModel, Float32MultiDimBroadcastSubshard1) {
  TestFloat32MultiDimBroadcast(1, kMultiDimBroadcastSubshardCount);
}
TEST(FloatAddOpModel, Float32MultiDimBroadcastSubshard2) {
  TestFloat32MultiDimBroadcast(2, kMultiDimBroadcastSubshardCount);
}
TEST(FloatAddOpModel, Float32MultiDimBroadcastSubshard3) {
  TestFloat32MultiDimBroadcast(3, kMultiDimBroadcastSubshardCount);
}
TEST(FloatAddOpModel, Float32MultiDimBroadcastSubshard4) {
  TestFloat32MultiDimBroadcast(4, kMultiDimBroadcastSubshardCount);
}
TEST(FloatAddOpModel, Float32MultiDimBroadcastSubshard5) {
  TestFloat32MultiDimBroadcast(5, kMultiDimBroadcastSubshardCount);
}
TEST(FloatAddOpModel, Float32MultiDimBroadcastSubshard6) {
  TestFloat32MultiDimBroadcast(6, kMultiDimBroadcastSubshardCount);
}
TEST(FloatAddOpModel, Float32MultiDimBroadcastSubshard7) {
  TestFloat32MultiDimBroadcast(7, kMultiDimBroadcastSubshardCount);
}
TEST(FloatAddOpModel, Float32MultiDimBroadcastSubshard8) {
  TestFloat32MultiDimBroadcast(8, kMultiDimBroadcastSubshardCount);
}
TEST(FloatAddOpModel, Float32MultiDimBroadcastSubshard9) {
  TestFloat32MultiDimBroadcast(9, kMultiDimBroadcastSubshardCount);
}

template <typename T>
class IntegerAddOpTest : public ::testing::Test {};

using Int16OrInt32Or64Types = ::testing::Types<int16_t, int32_t, int64_t>;
TYPED_TEST_SUITE(IntegerAddOpTest, Int16OrInt32Or64Types);

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
  for (uint32_t bm1 = 0; bm1 < (static_cast<uint32_t>(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (static_cast<uint32_t>(1) << 6); bm2++) {
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
      for (int input1_dims = 1; input1_dims <= 6; ++input1_dims) {
        for (int input2_dims = 1; input2_dims <= 6; ++input2_dims) {
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

TYPED_TEST(IntegerAddOpTest, IntegerMultiDimBroadcastSubshard0) {
  TestIntegerMultiDimBroadcast<TypeParam>(0, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerAddOpTest, IntegerMultiDimBroadcastSubshard1) {
  TestIntegerMultiDimBroadcast<TypeParam>(1, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerAddOpTest, IntegerMultiDimBroadcastSubshard2) {
  TestIntegerMultiDimBroadcast<TypeParam>(2, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerAddOpTest, IntegerMultiDimBroadcastSubshard3) {
  TestIntegerMultiDimBroadcast<TypeParam>(3, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerAddOpTest, IntegerMultiDimBroadcastSubshard4) {
  TestIntegerMultiDimBroadcast<TypeParam>(4, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerAddOpTest, IntegerMultiDimBroadcastSubshard5) {
  TestIntegerMultiDimBroadcast<TypeParam>(5, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerAddOpTest, IntegerMultiDimBroadcastSubshard6) {
  TestIntegerMultiDimBroadcast<TypeParam>(6, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerAddOpTest, IntegerMultiDimBroadcastSubshard7) {
  TestIntegerMultiDimBroadcast<TypeParam>(7, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerAddOpTest, IntegerMultiDimBroadcastSubshard8) {
  TestIntegerMultiDimBroadcast<TypeParam>(8, kMultiDimBroadcastSubshardCount);
}
TYPED_TEST(IntegerAddOpTest, IntegerMultiDimBroadcastSubshard9) {
  TestIntegerMultiDimBroadcast<TypeParam>(9, kMultiDimBroadcastSubshardCount);
}

TYPED_TEST(IntegerAddOpTest, NoActivation) {
  IntegerAddOpModel m(GetTensorType<TypeParam>(), {1, 2, 2, 1}, {1, 2, 2, 1},
                      ActivationFunctionType_NONE);
  m.PopulateTensor<TypeParam>(m.input1(), {-20, 2, 7, 8});
  m.PopulateTensor<TypeParam>(m.input2(), {1, 2, 3, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<TypeParam>(), ElementsAreArray({-19, 4, 10, 13}));
}

TYPED_TEST(IntegerAddOpTest, ActivationRELU_N1_TO_1) {
  IntegerAddOpModel m(GetTensorType<TypeParam>(), {1, 2, 2, 1}, {1, 2, 2, 1},
                      ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<TypeParam>(m.input1(), {-20, 2, 7, 8});
  m.PopulateTensor<TypeParam>(m.input2(), {1, 2, 3, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<TypeParam>(), ElementsAreArray({-1, 1, 1, 1}));
}

TYPED_TEST(IntegerAddOpTest, VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    IntegerAddOpModel m(GetTensorType<TypeParam>(), test_shapes[i],
                        test_shapes[i], ActivationFunctionType_NONE);
    m.PopulateTensor<TypeParam>(m.input1(), {-20, 2, 7, 8, 11, 20});
    m.PopulateTensor<TypeParam>(m.input2(), {1, 2, 3, 5, 11, 1});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput<TypeParam>(),
                ElementsAreArray({-19, 04, 10, 13, 22, 21}))
        << "With shape number " << i;
  }
}

TYPED_TEST(IntegerAddOpTest, WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    IntegerAddOpModel m(GetTensorType<TypeParam>(), test_shapes[i],
                        {},  // always a scalar
                        ActivationFunctionType_NONE);
    m.PopulateTensor<TypeParam>(m.input1(), {-20, 2, 7, 8, 11, 20});
    m.PopulateTensor<TypeParam>(m.input2(), {1});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput<TypeParam>(),
                ElementsAreArray({-19, 3, 8, 9, 12, 21}))
        << "With shape number " << i;
  }
}

TYPED_TEST(IntegerAddOpTest, Int32MultiDimBroadcast) {
  IntegerAddOpModel m(GetTensorType<TypeParam>(), {1, 2}, {2, 1},
                      ActivationFunctionType_NONE);
  m.PopulateTensor<TypeParam>(m.input1(), {3, 5});
  m.PopulateTensor<TypeParam>(m.input2(), {1, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<TypeParam>(), ElementsAreArray({4, 6, 7, 9}));
}

template <typename QuantizedType>
void TestQuantizedBroadcast(std::vector<int> input1_shape,
                            std::vector<int> input2_shape) {
  std::array<int, 6> input1_dims;
  std::array<int, 6> input2_dims;
  std::array<int, 6> output_dims;
  std::array<int, 6> input1_strides;
  std::array<int, 6> input2_strides;
  std::array<int, 6> output_strides;
  std::fill(input1_dims.begin(), input1_dims.end(), 1);
  std::fill(input2_dims.begin(), input2_dims.end(), 1);
  std::fill(output_dims.begin(), output_dims.end(), 1);
  std::copy(input1_shape.cbegin(), input1_shape.cend(),
            input1_dims.end() - input1_shape.size());
  std::copy(input2_shape.cbegin(), input2_shape.cend(),
            input2_dims.end() - input2_shape.size());

  for (size_t i = 0; i < 6; i++) {
    if (input1_dims[i] != 1 && input2_dims[i] != 1) {
      ASSERT_EQ(input1_dims[i], input2_dims[i]);
    }
    output_dims[i] = std::max(input1_dims[i], input2_dims[i]);
  }
  // Compute generalized strides.
  size_t input1_stride = 1, input2_stride = 1, output_stride = 1;
  for (size_t i = 6; i != 0; i--) {
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

  QuantizedAddOpModel m(
      {GetTensorType<QuantizedType>(), input1_shape, -0.5f, 0.5f},
      {GetTensorType<QuantizedType>(), input2_shape, -0.5f, 0.5f},
      {GetTensorType<QuantizedType>(), {}, -1.f, 1.f},
      ActivationFunctionType_NONE);
  m.QuantizeAndPopulate<QuantizedType>(m.input1(), input1);
  m.QuantizeAndPopulate<QuantizedType>(m.input2(), input2);
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
                         m * output_strides[4] + n * output_strides[5]] = x + y;
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
  std::vector<float> output = m.GetDequantizedOutput<QuantizedType>();
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
  for (uint32_t bm1 = 0; bm1 < (static_cast<uint32_t>(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (static_cast<int32_t>(1) << 6); bm2++) {
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
      for (int input1_dims = 1; input1_dims <= 6; ++input1_dims) {
        for (int input2_dims = 1; input2_dims <= 6; ++input2_dims) {
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

TEST(QuantizedAddOpModel, Int8QuantizedMultiDimBroadcastSubshard0) {
  TestQuantizedMultiDimBroadcast<int8_t>(0, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedAddOpModel, Int8QuantizedMultiDimBroadcastSubshard1) {
  TestQuantizedMultiDimBroadcast<int8_t>(1, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedAddOpModel, Int8QuantizedMultiDimBroadcastSubshard2) {
  TestQuantizedMultiDimBroadcast<int8_t>(2, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedAddOpModel, Int8QuantizedMultiDimBroadcastSubshard3) {
  TestQuantizedMultiDimBroadcast<int8_t>(3, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedAddOpModel, Int8QuantizedMultiDimBroadcastSubshard4) {
  TestQuantizedMultiDimBroadcast<int8_t>(4, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedAddOpModel, Int8QuantizedMultiDimBroadcastSubshard5) {
  TestQuantizedMultiDimBroadcast<int8_t>(5, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedAddOpModel, Int8QuantizedMultiDimBroadcastSubshard6) {
  TestQuantizedMultiDimBroadcast<int8_t>(6, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedAddOpModel, Int8QuantizedMultiDimBroadcastSubshard7) {
  TestQuantizedMultiDimBroadcast<int8_t>(7, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedAddOpModel, Int8QuantizedMultiDimBroadcastSubshard8) {
  TestQuantizedMultiDimBroadcast<int8_t>(8, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedAddOpModel, Int8QuantizedMultiDimBroadcastSubshard9) {
  TestQuantizedMultiDimBroadcast<int8_t>(9, kMultiDimBroadcastSubshardCount);
}

TEST(QuantizedAddOpModel, Uint8QuantizedMultiDimBroadcastSubshard0) {
  TestQuantizedMultiDimBroadcast<uint8_t>(0, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedAddOpModel, Uint8QuantizedMultiDimBroadcastSubshard1) {
  TestQuantizedMultiDimBroadcast<uint8_t>(1, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedAddOpModel, Uint8QuantizedMultiDimBroadcastSubshard2) {
  TestQuantizedMultiDimBroadcast<uint8_t>(2, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedAddOpModel, Uint8QuantizedMultiDimBroadcastSubshard3) {
  TestQuantizedMultiDimBroadcast<uint8_t>(3, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedAddOpModel, Uint8QuantizedMultiDimBroadcastSubshard4) {
  TestQuantizedMultiDimBroadcast<uint8_t>(4, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedAddOpModel, Uint8QuantizedMultiDimBroadcastSubshard5) {
  TestQuantizedMultiDimBroadcast<uint8_t>(5, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedAddOpModel, Uint8QuantizedMultiDimBroadcastSubshard6) {
  TestQuantizedMultiDimBroadcast<uint8_t>(6, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedAddOpModel, Uint8QuantizedMultiDimBroadcastSubshard7) {
  TestQuantizedMultiDimBroadcast<uint8_t>(7, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedAddOpModel, Uint8QuantizedMultiDimBroadcastSubshard8) {
  TestQuantizedMultiDimBroadcast<uint8_t>(8, kMultiDimBroadcastSubshardCount);
}
TEST(QuantizedAddOpModel, Uint8QuantizedMultiDimBroadcastSubshard9) {
  TestQuantizedMultiDimBroadcast<uint8_t>(9, kMultiDimBroadcastSubshardCount);
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedTestsNoActivation() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-1.0, 1.0);
  std::vector<std::vector<float>> inputs1 = {
      {0.1, 0.2, 0.3, 0.4}, {-0.8, 0.2, 0.4, 0.7}, {-0.8, 0.2, 0.7, 0.3}};
  std::vector<std::vector<float>> inputs2 = {
      {0.6, 0.4, 0.3, 0.1}, {0.6, 0.4, 0.5, -0.8}, {0.6, 0.4, -0.8, 0.5}};
  std::vector<std::vector<float>> results = {
      {0.7, 0.6, 0.6, 0.5}, {-0.2, 0.6, 0.9, -0.1}, {-0.2, 0.6, -0.1, 0.8}};
  for (size_t i = 0; i < inputs1.size(); ++i) {
    QuantizedAddOpModel m({tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                          {tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                          {tensor_type, {}, -1.0, 1.0},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<integer_dtype>(m.input1(), inputs1[i]);
    m.QuantizeAndPopulate<integer_dtype>(m.input2(), inputs2[i]);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        m.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear(results[i], kQuantizedTolerance)))
        << "With test number " << i;
  }
}

TEST(QuantizedAddOpModel, QuantizedTestsNoActivationUInt8) {
  QuantizedTestsNoActivation<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedAddOpModel, QuantizedTestsNoActivationInt8) {
  QuantizedTestsNoActivation<TensorType_INT8, int8_t>();
}

TEST(QuantizedAddOpModel, QuantizedTestsNoActivationInt16) {
  float kQuantizedTolerance = GetTolerance<int16_t>(-1.0, 1.0);
  std::vector<std::vector<float>> inputs1 = {{0.1, 0.2, 0.3, 0.4, 0.9, 0.7},
                                             {-0.8, 0.2, 0.4, 0.7, 0.1, 0.0},
                                             {-0.8, 0.2, 0.7, 0.3, 0.9, 0.1}};
  std::vector<std::vector<float>> inputs2 = {{0.6, 0.4, 0.3, 0.1, -0.1, 0.3},
                                             {0.6, 0.4, 0.5, -0.8, 0.0, -1.0},
                                             {0.6, 0.4, -0.8, 0.5, -0.9, 0.1}};
  std::vector<std::vector<float>> results = {{0.7, 0.6, 0.6, 0.5, 0.8, 1.0},
                                             {-0.2, 0.6, 0.9, -0.1, 0.1, -1.0},
                                             {-0.2, 0.6, -0.1, 0.8, 0.0, 0.2}};
  for (size_t i = 0; i < inputs1.size(); ++i) {
    QuantizedAddOpModel m({TensorType_INT16, {1, 2, 3, 1}, -1.0, 1.0},
                          {TensorType_INT16, {1, 2, 3, 1}, -1.0, 1.0},
                          {TensorType_INT16, {}, -1.0, 1.0},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<int16_t>(m.input1(), inputs1[i]);
    m.QuantizeAndPopulate<int16_t>(m.input2(), inputs2[i]);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        m.GetDequantizedOutputInt16(),
        ElementsAreArray(ArrayFloatNear(results[i], kQuantizedTolerance)))
        << "With test number " << i;
  }
}

template <enum TensorType tensor_type, typename integer_dtype>
void QuantizedTestsActivationRELU_N1_TO_1() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-1.0, 1.0);
  std::vector<std::vector<float>> inputs1 = {{-0.8, 0.2, 0.9, 0.7},
                                             {-0.8, 0.2, 0.7, 0.3}};
  std::vector<std::vector<float>> inputs2 = {{0.6, 0.4, 0.9, -0.8},
                                             {0.6, 0.4, -0.8, 0.5}};
  std::vector<std::vector<float>> results = {{-0.2, 0.6, 1.0, -0.1},
                                             {-0.2, 0.6, -0.1, 0.8}};
  for (size_t i = 0; i < inputs1.size(); ++i) {
    QuantizedAddOpModel m({tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                          {tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                          {tensor_type, {}, -1.0, 1.0},
                          ActivationFunctionType_RELU_N1_TO_1);
    m.QuantizeAndPopulate<integer_dtype>(m.input1(), inputs1[i]);
    m.QuantizeAndPopulate<integer_dtype>(m.input2(), inputs2[i]);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        m.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear(results[i], kQuantizedTolerance)))
        << "With test number " << i;
  }
}

TEST(QuantizedAddOpModel, QuantizedTestsActivationRELU_N1_TO_1UInt8) {
  QuantizedTestsActivationRELU_N1_TO_1<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedAddOpModel, QuantizedTestsActivationRELU_N1_TO_1Int8) {
  QuantizedTestsActivationRELU_N1_TO_1<TensorType_INT8, int8_t>();
}

template <enum TensorType tensor_type, typename integer_dtype>
void QuantizedVariousInputShapes() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-3.0, 3.0);
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    QuantizedAddOpModel m({tensor_type, test_shapes[i], -3.0, 3.0},
                          {tensor_type, test_shapes[i], -3.0, 3.0},
                          {tensor_type, {}, -3.0, 3.0},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<integer_dtype>(m.input1(),
                                         {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.QuantizeAndPopulate<integer_dtype>(m.input2(),
                                         {0.1, 0.3, 0.3, 0.5, 1.1, 0.1});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
                ElementsAreArray(ArrayFloatNear({-1.9, 0.5, 1.0, 1.3, 2.2, 2.1},
                                                kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedAddOpModel, QuantizedVariousInputShapesUInt8) {
  QuantizedVariousInputShapes<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedAddOpModel, QuantizedVariousInputShapesInt8) {
  QuantizedVariousInputShapes<TensorType_INT8, int8_t>();
}

template <enum TensorType tensor_type, typename integer_dtype>
void QuantizedWithScalarBroadcast() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-3.f, 3.f);
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    QuantizedAddOpModel model_fixture(
        {tensor_type, test_shapes[i], -3.f, 3.f}, {tensor_type, {}, -3.f, 3.f},
        {tensor_type, {}, -3.f, 3.f}, ActivationFunctionType_NONE);
    model_fixture.QuantizeAndPopulate<integer_dtype>(
        model_fixture.input1(), {-2.0f, 0.2f, 0.7f, 0.8f, 1.1f, 2.0f});
    model_fixture.QuantizeAndPopulate<integer_dtype>(model_fixture.input2(),
                                                     {0.1f});
    ASSERT_EQ(model_fixture.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        model_fixture.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear({-1.9f, 0.3f, 0.8f, 0.9f, 1.2f, 2.1f},
                                        kQuantizedTolerance)))
        << "With shape number " << i;
  }
  // Re-run with exchanged inputs.
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    QuantizedAddOpModel model_fixture(
        {tensor_type, {}, -3.f, 3.f}, {tensor_type, test_shapes[i], -3.f, 3.f},
        {tensor_type, {}, -3.f, 3.f}, ActivationFunctionType_NONE);
    model_fixture.QuantizeAndPopulate<integer_dtype>(model_fixture.input1(),
                                                     {0.1f});
    model_fixture.QuantizeAndPopulate<integer_dtype>(
        model_fixture.input2(), {-2.0f, 0.2f, 0.7f, 0.8f, 1.1f, 2.0f});
    ASSERT_EQ(model_fixture.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        model_fixture.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear({-1.9f, 0.3f, 0.8f, 0.9f, 1.2f, 2.1f},
                                        kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedAddOpModel, QuantizedWithScalarBroadcastUInt8) {
  QuantizedWithScalarBroadcast<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedAddOpModel, QuantizedWithScalarBroadcastInt8) {
  QuantizedWithScalarBroadcast<TensorType_INT8, int8_t>();
}

TEST(QuantizedAddOpModel, QuantizedWithScalarBroadcastInt16) {
  QuantizedWithScalarBroadcast<TensorType_INT16, int16_t>();
}

template <enum TensorType tensor_type, typename integer_dtype>
void QuantizedWithMixedBroadcast() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-3.f, 3.f);
  const std::vector<int> base_shape = {2, 3, 1, 2};
  std::vector<std::vector<int>> test_shapes = {
      {1, 1, 3, 2}, {1, 3, 1, 2}, {2, 1, 3, 1}, {2, 3, 1, 1}};
  std::vector<std::vector<float>> test_outputs = {
      {-0.1f, 2.6f,  -0.7f, 2.8f, 0.7f,  3.0f, 1.1f,  0.8f, 0.5f,
       1.0f,  1.9f,  1.4f,  1.0f, -0.8f, 0.4f, -0.6f, 1.8f, -0.2f,
       1.4f,  3.0f,  0.8f,  3.0f, 2.2f,  3.0f, -1.4f, 0.3f, -2.0f,
       0.5f,  -0.6f, 0.9f,  0.9f, -1.9f, 0.3f, -1.7f, 1.7f, -1.3f},
      {-0.1f, 2.6f, 0.5f, 1.0f, 1.8f, -0.2f, 1.4f, 3.0f, -2.0f, 0.5f, 1.7f,
       -1.3f},
      {-0.1f, 2.5f,  0.0f, 2.6f, -0.7f, 1.9f, 1.1f,  0.7f, 1.2f,
       0.8f,  0.5f,  0.1f, 1.0f, -0.9f, 1.1f, -0.8f, 0.4f, -1.5f,
       1.7f,  3.0f,  2.2f, 3.0f, 2.1f,  3.0f, -1.1f, 0.5f, -0.6f,
       1.0f,  -0.7f, 0.9f, 1.2f, -1.7f, 1.7f, -1.2f, 1.6f, -1.3f},
      {-0.1f, 2.5f, 1.2f, 0.8f, 0.4f, -1.5f, 1.7f, 3.0f, -0.6f, 1.0f, 1.6f,
       -1.3f}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    QuantizedAddOpModel model_fixture({tensor_type, base_shape, -3.f, 3.f},
                                      {tensor_type, test_shapes[i], -3.f, 3.f},
                                      {tensor_type, {}, -3.f, 3.f},
                                      ActivationFunctionType_NONE);
    model_fixture.QuantizeAndPopulate<integer_dtype>(
        model_fixture.input1(), {-0.3f, 2.3f, 0.9f, 0.5f, 0.8f, -1.1f, 1.2f,
                                 2.8f, -1.6f, 0.0f, 0.7f, -2.2f});
    model_fixture.QuantizeAndPopulate<integer_dtype>(
        model_fixture.input2(), {0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f});
    ASSERT_EQ(model_fixture.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        model_fixture.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear(test_outputs[i], kQuantizedTolerance)))
        << "With shape number " << i;
  }
  // Re-run with exchanged inputs.
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    QuantizedAddOpModel model_fixture({tensor_type, test_shapes[i], -3.f, 3.f},
                                      {tensor_type, base_shape, -3.f, 3.f},
                                      {tensor_type, {}, -3.f, 3.f},
                                      ActivationFunctionType_NONE);
    model_fixture.QuantizeAndPopulate<integer_dtype>(
        model_fixture.input1(), {0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f});
    model_fixture.QuantizeAndPopulate<integer_dtype>(
        model_fixture.input2(), {-0.3f, 2.3f, 0.9f, 0.5f, 0.8f, -1.1f, 1.2f,
                                 2.8f, -1.6f, 0.0f, 0.7f, -2.2f});
    ASSERT_EQ(model_fixture.Invoke(), kTfLiteOk);
    EXPECT_THAT(
        model_fixture.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear(test_outputs[i], kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedAddOpModel, QuantizedWithMixedBroadcastUInt8) {
  QuantizedWithMixedBroadcast<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedAddOpModel, QuantizedWithMixedBroadcastInt8) {
  QuantizedWithMixedBroadcast<TensorType_INT8, int8_t>();
}

TEST(QuantizedAddOpModel, QuantizedWithMixedBroadcastInt16) {
  QuantizedWithMixedBroadcast<TensorType_INT16, int16_t>();
}

template <enum TensorType tensor_type, typename integer_dtype>
void QuantizedWithGenericBroadcast() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-3.0, 3.0);
  std::vector<int> test_shape1 = {1, 3, 1};
  std::vector<int> test_shape2 = {2, 1, 2};
  QuantizedAddOpModel m({tensor_type, test_shape1, -3.0, 3.0},
                        {tensor_type, test_shape2, -3.0, 3.0},
                        {tensor_type, {}, -3.0, 3.0},
                        ActivationFunctionType_NONE);
  m.QuantizeAndPopulate<integer_dtype>(m.input1(), {0.1, 0.2, 0.3});
  m.QuantizeAndPopulate<integer_dtype>(m.input2(), {0.1, -0.2, 0.3, -0.4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear({0.2, -0.1, 0.3, 0., 0.4, 0.1,
                                               0.4, -0.3, 0.5, -0.2, 0.6, -0.1},
                                              kQuantizedTolerance)));
}

TEST(QuantizedAddOpModel, QuantizedWithGenericBroadcastUInt8) {
  QuantizedWithGenericBroadcast<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedAddOpModel, QuantizedWithGenericdBroadcastInt8) {
  QuantizedWithGenericBroadcast<TensorType_INT8, int8_t>();
}

TEST(QuantizedAddOpModel, QuantizedWithGenericdBroadcastInt16) {
  QuantizedWithGenericBroadcast<TensorType_INT16, int16_t>();
}

}  // namespace
}  // namespace tflite
