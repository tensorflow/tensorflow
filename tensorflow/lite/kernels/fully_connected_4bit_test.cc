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

#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/fully_connected.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {

class FullyConnected4BitOpModel : public SingleOpModel {
 public:
  FullyConnected4BitOpModel(
      int units, int batches, const TensorData& input,
      const TensorData& weights, const TensorData& output,
      std::vector<int8_t> weights_initializer, TfLiteRegistration* registration,
      ActivationFunctionType activation_func = ActivationFunctionType_RELU)
      : batches_(batches), units_(units) {
    // Calculate input_size_ from batch and input shape.
    int total_input_size = 1;
    for (size_t i = 0; i < input.shape.size(); ++i) {
      total_input_size *= input.shape[i];
    }
    input_size_ = total_input_size / batches_;
    input_ = AddInput(input);
    const std::vector<int8_t> quantized_data(weights_initializer);
    std::vector<int8_t> weight_data(quantized_data.size() / 2);
    for (int i = 0; i < quantized_data.size(); i++) {
      uint8_t val = quantized_data[i] & UINT8_C(15);
      if ((i % 2) == 0) {
        weight_data[i / 2] = val & INT8_C(15);
      } else {
        weight_data[i / 2] |= (val << 4);
      }
    }
    weights_ =
        AddConstInput<int8_t>(weights, weight_data.data(), weight_data.size());
    bias_ = AddInput({TensorType_FLOAT32, {units_}});
    output_ = AddOutput(output);
    FullyConnectedOptionsWeightsFormat weights_format =
        FullyConnectedOptionsWeightsFormat_DEFAULT;
    SetBuiltinOp(BuiltinOperator_FULLY_CONNECTED,
                 BuiltinOptions_FullyConnectedOptions,
                 CreateFullyConnectedOptions(builder_, activation_func,
                                             weights_format, true)
                     .Union());
    resolver_ = std::make_unique<SingleOpResolver>(
        BuiltinOperator_FULLY_CONNECTED, registration);
    BuildInterpreter({GetShape(input_), GetShape(weights_), GetShape(bias_)});
    SetUnitScale();
  }

  void SetUnitScale() {
    TfLiteTensor* t = interpreter_->tensor(weights_);
    t->type = kTfLiteInt4;
    t->params.scale = 1.0;
    auto filter_params =
        reinterpret_cast<TfLiteAffineQuantization*>(t->quantization.params);
    if (filter_params && filter_params->scale &&
        filter_params->scale->size > 0) {
      for (int i = 0; i < filter_params->scale->size; i++) {
        filter_params->scale->data[i] = 1.0;
      }
    }
  }
  void SetInput(const std::vector<float>& f) { PopulateTensor(input_, f); }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  void SetBias(const std::vector<float>& f) { PopulateTensor(bias_, f); }
  int input_size() { return input_size_; }
  int num_units() { return units_; }
  int num_batches() { return batches_; }

 protected:
  int input_;
  int weights_;
  int bias_;
  int output_;
  int batches_;
  int units_;
  int input_size_;
  bool use_native_int4_ = false;
};

TEST(Hybrid4BitFullyConnectedOpTest, SimpleTestHybridInt4) {
  int units = 5;
  int batches = 4;
  int cols = 40;
  FullyConnected4BitOpModel m(
      units, batches,
      /*input=*/{TensorType_FLOAT32, {batches, cols}},
      /*weights=*/{TensorType_INT4, {units, cols}, 0.0, 7.0, 1.0},
      /*output=*/{TensorType_FLOAT32, {units, batches}},
      {
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          1,  2, 3, 4, 5, 6, 7, 1, 2, -3, -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          1,  2, 3, 4, 5, 6, 7, 1, 2, -3, -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          1,  2, 3, 4, 5, 6, 7, 1, 2, -3, -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
      },
      ops::builtin::Register_FULLY_CONNECTED_GENERIC_OPT(),
      ActivationFunctionType_RELU);
  m.SetBias({1, 2, 3, 1, 2});
  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
  });
  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {393., 456., 457., 455., 394., 413., 476., 477., 475., 414.,
                   393., 456., 457., 455., 394., 393., 456., 457., 455., 394},
                  /*max_abs_error=*/1.3f)));
}

std::mt19937 random_engine(2023);
std::uniform_real_distribution<float> real_dist(0.f, 1.f);
std::uniform_int_distribution<int32_t> int_dist(-7, 7);

class Hybrid4BitFullyConnectedVsReferenceOpTests
    : public ::testing::TestWithParam<::testing::tuple<int, int, int>> {};

TEST_P(Hybrid4BitFullyConnectedVsReferenceOpTests, TestHybridInt4) {
  auto params = GetParam();
  int units = std::get<0>(params);
  int batches = std::get<1>(params);
  int cols = std::get<2>(params);
  std::vector<int8_t> weight_data(units * cols, 0);
  std::vector<float> input_data(batches * cols, 0);
  std::vector<float> bias_data(units, 0);
  for (int i = 0; i < units * cols; ++i) {
    weight_data[i] = int_dist(random_engine);
  }
  for (int i = 0; i < batches * cols; ++i) {
    input_data[i] = real_dist(random_engine);
  }
  for (int i = 0; i < units; ++i) {
    bias_data[i] = real_dist(random_engine);
  }
  FullyConnected4BitOpModel test(
      units, batches,
      /*input=*/{TensorType_FLOAT32, {batches, cols}},
      /*weights=*/{TensorType_INT4, {units, cols}, 0.0, 7.0, 1.0},
      /*output=*/{TensorType_FLOAT32, {units, batches}}, weight_data,
      ops::builtin::Register_FULLY_CONNECTED_GENERIC_OPT(),
      ActivationFunctionType_RELU);
  test.SetBias(bias_data);
  test.SetInput(input_data);
  test.Invoke();
  std::vector<float> test_data = test.GetOutput();
  FullyConnected4BitOpModel expected(
      units, batches,
      /*input=*/{TensorType_FLOAT32, {batches, cols}},
      /*weights=*/{TensorType_INT4, {units, cols}, 0.0, 7.0, 1.0},
      /*output=*/{TensorType_FLOAT32, {units, batches}}, weight_data,
      ops::builtin::Register_FULLY_CONNECTED_REF(),
      ActivationFunctionType_RELU);
  expected.SetBias(bias_data);
  expected.SetInput(input_data);
  expected.Invoke();
  std::vector<float> expected_data = expected.GetOutput();
  EXPECT_THAT(test_data, ElementsAreArray(ArrayFloatNear(
                             expected_data, /*max_abs_error=*/1e-3f)));
}

INSTANTIATE_TEST_SUITE_P(Hybrid4BitFullyConnectedVsReferenceOpTests,
                         Hybrid4BitFullyConnectedVsReferenceOpTests,
                         ::testing::ValuesIn({
                             std::make_tuple(4, 1, 32),
                             std::make_tuple(4, 1, 64),
                             std::make_tuple(5, 1, 128),
                             std::make_tuple(5, 4, 128),
                             std::make_tuple(5, 6, 128),
                             std::make_tuple(5, 1, 38),
                             std::make_tuple(5, 4, 72),
                             std::make_tuple(5, 6, 130),
                         }));
}  // namespace tflite
