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

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/delegates/xnnpack/batch_matrix_multiply_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

class BatchMatrixMultiplyTest : public testing::Test {
 public:
  // std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
  auto get_delegate(int num_threads = 1) {
    TfLiteXNNPackDelegateOptions delegate_options =
        TfLiteXNNPackDelegateOptionsDefault();
    delegate_options.flags |=
        TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
    delegate_options.num_threads = num_threads;
    return std::unique_ptr<TfLiteDelegate,
                           decltype(&TfLiteXNNPackDelegateDelete)>(
        TfLiteXNNPackDelegateCreate(&delegate_options),
        TfLiteXNNPackDelegateDelete);
  }

  int32_t shape_rng() {
    return std::uniform_int_distribution<int32_t>(2, 5)(rng_);
  }
  int32_t channels_rng() {
    return std::uniform_int_distribution<int32_t>(2, 9)(rng_);
  }

 private:
  std::random_device random_device_;
  std::mt19937 rng_ = std::mt19937(random_device_());
};

TEST_F(BatchMatrixMultiplyTest, 3D) {
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .InputADims({batch, height, input_channels})
      .InputBDims({batch, input_channels, output_channels})
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, DynamicallyQuantizedPerChannelWeights2D) {
  const auto height = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .InputADims({height, input_channels})
      .InputBDims({input_channels, output_channels})
      .InputBQuant(BatchMatrixMultiplyTester::kChannel)
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest,
       DynamicallyQuantizedPerChannelWeights2DTransposeB) {
  const auto height = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .InputADims({height, input_channels})
      .InputBDims({output_channels, input_channels})
      .InputBQuant(BatchMatrixMultiplyTester::kChannel)
      .TransposeB(true)
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, DynamicallyQuantizedPerTensorWeights3D) {
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .InputADims({batch, height, input_channels})
      .InputBDims({batch, input_channels, output_channels})
      .InputBQuant(BatchMatrixMultiplyTester::kTensor)
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest,
       DynamicallyQuantizedPerTensorWeights3DTransposeB) {
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .InputADims({batch, height, input_channels})
      .InputBDims({batch, output_channels, input_channels})
      .InputBQuant(BatchMatrixMultiplyTester::kTensor)
      .TransposeB(true)
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, DynamicallyQuantizedPerChannelWeights3D) {
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .InputADims({batch, height, input_channels})
      .InputBDims({batch, input_channels, output_channels})
      .InputBQuant(BatchMatrixMultiplyTester::kChannel)
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest,
       DynamicallyQuantizedPerChannelWeights3DTransposeB) {
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .InputADims({batch, height, input_channels})
      .InputBDims({batch, output_channels, input_channels})
      .InputBQuant(BatchMatrixMultiplyTester::kChannel)
      .TransposeB(true)
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, BroadcastOne3D) {
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .InputADims({batch, height, input_channels})
      .InputBDims({1, input_channels, output_channels})
      .Test(xnnpack_delegate.get());

  BatchMatrixMultiplyTester()
      .InputADims({1, height, input_channels})
      .InputBDims({batch, input_channels, output_channels})
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, BroadcastImplicit3D) {
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .InputADims({batch, height, input_channels})
      .InputBDims({input_channels, output_channels})
      .Test(xnnpack_delegate.get());

  BatchMatrixMultiplyTester()
      .InputADims({height, input_channels})
      .InputBDims({batch, input_channels, output_channels})
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, 4D) {
  const auto outer_batch = shape_rng();
  const auto inner_batch = shape_rng();
  const auto height = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .InputADims({outer_batch, inner_batch, height, input_channels})
      .InputBDims({outer_batch, inner_batch, input_channels, output_channels})
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, BroadcastOne4D) {
  const auto outer_batch = shape_rng();
  const auto inner_batch = shape_rng();
  const auto height = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .InputADims({1, inner_batch, height, input_channels})
      .InputBDims({outer_batch, inner_batch, input_channels, output_channels})
      .Test(xnnpack_delegate.get());
  BatchMatrixMultiplyTester()
      .InputADims({outer_batch, inner_batch, height, input_channels})
      .InputBDims({1, inner_batch, input_channels, output_channels})
      .Test(xnnpack_delegate.get());
  BatchMatrixMultiplyTester()
      .InputADims({outer_batch, 1, height, input_channels})
      .InputBDims({outer_batch, inner_batch, input_channels, output_channels})
      .Test(xnnpack_delegate.get());
  BatchMatrixMultiplyTester()
      .InputADims({outer_batch, inner_batch, height, input_channels})
      .InputBDims({outer_batch, 1, input_channels, output_channels})
      .Test(xnnpack_delegate.get());
  BatchMatrixMultiplyTester()
      .InputADims({1, 1, height, input_channels})
      .InputBDims({outer_batch, inner_batch, input_channels, output_channels})
      .Test(xnnpack_delegate.get());
  BatchMatrixMultiplyTester()
      .InputADims({outer_batch, inner_batch, height, input_channels})
      .InputBDims({1, 1, input_channels, output_channels})
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, BroadcastImplicit4D) {
  const auto outer_batch = shape_rng();
  const auto inner_batch = shape_rng();
  const auto height = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .InputADims({inner_batch, height, input_channels})
      .InputBDims({outer_batch, inner_batch, input_channels, output_channels})
      .Test(xnnpack_delegate.get());
  BatchMatrixMultiplyTester()
      .InputADims({outer_batch, inner_batch, height, input_channels})
      .InputBDims({inner_batch, input_channels, output_channels})
      .Test(xnnpack_delegate.get());
  BatchMatrixMultiplyTester()
      .InputADims({height, input_channels})
      .InputBDims({outer_batch, inner_batch, input_channels, output_channels})
      .Test(xnnpack_delegate.get());
  BatchMatrixMultiplyTester()
      .InputADims({outer_batch, inner_batch, height, input_channels})
      .InputBDims({input_channels, output_channels})
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, 4D_TransposeB) {
  const auto outer_batch = shape_rng();
  const auto inner_batch = shape_rng();
  const auto height = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .InputADims({outer_batch, inner_batch, height, input_channels})
      .InputBDims({outer_batch, inner_batch, output_channels, input_channels})
      .TransposeB(true)
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, MultiThreading) {
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate(/*num_threads=*/2);

  BatchMatrixMultiplyTester()
      .InputADims({batch, height, input_channels})
      .InputBDims({batch, input_channels, output_channels})
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, WeightsCache) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  std::unique_ptr<TfLiteXNNPackDelegateWeightsCache,
                  decltype(&TfLiteXNNPackDelegateWeightsCacheDelete)>
      weights_cache(TfLiteXNNPackDelegateWeightsCacheCreate(),
                    TfLiteXNNPackDelegateWeightsCacheDelete);
  delegate_options.weights_cache = weights_cache.get();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  BatchMatrixMultiplyTester()
      .InputADims({batch, height, input_channels})
      .InputBDims({batch, input_channels, output_channels})
      .WeightsCache(weights_cache.get())
      .Test(xnnpack_delegate.get());
}

// Builds a TFLite FlatBuffer model containing two BatchMatMul ops that share a
// single quantized INT8 weight tensor (input_b). This exercises the scale-
// expansion deduplication logic in the XNNPACK delegate.
std::vector<char> CreateSharedWeightsModel(
    const std::vector<int32_t>& input_a_shape,
    const std::vector<int32_t>& input_b_shape,
    const std::vector<int32_t>& output_shape,
    const std::vector<int8_t>& quantized_weights,
    const std::vector<float>& scales, const std::vector<int64_t>& zero_points,
    int quantized_dimension) {
  flatbuffers::FlatBufferBuilder builder;

  std::vector<flatbuffers::Offset<OperatorCode>> operator_codes{
      {CreateOperatorCode(builder, BuiltinOperator_BATCH_MATMUL)}};

  std::vector<flatbuffers::Offset<Buffer>> buffers{
      {CreateBuffer(builder, builder.CreateVector({})),
       CreateBuffer(builder, builder.CreateVector({})),
       CreateBuffer(builder, builder.CreateVector(
                                 reinterpret_cast<const uint8_t*>(
                                     quantized_weights.data()),
                                 sizeof(int8_t) * quantized_weights.size()))}};

  std::vector<flatbuffers::Offset<Tensor>> tensors;

  // Tensor 0: Input A1 (FLOAT32)
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(input_a_shape.data(), input_a_shape.size()),
      TensorType_FLOAT32, /*buffer=*/0));

  // Tensor 1: Input A2 (FLOAT32)
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(input_a_shape.data(), input_a_shape.size()),
      TensorType_FLOAT32, /*buffer=*/0));

  // Tensor 2: Shared Input B (INT8, Quantized, Buffer 2)
  flatbuffers::Offset<tflite::QuantizationParameters>
      filter_quantization_params = CreateQuantizationParameters(
          builder, /*min=*/0, /*max=*/0, builder.CreateVector<float>(scales),
          builder.CreateVector<int64_t>(zero_points),
          /*details_type=*/QuantizationDetails_NONE,
          /*details=*/0, quantized_dimension);

  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(input_b_shape.data(), input_b_shape.size()),
      /*type=*/TensorType_INT8,
      /*buffer=*/2,
      /*name=*/0, filter_quantization_params));

  // Tensor 3: Output 1 (FLOAT32)
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(output_shape.data(), output_shape.size()),
      TensorType_FLOAT32, /*buffer=*/0));

  // Tensor 4: Output 2 (FLOAT32)
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(output_shape.data(), output_shape.size()),
      TensorType_FLOAT32, /*buffer=*/0));

  std::vector<flatbuffers::Offset<Operator>> operators;

  // Op 0: BMM (input_a1, input_b) -> output1
  std::vector<int32_t> op0_inputs{{0, 2}};
  std::array<int32_t, 1> op0_outputs{{3}};
  flatbuffers::Offset<BatchMatMulOptions> batch_matmul_options0 =
      CreateBatchMatMulOptions(builder, /*adj_x=*/false, /*adj_y=*/false);
  operators.emplace_back(CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op0_inputs.data(), op0_inputs.size()),
      builder.CreateVector<int32_t>(op0_outputs.data(), op0_outputs.size()),
      BuiltinOptions_BatchMatMulOptions, batch_matmul_options0.Union()));

  // Op 1: BMM (input_a2, input_b) -> output2
  std::vector<int32_t> op1_inputs{{1, 2}};
  std::array<int32_t, 1> op1_outputs{{4}};
  flatbuffers::Offset<BatchMatMulOptions> batch_matmul_options1 =
      CreateBatchMatMulOptions(builder, /*adj_x=*/false, /*adj_y=*/false);
  operators.emplace_back(CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op1_inputs.data(), op1_inputs.size()),
      builder.CreateVector<int32_t>(op1_outputs.data(), op1_outputs.size()),
      BuiltinOptions_BatchMatMulOptions, batch_matmul_options1.Union()));

  std::array<int32_t, 2> subgraph_inputs{{0, 1}};
  std::array<int32_t, 2> subgraph_outputs{{3, 4}};
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(operators.data(), operators.size()));

  flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("Shared Batch Matrix Multiply model");

  flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

// Regression test for memory safety (ASAN/UAF) when multiple BatchMatMul ops
// share a single dynamically quantized weight tensor across delegate
// invocations.
TEST_F(BatchMatrixMultiplyTest, SharedDynamicallyQuantizedWeights) {
  const std::vector<int32_t> input_a_shape = {2, 3};
  const std::vector<int32_t> input_b_shape = {3, 4};
  const std::vector<int32_t> output_shape = {2, 4};

  const int32_t input_a_size = 2 * 3;
  const int32_t output_size = 2 * 4;

  std::vector<float> scales = {0.1f};
  std::vector<int64_t> zero_points = {0};
  int quantized_dimension = 0;

  std::vector<int8_t> quantized_weights = {1,  2,  3, 4, -1, -2,
                                           -3, -4, 0, 1, -1, 2};

  std::vector<char> buffer = CreateSharedWeightsModel(
      input_a_shape, input_b_shape, output_shape, quantized_weights, scales,
      zero_points, quantized_dimension);

  const Model* model = GetModel(buffer.data());

  std::unique_ptr<Interpreter> delegate_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &delegate_interpreter),
      kTfLiteOk);
  std::unique_ptr<Interpreter> default_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &default_interpreter),
      kTfLiteOk);

  ASSERT_TRUE(delegate_interpreter);
  ASSERT_TRUE(default_interpreter);

  ASSERT_EQ(delegate_interpreter->inputs().size(), 2);
  ASSERT_EQ(default_interpreter->inputs().size(), 2);

  ASSERT_EQ(delegate_interpreter->outputs().size(), 2);
  ASSERT_EQ(default_interpreter->outputs().size(), 2);

  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(default_interpreter->AllocateTensors(), kTfLiteOk);

  auto xnnpack_delegate = get_delegate();
  ASSERT_EQ(
      delegate_interpreter->ModifyGraphWithDelegate(xnnpack_delegate.get()),
      kTfLiteOk);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_rng_f32 = std::bind(
      std::uniform_real_distribution<float>(-1.0f, 1.0f), std::ref(rng));

  float* default_input1_data =
      default_interpreter->typed_input_tensor<float>(0);
  std::generate_n(default_input1_data, input_a_size, std::ref(input_rng_f32));
  float* delegate_input1_data =
      delegate_interpreter->typed_input_tensor<float>(0);
  std::copy_n(default_input1_data, input_a_size, delegate_input1_data);

  float* default_input2_data =
      default_interpreter->typed_input_tensor<float>(1);
  std::generate_n(default_input2_data, input_a_size, std::ref(input_rng_f32));
  float* delegate_input2_data =
      delegate_interpreter->typed_input_tensor<float>(1);
  std::copy_n(default_input2_data, input_a_size, delegate_input2_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  float* default_output1_data =
      default_interpreter->typed_output_tensor<float>(0);
  float* delegate_output1_data =
      delegate_interpreter->typed_output_tensor<float>(0);
  float* default_output2_data =
      default_interpreter->typed_output_tensor<float>(1);
  float* delegate_output2_data =
      delegate_interpreter->typed_output_tensor<float>(1);

  // Error tolerance derived from k * (scale_a * scale_b) where reduction dim
  // k=3 and maximum quantization scale step is 0.5f / 127.
  const float max_abs_error = 3 * 0.5f / 127;
  for (size_t i = 0; i < output_size; i++) {
    EXPECT_NEAR(default_output1_data[i], delegate_output1_data[i],
                max_abs_error);
    EXPECT_NEAR(default_output2_data[i], delegate_output2_data[i],
                max_abs_error);
  }
}

}  // namespace xnnpack
}  // namespace tflite
