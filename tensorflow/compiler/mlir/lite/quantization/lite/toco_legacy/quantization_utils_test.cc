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
// This file is the MLIR copy of part of
// third_party/tensorflow/lite/tools/optimize/quantization_utils_test.cc as part
// of the effort to decouple TFLite from MLIR.

#include "tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/quantization_utils.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/compiler/mlir/lite/core/absl_error_model_builder.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/test_util.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"

namespace {
std::string* g_test_model_dir = nullptr;
}  // namespace

namespace mlir {
namespace lite {
namespace toco_legacy {
namespace {

using mlir::TFL::FlatBufferModelAbslError;
using tflite::BuiltinOperator_CONV_2D;
using tflite::QuantizationParametersT;
using tflite::SubGraphT;
using tflite::TensorT;
using tflite::TensorType_FLOAT16;
using tflite::TensorType_FLOAT32;
using tflite::TensorType_INT8;

std::unique_ptr<FlatBufferModelAbslError> ReadModel(const char* model) {
  auto model_path = tsl::io::JoinPath(*g_test_model_dir, model);
  return FlatBufferModelAbslError::BuildFromFile(model_path.c_str());
}

std::unique_ptr<FlatBufferModelAbslError> ReadConvModel() {
  return ReadModel(mlir::lite::internal::kConvModelWith0Plus10Weights);
}

using ::testing::ElementsAreArray;

class QuantizationUtilsTest : public testing::Test {};

TEST_F(QuantizationUtilsTest, NumElements) {
  TensorT tensor;
  tensor.shape = {1, 2, 3, 4};
  uint64_t num_elements;
  TF_EXPECT_OK(NumElements(tensor, &num_elements));
  EXPECT_EQ(num_elements, 1 * 2 * 3 * 4);

  tensor.shape = {5};
  TF_EXPECT_OK(NumElements(tensor, &num_elements));
  EXPECT_EQ(num_elements, 5);

  tensor.shape = {};
  TF_EXPECT_OK(NumElements(tensor, &num_elements));
  // Scalars with empty shape have 1 element.
  EXPECT_EQ(num_elements, 1);

  tensor.shape = {1, 2, 3, -1};
  EXPECT_EQ(NumElements(tensor, &num_elements).code(),
            absl::StatusCode::kInternal);
}

TEST_F(QuantizationUtilsTest, SymmetricPerChannelQuantizationWithNullQParams) {
  // Set up an input with [3, 2, 2, 2] size and 0 is the channel index.
  const std::vector<float> input = {
      3.0, 2.0, 5.0,  -2.0, 3.0,  2.0,  5.0,  -2.0,  // Channel 1.
      1.0, 2.0, 3.0,  4.0,  5.0,  6.0,  7.0,  8.0,   // Channel 2.
      1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,  // Channel 3.
  };
  const int channel_index = 0;

  // Create holder for output scale and data.
  std::vector<float> output_scales(3);
  std::vector<int8_t> output_data(3 * 2 * 2 * 2);

  // Call SymmetricPerChannelQuantization with quant_params as a null pointer
  // and verify the result.
  TensorT tensor = TensorT();
  tensor.quantization = nullptr;
  tensor.shape = {3, 2, 2, 2};
  TF_EXPECT_OK(mlir::lite::toco_legacy::SymmetricPerChannelQuantization(
      &tensor, input.data(), channel_index, &output_scales, &output_data));
  const std::vector<float> expected_output_scales = {0.0393700786, 0.0629921257,
                                                     0.0472440943};
  const std::vector<int8_t> expected_output_data = {
      76, 51, 127, -51, 76,  51,  127,  -51,   // Channel 1.
      16, 32, 48,  64,  79,  95,  111,  127,   // Channel 2.
      21, 0,  -21, -42, -64, -85, -106, -127,  // Channel 3.
  };
  EXPECT_THAT(output_scales, ElementsAreArray(expected_output_scales));
  EXPECT_THAT(output_data, ElementsAreArray(expected_output_data));
}

TEST_F(QuantizationUtilsTest, SymmetricPerChannelQuantization) {
  // Set up an input with [3, 2, 2, 2] size and 0 is the channel index.
  const std::vector<float> input = {
      3.0, 2.0, 5.0,  -2.0, 3.0,  2.0,  5.0,  -2.0,  // Channel 1.
      1.0, 2.0, 3.0,  4.0,  5.0,  6.0,  7.0,  8.0,   // Channel 2.
      1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,  // Channel 3.
  };
  const int32_t channel_index = 0;

  // Create holder for output scale and data.
  std::vector<float> output_scales(3);
  std::vector<int8_t> output_data(3 * 2 * 2 * 2);

  // Initialize pointer to quantization parameters
  TensorT tensor = TensorT();
  tensor.quantization = std::make_unique<QuantizationParametersT>();
  tensor.shape = {3, 2, 2, 2};
  TF_EXPECT_OK(mlir::lite::toco_legacy::FillPerChannelMinMax(
      input.data(), tensor.shape, channel_index, tensor.quantization.get()));

  // Test that FillPerChanneMinMax worked
  const std::vector<float> expected_mins = {-2.0, 1.0, -6.0};
  const std::vector<float> expected_maxs = {5.0, 8.0, 1.0};
  EXPECT_THAT(tensor.quantization->min, ElementsAreArray(expected_mins));
  EXPECT_THAT(tensor.quantization->max, ElementsAreArray(expected_maxs));

  // Call SymmetricPerChannelQuantization with quant_params as a null pointer
  // and verify the result.
  TF_EXPECT_OK(mlir::lite::toco_legacy::SymmetricPerChannelQuantization(
      &tensor, input.data(), channel_index, &output_scales, &output_data));
  const std::vector<float> expected_output_scales = {0.0393700786, 0.0629921257,
                                                     0.0472440943};
  const std::vector<int8_t> expected_output_data = {
      76, 51, 127, -51, 76,  51,  127,  -51,   // Channel 1.
      16, 32, 48,  64,  79,  95,  111,  127,   // Channel 2.
      21, 0,  -21, -42, -64, -85, -106, -127,  // Channel 3.
  };
  EXPECT_THAT(output_scales, ElementsAreArray(expected_output_scales));
  EXPECT_THAT(output_data, ElementsAreArray(expected_output_data));
}

TEST_F(QuantizationUtilsTest, SymmetricPerChannelQuantization2DTensor) {
  // Set up an input with [3, 8] size and 0 is the channel index.
  const std::vector<float> input = {
      3.0, 2.0, 5.0,  -2.0, 3.0,  2.0,  5.0,  -2.0,  // Batch 1.
      1.0, 2.0, 3.0,  4.0,  5.0,  6.0,  7.0,  8.0,   // Batch 2.
      1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,  // Batch 3.
  };
  const int32_t channel_index = 1;

  // Create holder for output scale and data.
  std::vector<float> output_scales(8);
  std::vector<int8_t> output_data(3 * 8);

  // Initialize pointer to quantization parameters
  TensorT tensor = TensorT();
  tensor.quantization = std::make_unique<QuantizationParametersT>();
  tensor.shape = {3, 8};
  TF_EXPECT_OK(mlir::lite::toco_legacy::FillPerChannelMinMax(
      input.data(), tensor.shape, channel_index, tensor.quantization.get()));

  // Test that FillPerChanneMinMax worked
  const std::vector<float> expected_mins = {1.0,  0.0,  -1.0, -2.0,
                                            -3.0, -4.0, -5.0, -6.0};
  const std::vector<float> expected_maxs = {3.0, 2.0, 5.0, 4.0,
                                            5.0, 6.0, 7.0, 8.0};
  EXPECT_THAT(tensor.quantization->min, ElementsAreArray(expected_mins));
  EXPECT_THAT(tensor.quantization->max, ElementsAreArray(expected_maxs));

  // Call SymmetricPerChannelQuantization with quant_params as a null pointer
  // and verify the result.
  TF_EXPECT_OK(mlir::lite::toco_legacy::SymmetricPerChannelQuantization(
      &tensor, input.data(), channel_index, &output_scales, &output_data));
  const std::vector<float> expected_output_scales = {
      0.02362204724, 0.01574803149, 0.03937007874, 0.03149606299,
      0.03937007874, 0.04724409448, 0.05511811023, 0.06299212598};
  const std::vector<int8_t> expected_output_data = {
      127, 127, 127, -64, 76,  42,  91,  -32,  // Batch 1.
      42,  127, 76,  127, 127, 127, 127, 127,  // Batch 2.
      42,  0,   -25, -64, -76, -85, -91, -95,  // Batch 3.
  };
  EXPECT_THAT(output_scales, ElementsAreArray(expected_output_scales));
  EXPECT_THAT(output_data, ElementsAreArray(expected_output_data));
}

TEST_F(QuantizationUtilsTest, SymmetricPerChannelQuantizeValues) {
  // Set up an input with [3, 1, 1, 2] size and 0 is the channel index.
  const std::vector<float> input = {
      13.0, 21.0,  // Channel 1.
      21.0, 22.0,  // Channel 2.
      31.0, 40.0,  // Channel 3.
  };
  const std::vector<float> scales_inv = {2, 0.5, 3};
  const std::vector<int32_t> dimension = {3, 1, 1, 2};
  const int channel_index = 0;

  // Create holder for output data.
  std::vector<int8_t> output_data(3 * 1 * 1 * 2);

  // Call SymmetricPerChannelQuantizeValues and verify the result.
  SymmetricPerChannelQuantizeValues(input.data(), scales_inv, dimension,
                                    channel_index, &output_data);
  const std::vector<int8_t> expected_output_data = {
      26, 42,   // Channel 1.
      11, 11,   // Channel 2.
      93, 120,  // Channel 3.
  };
  EXPECT_THAT(output_data, ElementsAreArray(expected_output_data));
}

TEST_F(QuantizationUtilsTest, FillPerChannelMinMax) {
  // Set up an input with [3, 1, 1, 2] size.
  const std::vector<float> input = {
      13.0, 21.0,  // Channel 1.
      21.0, 22.0,  // Channel 2.
      31.0, 40.0,  // Channel 3.
  };

  // Initialize pointer to quantization parameters.
  QuantizationParametersT quantization_params = QuantizationParametersT();
  std::vector<int> dimension = {3, 1, 1, 2};
  int32_t channel_dim_idx = 0;
  const std::vector<float> expected_mins = {13.0, 21.0, 31.0};
  const std::vector<float> expected_maxs = {21.0, 22.0, 40.0};

  TF_EXPECT_OK(mlir::lite::toco_legacy::FillPerChannelMinMax(
      input.data(), dimension, channel_dim_idx, &quantization_params));

  EXPECT_EQ(quantization_params.min, expected_mins);
  EXPECT_EQ(quantization_params.max, expected_maxs);
  EXPECT_EQ(quantization_params.quantized_dimension, channel_dim_idx);
}

TEST_F(QuantizationUtilsTest, FillPerChannelMinMaxFillDim3) {
  // Set up an input with [3, 1, 1, 2] size.
  const std::vector<float> input = {
      // Channel 1, Channel 2
      13.0, 21.0, 21.0, 22.0, 31.0, 40.0,
  };

  // Initialize pointer to quantization parameters.
  QuantizationParametersT quantization_params = QuantizationParametersT();
  std::vector<int> dimension = {3, 1, 1, 2};
  int32_t channel_dim_idx = 3;
  const std::vector<float> expected_mins = {13.0, 21.0};
  const std::vector<float> expected_maxs = {31.0, 40.0};

  TF_EXPECT_OK(mlir::lite::toco_legacy::FillPerChannelMinMax(
      input.data(), dimension, channel_dim_idx, &quantization_params));

  EXPECT_EQ(quantization_params.min, expected_mins);
  EXPECT_EQ(quantization_params.max, expected_maxs);
  EXPECT_EQ(quantization_params.quantized_dimension, channel_dim_idx);
}

TEST_F(QuantizationUtilsTest, FillPerChannelMinMax2DTensor) {
  // Set up an input with [3, 2] size.
  const std::vector<float> input = {
      // Channel 1, Channel 2
      13.0, 21.0, 21.0, 22.0, 31.0, 40.0,
  };

  // Initialize pointer to quantization parameters.
  QuantizationParametersT quantization_params = QuantizationParametersT();
  std::vector<int> dimension = {3, 2};
  int32_t channel_dim_idx = 1;
  const std::vector<float> expected_mins = {13.0, 21.0};
  const std::vector<float> expected_maxs = {31.0, 40.0};

  TF_EXPECT_OK(mlir::lite::toco_legacy::FillPerChannelMinMax(
      input.data(), dimension, channel_dim_idx, &quantization_params));

  EXPECT_EQ(quantization_params.min, expected_mins);
  EXPECT_EQ(quantization_params.max, expected_maxs);
  EXPECT_EQ(quantization_params.quantized_dimension, channel_dim_idx);
}

TEST_F(QuantizationUtilsTest, SymmetricQuantizeTensorNullInputs) {
  EXPECT_EQ(SymmetricQuantizeTensor(nullptr, nullptr).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(QuantizationUtilsTest, SymmetricQuantizeTensorNullQuantParams) {
  // Conv model has weights between 0 and 10.
  // Quantize the weights tensor.
  ASSERT_TRUE(g_test_model_dir);
  ASSERT_FALSE(g_test_model_dir->empty());
  auto test_model = ReadConvModel();
  ASSERT_TRUE(test_model);
  auto readonly_model = test_model->GetModel();
  ASSERT_TRUE(readonly_model);
  ASSERT_TRUE(readonly_model->subgraphs());
  ASSERT_GE(readonly_model->subgraphs()->size(), 1);
  tflite::ModelT model;
  readonly_model->UnPackTo(&model);
  auto subgraph = model.subgraphs[0].get();
  auto conv_op = subgraph->operators.at(0).get();
  ASSERT_EQ(
      GetBuiltinCode(model.operator_codes.at(conv_op->opcode_index).get()),
      BuiltinOperator_CONV_2D);
  int32_t weights_tensor_idx = conv_op->inputs[1];
  TensorT* weights_tensor = subgraph->tensors.at(weights_tensor_idx).get();
  // Empty quantization parameters.
  weights_tensor->quantization = std::make_unique<QuantizationParametersT>();

  EXPECT_EQ(weights_tensor->type, TensorType_FLOAT32);
  size_t float_buffer_size =
      model.buffers.at(weights_tensor->buffer)->data.size();

  TF_EXPECT_OK(SymmetricQuantizeTensor(&model, weights_tensor));

  size_t quant_buffer_size =
      model.buffers.at(weights_tensor->buffer)->data.size();
  EXPECT_EQ(weights_tensor->type, TensorType_INT8);
  EXPECT_EQ(quant_buffer_size * 4, float_buffer_size);
}

TEST_F(QuantizationUtilsTest, SymmetricQuantizeTensor) {
  // Conv model has weights between 0 and 10.
  // Quantize the weights tensor.
  ASSERT_TRUE(g_test_model_dir);
  ASSERT_FALSE(g_test_model_dir->empty());
  auto test_model = ReadConvModel();
  ASSERT_TRUE(test_model);
  auto readonly_model = test_model->GetModel();
  ASSERT_TRUE(readonly_model);
  ASSERT_TRUE(readonly_model->subgraphs());
  ASSERT_GE(readonly_model->subgraphs()->size(), 1);
  tflite::ModelT model;
  readonly_model->UnPackTo(&model);
  auto subgraph = model.subgraphs[0].get();
  auto conv_op = subgraph->operators.at(0).get();
  ASSERT_EQ(
      GetBuiltinCode(model.operator_codes.at(conv_op->opcode_index).get()),
      BuiltinOperator_CONV_2D);
  int32_t weights_tensor_idx = conv_op->inputs[1];
  TensorT* weights_tensor = subgraph->tensors.at(weights_tensor_idx).get();

  EXPECT_EQ(weights_tensor->type, TensorType_FLOAT32);
  size_t float_buffer_size =
      model.buffers.at(weights_tensor->buffer)->data.size();

  TF_EXPECT_OK(SymmetricQuantizeTensor(&model, weights_tensor));

  size_t quant_buffer_size =
      model.buffers.at(weights_tensor->buffer)->data.size();
  EXPECT_EQ(weights_tensor->type, TensorType_INT8);
  EXPECT_EQ(quant_buffer_size * 4, float_buffer_size);
}

TEST_F(QuantizationUtilsTest, QuantizeFloat16Clamp) {
  // Create data.
  auto model = std::make_unique<ModelT>();
  auto subgraph = std::make_unique<tflite::SubGraphT>();
  auto tensor = std::make_unique<TensorT>();
  auto buffer = std::make_unique<tflite::BufferT>();
  constexpr int kNumElements = 6;
  const std::vector<float> weights = {2.0, 1.0, 65504., 65505, -65504., -99999};
  auto weights_reinterpreted_data =
      reinterpret_cast<const unsigned char*>(weights.data());
  buffer->data.assign(weights_reinterpreted_data,
                      weights_reinterpreted_data + weights.size() * 4);
  tensor->buffer = 0;
  tensor->shape = {1, kNumElements};

  // Wire the model.
  model->subgraphs.push_back(std::move(subgraph));
  model->subgraphs[0]->tensors.push_back(std::move(tensor));
  model->buffers.push_back(std::move(buffer));

  // Call and verify.
  TF_EXPECT_OK(QuantizeTensorFloat16(model.get(),
                                  model->subgraphs[0]->tensors[0].get()));
  auto weightsf16 = reinterpret_cast<Eigen::half*>(
      model->buffers[model->subgraphs[0]->tensors[0]->buffer]->data.data());
  std::vector<float> wf32(kNumElements);
  std::transform(weightsf16, weightsf16 + 6, wf32.begin(),
                 [](Eigen::half a) { return static_cast<float>(a); });

  EXPECT_THAT(wf32,
              ElementsAreArray({2.0, 1.0, 65504., 65504., -65504., -65504.}));
  EXPECT_EQ(model->subgraphs[0]->tensors[0]->type, TensorType_FLOAT16);
}

TEST_F(QuantizationUtilsTest, QuantizeFloat16) {
  // Conv model has weights between 0 and 10.
  // Quantize the weights tensor.
  ASSERT_TRUE(g_test_model_dir != nullptr);
  ASSERT_FALSE(g_test_model_dir->empty());
  auto test_model = ReadConvModel();
  ASSERT_TRUE(test_model);
  auto readonly_model = test_model->GetModel();
  ASSERT_TRUE(readonly_model);
  ASSERT_TRUE(readonly_model->subgraphs());
  ASSERT_GE(readonly_model->subgraphs()->size(), 1);
  tflite::ModelT model;
  readonly_model->UnPackTo(&model);
  auto subgraph = model.subgraphs[0].get();
  auto conv_op = subgraph->operators.at(0).get();
  ASSERT_EQ(
      GetBuiltinCode(model.operator_codes.at(conv_op->opcode_index).get()),
      BuiltinOperator_CONV_2D);
  int32_t weights_tensor_idx = conv_op->inputs[1];
  TensorT* weights_tensor = subgraph->tensors.at(weights_tensor_idx).get();

  EXPECT_EQ(weights_tensor->type, TensorType_FLOAT32);
  size_t float_buffer_size =
      model.buffers.at(weights_tensor->buffer)->data.size();

  TF_EXPECT_OK(QuantizeTensorFloat16(&model, weights_tensor));

  size_t quant_buffer_size =
      model.buffers.at(weights_tensor->buffer)->data.size();
  EXPECT_EQ(weights_tensor->type, TensorType_FLOAT16);
  EXPECT_EQ(quant_buffer_size * 2, float_buffer_size);
}

TEST_F(QuantizationUtilsTest, AddQuantizationParams) {
  // Create data.
  auto model = std::make_unique<ModelT>();
  auto subgraph = std::make_unique<tflite::SubGraphT>();
  auto tensor = std::make_unique<TensorT>();
  auto buffer = std::make_unique<tflite::BufferT>();
  const std::vector<float> scales = {0.5, 1.0, 1.5};
  const std::vector<int64_t> zero_points = {5, 10, 15};
  const int32_t quantizated_dimension = 3;
  const std::vector<uint8_t> buffer_data = {1, 2, 3, 4};
  const int32_t buffer_size = 4;
  tensor->buffer = 0;

  // Wire the model.
  model->subgraphs.push_back(std::move(subgraph));
  model->subgraphs[0]->tensors.push_back(std::move(tensor));
  model->buffers.push_back(std::move(buffer));

  // Call and verify.
  TF_EXPECT_OK(AddQuantizationParams(scales, zero_points, quantizated_dimension,
                                  buffer_data.data(), buffer_size,
                                  TensorType_INT8, model.get(),
                                  model->subgraphs[0]->tensors[0].get()));
  EXPECT_THAT(model->subgraphs[0]->tensors[0]->quantization->scale,
              ElementsAreArray(scales));
  EXPECT_THAT(model->subgraphs[0]->tensors[0]->quantization->zero_point,
              ElementsAreArray(zero_points));
  EXPECT_THAT(model->buffers[model->subgraphs[0]->tensors[0]->buffer]->data,
              ElementsAreArray(buffer_data));
  EXPECT_EQ(model->subgraphs[0]->tensors[0]->type, TensorType_INT8);
}


}  // namespace
}  // namespace toco_legacy
}  // namespace lite
}  // namespace mlir

int main(int argc, char** argv) {
  std::string model_file;
  const std::vector<tsl::Flag> flag_list = {
      tsl::Flag("test_model_file", &model_file,
                "Path to test tflite model file."),
  };

  const bool parse_result = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    std::cerr << "Required test_model_file\n";
    std::abort();
  }
  g_test_model_dir = new std::string(tsl::io::Dirname(model_file));
  ::tsl::port::InitMain(argv[0], &argc, &argv);
  return RUN_ALL_TESTS();
}
