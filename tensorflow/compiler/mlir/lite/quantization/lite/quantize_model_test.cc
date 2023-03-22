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
#include "tensorflow/compiler/mlir/lite/quantization/lite/quantize_model.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/tools/optimize/test_util.h"

// Note: branched from tensorflow/lite/tools/optimize/quantize_model_test.cc

namespace {
tensorflow::string* g_test_model_dir = nullptr;
}  // namespace

namespace tflite {
namespace optimize {
namespace {

using testing::Eq;
using testing::FloatEq;
using testing::FloatNear;
using testing::IsEmpty;
using testing::NotNull;
using testing::SizeIs;

// Unpacks the given flatbuffer model.
//
// This helper is useful as UnPackTo requires the input to not have any existing
// state so directly calling UnPackTo could lead to memory leaks if the model
// already had some state. Instead, the returned object from here can be used to
// overwrite existing model.
ModelT UnPackFlatBufferModel(const Model& flatbuffer_model) {
  ModelT model;
  flatbuffer_model.UnPackTo(&model);
  return model;
}

TfLiteStatus QuantizeModel(
    flatbuffers::FlatBufferBuilder* builder, ModelT* model,
    const TensorType& input_type, const TensorType& output_type,
    bool allow_float, const std::unordered_set<string>& operator_names,
    const TensorType& activations_type, ErrorReporter* error_reporter,
    bool disable_per_channel = false,
    const absl::flat_hash_set<std::string>& blocked_ops = {},
    const absl::flat_hash_set<std::string>& blocked_nodes = {}) {
  TensorType inference_tensor_type = activations_type;
  bool fully_quantize = !allow_float;
  auto status = mlir::lite::QuantizeModel(
      *model, input_type, output_type, inference_tensor_type,
      /*operator_names=*/{}, disable_per_channel, fully_quantize, builder,
      error_reporter, /*verify_numeric=*/false, /*whole_model_verify=*/false,
      /*legacy_float_scale=*/true, blocked_ops, blocked_nodes);
  if (status != kTfLiteOk) {
    return status;
  }
  std::string buffer(
      reinterpret_cast<const char*>(builder->GetCurrentBufferPointer()),
      builder->GetSize());

  auto flatbuffer_model =
      FlatBufferModel::BuildFromBuffer(buffer.c_str(), buffer.size());
  *model = UnPackFlatBufferModel(*flatbuffer_model->GetModel());
  return kTfLiteOk;
}

TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* model, const TensorType& input_type,
                           const TensorType& output_type, bool allow_float,
                           ErrorReporter* error_reporter) {
  return QuantizeModel(builder, model, input_type, output_type, allow_float,
                       /*operator_names=*/{}, TensorType_INT8, error_reporter);
}

TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* model, const TensorType& input_type,
                           const TensorType& output_type,
                           ErrorReporter* error_reporter) {
  return QuantizeModel(builder, model, input_type, output_type,
                       /*allow_float=*/false, error_reporter);
}

TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* model, ErrorReporter* error_reporter) {
  return QuantizeModel(builder, model, TensorType_FLOAT32, TensorType_FLOAT32,
                       /*allow_float=*/true, error_reporter);
}

TfLiteStatus QuantizeModelAllOperators(
    flatbuffers::FlatBufferBuilder* builder, ModelT* model,
    const TensorType& input_type, const TensorType& output_type,
    bool allow_float, const TensorType& activations_type,
    bool disable_per_channel, ErrorReporter* error_reporter) {
  return QuantizeModel(builder, model, input_type, output_type, allow_float,
                       /*operator_names=*/{}, activations_type, error_reporter,
                       disable_per_channel);
}

TfLiteStatus QuantizeModelAllOperators(flatbuffers::FlatBufferBuilder* builder,
                                       ModelT* model,
                                       const TensorType& input_type,
                                       const TensorType& output_type,
                                       bool allow_float,
                                       const TensorType& activations_type,
                                       ErrorReporter* error_reporter) {
  return QuantizeModel(builder, model, input_type, output_type, allow_float,
                       /*operator_names=*/{}, activations_type, error_reporter);
}

std::unique_ptr<FlatBufferModel> ReadModel(const string& model_name) {
  auto model_path = tensorflow::io::JoinPath(*g_test_model_dir, model_name);
  return FlatBufferModel::BuildFromFile(model_path.c_str());
}

template <typename T>
std::vector<T> GetAsVector(const flatbuffers::Vector<T>* vec) {
  return std::vector<T>(vec->begin(), vec->end());
}

void VerifyQuantizationScale(
    const QuantizationParameters& float_quant_params,
    const QuantizationParametersT& quantized_quant_params, const int bit_num,
    const bool symmetric) {
  const float eps = 1e-7;
  ASSERT_THAT(*float_quant_params.min(), SizeIs(1));
  ASSERT_THAT(*float_quant_params.max(), SizeIs(1));
  float float_min = std::min(0.f, float_quant_params.min()->Get(0));
  float float_max = std::max(0.f, float_quant_params.max()->Get(0));
  if (symmetric) {
    // When the symmetric case, ConvertStatsToQDQs in PrepareQuantizePass
    // updates the min and max values into bigger absolute values.
    float_max = std::max(std::abs(float_min), std::abs(float_max));
    float_min = -float_max;
  }

  ASSERT_THAT(quantized_quant_params.scale, SizeIs(1));
  ASSERT_THAT(quantized_quant_params.zero_point, SizeIs(1));
  float scale = (float_max - float_min) / ((1 << bit_num) - 1);
  EXPECT_THAT(scale, FloatNear(quantized_quant_params.scale[0], eps));
}

class QuantizeModelTest : public testing::Test {
 protected:
  QuantizeModelTest() {
    input_model_ = ReadModel(internal::kConvModelWith0Plus10Weights);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }

  std::unique_ptr<FlatBufferModel> input_model_;
  const Model* readonly_model_;
  tflite::ModelT model_;
  flatbuffers::FlatBufferBuilder builder_;
  internal::FailOnErrorReporter error_reporter_;
};

void ExpectEqualTensor(TensorT* tensor, TensorT* expected_tensor) {
  const float eps = 1e-7;
  EXPECT_THAT(expected_tensor, NotNull());
  EXPECT_THAT(tensor->is_variable, Eq(expected_tensor->is_variable));
  EXPECT_THAT(tensor->shape, Eq(expected_tensor->shape));
  EXPECT_THAT(tensor->type, Eq(expected_tensor->type));
  const auto quantization_params = tensor->quantization.get();
  const auto expected_quantization_params = expected_tensor->quantization.get();
  if (quantization_params != nullptr &&
      expected_quantization_params != nullptr) {
    for (int i = 0; i < quantization_params->scale.size(); ++i) {
      if (quantization_params->scale[i] > 3e-5) {
        EXPECT_THAT(quantization_params->scale[i],
                    FloatNear(expected_quantization_params->scale[i], eps));
      }
    }
    EXPECT_THAT(quantization_params->zero_point,
                Eq(expected_quantization_params->zero_point));
  }
}

TensorT* FindMatchingExpectedTensor(const SubGraphT& expected_graph,
                                    const ModelT& expected_model,
                                    const ModelT& quant_model,
                                    const OperatorT& quant_op, int idx) {
  const auto& builtin_code =
      GetBuiltinCode(quant_model.operator_codes[quant_op.opcode_index].get());
  for (const auto& expected_op : expected_graph.operators) {
    const auto& op_code =
        expected_model.operator_codes[expected_op->opcode_index].get();
    const auto& expected_code = GetBuiltinCode(op_code);
    if (expected_code == builtin_code) {
      return expected_graph.tensors[expected_op->inputs[idx]].get();
    }
  }
  return nullptr;
}

void ExpectSameModels(const ModelT& model, const ModelT& expected_model) {
  ASSERT_THAT(model.subgraphs, SizeIs(expected_model.subgraphs.size()));
  for (size_t subgraph_idx = 0; subgraph_idx < model.subgraphs.size();
       subgraph_idx++) {
    const auto graph = model.subgraphs[subgraph_idx].get();
    const auto expected_graph = expected_model.subgraphs[subgraph_idx].get();
    for (auto& op : graph->operators) {
      for (int idx = 0; idx < op->inputs.size(); idx++) {
        if (op->inputs[idx] < 0) {
          continue;
        }
        const auto& tensor = graph->tensors[op->inputs[idx]];
        auto* expected_tensor = FindMatchingExpectedTensor(
            *expected_graph, expected_model, model, *op, idx);
        if (!expected_tensor) {
          continue;
        }
        ExpectEqualTensor(tensor.get(), expected_tensor);
        if (expected_tensor->buffer > 0) {
          const int buffer_idx = tensor->buffer;
          const int expected_buffer_idx = expected_tensor->buffer;
          const auto buffer = model.buffers[buffer_idx].get()->data;
          const auto expected_buffer =
              expected_model.buffers[expected_buffer_idx].get()->data;
          EXPECT_THAT(buffer, Eq(expected_buffer));
        }
      }
    }
  }
}

class QuantizeConvModelTest : public QuantizeModelTest,
                              public testing::WithParamInterface<TensorType> {
 protected:
  QuantizeConvModelTest() {
    tensor_type_ = GetParam();
    input_model_ = ReadModel(internal::kConvModelWith0Plus10Weights);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
    // Flatbuffer is missing calibration data -- add dummy params.
    auto& subgraph = model_.subgraphs[0];
    auto* input = subgraph->tensors[subgraph->inputs[0]].get();
    auto* output = subgraph->tensors[subgraph->outputs[0]].get();
    input->quantization = std::make_unique<QuantizationParametersT>();
    output->quantization = std::make_unique<QuantizationParametersT>();
    input->quantization->min.push_back(0.0);
    output->quantization->min.push_back(0.0);
    input->quantization->max.push_back(6.0);
    output->quantization->max.push_back(6.0);
  }
  TensorType tensor_type_;
};

INSTANTIATE_TEST_SUITE_P(QuantizeConvModelTestInst, QuantizeConvModelTest,
                         testing::ValuesIn({TensorType_INT8}));

TEST_P(QuantizeConvModelTest, QuantizationSucceeds) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, tensor_type_, tensor_type_, /*allow_float=*/false,
      tensor_type_, &error_reporter_);
  EXPECT_THAT(status, Eq(kTfLiteOk));
  const uint8_t* buffer = builder_.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);
}

TEST_P(QuantizeConvModelTest, SkipUnspecifiedLayer) {
  auto status = QuantizeModel(
      &builder_, &model_, TensorType_FLOAT32, TensorType_FLOAT32,
      /*allow_float=*/true, /*operator_names=*/{}, TensorType_FLOAT32,
      &error_reporter_, /*disable_per_channel=*/false, {"CONV_2D"});
  EXPECT_THAT(status, Eq(kTfLiteOk));

  ModelT expected_model;
  readonly_model_->UnPackTo(&expected_model);
  // The resulting model should be the same.
  ExpectSameModels(model_, expected_model);
}

TEST_P(QuantizeConvModelTest, SkipUnspecifiedLayerByName) {
  auto status = QuantizeModel(
      &builder_, &model_, TensorType_FLOAT32, TensorType_FLOAT32,
      /*allow_float=*/true, /*operator_names=*/{}, TensorType_FLOAT32,
      &error_reporter_, /*disable_per_channel=*/false, /*blocked_ops=*/{},
      {"output"});
  EXPECT_THAT(status, Eq(kTfLiteOk));

  ModelT expected_model;
  readonly_model_->UnPackTo(&expected_model);
  // The resulting model should be the same.
  ExpectSameModels(model_, expected_model);
}

TEST_P(QuantizeConvModelTest, GraphIsFullyQuantized) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, tensor_type_, tensor_type_,
      /*allow_float=*/false, tensor_type_, &error_reporter_);
  EXPECT_THAT(status, Eq(kTfLiteOk));

  for (const auto& subgraph : model_.subgraphs) {
    for (const auto& tensor : subgraph->tensors) {
      EXPECT_TRUE(tensor->type == TensorType_INT32 ||
                  tensor->type == TensorType_INT8);
    }
  }
}

class QuantizeConvNoBiasModelTest : public QuantizeModelTest {
 protected:
  QuantizeConvNoBiasModelTest() {
    input_model_ = ReadModel(internal::kConvModelWithNoBias);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }
};

TEST_F(QuantizeConvNoBiasModelTest, QuantizationSucceeds) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  EXPECT_THAT(status, Eq(kTfLiteOk));
  const uint8_t* buffer = builder_.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);
}

class QuantizeSplitModelTest : public QuantizeModelTest {
 protected:
  QuantizeSplitModelTest() {
    input_model_ = ReadModel(internal::kModelSplit);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }
};

// There are two outputs for split with different scales, the resulting model
// should have the scales be hardcodes to the input scale value.
TEST_F(QuantizeSplitModelTest, QuantizeSplit) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  EXPECT_THAT(status, Eq(kTfLiteOk));

  // There is only one subgraph.
  const int32_t subgraph_idx = 0;
  const auto& subgraph = model_.subgraphs[subgraph_idx];
  const auto& readonly_subgraph =
      readonly_model_->subgraphs()->Get(subgraph_idx);

  // There should be two ops: the split and add in the original model.
  EXPECT_THAT(*readonly_subgraph->operators(), SizeIs(2));
  EXPECT_THAT(subgraph->operators, SizeIs(2));
  const auto& split = subgraph->operators[0];
  const auto& add = subgraph->operators[1];
  EXPECT_THAT(GetBuiltinCode(model_.operator_codes[split->opcode_index].get()),
              Eq(BuiltinOperator_SPLIT));
  EXPECT_THAT(GetBuiltinCode(model_.operator_codes[add->opcode_index].get()),
              Eq(BuiltinOperator_ADD));

  // There should be 5 tensors: input, output, split, split/split_dim, split:1.
  // Tensor indices could be different between original and quantized.
  EXPECT_THAT(subgraph->tensors, SizeIs(5));
  const int input_idx = 0;
  EXPECT_THAT(subgraph->tensors[input_idx]->type, Eq(TensorType_INT8));
  EXPECT_THAT(subgraph->tensors[input_idx]->name, Eq("input"));
  EXPECT_THAT(subgraph->tensors[input_idx]->quantization->scale, SizeIs(1));
  EXPECT_THAT(subgraph->tensors[input_idx]->quantization->zero_point,
              SizeIs(1));
  EXPECT_THAT(subgraph->tensors[input_idx]->quantization->scale[0],
              FloatEq(1.0));
  EXPECT_THAT(subgraph->tensors[input_idx]->quantization->zero_point[0],
              Eq(-128));
  const int output_idx = 4;
  EXPECT_THAT(subgraph->tensors[output_idx]->type, Eq(TensorType_INT8));
  EXPECT_THAT(subgraph->tensors[output_idx]->name, Eq("output"));
  EXPECT_THAT(subgraph->tensors[output_idx]->quantization->scale, SizeIs(1));
  EXPECT_THAT(subgraph->tensors[output_idx]->quantization->zero_point,
              SizeIs(1));
  EXPECT_THAT(subgraph->tensors[output_idx]->quantization->scale[0],
              FloatEq(1.0));
  EXPECT_THAT(subgraph->tensors[output_idx]->quantization->zero_point[0],
              Eq(-128));
  const int split0_idx = 2;
  EXPECT_THAT(subgraph->tensors[split0_idx]->type, Eq(TensorType_INT8));
  EXPECT_THAT(subgraph->tensors[split0_idx]->name, Eq("split;split:1"));
  EXPECT_THAT(subgraph->tensors[split0_idx]->quantization->scale, SizeIs(1));
  EXPECT_THAT(subgraph->tensors[split0_idx]->quantization->zero_point,
              SizeIs(1));
  EXPECT_THAT(subgraph->tensors[split0_idx]->quantization->scale[0],
              FloatEq(1.0));
  EXPECT_THAT(subgraph->tensors[split0_idx]->quantization->zero_point[0],
              Eq(-128));
  const int split1_idx = 3;
  EXPECT_THAT(subgraph->tensors[split1_idx]->type, Eq(TensorType_INT8));
  EXPECT_THAT(subgraph->tensors[split1_idx]->name, Eq("split;split:11"));
  EXPECT_THAT(subgraph->tensors[split1_idx]->quantization->scale, SizeIs(1));
  EXPECT_THAT(subgraph->tensors[split1_idx]->quantization->zero_point,
              SizeIs(1));
  EXPECT_THAT(subgraph->tensors[split1_idx]->quantization->scale[0],
              FloatEq(1.0));
  EXPECT_THAT(subgraph->tensors[split1_idx]->quantization->zero_point[0],
              Eq(-128));

  // check op and versioning.
  EXPECT_THAT(model_.operator_codes, SizeIs(2));
  EXPECT_THAT(GetBuiltinCode(model_.operator_codes[0].get()),
              Eq(BuiltinOperator_SPLIT));
  ASSERT_THAT(model_.operator_codes[0]->version, Eq(2));
}

class QuantizeConvModel2Test : public QuantizeModelTest,
                               public testing::WithParamInterface<TensorType> {
 protected:
  QuantizeConvModel2Test() {
    tensor_type_ = GetParam();
    input_model_ = ReadModel(internal::kConvModelWith0Plus10Weights);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
    auto& subgraph = model_.subgraphs[0];
    auto* input = subgraph->tensors[subgraph->inputs[0]].get();
    auto* output = subgraph->tensors[subgraph->outputs[0]].get();
    input->quantization = std::make_unique<QuantizationParametersT>();
    output->quantization = std::make_unique<QuantizationParametersT>();
    input->quantization->min.push_back(0.0);
    output->quantization->min.push_back(0.0);
    input->quantization->max.push_back(6.0);
    output->quantization->max.push_back(6.0);
  }

  TensorType tensor_type_;
};

INSTANTIATE_TEST_SUITE_P(QuantizeConvModel2TestInst, QuantizeConvModel2Test,
                         testing::ValuesIn({TensorType_INT8}));

TEST_P(QuantizeConvModel2Test, VerifyConvQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, tensor_type_, tensor_type_, /*allow_float=*/false,
      tensor_type_, &error_reporter_);
  ASSERT_THAT(status, Eq(kTfLiteOk));
  const auto& subgraph = model_.subgraphs[0];
  auto conv_op = subgraph->operators[0].get();
  const int input_tensor_idx = 0;
  const int weights_tensor_idx = 1;
  const int bias_tensor_index = 2;
  const int output_tensor_idx = 0;
  const auto bias_tensor =
      subgraph->tensors[conv_op->inputs[bias_tensor_index]].get();
  const auto input_tensor =
      subgraph->tensors[conv_op->inputs[input_tensor_idx]].get();
  const auto weights_tensor =
      subgraph->tensors[conv_op->inputs[weights_tensor_idx]].get();
  const auto output_tensor =
      subgraph->tensors[conv_op->outputs[output_tensor_idx]].get();

  EXPECT_THAT(bias_tensor->type,
              Eq(tensor_type_ == TensorType_INT8 ? TensorType_INT32
                                                 : TensorType_INT64));
  EXPECT_THAT(input_tensor->type, Eq(tensor_type_));
  EXPECT_THAT(weights_tensor->type, Eq(TensorType_INT8));

  ASSERT_TRUE(weights_tensor->quantization);
  ASSERT_TRUE(bias_tensor->quantization);
  ASSERT_TRUE(weights_tensor->quantization);
  const std::vector<float>& bias_scales = bias_tensor->quantization->scale;
  const std::vector<float>& weights_scales =
      weights_tensor->quantization->scale;
  const std::vector<int64_t>& weights_zero_points =
      weights_tensor->quantization->zero_point;
  const int out_channel_size = weights_tensor->shape[0];
  ASSERT_THAT(bias_scales, SizeIs(out_channel_size));
  ASSERT_THAT(weights_scales, SizeIs(out_channel_size));
  ASSERT_THAT(weights_zero_points, SizeIs(out_channel_size));
  ASSERT_THAT(input_tensor->quantization->scale, SizeIs(1));
  ASSERT_THAT(output_tensor->quantization->scale, SizeIs(1));

  const float eps = 1e-7;

  // Bias scale should be input * per_channel_weight_scale.
  for (size_t i = 0; i < out_channel_size; i++) {
    EXPECT_THAT(bias_scales[i], FloatNear(input_tensor->quantization->scale[0] *
                                              weights_scales[i],
                                          eps));
  }

  const auto bias_buffer = model_.buffers[bias_tensor->buffer].get();
  auto control_size = tensor_type_ == TensorType_INT8
                          ? sizeof(int32_t) * bias_tensor->shape[0]
                          : sizeof(int64_t) * bias_tensor->shape[0];

  const auto float_op =
      readonly_model_->subgraphs()->Get(0)->operators()->Get(0);
  const auto original_bias_tensor =
      readonly_model_->subgraphs()->Get(0)->tensors()->Get(
          float_op->inputs()->Get(2));
  ASSERT_THAT(bias_buffer->data, SizeIs(control_size));
  const auto original_bias_buffer =
      readonly_model_->buffers()->Get(original_bias_tensor->buffer());
  const float* bias_float_buffer =
      reinterpret_cast<const float*>(original_bias_buffer->data()->data());

  if (tensor_type_ == TensorType_INT8) {
    int32_t* bias_values = reinterpret_cast<int32_t*>(bias_buffer->data.data());
    for (size_t i = 0; i < out_channel_size; i++) {
      auto dequantized_value = bias_values[i] * bias_scales[i];
      EXPECT_THAT(dequantized_value,
                  FloatNear(bias_float_buffer[i], bias_scales[i] / 2));
    }
  }

  const auto weights_buffer = model_.buffers[weights_tensor->buffer].get();
  const auto original_weights_tensor =
      readonly_model_->subgraphs()->Get(0)->tensors()->Get(
          float_op->inputs()->Get(1));
  const auto original_weights_buffer =
      readonly_model_->buffers()->Get(original_weights_tensor->buffer());
  const int8_t* weight_values =
      reinterpret_cast<int8_t*>(weights_buffer->data.data());
  const float* weights_float_buffer =
      reinterpret_cast<const float*>(original_weights_buffer->data()->data());
  ASSERT_THAT(sizeof(float) * weights_buffer->data.size(),
              Eq(original_weights_buffer->data()->size()));
  int num_values_in_channel = weights_buffer->data.size() / out_channel_size;
  for (size_t channel_idx = 0; channel_idx < out_channel_size; channel_idx++) {
    for (size_t j = 0; j < num_values_in_channel; j++) {
      size_t element_idx = channel_idx * out_channel_size + j;
      auto scale = weights_scales[channel_idx];
      auto zero_point = weights_zero_points[channel_idx];
      auto dequantized_value = weight_values[element_idx] * scale;
      EXPECT_THAT(dequantized_value,
                  FloatNear(weights_float_buffer[element_idx], scale / 2));
      EXPECT_THAT(zero_point, Eq(0));
    }
  }

  // check op and versioning.
  EXPECT_THAT(model_.operator_codes, SizeIs(1));
  EXPECT_THAT(GetBuiltinCode(model_.operator_codes[0].get()),
              Eq(BuiltinOperator_CONV_2D));
  ASSERT_THAT(model_.operator_codes[0]->version, Eq(3));
}

TEST_P(QuantizeConvModel2Test, VerifyConvDisablePerChannelQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, tensor_type_, tensor_type_, /*allow_float=*/false,
      tensor_type_, /*disable_per_channel=*/true, &error_reporter_);
  ASSERT_THAT(status, Eq(kTfLiteOk));
  const auto& subgraph = model_.subgraphs[0];
  auto conv_op = subgraph->operators[0].get();
  const int input_tensor_idx = 0;
  const int weights_tensor_idx = 1;
  const int bias_tensor_index = 2;
  const int output_tensor_idx = 0;
  const auto bias_tensor =
      subgraph->tensors[conv_op->inputs[bias_tensor_index]].get();
  const auto input_tensor =
      subgraph->tensors[conv_op->inputs[input_tensor_idx]].get();
  const auto weights_tensor =
      subgraph->tensors[conv_op->inputs[weights_tensor_idx]].get();
  const auto output_tensor =
      subgraph->tensors[conv_op->outputs[output_tensor_idx]].get();

  EXPECT_THAT(bias_tensor->type,
              Eq(tensor_type_ == TensorType_INT8 ? TensorType_INT32
                                                 : TensorType_INT64));
  EXPECT_THAT(input_tensor->type, Eq(tensor_type_));
  EXPECT_THAT(weights_tensor->type, Eq(TensorType_INT8));

  ASSERT_TRUE(weights_tensor->quantization);
  ASSERT_TRUE(bias_tensor->quantization);
  ASSERT_TRUE(weights_tensor->quantization);
  const std::vector<float>& bias_scales = bias_tensor->quantization->scale;
  const std::vector<float>& weights_scales =
      weights_tensor->quantization->scale;
  const std::vector<int64_t>& weights_zero_points =
      weights_tensor->quantization->zero_point;

  const int out_channel_size = 1;
  ASSERT_THAT(bias_scales, SizeIs(out_channel_size));
  ASSERT_THAT(weights_scales, SizeIs(out_channel_size));
  ASSERT_THAT(weights_zero_points, SizeIs(out_channel_size));
  ASSERT_THAT(input_tensor->quantization->scale, SizeIs(1));
  ASSERT_THAT(output_tensor->quantization->scale, SizeIs(1));

  const float eps = 1e-7;

  // Bias scale should be input * per_channel_weight_scale.
  for (size_t i = 0; i < out_channel_size; i++) {
    EXPECT_THAT(bias_scales[i], FloatNear(input_tensor->quantization->scale[0] *
                                              weights_scales[i],
                                          eps));
  }

  const auto bias_buffer = model_.buffers[bias_tensor->buffer].get();
  auto control_size = tensor_type_ == TensorType_INT8
                          ? sizeof(int32_t) * bias_tensor->shape[0]
                          : sizeof(int64_t) * bias_tensor->shape[0];

  ASSERT_THAT(bias_buffer->data, SizeIs(control_size));
  const auto float_op =
      readonly_model_->subgraphs()->Get(0)->operators()->Get(0);
  const auto original_bias_tensor =
      readonly_model_->subgraphs()->Get(0)->tensors()->Get(
          float_op->inputs()->Get(2));
  ASSERT_THAT(bias_buffer->data, SizeIs(control_size));
  const auto original_bias_buffer =
      readonly_model_->buffers()->Get(original_bias_tensor->buffer());
  const float* bias_float_buffer =
      reinterpret_cast<const float*>(original_bias_buffer->data()->data());

  if (tensor_type_ == TensorType_INT8) {
    int32_t* bias_values = reinterpret_cast<int32_t*>(bias_buffer->data.data());
    for (size_t i = 0; i < out_channel_size; i++) {
      auto dequantized_value = bias_values[i] * bias_scales[i];
      EXPECT_THAT(dequantized_value,
                  FloatNear(bias_float_buffer[i], bias_scales[i] / 2));
    }
  }

  const auto weights_buffer = model_.buffers[weights_tensor->buffer].get();
  const auto original_weights_tensor =
      readonly_model_->subgraphs()->Get(0)->tensors()->Get(
          float_op->inputs()->Get(1));
  const auto original_weights_buffer =
      readonly_model_->buffers()->Get(original_weights_tensor->buffer());
  const int8_t* weight_values =
      reinterpret_cast<int8_t*>(weights_buffer->data.data());
  const float* weights_float_buffer =
      reinterpret_cast<const float*>(original_weights_buffer->data()->data());
  ASSERT_THAT(sizeof(float) * weights_buffer->data.size(),
              Eq(original_weights_buffer->data()->size()));
  int num_values_in_channel = weights_buffer->data.size() / out_channel_size;
  for (size_t channel_idx = 0; channel_idx < out_channel_size; channel_idx++) {
    for (size_t j = 0; j < num_values_in_channel; j++) {
      size_t element_idx = channel_idx * out_channel_size + j;
      auto scale = weights_scales[channel_idx];
      auto zero_point = weights_zero_points[channel_idx];
      auto dequantized_value = weight_values[element_idx] * scale;
      EXPECT_THAT(dequantized_value,
                  FloatNear(weights_float_buffer[element_idx], scale / 2));
      EXPECT_THAT(zero_point, Eq(0));
    }
  }

  // check op and versioning.
  EXPECT_THAT(model_.operator_codes, SizeIs(1));
  EXPECT_THAT(GetBuiltinCode(model_.operator_codes[0].get()),
              Eq(BuiltinOperator_CONV_2D));
  ASSERT_THAT(model_.operator_codes[0]->version, Eq(3));
}

class QuantizeSoftmaxTest : public QuantizeModelTest {
 protected:
  QuantizeSoftmaxTest() {
    input_model_ = ReadModel(internal::kSingleSoftmaxModelMinMinus5MaxPlus5);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }
};

TEST_F(QuantizeSoftmaxTest, VerifySoftmaxQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  ASSERT_THAT(status, Eq(kTfLiteOk));

  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  // Model has a single softmax op.
  ASSERT_THAT(op->opcode_index, Eq(0));
  ASSERT_THAT(GetBuiltinCode(model_.operator_codes[0].get()),
              Eq(BuiltinOperator_SOFTMAX));

  ASSERT_THAT(op->inputs, SizeIs(1));
  ASSERT_THAT(op->outputs, SizeIs(1));
  auto float_graph = readonly_model_->subgraphs()->Get(0);

  // Verify input.
  ASSERT_THAT(float_graph->tensors()->Get(op->inputs[0])->type(),
              Eq(TensorType_FLOAT32));
  ASSERT_THAT(float_graph->tensors()->Get(op->outputs[0])->type(),
              Eq(TensorType_FLOAT32));

  EXPECT_THAT(subgraph->tensors[op->inputs[0]].get()->type,
              Eq(TensorType_INT8));
  EXPECT_THAT(subgraph->tensors[op->outputs[0]].get()->type,
              Eq(TensorType_INT8));

  auto float_input_quant_params =
      float_graph->tensors()->Get(op->inputs[0])->quantization();
  auto input_quant_params =
      subgraph->tensors[op->inputs[0]]->quantization.get();
  VerifyQuantizationScale(*float_input_quant_params, *input_quant_params,
                          /*bit_num=*/8, /*symmetric=*/false);

  // Verify output.
  auto float_output_quant_params =
      float_graph->tensors()->Get(op->outputs[0])->quantization();
  auto output_quant_params =
      subgraph->tensors[op->outputs[0]]->quantization.get();
  ASSERT_THAT(*float_output_quant_params->min(), SizeIs(1));
  ASSERT_THAT(*float_output_quant_params->max(), SizeIs(1));

  ASSERT_THAT(output_quant_params->scale, SizeIs(1));
  ASSERT_THAT(output_quant_params->zero_point, SizeIs(1));
  ASSERT_THAT(1.0f / 256.0f, Eq(output_quant_params->scale[0]));
  ASSERT_THAT(-128, Eq(output_quant_params->zero_point[0]));

  // check op and versioning.
  EXPECT_THAT(model_.operator_codes, SizeIs(1));
  EXPECT_THAT(GetBuiltinCode(model_.operator_codes[0].get()),
              Eq(BuiltinOperator_SOFTMAX));
  ASSERT_THAT(model_.operator_codes[0]->version, Eq(2));
}

class QuantizeAvgPoolTest : public QuantizeModelTest {
 protected:
  QuantizeAvgPoolTest() {
    input_model_ = ReadModel(internal::kSingleAvgPoolModelMinMinus5MaxPlus5);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }
};

TEST_F(QuantizeAvgPoolTest, VerifyAvgPoolQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  ASSERT_THAT(status, Eq(kTfLiteOk));

  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  // Model has a single AveragePool op.
  ASSERT_THAT(op->opcode_index, Eq(0));
  ASSERT_THAT(GetBuiltinCode(model_.operator_codes[0].get()),
              Eq(BuiltinOperator_AVERAGE_POOL_2D));

  ASSERT_THAT(op->inputs, SizeIs(1));
  ASSERT_THAT(op->outputs, SizeIs(1));

  auto float_graph = readonly_model_->subgraphs()->Get(0);
  ASSERT_THAT(float_graph->tensors()->Get(op->inputs[0])->type(),
              Eq(TensorType_FLOAT32));
  ASSERT_THAT(float_graph->tensors()->Get(op->outputs[0])->type(),
              Eq(TensorType_FLOAT32));

  EXPECT_THAT(subgraph->tensors[op->inputs[0]].get()->type,
              Eq(TensorType_INT8));
  EXPECT_THAT(subgraph->tensors[op->outputs[0]].get()->type,
              Eq(TensorType_INT8));

  auto float_input_quant_params =
      float_graph->tensors()->Get(op->inputs[0])->quantization();
  auto input_quant_params =
      subgraph->tensors[op->inputs[0]]->quantization.get();
  VerifyQuantizationScale(*float_input_quant_params, *input_quant_params,
                          /*bit_num=*/8, /*symmetric=*/false);

  auto float_output_quant_params =
      float_graph->tensors()->Get(op->outputs[0])->quantization();
  auto output_quant_params =
      subgraph->tensors[op->outputs[0]]->quantization.get();
  ASSERT_THAT(*float_output_quant_params->min(), SizeIs(1));
  ASSERT_THAT(*float_output_quant_params->max(), SizeIs(1));
  ASSERT_THAT(output_quant_params->scale, SizeIs(1));

  // Make sure the input min/maxes are propagated to outputs.
  EXPECT_THAT(input_quant_params->scale[0], Eq(output_quant_params->scale[0]));

  // check op and versioning.
  EXPECT_THAT(model_.operator_codes, SizeIs(1));
  EXPECT_THAT(GetBuiltinCode(model_.operator_codes[0].get()),
              Eq(BuiltinOperator_AVERAGE_POOL_2D));
  ASSERT_THAT(model_.operator_codes[0]->version, Eq(2));
}

class QuantizeMultiInputAddWithReshapeTest : public QuantizeModelTest {
 protected:
  QuantizeMultiInputAddWithReshapeTest() {
    input_model_ = ReadModel(internal::kMultiInputAddWithReshape);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }
};

TEST_F(QuantizeMultiInputAddWithReshapeTest, VerifyReshapeQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);

  ASSERT_THAT(status, Eq(kTfLiteOk));

  // Verify Reshape is quantized.
  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[1].get();
  ASSERT_THAT(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
              Eq(BuiltinOperator_RESHAPE));

  ASSERT_THAT(op->inputs, SizeIs(2));
  ASSERT_THAT(op->outputs, SizeIs(1));

  auto float_graph = readonly_model_->subgraphs()->Get(0);
  auto float_op = float_graph->operators()->Get(1);
  ASSERT_THAT(float_graph->tensors()->Get(float_op->inputs()->Get(0))->type(),
              Eq(TensorType_FLOAT32));
  ASSERT_THAT(float_graph->tensors()->Get(float_op->outputs()->Get(0))->type(),
              Eq(TensorType_FLOAT32));

  EXPECT_THAT(subgraph->tensors[op->inputs[0]].get()->type,
              Eq(TensorType_INT8));
  EXPECT_THAT(subgraph->tensors[op->outputs[0]].get()->type,
              Eq(TensorType_INT8));
  auto float_input_quant_params =
      float_graph->tensors()->Get(op->inputs[0])->quantization();
  auto input_quant_params =
      subgraph->tensors[op->inputs[0]]->quantization.get();
  VerifyQuantizationScale(*float_input_quant_params, *input_quant_params,
                          /*bit_num=*/8, /*symmetric=*/false);

  auto float_output_quant_params =
      float_graph->tensors()->Get(float_op->outputs()->Get(0))->quantization();
  auto output_quant_params =
      subgraph->tensors[op->outputs[0]]->quantization.get();
  ASSERT_THAT(*float_output_quant_params->min(), SizeIs(1));
  ASSERT_THAT(*float_output_quant_params->max(), SizeIs(1));
  ASSERT_THAT(output_quant_params->scale, SizeIs(1));

  // check op and versioning.
  EXPECT_THAT(model_.operator_codes, SizeIs(2));
  EXPECT_THAT(GetBuiltinCode(model_.operator_codes[0].get()),
              Eq(BuiltinOperator_ADD));
  ASSERT_THAT(model_.operator_codes[0]->version, Eq(2));
  EXPECT_THAT(GetBuiltinCode(model_.operator_codes[1].get()),
              Eq(BuiltinOperator_RESHAPE));
  ASSERT_THAT(model_.operator_codes[1]->version, Eq(1));
}

TEST_F(QuantizeMultiInputAddWithReshapeTest, VerifyAddQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  ASSERT_THAT(status, Eq(kTfLiteOk));

  // Verify ADD is quantized.
  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  ASSERT_THAT(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
              Eq(BuiltinOperator_ADD));

  ASSERT_THAT(op->inputs, SizeIs(2));
  ASSERT_THAT(op->outputs, SizeIs(1));

  auto float_graph = readonly_model_->subgraphs()->Get(0);
  auto float_op = float_graph->operators()->Get(0);
  const int float_input0_idx = float_op->inputs()->Get(0);
  const int float_input1_idx = float_op->inputs()->Get(1);
  const int float_output_idx = float_op->outputs()->Get(0);
  ASSERT_THAT(float_graph->tensors()->Get(float_input0_idx)->type(),
              Eq(TensorType_FLOAT32));
  ASSERT_THAT(float_graph->tensors()->Get(float_input1_idx)->type(),
              Eq(TensorType_FLOAT32));
  ASSERT_THAT(float_graph->tensors()->Get(float_output_idx)->type(),
              Eq(TensorType_FLOAT32));

  for (size_t input_idx = 0; input_idx < 2; ++input_idx) {
    EXPECT_THAT(subgraph->tensors[op->inputs[input_idx]].get()->type,
                Eq(TensorType_INT8));
    auto float_input_quant_params =
        float_graph->tensors()
            ->Get(float_op->inputs()->Get(input_idx))
            ->quantization();
    auto input_quant_params =
        subgraph->tensors[op->inputs[input_idx]]->quantization.get();
    VerifyQuantizationScale(*float_input_quant_params, *input_quant_params,
                            /*bit_num=*/8, /*symmetric=*/false);
  }

  EXPECT_THAT(subgraph->tensors[op->outputs[0]].get()->type,
              Eq(TensorType_INT8));
  auto float_output_quant_params =
      float_graph->tensors()->Get(op->outputs[0])->quantization();
  auto output_quant_params =
      subgraph->tensors[op->outputs[0]]->quantization.get();
  ASSERT_THAT(*float_output_quant_params->min(), SizeIs(1));
  ASSERT_THAT(*float_output_quant_params->max(), SizeIs(1));
  ASSERT_THAT(output_quant_params->scale, SizeIs(1));

  // check op and versioning.
  EXPECT_THAT(model_.operator_codes, SizeIs(2));
  EXPECT_THAT(GetBuiltinCode(model_.operator_codes[0].get()),
              Eq(BuiltinOperator_ADD));
  ASSERT_THAT(model_.operator_codes[0]->version, Eq(2));
  EXPECT_THAT(GetBuiltinCode(model_.operator_codes[1].get()),
              Eq(BuiltinOperator_RESHAPE));
  ASSERT_THAT(model_.operator_codes[1]->version, Eq(1));
}

class QuantizeConstInputTest : public QuantizeModelTest,
                               public testing::WithParamInterface<TensorType> {
 protected:
  QuantizeConstInputTest() {
    tensor_type_ = GetParam();
    input_model_ = ReadModel(internal::kConstInputAddModel);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }

  TensorType tensor_type_;
};
INSTANTIATE_TEST_SUITE_P(QuantizeConstInputTestInst, QuantizeConstInputTest,
                         testing::ValuesIn({TensorType_INT8}));

TEST_P(QuantizeConstInputTest, VerifyConstOpInput) {
  auto status =
      QuantizeModelAllOperators(
          &builder_, &model_, tensor_type_, tensor_type_, /*allow_float=*/false,
          tensor_type_, &error_reporter_);
  ASSERT_THAT(status, Eq(kTfLiteOk));

  // Verify ConstOp is quantized.
  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  ASSERT_THAT(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
              Eq(BuiltinOperator_ADD));

  ASSERT_THAT(op->inputs, SizeIs(2));
  ASSERT_THAT(op->outputs, SizeIs(1));

  auto float_graph = readonly_model_->subgraphs()->Get(0);
  ASSERT_THAT(float_graph->tensors()->Get(op->inputs[0])->type(),
              Eq(TensorType_FLOAT32));
  ASSERT_THAT(float_graph->tensors()->Get(op->outputs[0])->type(),
              Eq(TensorType_FLOAT32));

  for (size_t input_idx = 0; input_idx < 2; ++input_idx) {
    EXPECT_THAT(subgraph->tensors[op->inputs[input_idx]].get()->type,
                Eq(tensor_type_));
  }

  EXPECT_THAT(subgraph->tensors[op->outputs[0]].get()->type, Eq(tensor_type_));

  // check op and versioning.
  EXPECT_THAT(model_.operator_codes, SizeIs(1));
  EXPECT_THAT(GetBuiltinCode(model_.operator_codes[0].get()),
              Eq(BuiltinOperator_ADD));
  ASSERT_THAT(model_.operator_codes[0]->version, Eq(2));
}

class QuantizeArgMaxTest : public QuantizeModelTest {
 protected:
  QuantizeArgMaxTest() {
    input_model_ = ReadModel(internal::kModelWithArgMaxOp);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }
};

TEST_F(QuantizeArgMaxTest, VerifyArgMax) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  ASSERT_THAT(status, Eq(kTfLiteOk));

  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  ASSERT_THAT(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
              Eq(BuiltinOperator_ARG_MAX));

  ASSERT_THAT(op->inputs, SizeIs(2));
  ASSERT_THAT(op->outputs, SizeIs(1));

  auto float_graph = readonly_model_->subgraphs()->Get(0);
  auto float_op = float_graph->operators()->Get(0);
  // Verify ArgMax input is quantized.
  ASSERT_THAT(float_graph->tensors()->Get(float_op->inputs()->Get(0))->type(),
              Eq(TensorType_FLOAT32));
  EXPECT_THAT(subgraph->tensors[op->inputs[0]].get()->type,
              Eq(TensorType_INT8));

  // Verify ArgMax input axis should still be the same type.
  ASSERT_THAT(float_graph->tensors()->Get(float_op->inputs()->Get(1))->type(),
              Eq(subgraph->tensors[op->inputs[1]].get()->type));

  // The output of ArgMax should still be the same type.
  ASSERT_THAT(float_graph->tensors()->Get(float_op->outputs()->Get(0))->type(),
              Eq(subgraph->tensors[op->outputs[0]].get()->type));

  // check op and versioning.
  EXPECT_THAT(model_.operator_codes, SizeIs(1));
  EXPECT_THAT(GetBuiltinCode(model_.operator_codes[0].get()),
              Eq(BuiltinOperator_ARG_MAX));
  ASSERT_THAT(model_.operator_codes[0]->version, Eq(2));
}

class QuantizeLSTMTest : public QuantizeModelTest {
 protected:
  QuantizeLSTMTest() {
    input_model_ = ReadModel(internal::kLstmCalibrated);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }
};

TEST_F(QuantizeLSTMTest, VerifyLSTM) {
  // Quantize model.
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_FLOAT32, TensorType_FLOAT32, true,
      TensorType_INT8, &error_reporter_);
  ASSERT_THAT(status, Eq(kTfLiteOk));

  // Read expected model.
  auto expected_fb_model = ReadModel(internal::kLstmQuantized);
  auto expected_read_only_model = expected_fb_model->GetModel();
  ModelT expected_model;
  expected_read_only_model->UnPackTo(&expected_model);

  ExpectSameModels(model_, expected_model);
}

class QuantizeLSTM2Test : public QuantizeModelTest {
 protected:
  QuantizeLSTM2Test() {
    input_model_ = ReadModel(internal::kLstmCalibrated2);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }
};

TEST_F(QuantizeLSTM2Test, VerifyLSTM) {
  // Quantize model.
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_FLOAT32, TensorType_FLOAT32,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  ASSERT_THAT(status, Eq(kTfLiteOk));

  // Read expected model.
  auto expected_fb_model = ReadModel(internal::kLstmQuantized2);
  auto expected_read_only_model = expected_fb_model->GetModel();
  ModelT expected_model;
  expected_read_only_model->UnPackTo(&expected_model);

  ExpectSameModels(model_, expected_model);
}

class QuantizeUnidirectionalSequenceLSTMTest : public QuantizeModelTest {
 protected:
  QuantizeUnidirectionalSequenceLSTMTest() {
    input_model_ = ReadModel(internal::kUnidirectionalSequenceLstmCalibrated);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }
};

TEST_F(QuantizeUnidirectionalSequenceLSTMTest,
       VerifyUnidirectionalSequenceLSTM) {
  // Quantize model.
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_FLOAT32, TensorType_FLOAT32,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  ASSERT_THAT(status, Eq(kTfLiteOk));

  // Read expected model.
  auto expected_fb_model =
      ReadModel(internal::kUnidirectionalSequenceLstmQuantized);
  auto expected_read_only_model = expected_fb_model->GetModel();
  ModelT expected_model;
  expected_read_only_model->UnPackTo(&expected_model);

  ExpectSameModels(model_, expected_model);
}

class QuantizeSVDFTest : public QuantizeModelTest {
 protected:
  QuantizeSVDFTest() {
    input_model_ = ReadModel(internal::kSvdfCalibrated);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }
};

TEST_F(QuantizeSVDFTest, VerifySVDF) {
  // Quantize model.
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  ASSERT_THAT(status, Eq(kTfLiteOk));

  // Read expected model.
  auto expected_fb_model = ReadModel(internal::kSvdfQuantized);
  auto expected_read_only_model = expected_fb_model->GetModel();
  ModelT expected_model;
  expected_read_only_model->UnPackTo(&expected_model);

  ExpectSameModels(model_, expected_model);
}

class QuantizeFCTest : public QuantizeModelTest {
 protected:
  QuantizeFCTest() {
    input_model_ = ReadModel(internal::kModelWithFCOp);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }
};

TEST_F(QuantizeFCTest, VerifyFC8x8) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  ASSERT_THAT(status, Eq(kTfLiteOk));

  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  ASSERT_THAT(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
              Eq(BuiltinOperator_FULLY_CONNECTED));

  ASSERT_THAT(op->inputs, SizeIs(3));
  ASSERT_THAT(op->outputs, SizeIs(1));

  auto float_graph = readonly_model_->subgraphs()->Get(0);
  // Verify FC input tensor and weight are int8 quantized.
  auto float_op = float_graph->operators()->Get(0);
  ASSERT_THAT(float_graph->tensors()->Get(float_op->inputs()->Get(0))->type(),
              Eq(TensorType_FLOAT32));
  EXPECT_THAT(subgraph->tensors[op->inputs[0]].get()->type,
              Eq(TensorType_INT8));
  ASSERT_THAT(float_graph->tensors()->Get(float_op->inputs()->Get(1))->type(),
              Eq(TensorType_FLOAT32));
  EXPECT_THAT(subgraph->tensors[op->inputs[1]].get()->type,
              Eq(TensorType_INT8));

  // Verify FC bias should be int32 quantized.
  ASSERT_THAT(float_graph->tensors()->Get(float_op->inputs()->Get(2))->type(),
              Eq(TensorType_FLOAT32));
  EXPECT_THAT(subgraph->tensors[op->inputs[2]].get()->type,
              Eq(TensorType_INT32));

  // The output tensor of FC should be int8 quantized.
  ASSERT_THAT(float_graph->tensors()->Get(float_op->outputs()->Get(0))->type(),
              Eq(TensorType_FLOAT32));
  EXPECT_THAT(subgraph->tensors[op->outputs[0]].get()->type,
              Eq(TensorType_INT8));

  // Check op and versioning.
  EXPECT_THAT(model_.operator_codes, SizeIs(1));
  EXPECT_THAT(GetBuiltinCode(model_.operator_codes[0].get()),
              Eq(BuiltinOperator_FULLY_CONNECTED));
  ASSERT_THAT(model_.operator_codes[0]->version, Eq(5));

  // check the scale
  auto float_input_quant_params =
      float_graph->tensors()->Get(op->inputs[0])->quantization();
  auto input_quant_params =
      subgraph->tensors[op->inputs[0]]->quantization.get();
  VerifyQuantizationScale(*float_input_quant_params, *input_quant_params,
                          /*bit_num=*/8, /*symmetric=*/false);
}

TEST_F(QuantizeFCTest, VerifyFCFor16x8) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT16, &error_reporter_);
  ASSERT_THAT(status, Eq(kTfLiteOk));

  const std::unique_ptr<tflite::SubGraphT>& subgraph = model_.subgraphs[0];
  const tflite::OperatorT* op = subgraph->operators[0].get();
  ASSERT_THAT(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
              Eq(BuiltinOperator_FULLY_CONNECTED));

  ASSERT_THAT(op->inputs, SizeIs(3));
  ASSERT_THAT(op->outputs, SizeIs(1));

  const SubGraph* float_graph = readonly_model_->subgraphs()->Get(0);
  // Verify FC input tesnor and weight are int16 and int8 quantized.
  const Operator* float_op = float_graph->operators()->Get(0);
  ASSERT_THAT(float_graph->tensors()->Get(float_op->inputs()->Get(0))->type(),
              Eq(TensorType_FLOAT32));
  EXPECT_THAT(subgraph->tensors[op->inputs[0]].get()->type,
              Eq(TensorType_INT16));
  ASSERT_THAT(float_graph->tensors()->Get(float_op->inputs()->Get(1))->type(),
              Eq(TensorType_FLOAT32));
  EXPECT_THAT(subgraph->tensors[op->inputs[1]].get()->type,
              Eq(TensorType_INT8));

  // Verify FC bias should be int32 quantized.
  ASSERT_THAT(float_graph->tensors()->Get(float_op->inputs()->Get(2))->type(),
              Eq(TensorType_FLOAT32));
  EXPECT_THAT(subgraph->tensors[op->inputs[2]].get()->type,
              Eq(TensorType_INT32));

  // The output tensor of FC should be quantized as int16.
  ASSERT_THAT(float_graph->tensors()->Get(float_op->outputs()->Get(0))->type(),
              Eq(TensorType_FLOAT32));
  EXPECT_THAT(subgraph->tensors[op->outputs[0]].get()->type,
              Eq(TensorType_INT16));

  // Check op and versioning.
  EXPECT_THAT(model_.operator_codes, SizeIs(1));
  EXPECT_THAT(GetBuiltinCode(model_.operator_codes[0].get()),
              Eq(BuiltinOperator_FULLY_CONNECTED));
  ASSERT_THAT(model_.operator_codes[0]->version, Eq(11));

  // Check the scale value. The scale value will be smaller than the int8 scale
  // since the scale is calculated by dividing by 2^bit_num.
  const int32_t target_input_index = op->inputs[0];
  const QuantizationParameters* float_input_quant_params =
      float_graph->tensors()->Get(target_input_index)->quantization();
  const QuantizationParametersT* input_quant_params =
      subgraph->tensors[target_input_index]->quantization.get();
  VerifyQuantizationScale(*float_input_quant_params, *input_quant_params,
                          /*bit_num=*/16, /*symmetric=*/true);
}

class QuantizeCustomOpTest
    : public QuantizeModelTest,
      public ::testing::WithParamInterface<tflite::TensorType> {
 protected:
  QuantizeCustomOpTest() {
    input_model_ = ReadModel(internal::kModelMixed);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }
};

TEST_P(QuantizeCustomOpTest, VerifyMixedQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, GetParam(), GetParam(),
      /*allow_float=*/true, GetParam(), &error_reporter_);
  ASSERT_THAT(status, Eq(kTfLiteOk));
  const auto& subgraph = model_.subgraphs[0];
  auto float_graph = readonly_model_->subgraphs()->Get(0);
  // The original model reshape->custom->custom->squeeze.
  ASSERT_THAT(*float_graph->operators(), SizeIs(4));
  // The resulting model should be:
  // reshape->dequantize->custom->custom->quantize->squeeze.
  ASSERT_THAT(subgraph->operators, SizeIs(6));
  const std::vector<BuiltinOperator> op_codes = {
      BuiltinOperator_RESHAPE,  BuiltinOperator_DEQUANTIZE,
      BuiltinOperator_CUSTOM,   BuiltinOperator_CUSTOM,
      BuiltinOperator_QUANTIZE, BuiltinOperator_SQUEEZE};
  const std::vector<TensorType> op_input_types = {
      GetParam(),         GetParam(),         TensorType_FLOAT32,
      TensorType_FLOAT32, TensorType_FLOAT32, GetParam()};
  for (int i = 0; i < subgraph->operators.size(); ++i) {
    OperatorT* op = subgraph->operators[i].get();
    ASSERT_THAT(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
                Eq(op_codes[i]));
    ASSERT_THAT(subgraph->tensors[op->inputs[0]]->type, Eq(op_input_types[i]));
  }
}

INSTANTIATE_TEST_SUITE_P(QuantizeCustomOpTest, QuantizeCustomOpTest,
                         ::testing::Values(TensorType_INT8));

class QuantizePackTest : public QuantizeModelTest {
 protected:
  QuantizePackTest() {
    input_model_ = ReadModel(internal::kModelPack);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }
};

TEST_F(QuantizePackTest, VerifyPack) {
  auto status = QuantizeModel(&builder_, &model_, &error_reporter_);

  ASSERT_THAT(status, Eq(kTfLiteOk));

  const auto subgraph = model_.subgraphs[0].get();

  // The model should only have 3 inputs and 1 output.
  EXPECT_THAT(subgraph->inputs, SizeIs(3));
  EXPECT_THAT(subgraph->outputs, SizeIs(1));

  const auto& op1 = subgraph->operators[1].get();
  const auto& op2 = subgraph->operators[2].get();
  const auto& op3 = subgraph->operators[3].get();
  const auto& op4 = subgraph->operators[4].get();

  ASSERT_THAT(GetBuiltinCode(model_.operator_codes[op1->opcode_index].get()),
              Eq(BuiltinOperator_QUANTIZE));
  ASSERT_THAT(GetBuiltinCode(model_.operator_codes[op2->opcode_index].get()),
              Eq(BuiltinOperator_QUANTIZE));
  ASSERT_THAT(GetBuiltinCode(model_.operator_codes[op3->opcode_index].get()),
              Eq(BuiltinOperator_PACK));
  ASSERT_THAT(GetBuiltinCode(model_.operator_codes[op4->opcode_index].get()),
              Eq(BuiltinOperator_DEQUANTIZE));

  const auto& pack_input0 = subgraph->tensors[op3->inputs[0]].get();
  const auto& pack_input1 = subgraph->tensors[op3->inputs[1]].get();
  const auto& pack_input2 = subgraph->tensors[op3->inputs[2]].get();

  const auto& pack_output = subgraph->tensors[op3->outputs[0]].get();

  // Check quantization parameters for input and output.
  EXPECT_THAT(pack_input0->quantization->scale[0],
              FloatEq(pack_input1->quantization->scale[0]));
  EXPECT_THAT(pack_input1->quantization->scale[0],
              FloatEq(pack_input2->quantization->scale[0]));
  EXPECT_THAT(pack_input0->quantization->zero_point[0],
              Eq(pack_input1->quantization->zero_point[0]));
  EXPECT_THAT(pack_input1->quantization->zero_point[0],
              Eq(pack_input2->quantization->zero_point[0]));

  EXPECT_THAT(pack_input1->quantization->scale[0],
              FloatEq(pack_output->quantization->scale[0]));
  EXPECT_THAT(pack_input1->quantization->zero_point[0],
              Eq(pack_output->quantization->zero_point[0]));

  // Check type of input and output.
  EXPECT_THAT(pack_output->type, Eq(TensorType_INT8));
  EXPECT_THAT(pack_input0->type, Eq(TensorType_INT8));
  EXPECT_THAT(pack_input1->type, Eq(TensorType_INT8));
  EXPECT_THAT(pack_input2->type, Eq(TensorType_INT8));
}

class QuantizeMinimumMaximumTest
    : public QuantizeModelTest,
      public testing::WithParamInterface<const char*> {
 protected:
  QuantizeMinimumMaximumTest() {
    input_model_ = ReadModel(GetParam());
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }
};

TEST_P(QuantizeMinimumMaximumTest, VerifyMinimumMaximum) {
  auto status = QuantizeModel(&builder_, &model_, &error_reporter_);
  ASSERT_THAT(status, Eq(kTfLiteOk));
  const auto& subgraph = model_.subgraphs[0];
  // Check that the first op is Quantize and the last is Dequant.
  const auto& quant_op = subgraph->operators[0];
  const auto& dequant_op = subgraph->operators[subgraph->operators.size() - 1];
  const int32_t quant_idx = quant_op->opcode_index;
  const int32_t dequant_idx = dequant_op->opcode_index;
  EXPECT_THAT(GetBuiltinCode(model_.operator_codes[quant_idx].get()),
              Eq(BuiltinOperator_QUANTIZE));
  EXPECT_THAT(GetBuiltinCode(model_.operator_codes[dequant_idx].get()),
              Eq(BuiltinOperator_DEQUANTIZE));

  const auto& op = subgraph->operators[1].get();

  // Check that we have MINIMUM or MAXIMUM operator.
  auto op_builtin_code =
      GetBuiltinCode(model_.operator_codes[op->opcode_index].get());
  ASSERT_TRUE(op_builtin_code == tflite::BuiltinOperator_MINIMUM ||
              op_builtin_code == tflite::BuiltinOperator_MAXIMUM);

  // Check that we have two inputs and one output.
  ASSERT_THAT(op->inputs, SizeIs(2));
  ASSERT_THAT(op->outputs, SizeIs(1));

  // Check that all is quantized.
  auto output = subgraph->tensors[op->outputs[0]].get();
  auto input1 = subgraph->tensors[op->inputs[0]].get();
  auto input2 = subgraph->tensors[op->inputs[1]].get();

  EXPECT_THAT(output->type, Eq(TensorType_INT8));
  EXPECT_THAT(input1->type, Eq(TensorType_INT8));
  EXPECT_THAT(input2->type, Eq(TensorType_INT8));

  // Check if the quantization params of the minimum/maximum inputs match
  // after requantization
  EXPECT_THAT(input1->quantization->scale, Eq(input2->quantization->scale));
  EXPECT_THAT(input1->quantization->zero_point,
              Eq(input2->quantization->zero_point));

  // Check the input quantization params match the output ones.
  EXPECT_THAT(output->quantization->scale, Eq(input1->quantization->scale));
  EXPECT_THAT(output->quantization->zero_point,
              Eq(input1->quantization->zero_point));
  EXPECT_THAT(output->quantization->scale, Eq(input2->quantization->scale));
  EXPECT_THAT(output->quantization->zero_point,
              Eq(input2->quantization->zero_point));
}

INSTANTIATE_TEST_SUITE_P(MinimumMaximumTestInst, QuantizeMinimumMaximumTest,
                         testing::ValuesIn({internal::kModelWithMinimumOp,
                                            internal::kModelWithMaximumOp}));

class QuantizeUnpackTest : public QuantizeModelTest {
 protected:
  QuantizeUnpackTest() {
    input_model_ = ReadModel(internal::kModelWithUnpack);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }
};

TEST_F(QuantizeUnpackTest, VerifyUnpack) {
  auto status = QuantizeModel(&builder_, &model_, &error_reporter_);

  ASSERT_THAT(status, Eq(kTfLiteOk));

  const auto subgraph = model_.subgraphs[0].get();
  auto op = subgraph->operators[1].get();

  auto float_graph = readonly_model_->subgraphs()->Get(0);

  ASSERT_THAT(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
              Eq(BuiltinOperator_UNPACK));

  // Get unpack input and output tensors
  auto unpack_input = subgraph->tensors[op->inputs[0]].get();
  auto unpack_output_0 = subgraph->tensors[op->outputs[0]].get();
  auto unpack_output_1 = subgraph->tensors[op->outputs[1]].get();

  // Verify Unpack input is quantized.
  ASSERT_THAT(float_graph->tensors()->Get(op->inputs[0])->type(),
              Eq(TensorType_FLOAT32));
  EXPECT_THAT(unpack_input->type, Eq(TensorType_INT8));

  // The model should only have one input and 2 outputs.
  EXPECT_THAT(subgraph->inputs, SizeIs(1));
  EXPECT_THAT(subgraph->outputs, SizeIs(2));

  // Ensure quantization parameters before and after unpack
  // are preserved after quantization for all outputs of
  // unpack.
  EXPECT_THAT(unpack_input->quantization->scale[0],
              FloatEq(unpack_output_0->quantization->scale[0]));
  EXPECT_THAT(unpack_input->quantization->scale[0],
              FloatEq(unpack_output_1->quantization->scale[0]));
  EXPECT_THAT(unpack_input->quantization->zero_point[0],
              Eq(unpack_output_0->quantization->zero_point[0]));
  EXPECT_THAT(unpack_input->quantization->zero_point[0],
              Eq(unpack_output_1->quantization->zero_point[0]));
}

class QuantizeBroadcastToModelTest
    : public QuantizeModelTest,
      public testing::WithParamInterface<TensorType> {
 protected:
  QuantizeBroadcastToModelTest() {
    tensor_type_ = GetParam();
    input_model_ = ReadModel(internal::kModelWithBroadcastToOp);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }
  TensorType tensor_type_;
};

INSTANTIATE_TEST_SUITE_P(QuantizeBroadcastToModelTestInst,
                         QuantizeBroadcastToModelTest,
                         testing::ValuesIn({TensorType_INT8}));

TEST_P(QuantizeBroadcastToModelTest, VerifyBroadcastToQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, tensor_type_, tensor_type_, /*allow_float=*/false,
      tensor_type_, &error_reporter_);
  EXPECT_THAT(status, Eq(kTfLiteOk));

  // There is only one subgraph.
  const int32_t subgraph_idx = 0;
  const auto& subgraph = model_.subgraphs[subgraph_idx];
  const auto& readonly_subgraph =
      readonly_model_->subgraphs()->Get(subgraph_idx);

  // There should be a single broadcast_to op.
  EXPECT_THAT(*readonly_subgraph->operators(), SizeIs(1));
  EXPECT_THAT(subgraph->operators, SizeIs(1));
  const auto& broadcast_to = subgraph->operators[0];
  EXPECT_THAT(model_.operator_codes[broadcast_to->opcode_index]->builtin_code,
              Eq(BuiltinOperator_BROADCAST_TO));

  // There should be 3 tensors: input, output, and BroadcastTo/shape.
  EXPECT_THAT(subgraph->tensors, SizeIs(3));

  // Input Tensor
  EXPECT_THAT(subgraph->tensors[0]->type, Eq(tensor_type_));
  EXPECT_THAT(subgraph->tensors[0]->name, Eq("input_1"));
  EXPECT_THAT(subgraph->tensors[0]->quantization->scale, SizeIs(1));
  EXPECT_THAT(subgraph->tensors[0]->quantization->zero_point, SizeIs(1));

  // Output Tensor. The name given in the generated
  // .bin test file is 'Identity' and should be preserved
  EXPECT_THAT(subgraph->tensors[2]->type, Eq(tensor_type_));
  EXPECT_THAT(subgraph->tensors[2]->name, Eq("Identity"));
  EXPECT_THAT(subgraph->tensors[2]->quantization->scale, SizeIs(1));
  EXPECT_THAT(subgraph->tensors[2]->quantization->zero_point, SizeIs(1));

  // The BroadCastTo shape is of type INT32 and should not be quantized
  EXPECT_THAT(subgraph->tensors[1]->type, Eq(TensorType_INT32));
  EXPECT_THAT(subgraph->tensors[1]->name,
              Eq("model/tf.broadcast_to/BroadcastTo/shape"));
  EXPECT_THAT(subgraph->tensors[1]->quantization->scale, IsEmpty());
  EXPECT_THAT(subgraph->tensors[1]->quantization->zero_point, IsEmpty());

  // check op and versioning.
  EXPECT_THAT(model_.operator_codes, SizeIs(1));
  EXPECT_THAT(model_.operator_codes[0]->builtin_code,
              Eq(BuiltinOperator_BROADCAST_TO));
  ASSERT_THAT(model_.operator_codes[0]->version, Eq(3));
}

class QuantizeGatherNDModelTest
    : public QuantizeModelTest,
      public testing::WithParamInterface<TensorType> {
 protected:
  QuantizeGatherNDModelTest() {
    tensor_type_ = GetParam();
    input_model_ = ReadModel(internal::kModelWithGatherNDOp);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }

  TensorType tensor_type_;
};

INSTANTIATE_TEST_SUITE_P(QuantizeGatherNDModelTestInst,
                         QuantizeGatherNDModelTest,
                         testing::ValuesIn({TensorType_INT8}));

TEST_P(QuantizeGatherNDModelTest, QuantizeGatherND) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, tensor_type_, tensor_type_, /*allow_float=*/false,
      tensor_type_, &error_reporter_);
  EXPECT_THAT(status, Eq(kTfLiteOk));

  // There is only one subgraph.
  const int32_t subgraph_idx = 0;
  const auto& subgraph = model_.subgraphs[subgraph_idx];
  const auto& readonly_subgraph =
      readonly_model_->subgraphs()->Get(subgraph_idx);

  // There should be a single gather_nd op.
  EXPECT_THAT(*readonly_subgraph->operators(), SizeIs(1));
  EXPECT_THAT(subgraph->operators, SizeIs(1));
  const auto& gather_nd = subgraph->operators[0];
  EXPECT_THAT(model_.operator_codes[gather_nd->opcode_index]->builtin_code,
              Eq(BuiltinOperator_GATHER_ND));

  // There should be 3 tensors: input, output, and indices.
  EXPECT_THAT(subgraph->tensors, SizeIs(3));

  // Input Tensor
  EXPECT_THAT(subgraph->tensors[0]->type, Eq(tensor_type_));
  EXPECT_THAT(subgraph->tensors[0]->name, Eq("input"));
  EXPECT_THAT(subgraph->tensors[0]->quantization->scale, SizeIs(1));
  EXPECT_THAT(subgraph->tensors[0]->quantization->zero_point, SizeIs(1));

  // Output Tensor
  EXPECT_THAT(subgraph->tensors[2]->type, Eq(tensor_type_));
  EXPECT_THAT(subgraph->tensors[2]->name, Eq("output"));
  EXPECT_THAT(subgraph->tensors[2]->quantization->scale, SizeIs(1));
  EXPECT_THAT(subgraph->tensors[2]->quantization->zero_point, SizeIs(1));

  // The gather indices are of type INT32 and should not be quantized
  EXPECT_THAT(subgraph->tensors[1]->type, Eq(TensorType_INT32));
  EXPECT_THAT(subgraph->tensors[1]->name, Eq("indices"));
  EXPECT_THAT(subgraph->tensors[1]->quantization->scale, IsEmpty());
  EXPECT_THAT(subgraph->tensors[1]->quantization->zero_point, IsEmpty());

  // Check op and versioning.
  EXPECT_THAT(model_.operator_codes, SizeIs(1));
  EXPECT_THAT(model_.operator_codes[0]->builtin_code,
              Eq(BuiltinOperator_GATHER_ND));
  ASSERT_THAT(model_.operator_codes[0]->version, Eq(1));
}

class QuantizeWhereModelTest : public QuantizeModelTest {
 protected:
  QuantizeWhereModelTest() {
    input_model_ = ReadModel(internal::kModelWithWhereOp);
    readonly_model_ = input_model_->GetModel();
    model_ = UnPackFlatBufferModel(*readonly_model_);
  }
};

TEST_F(QuantizeWhereModelTest, QuantizeWhere) {
  // Where operator takes a BOOL tensor as input
  // and outputs INT64 indices, both of which
  // should not be quantized
  auto status = QuantizeModel(&builder_, &model_, TensorType_BOOL,
                              TensorType_INT64, &error_reporter_);
  EXPECT_THAT(status, Eq(kTfLiteOk));

  // There is only one subgraph.
  const int32_t subgraph_idx = 0;
  const auto& subgraph = model_.subgraphs[subgraph_idx];
  const auto& readonly_subgraph =
      readonly_model_->subgraphs()->Get(subgraph_idx);

  // There should be a single where op.
  EXPECT_THAT(*readonly_subgraph->operators(), SizeIs(1));
  EXPECT_THAT(subgraph->operators, SizeIs(1));
  const auto& where = subgraph->operators[0];
  EXPECT_THAT(model_.operator_codes[where->opcode_index]->builtin_code,
              Eq(BuiltinOperator_WHERE));

  // There should be 2 tensors: input and output.
  EXPECT_THAT(subgraph->tensors, SizeIs(2));

  // Testing input tensor type and ensuring it
  // was not quantized
  EXPECT_THAT(subgraph->tensors[0]->type, Eq(TensorType_BOOL));
  EXPECT_THAT(subgraph->tensors[0]->name, Eq("input"));
  EXPECT_THAT(subgraph->tensors[0]->quantization->scale, IsEmpty());
  EXPECT_THAT(subgraph->tensors[0]->quantization->zero_point, IsEmpty());

  // Testing output (indices) tensor type and ensuring it
  // was not quantized
  EXPECT_THAT(subgraph->tensors[1]->type, Eq(TensorType_INT64));
  EXPECT_THAT(subgraph->tensors[1]->name, Eq("indices"));
  EXPECT_THAT(subgraph->tensors[1]->quantization->scale, IsEmpty());
  EXPECT_THAT(subgraph->tensors[1]->quantization->zero_point, IsEmpty());

  // check op and versioning.
  EXPECT_THAT(model_.operator_codes, SizeIs(1));
  EXPECT_THAT(model_.operator_codes[0]->builtin_code,
              Eq(BuiltinOperator_WHERE));
  ASSERT_THAT(model_.operator_codes[0]->version, Eq(1));
}

}  // namespace
}  // namespace optimize
}  // namespace tflite

int main(int argc, char** argv) {
  tensorflow::string model_file;
  const std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("test_model_file", &model_file,
                       "Path to test tflite model file."),
  };

  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    std::cerr << "Required test_model_file\n";
    std::abort();
  }
  g_test_model_dir =
      new tensorflow::string(tensorflow::io::Dirname(model_file));
  ::tensorflow::port::InitMain(argv[0], &argc, &argv);
  return RUN_ALL_TESTS();
}
