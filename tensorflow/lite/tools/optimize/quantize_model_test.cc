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
#include "tensorflow/lite/tools/optimize/quantize_model.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
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
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/tools/optimize/test_util.h"

// Note: More rigorous model tests can be found in subgraph_quantizer_test.cc

namespace {
tensorflow::string* g_test_model_dir = nullptr;
}  // namespace

namespace tflite {
namespace optimize {
namespace {

std::unique_ptr<FlatBufferModel> ReadModel(const string& model_name) {
  auto model_path = tensorflow::io::JoinPath(*g_test_model_dir, model_name);
  return FlatBufferModel::BuildFromFile(model_path.c_str());
}

template <typename T>
std::vector<T> GetAsVector(const flatbuffers::Vector<T>* vec) {
  return std::vector<T>(vec->begin(), vec->end());
}

void VerifyAsymmetricQuantizationScale(
    const QuantizationParameters& float_quant_params,
    const QuantizationParametersT& quantized_quant_params) {
  const float eps = 1e-7;
  ASSERT_EQ(float_quant_params.min()->size(), 1);
  ASSERT_EQ(float_quant_params.max()->size(), 1);
  float float_min = std::min(0.f, float_quant_params.min()->Get(0));
  float float_max = std::max(0.f, float_quant_params.max()->Get(0));

  ASSERT_EQ(quantized_quant_params.scale.size(), 1);
  ASSERT_EQ(quantized_quant_params.zero_point.size(), 1);

  float scale = (float_max - float_min) / 255;
  EXPECT_NEAR(scale, quantized_quant_params.scale[0], eps);
}

TensorType GetBiasTensorType(TensorType& activation_type) {
  return activation_type == TensorType_INT16 ? TensorType_INT64
                                             : TensorType_INT32;
}

class QuantizeModelTest : public testing::Test {
 protected:
  QuantizeModelTest() {
    input_model_ = ReadModel(internal::kConvModelWith0Plus10Weights);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }

  std::unique_ptr<FlatBufferModel> input_model_;
  const Model* readonly_model_;
  tflite::ModelT model_;
  flatbuffers::FlatBufferBuilder builder_;
  internal::FailOnErrorReporter error_reporter_;
};

void ExpectSameModels(const ModelT& model, const ModelT& expected_model) {
  ASSERT_EQ(model.subgraphs.size(), expected_model.subgraphs.size());
  for (size_t subgraph_idx = 0; subgraph_idx < model.subgraphs.size();
       subgraph_idx++) {
    const auto graph = model.subgraphs[subgraph_idx].get();
    const auto expected_graph = expected_model.subgraphs[subgraph_idx].get();
    ASSERT_EQ(graph->tensors.size(), expected_graph->tensors.size());
    for (size_t i = 0; i < graph->tensors.size(); i++) {
      const auto tensor = graph->tensors[i].get();
      const auto expected_tensor = expected_graph->tensors[i].get();
      EXPECT_EQ(tensor->buffer, expected_tensor->buffer);
      EXPECT_EQ(tensor->is_variable, expected_tensor->is_variable);
      EXPECT_EQ(tensor->shape, expected_tensor->shape);
      EXPECT_EQ(tensor->name, expected_tensor->name);
      EXPECT_EQ(tensor->type, expected_tensor->type);
      const auto quantization_params = tensor->quantization.get();
      const auto expected_quantization_params =
          expected_tensor->quantization.get();
      if (quantization_params != nullptr ||
          expected_quantization_params != nullptr) {
        EXPECT_NE(quantization_params, nullptr);
        EXPECT_NE(expected_quantization_params, nullptr);
        EXPECT_EQ(quantization_params->scale,
                  expected_quantization_params->scale);
        EXPECT_EQ(quantization_params->zero_point,
                  expected_quantization_params->zero_point);
      }
    }
  }
  ASSERT_EQ(model.buffers.size(), expected_model.buffers.size());
  for (size_t buffer_idx = 0; buffer_idx < model.buffers.size(); ++buffer_idx) {
    const auto buffer = model.buffers[buffer_idx].get()->data;
    const auto expected_buffer = expected_model.buffers[buffer_idx].get()->data;
    EXPECT_EQ(buffer, expected_buffer);
  }
  // TODO(jianlijianli): Compare operators as well.
}

class QuantizeConvModelTest : public QuantizeModelTest,
                              public testing::WithParamInterface<TensorType> {
 protected:
  QuantizeConvModelTest() {
    tensor_type_ = GetParam();
    bias_type_ = GetBiasTensorType(tensor_type_);
    input_model_ = ReadModel(internal::kConvModelWith0Plus10Weights);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
  TensorType tensor_type_;
  TensorType bias_type_;
};

INSTANTIATE_TEST_SUITE_P(QuantizeConvModelTestInst, QuantizeConvModelTest,
                         testing::ValuesIn({TensorType_INT8,
                                            TensorType_INT16}));

TEST_P(QuantizeConvModelTest, QuantizationSucceeds) {
  auto status = QuantizeModelAllOperators(&builder_, &model_, tensor_type_,
                                          tensor_type_, false, tensor_type_,
                                          bias_type_, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);
  const uint8_t* buffer = builder_.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);
}

TEST_P(QuantizeConvModelTest, SkipUnspecifiedLayer) {
  auto status =
      QuantizeModel(&builder_, &model_, TensorType_FLOAT32, TensorType_FLOAT32,
                    /*allow_float=*/true, {}, TensorType_FLOAT32,
                    TensorType_FLOAT32, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);
  ASSERT_EQ(model_.subgraphs.size(), readonly_model_->subgraphs()->size());
  // The resulting model should be the same.
  ASSERT_EQ(model_.subgraphs.size(), readonly_model_->subgraphs()->size());
  for (size_t subgraph_idx = 0; subgraph_idx < model_.subgraphs.size();
       subgraph_idx++) {
    const auto quantized_graph = model_.subgraphs[subgraph_idx].get();
    const auto float_graph = readonly_model_->subgraphs()->Get(subgraph_idx);
    ASSERT_EQ(quantized_graph->tensors.size(), float_graph->tensors()->size());
    for (size_t i = 0; i < quantized_graph->tensors.size(); i++) {
      const auto quant_tensor = quantized_graph->tensors[i].get();
      const auto float_tensor = float_graph->tensors()->Get(i);
      EXPECT_EQ(quant_tensor->buffer, float_tensor->buffer());
      EXPECT_EQ(quant_tensor->is_variable, float_tensor->is_variable());
      EXPECT_EQ(quant_tensor->shape, GetAsVector(float_tensor->shape()));
      EXPECT_EQ(quant_tensor->name, float_tensor->name()->str());
      EXPECT_EQ(quant_tensor->type, float_tensor->type());
    }
  }
}

TEST_P(QuantizeConvModelTest, TensorShapesAndStructureIsUnchanged) {
  auto status = QuantizeModelAllOperators(&builder_, &model_, tensor_type_,
                                          tensor_type_, false, tensor_type_,
                                          bias_type_, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);
  ASSERT_EQ(model_.subgraphs.size(), readonly_model_->subgraphs()->size());
  for (size_t subgraph_idx = 0; subgraph_idx < model_.subgraphs.size();
       subgraph_idx++) {
    const auto quantized_graph = model_.subgraphs[subgraph_idx].get();
    const auto float_graph = readonly_model_->subgraphs()->Get(subgraph_idx);
    ASSERT_EQ(quantized_graph->tensors.size(), float_graph->tensors()->size());
    for (size_t i = 0; i < quantized_graph->tensors.size(); i++) {
      const auto quant_tensor = quantized_graph->tensors[i].get();
      const auto float_tensor = float_graph->tensors()->Get(i);
      EXPECT_EQ(quant_tensor->buffer, float_tensor->buffer());
      EXPECT_EQ(quant_tensor->is_variable, float_tensor->is_variable());
      EXPECT_EQ(quant_tensor->shape, GetAsVector(float_tensor->shape()));
      EXPECT_EQ(quant_tensor->name, float_tensor->name()->str());
    }
  }
  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_CONV_2D);
  EXPECT_EQ(model_.operator_codes[0]->version, 3);
}

TEST_P(QuantizeConvModelTest, OperatorsAreUnchanged) {
  auto status = QuantizeModelAllOperators(&builder_, &model_, tensor_type_,
                                          tensor_type_, false, tensor_type_,
                                          bias_type_, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);
  ASSERT_EQ(model_.operator_codes.size(),
            readonly_model_->operator_codes()->size());
  for (size_t i = 0; i < model_.operator_codes.size(); i++) {
    const auto float_model_op = readonly_model_->operator_codes()->Get(i);
    EXPECT_EQ(GetBuiltinCode(model_.operator_codes[i].get()),
              GetBuiltinCode(float_model_op));
    if (GetBuiltinCode(model_.operator_codes[i].get()) ==
        BuiltinOperator_CONV_2D) {
      EXPECT_EQ(model_.operator_codes[i]->version, 3);
    } else {
      EXPECT_EQ(model_.operator_codes[i]->version, 2);
    }
  }

  ASSERT_EQ(model_.subgraphs.size(), readonly_model_->subgraphs()->size());
  for (size_t subgraph_idx = 0; subgraph_idx < model_.subgraphs.size();
       subgraph_idx++) {
    const auto quantized_graph = model_.subgraphs[subgraph_idx].get();
    const auto float_graph = readonly_model_->subgraphs()->Get(subgraph_idx);
    ASSERT_EQ(quantized_graph->operators.size(),
              float_graph->operators()->size());
    for (size_t i = 0; i < quantized_graph->operators.size(); i++) {
      const auto quant_op = quantized_graph->operators[i].get();
      const auto float_op = float_graph->operators()->Get(i);
      EXPECT_EQ(quant_op->inputs, GetAsVector(float_op->inputs()));
      EXPECT_EQ(quant_op->outputs, GetAsVector(float_op->outputs()));
      EXPECT_EQ(quant_op->opcode_index, float_op->opcode_index());
    }
  }
}

TEST_P(QuantizeConvModelTest, GraphIsFullyQuantized) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, tensor_type_, tensor_type_,
      /*allow_float*/ false, tensor_type_, bias_type_, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);
  for (const auto& subgraph : model_.subgraphs) {
    for (const auto& tensor : subgraph->tensors) {
      if (tensor_type_ == TensorType_INT8) {
        EXPECT_TRUE(tensor->type == TensorType_INT32 ||
                    tensor->type == TensorType_INT8);
      } else if (tensor_type_ == TensorType_INT16) {
        EXPECT_TRUE(tensor->type == TensorType_INT64 ||  // bias
                    tensor->type == TensorType_INT8 ||   // weights
                    tensor->type == TensorType_INT16);   // activations
      }
    }
  }
}

TEST_P(QuantizeConvModelTest, FloatInputAndOutput) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_FLOAT32, TensorType_FLOAT32,
      /*allow_float*/ false, tensor_type_, bias_type_, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);

  for (int32_t subgraph_idx = 0; subgraph_idx < model_.subgraphs.size();
       ++subgraph_idx) {
    const auto& subgraph = model_.subgraphs[subgraph_idx];
    const auto& readonly_subgraph =
        readonly_model_->subgraphs()->Get(subgraph_idx);
    // The model has one input and output, so the converted model should have
    // two extra ops, a Quantize and Dequantize.
    EXPECT_EQ(subgraph->operators.size(),
              readonly_subgraph->operators()->size() + 2);
    // Check that the first op is Quantize and the last is Dequant.
    const auto& quant_op = subgraph->operators[0];
    const auto& dequant_op =
        subgraph->operators[subgraph->operators.size() - 1];
    const int32_t quant_idx = quant_op->opcode_index;
    const int32_t dequant_idx = dequant_op->opcode_index;
    EXPECT_EQ(GetBuiltinCode(model_.operator_codes[quant_idx].get()),
              BuiltinOperator_QUANTIZE);
    EXPECT_EQ(GetBuiltinCode(model_.operator_codes[dequant_idx].get()),
              BuiltinOperator_DEQUANTIZE);
    // The model should only have one input and output.
    EXPECT_EQ(subgraph->inputs.size(), 1);
    EXPECT_EQ(subgraph->outputs.size(), 1);
    const int32_t input_idx = subgraph->inputs[0];
    const int32_t output_idx = subgraph->outputs[0];
    // Ensure: new input -> Quant -> old input.
    EXPECT_EQ(quant_op->inputs[0], input_idx);
    EXPECT_EQ(quant_op->outputs[0], readonly_subgraph->inputs()->Get(0));
    // Ensure: old output -> dequant -> new output.
    EXPECT_EQ(dequant_op->inputs[0], readonly_subgraph->outputs()->Get(0));
    EXPECT_EQ(dequant_op->outputs[0], output_idx);
    // The input and output types should be float.
    EXPECT_EQ(subgraph->tensors[input_idx]->type, TensorType_FLOAT32);
    EXPECT_EQ(subgraph->tensors[input_idx]->name, "input");
    EXPECT_EQ(subgraph->tensors[output_idx]->type, TensorType_FLOAT32);
    EXPECT_EQ(subgraph->tensors[output_idx]->name, "output");
    // The original input and output has been renamed.
    std::string control_suffix =
        (tensor_type_ == TensorType_INT16) ? "int16" : "int8";
    EXPECT_EQ(subgraph->tensors[quant_op->outputs[0]]->name,
              "input_" + control_suffix);
    EXPECT_EQ(subgraph->tensors[dequant_op->inputs[0]]->name,
              "output_" + control_suffix);
    for (int tensor_idx = 0; tensor_idx < subgraph->tensors.size();
         ++tensor_idx) {
      const auto& tensor = subgraph->tensors[tensor_idx];
      if (input_idx != tensor_idx && output_idx != tensor_idx) {
        if (tensor_type_ == TensorType_INT8) {
          EXPECT_TRUE(tensor->type == TensorType_INT32 ||
                      tensor->type == TensorType_INT8);
        } else if (tensor_type_ == TensorType_INT16) {
          EXPECT_TRUE(tensor->type == TensorType_INT64 ||  // bias
                      tensor->type == TensorType_INT8 ||   // weights
                      tensor->type == TensorType_INT16);   // activations
        }
      }
    }
  }
}

TEST_P(QuantizeConvModelTest, Uint8InputAndOutput) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_UINT8, TensorType_UINT8, false,
      TensorType_INT8, TensorType_INT32, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);

  for (int32_t subgraph_idx = 0; subgraph_idx < model_.subgraphs.size();
       ++subgraph_idx) {
    const auto& subgraph = model_.subgraphs[subgraph_idx];
    const auto& readonly_subgraph =
        readonly_model_->subgraphs()->Get(subgraph_idx);
    // The model has one input and output, so the converted model should have
    // two extra ops, a Quantize and Dequantize.
    EXPECT_EQ(subgraph->operators.size(),
              readonly_subgraph->operators()->size() + 2);
    // Check that the first op is Quantize and the last is Dequant.
    const auto& quant_op_uint8_int8 = subgraph->operators[0];
    const auto& quant_op_int8_uint8 =
        subgraph->operators[subgraph->operators.size() - 1];
    const int32_t quant_op_uint8_int8_idx = quant_op_uint8_int8->opcode_index;
    const int32_t quant_op_int8_uint8_idx = quant_op_int8_uint8->opcode_index;
    EXPECT_EQ(
        GetBuiltinCode(model_.operator_codes[quant_op_uint8_int8_idx].get()),
        BuiltinOperator_QUANTIZE);
    EXPECT_EQ(
        GetBuiltinCode(model_.operator_codes[quant_op_int8_uint8_idx].get()),
        BuiltinOperator_QUANTIZE);
    // The model should only have one input and output.
    EXPECT_EQ(subgraph->inputs.size(), 1);
    EXPECT_EQ(subgraph->outputs.size(), 1);
    const int32_t input_idx = subgraph->inputs[0];
    const int32_t output_idx = subgraph->outputs[0];
    // Ensure: new input -> Quant -> old input.
    EXPECT_EQ(quant_op_uint8_int8->inputs[0], input_idx);
    EXPECT_EQ(quant_op_uint8_int8->outputs[0],
              readonly_subgraph->inputs()->Get(0));
    // Ensure: old output -> dequant -> new output.
    EXPECT_EQ(quant_op_int8_uint8->inputs[0],
              readonly_subgraph->outputs()->Get(0));
    EXPECT_EQ(quant_op_int8_uint8->outputs[0], output_idx);
    // The input and output types should be uint8.
    EXPECT_EQ(subgraph->tensors[input_idx]->type, TensorType_UINT8);
    EXPECT_EQ(subgraph->tensors[input_idx]->name, "input");
    EXPECT_EQ(subgraph->tensors[input_idx]->quantization->scale.size(), 1);
    EXPECT_FLOAT_EQ(subgraph->tensors[input_idx]->quantization->scale[0],
                    0.0392156877);
    EXPECT_EQ(subgraph->tensors[input_idx]->quantization->zero_point.size(), 1);
    EXPECT_EQ(subgraph->tensors[input_idx]->quantization->zero_point[0], 0);
    EXPECT_EQ(subgraph->tensors[output_idx]->type, TensorType_UINT8);
    EXPECT_EQ(subgraph->tensors[output_idx]->name, "output");
    EXPECT_EQ(subgraph->tensors[output_idx]->quantization->scale.size(), 1);
    EXPECT_FLOAT_EQ(subgraph->tensors[output_idx]->quantization->scale[0],
                    0.0392156877);
    EXPECT_EQ(subgraph->tensors[output_idx]->quantization->zero_point.size(),
              1);
    EXPECT_EQ(subgraph->tensors[output_idx]->quantization->zero_point[0], 0);
    // The original input and output has been renamed.
    EXPECT_EQ(subgraph->tensors[quant_op_uint8_int8->outputs[0]]->name,
              "input_int8");
    EXPECT_EQ(subgraph->tensors[quant_op_int8_uint8->inputs[0]]->name,
              "output_int8");
    for (int tensor_idx = 0; tensor_idx < subgraph->tensors.size();
         ++tensor_idx) {
      const auto& tensor = subgraph->tensors[tensor_idx];
      if (input_idx != tensor_idx && output_idx != tensor_idx) {
        EXPECT_TRUE(tensor->type == TensorType_INT32 ||
                    tensor->type == TensorType_INT8);
      }
    }
  }
}

class QuantizeConvNoBiasModelTest : public QuantizeModelTest {
 protected:
  QuantizeConvNoBiasModelTest() {
    input_model_ = ReadModel(internal::kConvModelWithNoBias);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeConvNoBiasModelTest, QuantizationSucceeds) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8, false,
      TensorType_INT8, TensorType_INT32, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);
  const uint8_t* buffer = builder_.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);
}

class QuantizeConcatModelTest : public QuantizeModelTest,
                                public testing::WithParamInterface<TensorType> {
 protected:
  QuantizeConcatModelTest() {
    input_model_ = ReadModel(internal::kFloatConcatMax5Max10Max10);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }

  void SetUp() override {
    tensor_type_ = GetParam();
    bias_type_ = GetBiasTensorType(tensor_type_);
  }

  TensorType tensor_type_;
  TensorType bias_type_;
};

// There are two inputs for concat, "input0" and "input1". "input0" has [0, 5]
// as min/max and "input1" has [0, 10] as min/max. The output "output" for
// concat has [0, 10] as min/max.
// After applyging QuantizeModel(), "input0" will have a requant op added, along
// with a tensor "input0_reqaunt" that has [0, 10] as min/max. So the topology
// becomes:
// input0 -> requant -> input0_requant \
//                                       concat - output
//                              input1 /
TEST_P(QuantizeConcatModelTest, AddRequantBeforeConcat) {
  auto status = QuantizeModelAllOperators(&builder_, &model_, tensor_type_,
                                          tensor_type_, false, tensor_type_,
                                          bias_type_, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);

  // There is only one subgraph.
  const int32_t subgraph_idx = 0;
  const auto& subgraph = model_.subgraphs[subgraph_idx];
  const auto& readonly_subgraph =
      readonly_model_->subgraphs()->Get(subgraph_idx);

  // There should be two ops: quant and concat.
  EXPECT_EQ(readonly_subgraph->operators()->size(), 1);
  EXPECT_EQ(subgraph->operators.size(), 2);
  const auto& requant = subgraph->operators[0];
  const auto& concat = subgraph->operators[1];
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[requant->opcode_index].get()),
            BuiltinOperator_QUANTIZE);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[concat->opcode_index].get()),
            BuiltinOperator_CONCATENATION);

  auto zero_point_control = tensor_type_ == TensorType_INT8 ? -128 : 0;
  /*
     input0_scale_control
        INT8: (5-0) / (2^8 - 1)
        INT16: (5-0) / (2^16 / 2 - 1)
     input1_scale
        INT8: (10-0) / (2^8 - 1)
        INT16: (10-0) / (2^16 / 2 - 1)
  */
  auto input0_scale_control =
      tensor_type_ == TensorType_INT8 ? 0.019607844 : 0.00015259254;
  auto input1_scale =
      tensor_type_ == TensorType_INT8 ? 0.039215688 : 0.00030518509;

  // There should be 4 tensors: input0, input1, input0_requantized, output.
  EXPECT_EQ(subgraph->tensors.size(), 4);
  EXPECT_EQ(subgraph->tensors[0]->type, tensor_type_);
  EXPECT_EQ(subgraph->tensors[0]->name, "input0");
  EXPECT_EQ(subgraph->tensors[0]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[0]->quantization->zero_point.size(), 1);
  EXPECT_FLOAT_EQ(subgraph->tensors[0]->quantization->scale[0],
                  input0_scale_control);
  EXPECT_FLOAT_EQ(subgraph->tensors[0]->quantization->zero_point[0],
                  zero_point_control);
  EXPECT_EQ(subgraph->tensors[1]->type, tensor_type_);
  EXPECT_EQ(subgraph->tensors[1]->name, "input1");
  EXPECT_EQ(subgraph->tensors[1]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[1]->quantization->zero_point.size(), 1);
  EXPECT_FLOAT_EQ(subgraph->tensors[1]->quantization->scale[0], input1_scale);
  EXPECT_FLOAT_EQ(subgraph->tensors[1]->quantization->zero_point[0],
                  zero_point_control);
  EXPECT_EQ(subgraph->tensors[2]->type, tensor_type_);
  EXPECT_EQ(subgraph->tensors[2]->name, "output");
  EXPECT_EQ(subgraph->tensors[2]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[2]->quantization->zero_point.size(), 1);
  EXPECT_FLOAT_EQ(subgraph->tensors[2]->quantization->scale[0], input1_scale);
  EXPECT_FLOAT_EQ(subgraph->tensors[2]->quantization->zero_point[0],
                  zero_point_control);
  EXPECT_EQ(subgraph->tensors[3]->type, tensor_type_);
  EXPECT_EQ(subgraph->tensors[3]->name, "input0_requantized");
  EXPECT_EQ(subgraph->tensors[3]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[3]->quantization->zero_point.size(), 1);
  EXPECT_FLOAT_EQ(subgraph->tensors[3]->quantization->scale[0], input1_scale);
  EXPECT_FLOAT_EQ(subgraph->tensors[3]->quantization->zero_point[0],
                  zero_point_control);

  // The connection should be what is described in the comment.
  EXPECT_EQ(requant->inputs.size(), 1);
  EXPECT_EQ(requant->outputs.size(), 1);
  EXPECT_EQ(requant->inputs[0], 0);
  EXPECT_EQ(requant->outputs[0], 3);
  EXPECT_EQ(concat->inputs.size(), 2);
  EXPECT_EQ(concat->outputs.size(), 1);
  EXPECT_EQ(concat->inputs[0], 3);
  EXPECT_EQ(concat->inputs[1], 1);
  EXPECT_EQ(concat->outputs[0], 2);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 2);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_CONCATENATION);
  EXPECT_EQ(model_.operator_codes[0]->version, 2);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[1].get()),
            BuiltinOperator_QUANTIZE);
  EXPECT_EQ(model_.operator_codes[1]->version, 2);
}
INSTANTIATE_TEST_SUITE_P(QuantizeConcatModelInst, QuantizeConcatModelTest,
                         testing::ValuesIn({TensorType_INT8,
                                            TensorType_INT16}));
class QuantizeSplitModelTest : public QuantizeModelTest {
 protected:
  QuantizeSplitModelTest() {
    input_model_ = ReadModel(internal::kModelSplit);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

// There are two outputs for split with different scales, the resulting model
// should have the scales be hardcodes to the input scale value.
TEST_F(QuantizeSplitModelTest, QuantizeSplit) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8, false,
      TensorType_INT8, TensorType_INT32, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);

  // There is only one subgraph.
  const int32_t subgraph_idx = 0;
  const auto& subgraph = model_.subgraphs[subgraph_idx];
  const auto& readonly_subgraph =
      readonly_model_->subgraphs()->Get(subgraph_idx);

  // There should be two ops: the split and add in the original model.
  EXPECT_EQ(readonly_subgraph->operators()->size(), 2);
  EXPECT_EQ(subgraph->operators.size(), 2);
  const auto& split = subgraph->operators[0];
  const auto& add = subgraph->operators[1];
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[split->opcode_index].get()),
            BuiltinOperator_SPLIT);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[add->opcode_index].get()),
            BuiltinOperator_ADD);

  // There should be 5 tensors: input, output, split, split/split_dim, split:1.
  EXPECT_EQ(subgraph->tensors.size(), 5);

  EXPECT_EQ(subgraph->tensors[0]->type, TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[0]->name, "input");
  EXPECT_EQ(subgraph->tensors[0]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[0]->quantization->zero_point.size(), 1);
  EXPECT_FLOAT_EQ(subgraph->tensors[0]->quantization->scale[0], 1.0);
  EXPECT_FLOAT_EQ(subgraph->tensors[0]->quantization->zero_point[0], -128);
  EXPECT_EQ(subgraph->tensors[1]->type, TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[1]->name, "output");
  EXPECT_EQ(subgraph->tensors[1]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[1]->quantization->zero_point.size(), 1);
  EXPECT_FLOAT_EQ(subgraph->tensors[1]->quantization->scale[0], 1.0);
  EXPECT_FLOAT_EQ(subgraph->tensors[1]->quantization->zero_point[0], -128);
  EXPECT_EQ(subgraph->tensors[2]->type, TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[2]->name, "split");
  EXPECT_EQ(subgraph->tensors[2]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[2]->quantization->zero_point.size(), 1);
  EXPECT_FLOAT_EQ(subgraph->tensors[2]->quantization->scale[0], 1.0);
  EXPECT_FLOAT_EQ(subgraph->tensors[2]->quantization->zero_point[0], -128);
  EXPECT_EQ(subgraph->tensors[4]->type, TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[4]->name, "split:1");
  EXPECT_EQ(subgraph->tensors[4]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[4]->quantization->zero_point.size(), 1);
  EXPECT_FLOAT_EQ(subgraph->tensors[4]->quantization->scale[0], 1.0);
  EXPECT_FLOAT_EQ(subgraph->tensors[4]->quantization->zero_point[0], -128);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 2);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[1].get()),
            BuiltinOperator_SPLIT);
  EXPECT_EQ(model_.operator_codes[0]->version, 2);
}

class QuantizeConvModel1Test : public QuantizeModelTest {
 protected:
  QuantizeConvModel1Test() {
    input_model_ = ReadModel(internal::kConvModelWithMinus128Plus127Weights);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeConvModel1Test, VerifyConvQuantizationWithUnitScale) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8, false,
      TensorType_INT8, TensorType_INT32, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);
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

  EXPECT_EQ(bias_tensor->type, TensorType_INT32);
  EXPECT_EQ(input_tensor->type, TensorType_INT8);
  EXPECT_EQ(weights_tensor->type, TensorType_INT8);

  ASSERT_TRUE(weights_tensor->quantization);
  const int out_channel_size = weights_tensor->shape[0];
  ASSERT_TRUE(bias_tensor->quantization);
  ASSERT_TRUE(weights_tensor->quantization);
  const std::vector<float>& bias_scales = bias_tensor->quantization->scale;
  const std::vector<float>& weights_scales =
      weights_tensor->quantization->scale;

  const std::vector<int64_t>& weights_zero_points =
      weights_tensor->quantization->zero_point;

  ASSERT_EQ(bias_scales.size(), out_channel_size);
  ASSERT_EQ(weights_scales.size(), out_channel_size);
  ASSERT_EQ(weights_zero_points.size(), out_channel_size);
  ASSERT_EQ(input_tensor->quantization->scale.size(), 1);
  ASSERT_EQ(output_tensor->quantization->scale.size(), 1);

  for (size_t i = 0; i < out_channel_size; i++) {
    EXPECT_EQ(weights_scales[i], 1);
    EXPECT_EQ(bias_scales[i], 1);
    EXPECT_EQ(weights_zero_points[i], 0);
  }

  EXPECT_EQ(input_tensor->quantization->scale[0], 1);
  EXPECT_EQ(output_tensor->quantization->scale[0], 1);

  const auto bias_buffer = model_.buffers[bias_tensor->buffer].get();
  ASSERT_EQ(bias_buffer->data.size(), sizeof(int32_t) * bias_tensor->shape[0]);
  const int32_t* bias_values =
      reinterpret_cast<int32_t*>(bias_buffer->data.data());
  const auto original_bias_buffer =
      readonly_model_->buffers()->Get(bias_tensor->buffer);
  const float* bias_float_buffer =
      reinterpret_cast<const float*>(original_bias_buffer->data()->data());

  const float eps = 1e-7;
  for (size_t i = 0; i < bias_tensor->shape[0]; i++) {
    const float bias_scale =
        input_tensor->quantization->scale[0] * weights_scales[i];
    auto dequantized_value = bias_values[i] * bias_scale;
    EXPECT_NEAR(dequantized_value, bias_float_buffer[i], eps);
  }

  const auto weights_buffer = model_.buffers[weights_tensor->buffer].get();
  const auto original_weights_buffer =
      readonly_model_->buffers()->Get(weights_tensor->buffer);
  const int8_t* weight_values =
      reinterpret_cast<int8_t*>(weights_buffer->data.data());
  const float* weights_float_buffer =
      reinterpret_cast<const float*>(original_weights_buffer->data()->data());
  ASSERT_EQ(sizeof(float) * weights_buffer->data.size(),
            original_weights_buffer->data()->size());
  int num_values_in_channel = weights_buffer->data.size() / out_channel_size;
  for (size_t channel_idx = 0; channel_idx < out_channel_size; channel_idx++) {
    for (size_t j = 0; j < num_values_in_channel; j++) {
      size_t element_idx = channel_idx * out_channel_size + j;
      auto dequantized_value =
          weight_values[element_idx] * weights_scales[channel_idx];
      EXPECT_NEAR(dequantized_value, weights_float_buffer[element_idx], eps);
    }
  }

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_CONV_2D);
  EXPECT_EQ(model_.operator_codes[0]->version, 3);
}

class QuantizeConvModel2Test : public QuantizeModelTest,
                               public testing::WithParamInterface<TensorType> {
 protected:
  QuantizeConvModel2Test() {
    tensor_type_ = GetParam();
    bias_type_ = GetBiasTensorType(tensor_type_);
    input_model_ = ReadModel(internal::kConvModelWith0Plus10Weights);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }

  TensorType tensor_type_;
  TensorType bias_type_;
};
INSTANTIATE_TEST_SUITE_P(QuantizeConvModel2TestInst, QuantizeConvModel2Test,
                         testing::ValuesIn({TensorType_INT8,
                                            TensorType_INT16}));

TEST_P(QuantizeConvModel2Test, VerifyConvQuantization) {
  auto status = QuantizeModelAllOperators(&builder_, &model_, tensor_type_,
                                          tensor_type_, false, tensor_type_,
                                          bias_type_, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);
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

  EXPECT_EQ(bias_tensor->type, tensor_type_ == TensorType_INT8
                                   ? TensorType_INT32
                                   : TensorType_INT64);
  EXPECT_EQ(input_tensor->type, tensor_type_);
  EXPECT_EQ(weights_tensor->type, TensorType_INT8);

  ASSERT_TRUE(weights_tensor->quantization);
  ASSERT_TRUE(bias_tensor->quantization);
  ASSERT_TRUE(weights_tensor->quantization);
  const std::vector<float>& bias_scales = bias_tensor->quantization->scale;
  const std::vector<float>& weights_scales =
      weights_tensor->quantization->scale;
  const std::vector<int64_t>& weights_zero_points =
      weights_tensor->quantization->zero_point;
  const int out_channel_size = weights_tensor->shape[0];
  ASSERT_EQ(bias_scales.size(), out_channel_size);
  ASSERT_EQ(weights_scales.size(), out_channel_size);
  ASSERT_EQ(weights_zero_points.size(), out_channel_size);
  ASSERT_EQ(input_tensor->quantization->scale.size(), 1);
  ASSERT_EQ(output_tensor->quantization->scale.size(), 1);

  const float eps = 1e-7;

  // Bias scale should be input * per_channel_weight_scale.
  for (size_t i = 0; i < out_channel_size; i++) {
    EXPECT_NEAR(bias_scales[i],
                input_tensor->quantization->scale[0] * weights_scales[i], eps);
  }

  const auto bias_buffer = model_.buffers[bias_tensor->buffer].get();
  auto control_size = tensor_type_ == TensorType_INT8
                          ? sizeof(int32_t) * bias_tensor->shape[0]
                          : sizeof(int64_t) * bias_tensor->shape[0];

  ASSERT_EQ(bias_buffer->data.size(), control_size);
  const auto original_bias_buffer =
      readonly_model_->buffers()->Get(bias_tensor->buffer);
  const float* bias_float_buffer =
      reinterpret_cast<const float*>(original_bias_buffer->data()->data());

  if (tensor_type_ == TensorType_INT8) {
    int32_t* bias_values = reinterpret_cast<int32_t*>(bias_buffer->data.data());
    for (size_t i = 0; i < out_channel_size; i++) {
      auto dequantized_value = bias_values[i] * bias_scales[i];
      EXPECT_NEAR(dequantized_value, bias_float_buffer[i], bias_scales[i] / 2);
    }
  } else if (tensor_type_ == TensorType_INT16) {
    int64_t* bias_values = reinterpret_cast<int64_t*>(bias_buffer->data.data());
    for (size_t i = 0; i < out_channel_size; i++) {
      auto dequantized_value = bias_values[i] * bias_scales[i];
      EXPECT_NEAR(dequantized_value, bias_float_buffer[i], bias_scales[i] / 2);
    }
  }

  const auto weights_buffer = model_.buffers[weights_tensor->buffer].get();
  const auto original_weights_buffer =
      readonly_model_->buffers()->Get(weights_tensor->buffer);
  const int8_t* weight_values =
      reinterpret_cast<int8_t*>(weights_buffer->data.data());
  const float* weights_float_buffer =
      reinterpret_cast<const float*>(original_weights_buffer->data()->data());
  ASSERT_EQ(sizeof(float) * weights_buffer->data.size(),
            original_weights_buffer->data()->size());
  int num_values_in_channel = weights_buffer->data.size() / out_channel_size;
  for (size_t channel_idx = 0; channel_idx < out_channel_size; channel_idx++) {
    for (size_t j = 0; j < num_values_in_channel; j++) {
      size_t element_idx = channel_idx * out_channel_size + j;
      auto scale = weights_scales[channel_idx];
      auto zero_point = weights_zero_points[channel_idx];
      auto dequantized_value = weight_values[element_idx] * scale;
      EXPECT_NEAR(dequantized_value, weights_float_buffer[element_idx],
                  scale / 2);
      EXPECT_EQ(zero_point, 0);
    }
  }

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_CONV_2D);
  EXPECT_EQ(model_.operator_codes[0]->version, 3);
}

TEST_P(QuantizeConvModel2Test, VerifyConvDisablePerChannelQuantization) {
  auto status =
      QuantizeModelAllOperators(&builder_, &model_, tensor_type_, tensor_type_,
                                false, tensor_type_, bias_type_,
                                /*disable_per_channel=*/true, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);
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

  EXPECT_EQ(bias_tensor->type, tensor_type_ == TensorType_INT8
                                   ? TensorType_INT32
                                   : TensorType_INT64);
  EXPECT_EQ(input_tensor->type, tensor_type_);
  EXPECT_EQ(weights_tensor->type, TensorType_INT8);

  ASSERT_TRUE(weights_tensor->quantization);
  ASSERT_TRUE(bias_tensor->quantization);
  ASSERT_TRUE(weights_tensor->quantization);
  const std::vector<float>& bias_scales = bias_tensor->quantization->scale;
  const std::vector<float>& weights_scales =
      weights_tensor->quantization->scale;
  const std::vector<int64_t>& weights_zero_points =
      weights_tensor->quantization->zero_point;

  const int out_channel_size = 1;
  ASSERT_EQ(bias_scales.size(), out_channel_size);
  ASSERT_EQ(weights_scales.size(), out_channel_size);
  ASSERT_EQ(weights_zero_points.size(), out_channel_size);
  ASSERT_EQ(input_tensor->quantization->scale.size(), 1);
  ASSERT_EQ(output_tensor->quantization->scale.size(), 1);

  const float eps = 1e-7;

  // Bias scale should be input * per_channel_weight_scale.
  for (size_t i = 0; i < out_channel_size; i++) {
    EXPECT_NEAR(bias_scales[i],
                input_tensor->quantization->scale[0] * weights_scales[i], eps);
  }

  const auto bias_buffer = model_.buffers[bias_tensor->buffer].get();
  auto control_size = tensor_type_ == TensorType_INT8
                          ? sizeof(int32_t) * bias_tensor->shape[0]
                          : sizeof(int64_t) * bias_tensor->shape[0];

  ASSERT_EQ(bias_buffer->data.size(), control_size);
  const auto original_bias_buffer =
      readonly_model_->buffers()->Get(bias_tensor->buffer);
  const float* bias_float_buffer =
      reinterpret_cast<const float*>(original_bias_buffer->data()->data());

  if (tensor_type_ == TensorType_INT8) {
    int32_t* bias_values = reinterpret_cast<int32_t*>(bias_buffer->data.data());
    for (size_t i = 0; i < out_channel_size; i++) {
      auto dequantized_value = bias_values[i] * bias_scales[i];
      EXPECT_NEAR(dequantized_value, bias_float_buffer[i], bias_scales[i] / 2);
    }
  } else if (tensor_type_ == TensorType_INT16) {
    int64_t* bias_values = reinterpret_cast<int64_t*>(bias_buffer->data.data());
    for (size_t i = 0; i < out_channel_size; i++) {
      auto dequantized_value = bias_values[i] * bias_scales[i];
      EXPECT_NEAR(dequantized_value, bias_float_buffer[i], bias_scales[i] / 2);
    }
  }

  const auto weights_buffer = model_.buffers[weights_tensor->buffer].get();
  const auto original_weights_buffer =
      readonly_model_->buffers()->Get(weights_tensor->buffer);
  const int8_t* weight_values =
      reinterpret_cast<int8_t*>(weights_buffer->data.data());
  const float* weights_float_buffer =
      reinterpret_cast<const float*>(original_weights_buffer->data()->data());
  ASSERT_EQ(sizeof(float) * weights_buffer->data.size(),
            original_weights_buffer->data()->size());
  int num_values_in_channel = weights_buffer->data.size() / out_channel_size;
  for (size_t channel_idx = 0; channel_idx < out_channel_size; channel_idx++) {
    for (size_t j = 0; j < num_values_in_channel; j++) {
      size_t element_idx = channel_idx * out_channel_size + j;
      auto scale = weights_scales[channel_idx];
      auto zero_point = weights_zero_points[channel_idx];
      auto dequantized_value = weight_values[element_idx] * scale;
      EXPECT_NEAR(dequantized_value, weights_float_buffer[element_idx],
                  scale / 2);
      EXPECT_EQ(zero_point, 0);
    }
  }

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_CONV_2D);
  EXPECT_EQ(model_.operator_codes[0]->version, 3);
}

class QuantizeSoftmaxTest : public QuantizeModelTest {
 protected:
  QuantizeSoftmaxTest() {
    input_model_ = ReadModel(internal::kSingleSoftmaxModelMinMinus5MaxPlus5);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeSoftmaxTest, VerifySoftmaxQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8, false,
      TensorType_INT8, TensorType_INT32, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  // Model has a single softmax op.
  ASSERT_EQ(op->opcode_index, 0);
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_SOFTMAX);

  ASSERT_EQ(op->inputs.size(), 1);
  ASSERT_EQ(op->outputs.size(), 1);
  auto float_graph = readonly_model_->subgraphs()->Get(0);

  // Verify input.
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[0])->type(),
            TensorType_FLOAT32);
  ASSERT_EQ(float_graph->tensors()->Get(op->outputs[0])->type(),
            TensorType_FLOAT32);

  EXPECT_EQ(subgraph->tensors[op->inputs[0]].get()->type, TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[op->outputs[0]].get()->type, TensorType_INT8);

  auto float_input_quant_params =
      float_graph->tensors()->Get(op->inputs[0])->quantization();
  auto input_quant_params =
      subgraph->tensors[op->inputs[0]]->quantization.get();
  VerifyAsymmetricQuantizationScale(*float_input_quant_params,
                                    *input_quant_params);

  // Verify output.
  auto float_output_quant_params =
      float_graph->tensors()->Get(op->outputs[0])->quantization();
  auto output_quant_params =
      subgraph->tensors[op->outputs[0]]->quantization.get();
  ASSERT_EQ(float_output_quant_params->min()->size(), 1);
  ASSERT_EQ(float_output_quant_params->max()->size(), 1);

  ASSERT_EQ(output_quant_params->scale.size(), 1);
  ASSERT_EQ(output_quant_params->zero_point.size(), 1);
  ASSERT_EQ(1.0f / 256.0f, output_quant_params->scale[0]);
  ASSERT_EQ(-128, output_quant_params->zero_point[0]);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_SOFTMAX);
  EXPECT_EQ(model_.operator_codes[0]->version, 2);
}

class QuantizeAvgPoolTest : public QuantizeModelTest {
 protected:
  QuantizeAvgPoolTest() {
    input_model_ = ReadModel(internal::kSingleAvgPoolModelMinMinus5MaxPlus5);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeAvgPoolTest, VerifyAvgPoolQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8, false,
      TensorType_INT8, TensorType_INT32, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  // Model has a single AveragePool op.
  ASSERT_EQ(op->opcode_index, 0);
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_AVERAGE_POOL_2D);

  ASSERT_EQ(op->inputs.size(), 1);
  ASSERT_EQ(op->outputs.size(), 1);

  auto float_graph = readonly_model_->subgraphs()->Get(0);
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[0])->type(),
            TensorType_FLOAT32);
  ASSERT_EQ(float_graph->tensors()->Get(op->outputs[0])->type(),
            TensorType_FLOAT32);

  EXPECT_EQ(subgraph->tensors[op->inputs[0]].get()->type, TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[op->outputs[0]].get()->type, TensorType_INT8);

  auto float_input_quant_params =
      float_graph->tensors()->Get(op->inputs[0])->quantization();
  auto input_quant_params =
      subgraph->tensors[op->inputs[0]]->quantization.get();
  VerifyAsymmetricQuantizationScale(*float_input_quant_params,
                                    *input_quant_params);

  auto float_output_quant_params =
      float_graph->tensors()->Get(op->outputs[0])->quantization();
  auto output_quant_params =
      subgraph->tensors[op->outputs[0]]->quantization.get();
  ASSERT_EQ(float_output_quant_params->min()->size(), 1);
  ASSERT_EQ(float_output_quant_params->max()->size(), 1);
  ASSERT_EQ(output_quant_params->min.size(), 1);
  ASSERT_EQ(output_quant_params->max.size(), 1);

  // Make sure the input min/maxes are propagated to outputs.
  EXPECT_EQ(input_quant_params->min[0], output_quant_params->min[0]);
  EXPECT_EQ(input_quant_params->max[0], output_quant_params->max[0]);
  EXPECT_EQ(input_quant_params->scale[0], output_quant_params->scale[0]);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_AVERAGE_POOL_2D);
  EXPECT_EQ(model_.operator_codes[0]->version, 2);
}

class QuantizeMultiInputAddWithReshapeTest : public QuantizeModelTest {
 protected:
  QuantizeMultiInputAddWithReshapeTest() {
    input_model_ = ReadModel(internal::kMultiInputAddWithReshape);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeMultiInputAddWithReshapeTest, VerifyReshapeQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8, false,
      TensorType_INT8, TensorType_INT32, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  // Verify Reshape is quantized.
  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[1].get();
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
            BuiltinOperator_RESHAPE);

  ASSERT_EQ(op->inputs.size(), 2);
  ASSERT_EQ(op->outputs.size(), 1);

  auto float_graph = readonly_model_->subgraphs()->Get(0);
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[0])->type(),
            TensorType_FLOAT32);
  ASSERT_EQ(float_graph->tensors()->Get(op->outputs[0])->type(),
            TensorType_FLOAT32);

  EXPECT_EQ(subgraph->tensors[op->inputs[0]].get()->type, TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[op->outputs[0]].get()->type, TensorType_INT8);
  auto float_input_quant_params =
      float_graph->tensors()->Get(op->inputs[0])->quantization();
  auto input_quant_params =
      subgraph->tensors[op->inputs[0]]->quantization.get();
  VerifyAsymmetricQuantizationScale(*float_input_quant_params,
                                    *input_quant_params);

  auto float_output_quant_params =
      float_graph->tensors()->Get(op->outputs[0])->quantization();
  auto output_quant_params =
      subgraph->tensors[op->outputs[0]]->quantization.get();
  ASSERT_EQ(float_output_quant_params->min()->size(), 1);
  ASSERT_EQ(float_output_quant_params->max()->size(), 1);
  ASSERT_EQ(output_quant_params->min.size(), 1);
  ASSERT_EQ(output_quant_params->max.size(), 1);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 2);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_ADD);
  EXPECT_EQ(model_.operator_codes[0]->version, 2);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[1].get()),
            BuiltinOperator_RESHAPE);
  EXPECT_EQ(model_.operator_codes[1]->version, 1);
}

TEST_F(QuantizeMultiInputAddWithReshapeTest, VerifyAddQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8, false,
      TensorType_INT8, TensorType_INT32, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  // Verify ADD is quantized.
  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
            BuiltinOperator_ADD);

  ASSERT_EQ(op->inputs.size(), 2);
  ASSERT_EQ(op->outputs.size(), 1);

  auto float_graph = readonly_model_->subgraphs()->Get(0);
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[0])->type(),
            TensorType_FLOAT32);
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[1])->type(),
            TensorType_FLOAT32);
  ASSERT_EQ(float_graph->tensors()->Get(op->outputs[0])->type(),
            TensorType_FLOAT32);

  for (size_t input_idx = 0; input_idx < 2; ++input_idx) {
    EXPECT_EQ(subgraph->tensors[op->inputs[input_idx]].get()->type,
              TensorType_INT8);
    auto float_input_quant_params =
        float_graph->tensors()->Get(op->inputs[input_idx])->quantization();
    auto input_quant_params =
        subgraph->tensors[op->inputs[input_idx]]->quantization.get();
    VerifyAsymmetricQuantizationScale(*float_input_quant_params,
                                      *input_quant_params);
  }

  EXPECT_EQ(subgraph->tensors[op->outputs[0]].get()->type, TensorType_INT8);
  auto float_output_quant_params =
      float_graph->tensors()->Get(op->outputs[0])->quantization();
  auto output_quant_params =
      subgraph->tensors[op->outputs[0]]->quantization.get();
  ASSERT_EQ(float_output_quant_params->min()->size(), 1);
  ASSERT_EQ(float_output_quant_params->max()->size(), 1);
  ASSERT_EQ(output_quant_params->min.size(), 1);
  ASSERT_EQ(output_quant_params->max.size(), 1);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 2);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_ADD);
  EXPECT_EQ(model_.operator_codes[0]->version, 2);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[1].get()),
            BuiltinOperator_RESHAPE);
  EXPECT_EQ(model_.operator_codes[1]->version, 1);
}

class QuantizeConstInputTest : public QuantizeModelTest,
                               public testing::WithParamInterface<TensorType> {
 protected:
  QuantizeConstInputTest() {
    tensor_type_ = GetParam();
    bias_type_ = GetBiasTensorType(tensor_type_);
    input_model_ = ReadModel(internal::kConstInputAddModel);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }

  TensorType tensor_type_;
  TensorType bias_type_;
};
INSTANTIATE_TEST_SUITE_P(QuantizeConstInputTestInst, QuantizeConstInputTest,
                         testing::ValuesIn({TensorType_INT8,
                                            TensorType_INT16}));

TEST_P(QuantizeConstInputTest, VerifyConstOpInput) {
  auto status = QuantizeModelAllOperators(&builder_, &model_, tensor_type_,
                                          tensor_type_, false, tensor_type_,
                                          bias_type_, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  // Verify ConstOp is quantized.
  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
            BuiltinOperator_ADD);

  ASSERT_EQ(op->inputs.size(), 2);
  ASSERT_EQ(op->outputs.size(), 1);

  auto float_graph = readonly_model_->subgraphs()->Get(0);
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[0])->type(),
            TensorType_FLOAT32);
  ASSERT_EQ(float_graph->tensors()->Get(op->outputs[0])->type(),
            TensorType_FLOAT32);

  for (size_t input_idx = 0; input_idx < 2; ++input_idx) {
    EXPECT_EQ(subgraph->tensors[op->inputs[input_idx]].get()->type,
              tensor_type_);
  }

  EXPECT_EQ(subgraph->tensors[op->outputs[0]].get()->type, tensor_type_);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_ADD);
  EXPECT_EQ(model_.operator_codes[0]->version, 2);

  // check that in case of int16 activations, pot_scale_int16 parameter is set
  // to false.
  if (tensor_type_ == TensorType_INT16) {
    EXPECT_EQ(subgraph->operators[0]
                  .get()
                  ->builtin_options.AsAddOptions()
                  ->pot_scale_int16,
              false);
  }
}
class QuantizeArgMaxTest : public QuantizeModelTest {
 protected:
  QuantizeArgMaxTest() {
    input_model_ = ReadModel(internal::kModelWithArgMaxOp);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeArgMaxTest, VerifyArgMax) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8, false,
      TensorType_INT8, TensorType_INT32, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
            BuiltinOperator_ARG_MAX);

  ASSERT_EQ(op->inputs.size(), 2);
  ASSERT_EQ(op->outputs.size(), 1);

  auto float_graph = readonly_model_->subgraphs()->Get(0);
  // Verify ArgMax input is quantized.
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[0])->type(),
            TensorType_FLOAT32);
  EXPECT_EQ(subgraph->tensors[op->inputs[0]].get()->type, TensorType_INT8);

  // Verify ArgMax input axis should still be the same type.
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[1])->type(),
            subgraph->tensors[op->inputs[1]].get()->type);

  // The output of ArgMax should still be the same type.
  ASSERT_EQ(float_graph->tensors()->Get(op->outputs[0])->type(),
            subgraph->tensors[op->outputs[0]].get()->type);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_ARG_MAX);
  EXPECT_EQ(model_.operator_codes[0]->version, 2);
}

class QuantizeLSTMTest : public QuantizeModelTest {
 protected:
  QuantizeLSTMTest() {
    input_model_ = ReadModel(internal::kLstmCalibrated);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeLSTMTest, VerifyLSTM) {
  // Quantize model.
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_FLOAT32, TensorType_FLOAT32, false,
      TensorType_INT8, TensorType_INT32, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

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
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeLSTM2Test, VerifyLSTM) {
  // Quantize model.
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_FLOAT32, TensorType_FLOAT32, false,
      TensorType_INT8, TensorType_INT32, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

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
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeUnidirectionalSequenceLSTMTest,
       VerifyUnidirectionalSequenceLSTM) {
  // Quantize model.
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_FLOAT32, TensorType_FLOAT32, false,
      TensorType_INT8, TensorType_INT32, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

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
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeSVDFTest, VerifySVDF) {
  // Quantize model.
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8, false,
      TensorType_INT8, TensorType_INT32, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  // Read expected model.
  auto expected_fb_model = ReadModel(internal::kSvdfQuantized);
  auto expected_read_only_model = expected_fb_model->GetModel();
  ModelT expected_model;
  expected_read_only_model->UnPackTo(&expected_model);

  // Comparison.
  ASSERT_EQ(model_.subgraphs.size(), expected_model.subgraphs.size());
  for (size_t subgraph_idx = 0; subgraph_idx < model_.subgraphs.size();
       subgraph_idx++) {
    const auto graph = model_.subgraphs[subgraph_idx].get();
    const auto expected_graph = expected_model.subgraphs[subgraph_idx].get();
    ASSERT_EQ(graph->tensors.size(), expected_graph->tensors.size());
    for (size_t i = 0; i < graph->tensors.size(); i++) {
      const auto tensor = graph->tensors[i].get();
      const auto expected_tensor = expected_graph->tensors[i].get();
      EXPECT_EQ(tensor->buffer, expected_tensor->buffer);
      EXPECT_EQ(tensor->is_variable, expected_tensor->is_variable);
      EXPECT_EQ(tensor->shape, expected_tensor->shape);
      EXPECT_EQ(tensor->name, expected_tensor->name);
      EXPECT_EQ(tensor->type, expected_tensor->type);
      const auto quantization_params = tensor->quantization.get();
      const auto expected_quantization_params =
          expected_tensor->quantization.get();
      if (quantization_params != nullptr ||
          expected_quantization_params != nullptr) {
        EXPECT_NE(quantization_params, nullptr);
        EXPECT_NE(expected_quantization_params, nullptr);
        EXPECT_EQ(quantization_params->scale,
                  expected_quantization_params->scale);
        EXPECT_EQ(quantization_params->zero_point,
                  expected_quantization_params->zero_point);
      }
    }
  }
  ASSERT_EQ(model_.buffers.size(), expected_model.buffers.size());
  for (size_t buffer_idx = 0; buffer_idx < model_.buffers.size();
       ++buffer_idx) {
    const auto buffer = model_.buffers[buffer_idx].get()->data;
    const auto expected_buffer = expected_model.buffers[buffer_idx].get()->data;
    EXPECT_EQ(buffer, expected_buffer);
  }
}

class QuantizeFCTest : public QuantizeModelTest {
 protected:
  QuantizeFCTest() {
    input_model_ = ReadModel(internal::kModelWithFCOp);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeFCTest, VerifyFC) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8, false,
      TensorType_INT8, TensorType_INT32, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
            BuiltinOperator_FULLY_CONNECTED);

  ASSERT_EQ(op->inputs.size(), 3);
  ASSERT_EQ(op->outputs.size(), 1);

  auto float_graph = readonly_model_->subgraphs()->Get(0);
  // Verify FC input and weight is quantized.
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[0])->type(),
            TensorType_FLOAT32);
  EXPECT_EQ(subgraph->tensors[op->inputs[0]].get()->type, TensorType_INT8);
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[1])->type(),
            TensorType_FLOAT32);
  EXPECT_EQ(subgraph->tensors[op->inputs[1]].get()->type, TensorType_INT8);

  // Verify FC bias should be int32 quantized.
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[2])->type(),
            TensorType_FLOAT32);
  EXPECT_EQ(subgraph->tensors[op->inputs[2]].get()->type, TensorType_INT32);

  // The output of FC should be quantized.
  ASSERT_EQ(float_graph->tensors()->Get(op->outputs[0])->type(),
            TensorType_FLOAT32);
  EXPECT_EQ(subgraph->tensors[op->outputs[0]].get()->type, TensorType_INT8);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 2);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_FULLY_CONNECTED);
  EXPECT_EQ(model_.operator_codes[0]->version, 4);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[1].get()),
            BuiltinOperator_RESHAPE);
  EXPECT_EQ(model_.operator_codes[1]->version, 1);
}

class QuantizeCustomOpTest
    : public QuantizeModelTest,
      public ::testing::WithParamInterface<tflite::TensorType> {
 protected:
  QuantizeCustomOpTest() {
    tensor_type_ = GetParam();
    bias_type_ = GetBiasTensorType(tensor_type_);
    input_model_ = ReadModel(internal::kModelMixed);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }

  TensorType tensor_type_;
  TensorType bias_type_;
};

TEST_P(QuantizeCustomOpTest, VerifyMixedQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, tensor_type_, tensor_type_,
      /*allow_float=*/true, tensor_type_, bias_type_, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);
  const auto& subgraph = model_.subgraphs[0];
  auto float_graph = readonly_model_->subgraphs()->Get(0);
  // The original model reshape->custom->custom->squeeze.
  ASSERT_EQ(float_graph->operators()->size(), 4);
  // The resulting model should be:
  // reshape->dequantize->custom->custom->quantize->squeeze.
  ASSERT_EQ(subgraph->operators.size(), 6);
  const std::vector<BuiltinOperator> op_codes = {
      BuiltinOperator_RESHAPE,  BuiltinOperator_DEQUANTIZE,
      BuiltinOperator_CUSTOM,   BuiltinOperator_CUSTOM,
      BuiltinOperator_QUANTIZE, BuiltinOperator_SQUEEZE};
  const std::vector<TensorType> op_input_types = {
      GetParam(),         GetParam(),         TensorType_FLOAT32,
      TensorType_FLOAT32, TensorType_FLOAT32, GetParam()};
  for (int i = 0; i < subgraph->operators.size(); ++i) {
    OperatorT* op = subgraph->operators[i].get();
    ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
              op_codes[i]);
    ASSERT_EQ(subgraph->tensors[op->inputs[0]]->type, op_input_types[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(QuantizeCustomOpTest, QuantizeCustomOpTest,
                         ::testing::Values(TensorType_INT8, TensorType_INT16));

class QuantizeOp16x8Test : public QuantizeModelTest {
 protected:
  QuantizeOp16x8Test() {
    input_model_ = ReadModel(internal::kModelMixed16x8);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeOp16x8Test, VerifyMixedQuantization16x8) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT16, TensorType_FLOAT32,
      /*allow_float=*/true, TensorType_INT16, TensorType_INT64,
      &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);
  const auto& subgraph = model_.subgraphs[0];
  auto float_graph = readonly_model_->subgraphs()->Get(0);
  // The original model conv_2d->log_softmax
  ASSERT_EQ(float_graph->operators()->size(), 2);
  // The resulting model should be:
  // conv_2d->dequantize->log_softmax
  ASSERT_EQ(subgraph->operators.size(), 3);
  const std::vector<BuiltinOperator> op_codes = {BuiltinOperator_CONV_2D,
                                                 BuiltinOperator_DEQUANTIZE,
                                                 BuiltinOperator_LOG_SOFTMAX};
  const std::vector<TensorType> op_input_types = {
      TensorType_INT16, TensorType_INT16, TensorType_FLOAT32};
  for (int i = 0; i < subgraph->operators.size(); ++i) {
    OperatorT* op = subgraph->operators[i].get();
    ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
              op_codes[i]);
    ASSERT_EQ(subgraph->tensors[op->inputs[0]]->type, op_input_types[i]);
  }
}

class QuantizePackTest : public QuantizeModelTest {
 protected:
  QuantizePackTest() {
    input_model_ = ReadModel(internal::kModelPack);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizePackTest, VerifyPack) {
  auto status = QuantizeModel(&builder_, &model_, &error_reporter_);

  ASSERT_EQ(kTfLiteOk, status);

  const auto subgraph = model_.subgraphs[0].get();

  // The model should only have 3 inputs and 1 output.
  EXPECT_EQ(subgraph->inputs.size(), 3);
  EXPECT_EQ(subgraph->outputs.size(), 1);

  const auto& op1 = subgraph->operators[1].get();
  const auto& op2 = subgraph->operators[2].get();
  const auto& op3 = subgraph->operators[3].get();
  const auto& op4 = subgraph->operators[4].get();

  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op1->opcode_index].get()),
            BuiltinOperator_QUANTIZE);
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op2->opcode_index].get()),
            BuiltinOperator_QUANTIZE);
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op3->opcode_index].get()),
            BuiltinOperator_PACK);
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op4->opcode_index].get()),
            BuiltinOperator_DEQUANTIZE);

  const auto& pack_input0 = subgraph->tensors[op3->inputs[0]].get();
  const auto& pack_input1 = subgraph->tensors[op3->inputs[1]].get();
  const auto& pack_input2 = subgraph->tensors[op3->inputs[2]].get();

  const auto& pack_output = subgraph->tensors[op3->outputs[0]].get();

  // Check quantization parameters for input and output.
  EXPECT_FLOAT_EQ(pack_input0->quantization->scale[0],
                  pack_input1->quantization->scale[0]);
  EXPECT_FLOAT_EQ(pack_input1->quantization->scale[0],
                  pack_input2->quantization->scale[0]);
  EXPECT_FLOAT_EQ(pack_input0->quantization->zero_point[0],
                  pack_input1->quantization->zero_point[0]);
  EXPECT_FLOAT_EQ(pack_input1->quantization->zero_point[0],
                  pack_input2->quantization->zero_point[0]);

  EXPECT_FLOAT_EQ(pack_input1->quantization->scale[0],
                  pack_output->quantization->scale[0]);
  EXPECT_FLOAT_EQ(pack_input1->quantization->zero_point[0],
                  pack_output->quantization->zero_point[0]);

  // Check type of input and output.
  EXPECT_EQ(pack_output->type, TensorType_INT8);
  EXPECT_EQ(pack_input0->type, TensorType_INT8);
  EXPECT_EQ(pack_input1->type, TensorType_INT8);
  EXPECT_EQ(pack_input2->type, TensorType_INT8);
}

class QuantizeMinimumMaximumTest
    : public QuantizeModelTest,
      public testing::WithParamInterface<const char*> {
 protected:
  QuantizeMinimumMaximumTest() {
    input_model_ = ReadModel(GetParam());
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_P(QuantizeMinimumMaximumTest, VerifyMinimumMaximum) {
  auto status = QuantizeModel(&builder_, &model_, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);
  const auto& subgraph = model_.subgraphs[0];

  // Check that the first op is Quantize and the last is Dequant.
  const auto& quant_op = subgraph->operators[0];
  const auto& dequant_op = subgraph->operators[subgraph->operators.size() - 1];
  const int32_t quant_idx = quant_op->opcode_index;
  const int32_t dequant_idx = dequant_op->opcode_index;
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[quant_idx].get()),
            BuiltinOperator_QUANTIZE);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[dequant_idx].get()),
            BuiltinOperator_DEQUANTIZE);
  const auto& requant1 = subgraph->operators[1].get();
  // Check that we have RE operator.
  auto requant1_builtin_code =
      GetBuiltinCode(model_.operator_codes[requant1->opcode_index].get());
  ASSERT_TRUE(requant1_builtin_code == tflite::BuiltinOperator_QUANTIZE);

  // Constant is quantized rather than adding requant.
  const auto& op = subgraph->operators[2].get();

  // Check that we have MINIMUM or MAXIMUM operator.
  auto op_builtin_code =
      GetBuiltinCode(model_.operator_codes[op->opcode_index].get());
  ASSERT_TRUE(op_builtin_code == tflite::BuiltinOperator_MINIMUM ||
              op_builtin_code == tflite::BuiltinOperator_MAXIMUM);

  // Check that we have two inputs and one output.
  ASSERT_EQ(op->inputs.size(), 2);
  ASSERT_EQ(op->outputs.size(), 1);

  // Check that all is quantized.
  auto output = subgraph->tensors[op->outputs[0]].get();
  auto input1 = subgraph->tensors[op->outputs[0]].get();
  auto input2 = subgraph->tensors[op->outputs[0]].get();

  EXPECT_EQ(output->type, TensorType_INT8);
  EXPECT_EQ(input1->type, TensorType_INT8);
  EXPECT_EQ(input2->type, TensorType_INT8);

  // Check if the quantization params of the minimum/maximum inputs match
  // after requantization
  EXPECT_EQ(input1->quantization->scale, input2->quantization->scale);
  EXPECT_EQ(input1->quantization->zero_point, input2->quantization->zero_point);

  // Check the input quantization params match the output ones.
  EXPECT_EQ(output->quantization->scale, input1->quantization->scale);
  EXPECT_EQ(output->quantization->zero_point, input1->quantization->zero_point);
  EXPECT_EQ(output->quantization->scale, input2->quantization->scale);
  EXPECT_EQ(output->quantization->zero_point, input2->quantization->zero_point);

  EXPECT_EQ(subgraph->tensors.size(), 6);

  EXPECT_EQ(subgraph->tensors[0]->name, "input_int8");
  EXPECT_EQ(subgraph->tensors[1]->name, "output_int8");
  EXPECT_EQ(subgraph->tensors[2]->name, "output/y");
  EXPECT_EQ(subgraph->tensors[3]->name, "input_requantized");
  EXPECT_EQ(subgraph->tensors[4]->name, "input");
  EXPECT_EQ(subgraph->tensors[5]->name, "output");
}

INSTANTIATE_TEST_SUITE_P(MinimumMaximumTestInst, QuantizeMinimumMaximumTest,
                         testing::ValuesIn({internal::kModelWithMinimumOp,
                                            internal::kModelWithMaximumOp}));

class QuantizeUnpackTest : public QuantizeModelTest {
 protected:
  QuantizeUnpackTest() {
    input_model_ = ReadModel(internal::kModelWithUnpack);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};
TEST_F(QuantizeUnpackTest, VerifyUnpack) {
  auto status = QuantizeModel(&builder_, &model_, &error_reporter_);

  ASSERT_EQ(kTfLiteOk, status);

  const auto subgraph = model_.subgraphs[0].get();
  auto op = subgraph->operators[1].get();

  auto float_graph = readonly_model_->subgraphs()->Get(0);

  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
            BuiltinOperator_UNPACK);

  // Get unpack input and output tensors
  auto unpack_input = subgraph->tensors[op->inputs[0]].get();
  auto unpack_output_0 = subgraph->tensors[op->outputs[0]].get();
  auto unpack_output_1 = subgraph->tensors[op->outputs[1]].get();

  // Verify Unpack input is quantized.
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[0])->type(),
            TensorType_FLOAT32);
  EXPECT_EQ(unpack_input->type, TensorType_INT8);

  // The model should only have one input and 2 outputs.
  EXPECT_EQ(subgraph->inputs.size(), 1);
  EXPECT_EQ(subgraph->outputs.size(), 2);

  // Ensure quantization parameters before and after unpack
  // are preserved after quantization for all outputs of
  // unpack.
  EXPECT_FLOAT_EQ(unpack_input->quantization->scale[0],
                  unpack_output_0->quantization->scale[0]);
  EXPECT_FLOAT_EQ(unpack_input->quantization->scale[0],
                  unpack_output_1->quantization->scale[0]);
  EXPECT_FLOAT_EQ(unpack_input->quantization->zero_point[0],
                  unpack_output_0->quantization->zero_point[0]);
  EXPECT_FLOAT_EQ(unpack_input->quantization->zero_point[0],
                  unpack_output_1->quantization->zero_point[0]);
}

class QuantizeTransposeTest : public QuantizeModelTest {
 protected:
  QuantizeTransposeTest() {
    input_model_ = ReadModel(internal::kModelWithTranspose);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeTransposeTest, VerifyTranspose) {
  auto status = QuantizeModel(&builder_, &model_, &error_reporter_);

  ASSERT_EQ(kTfLiteOk, status);

  const auto subgraph = model_.subgraphs[0].get();
  auto op = subgraph->operators[1].get();

  auto float_graph = readonly_model_->subgraphs()->Get(0);

  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
            BuiltinOperator_TRANSPOSE);

  // The model should only have one input and one outputs.
  EXPECT_EQ(subgraph->inputs.size(), 1);
  EXPECT_EQ(subgraph->outputs.size(), 1);

  // Get transpose input and output tensors
  auto transpose_input = subgraph->tensors[op->inputs[0]].get();
  auto transpose_output = subgraph->tensors[op->outputs[0]].get();

  // Verify transpose input is quantized.
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[0])->type(),
            TensorType_FLOAT32);
  EXPECT_EQ(transpose_input->type, TensorType_INT8);

  // Ensure quantization parameters before and after transpose
  // are preserved after quantization for all outputs of
  // transpose.
  EXPECT_FLOAT_EQ(transpose_input->quantization->scale[0],
                  transpose_output->quantization->scale[0]);
  EXPECT_EQ(transpose_input->quantization->zero_point[0],
            transpose_output->quantization->zero_point[0]);
}

class QuantizeQatTest : public QuantizeModelTest {
 protected:
  QuantizeQatTest() {
    input_model_ = ReadModel(internal::kQatModelWithFc);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeQatTest, VerifySingleQuantize) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_FLOAT32, TensorType_FLOAT32, false,
      TensorType_INT8, TensorType_INT32, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
            BuiltinOperator_QUANTIZE);
  op = subgraph->operators[1].get();
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
            BuiltinOperator_RESHAPE);
  op = subgraph->operators[2].get();
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
            BuiltinOperator_FULLY_CONNECTED);

  ASSERT_EQ(op->inputs.size(), 3);
  ASSERT_EQ(op->outputs.size(), 1);

  auto qat_graph = readonly_model_->subgraphs()->Get(0);
  // Verify FC input and weight is quantized.
  ASSERT_EQ(qat_graph->tensors()->Get(op->inputs[0])->type(), TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[op->inputs[0]].get()->type, TensorType_INT8);
  ASSERT_EQ(qat_graph->tensors()->Get(op->inputs[1])->type(), TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[op->inputs[1]].get()->type, TensorType_INT8);

  // Verify FC bias should be int32 quantized.
  ASSERT_EQ(qat_graph->tensors()->Get(op->inputs[2])->type(), TensorType_INT32);
  EXPECT_EQ(subgraph->tensors[op->inputs[2]].get()->type, TensorType_INT32);

  // The output of FC should be quantized.
  ASSERT_EQ(qat_graph->tensors()->Get(op->outputs[0])->type(), TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[op->outputs[0]].get()->type, TensorType_INT8);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 4);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_QUANTIZE);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[1].get()),
            BuiltinOperator_RESHAPE);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[2].get()),
            BuiltinOperator_FULLY_CONNECTED);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[3].get()),
            BuiltinOperator_DEQUANTIZE);
  EXPECT_EQ(model_.operator_codes[1]->version, 1);
  EXPECT_EQ(model_.operator_codes[2]->version, 4);
}

class QuantizeBroadcastToModelTest
    : public QuantizeModelTest,
      public testing::WithParamInterface<TensorType> {
 protected:
  QuantizeBroadcastToModelTest() {
    tensor_type_ = GetParam();
    bias_type_ = GetBiasTensorType(tensor_type_);
    input_model_ = ReadModel(internal::kModelWithBroadcastToOp);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
  TensorType tensor_type_;
  TensorType bias_type_;
};

INSTANTIATE_TEST_SUITE_P(QuantizeBroadcastToModelTestInst,
                         QuantizeBroadcastToModelTest,
                         testing::ValuesIn({TensorType_INT8,
                                            TensorType_INT16}));

TEST_P(QuantizeBroadcastToModelTest, VerifyBroadcastToQuantization) {
  auto status = QuantizeModelAllOperators(&builder_, &model_, tensor_type_,
                                          tensor_type_, false, tensor_type_,
                                          bias_type_, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);

  // There is only one subgraph.
  const int32_t subgraph_idx = 0;
  const auto& subgraph = model_.subgraphs[subgraph_idx];
  const auto& readonly_subgraph =
      readonly_model_->subgraphs()->Get(subgraph_idx);

  // There should be a single broadcast_to op.
  EXPECT_EQ(readonly_subgraph->operators()->size(), 1);
  EXPECT_EQ(subgraph->operators.size(), 1);
  const auto& broadcast_to = subgraph->operators[0];
  EXPECT_EQ(model_.operator_codes[broadcast_to->opcode_index]->builtin_code,
            BuiltinOperator_BROADCAST_TO);

  // There should be 3 tensors: input, output, and BroadcastTo/shape.
  EXPECT_EQ(subgraph->tensors.size(), 3);

  // Input Tensor
  EXPECT_EQ(subgraph->tensors[0]->type, tensor_type_);
  EXPECT_EQ(subgraph->tensors[0]->name, "input_1");
  EXPECT_EQ(subgraph->tensors[0]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[0]->quantization->zero_point.size(), 1);

  // Output Tensor. The name given in the generated
  // .bin test file is 'Identity' and should be preserved
  EXPECT_EQ(subgraph->tensors[2]->type, tensor_type_);
  EXPECT_EQ(subgraph->tensors[2]->name, "Identity");
  EXPECT_EQ(subgraph->tensors[2]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[2]->quantization->zero_point.size(), 1);

  // The BroadCastTo shape is of type INT32 and should not be quantized
  EXPECT_EQ(subgraph->tensors[1]->type, TensorType_INT32);
  EXPECT_EQ(subgraph->tensors[1]->name,
            "model/tf.broadcast_to/BroadcastTo/shape");
  EXPECT_EQ(subgraph->tensors[1]->quantization->scale.size(), 0);
  EXPECT_EQ(subgraph->tensors[1]->quantization->zero_point.size(), 0);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(model_.operator_codes[0]->builtin_code,
            BuiltinOperator_BROADCAST_TO);
  EXPECT_EQ(model_.operator_codes[0]->version, 3);
}

class QuantizeGatherNDModelTest
    : public QuantizeModelTest,
      public testing::WithParamInterface<TensorType> {
 protected:
  QuantizeGatherNDModelTest() {
    tensor_type_ = GetParam();
    bias_type_ = GetBiasTensorType(tensor_type_);
    input_model_ = ReadModel(internal::kModelWithGatherNDOp);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }

  TensorType tensor_type_;
  TensorType bias_type_;
};

INSTANTIATE_TEST_SUITE_P(QuantizeGatherNDModelTestInst,
                         QuantizeGatherNDModelTest,
                         testing::ValuesIn({TensorType_INT8,
                                            TensorType_INT16}));

TEST_P(QuantizeGatherNDModelTest, QuantizeGatherND) {
  auto status = QuantizeModelAllOperators(&builder_, &model_, tensor_type_,
                                          tensor_type_, false, tensor_type_,
                                          bias_type_, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);

  // There is only one subgraph.
  const int32_t subgraph_idx = 0;
  const auto& subgraph = model_.subgraphs[subgraph_idx];
  const auto& readonly_subgraph =
      readonly_model_->subgraphs()->Get(subgraph_idx);

  // There should be a single gather_nd op.
  EXPECT_EQ(readonly_subgraph->operators()->size(), 1);
  EXPECT_EQ(subgraph->operators.size(), 1);
  const auto& gather_nd = subgraph->operators[0];
  EXPECT_EQ(model_.operator_codes[gather_nd->opcode_index]->builtin_code,
            BuiltinOperator_GATHER_ND);

  // There should be 3 tensors: input, output, and indices.
  EXPECT_EQ(subgraph->tensors.size(), 3);

  // Input Tensor
  EXPECT_EQ(subgraph->tensors[0]->type, tensor_type_);
  EXPECT_EQ(subgraph->tensors[0]->name, "input");
  EXPECT_EQ(subgraph->tensors[0]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[0]->quantization->zero_point.size(), 1);

  // Output Tensor
  EXPECT_EQ(subgraph->tensors[2]->type, tensor_type_);
  EXPECT_EQ(subgraph->tensors[2]->name, "output");
  EXPECT_EQ(subgraph->tensors[2]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[2]->quantization->zero_point.size(), 1);

  // The gather indices are of type INT32 and should not be quantized
  EXPECT_EQ(subgraph->tensors[1]->type, TensorType_INT32);
  EXPECT_EQ(subgraph->tensors[1]->name, "indices");
  EXPECT_EQ(subgraph->tensors[1]->quantization->scale.size(), 0);
  EXPECT_EQ(subgraph->tensors[1]->quantization->zero_point.size(), 0);

  // Check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(model_.operator_codes[0]->builtin_code, BuiltinOperator_GATHER_ND);
  EXPECT_EQ(model_.operator_codes[0]->version, 3);
}

class QuantizeWhereModelTest : public QuantizeModelTest {
 protected:
  QuantizeWhereModelTest() {
    input_model_ = ReadModel(internal::kModelWithWhereOp);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeWhereModelTest, QuantizeWhere) {
  // Where operator takes a BOOL tensor as input
  // and outputs INT64 indices, both of which
  // should not be quantized
  auto status = QuantizeModel(&builder_, &model_, TensorType_BOOL,
                              TensorType_INT64, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);

  // There is only one subgraph.
  const int32_t subgraph_idx = 0;
  const auto& subgraph = model_.subgraphs[subgraph_idx];
  const auto& readonly_subgraph =
      readonly_model_->subgraphs()->Get(subgraph_idx);

  // There should be a single where op.
  EXPECT_EQ(readonly_subgraph->operators()->size(), 1);
  EXPECT_EQ(subgraph->operators.size(), 1);
  const auto& where = subgraph->operators[0];
  EXPECT_EQ(model_.operator_codes[where->opcode_index]->builtin_code,
            BuiltinOperator_WHERE);

  // There should be 2 tensors: input and output.
  EXPECT_EQ(subgraph->tensors.size(), 2);

  // Testing input tensor type and ensuring it
  // was not quantized
  EXPECT_EQ(subgraph->tensors[0]->type, TensorType_BOOL);
  EXPECT_EQ(subgraph->tensors[0]->name, "input");
  EXPECT_EQ(subgraph->tensors[0]->quantization->scale.size(), 0);
  EXPECT_EQ(subgraph->tensors[0]->quantization->zero_point.size(), 0);

  // Testing output (indices) tensor type and ensuring it
  // was not quantized
  EXPECT_EQ(subgraph->tensors[1]->type, TensorType_INT64);
  EXPECT_EQ(subgraph->tensors[1]->name, "indices");
  EXPECT_EQ(subgraph->tensors[1]->quantization->scale.size(), 0);
  EXPECT_EQ(subgraph->tensors[1]->quantization->zero_point.size(), 0);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(model_.operator_codes[0]->builtin_code, BuiltinOperator_WHERE);
  EXPECT_EQ(model_.operator_codes[0]->version, 1);
}

enum struct ModifyRangeType {
  kNone = 0,
  kAll = 1,
  kReadOnly = 2,
  kAssignOnly = 3,
};

struct TestType {
  TensorType tensor_type;
  ModifyRangeType modify_range;
};

struct BiasTestType {
  TensorType tensor_type;
  TensorType bias_type;
  bool is_valid_bias_type;
};

class QuantizeResourcesModelTest
    : public QuantizeModelTest,
      public testing::WithParamInterface<TestType> {
 protected:
  QuantizeResourcesModelTest() {
    TestType obj = GetParam();
    tensor_type_ = obj.tensor_type;
    modify_range_ = obj.modify_range;
    bias_type_ = GetBiasTensorType(tensor_type_);
    input_model_ = ReadModel(internal::kModelWithResourceVarsCalibrated);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_, nullptr);
    if (modify_range_ != ModifyRangeType::kNone) {
      ModifyRange(&model_);
    }
  }
  void ModifyRange(ModelT* model) {
    // Modify ranges to test when min/max of the primary subgraph variable
    // is smaller than the initializer subgraph.
    const bool do_read = (modify_range_ == ModifyRangeType::kAll ||
                          modify_range_ == ModifyRangeType::kReadOnly);
    const bool do_assign = (modify_range_ == ModifyRangeType::kAll ||
                            modify_range_ == ModifyRangeType::kAssignOnly);
    SubGraphT* subgraph = model->subgraphs[0].get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); ++op_idx) {
      OperatorT* op = subgraph->operators[op_idx].get();
      const BuiltinOperator op_code =
          GetBuiltinCode(model_.operator_codes[op->opcode_index].get());
      TensorT* var_tensor;
      if (op_code == BuiltinOperator_ASSIGN_VARIABLE && do_assign) {
        var_tensor = subgraph->tensors[op->inputs[1]].get();
      } else if (op_code == BuiltinOperator_READ_VARIABLE && do_read) {
        var_tensor = subgraph->tensors[op->outputs[0]].get();
      } else {
        continue;
      }
      // This value is lower than the initial values, so should be replaced
      var_tensor->quantization->max[0] = 12.5;
    }
  }
  TensorType tensor_type_;
  TensorType bias_type_;
  ModifyRangeType modify_range_ = ModifyRangeType::kAll;
};

INSTANTIATE_TEST_SUITE_P(QuantizeResourcesModelTest, QuantizeResourcesModelTest,
                         testing::ValuesIn<TestType>(
                             {{TensorType_INT8, ModifyRangeType::kNone},
                              {TensorType_INT8, ModifyRangeType::kAll},
                              {TensorType_INT8, ModifyRangeType::kReadOnly},
                              {TensorType_INT8, ModifyRangeType::kAssignOnly},
                              {TensorType_INT16, ModifyRangeType::kNone},
                              {TensorType_INT16, ModifyRangeType::kAll},
                              {TensorType_INT16, ModifyRangeType::kReadOnly},
                              {TensorType_INT16,
                               ModifyRangeType::kAssignOnly}}));

TEST_P(QuantizeResourcesModelTest, GraphIsFullyQuantized) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, tensor_type_, tensor_type_,
      /*allow_float*/ false, tensor_type_, bias_type_, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);
  std::vector<QuantizationParametersT*> quant_params;
  const float quant_eps = tensor_type_ == TensorType_INT8 ? 1e-1 : 1e-2;
  for (const auto& subgraph : model_.subgraphs) {
    for (const auto& tensor : subgraph->tensors) {
      if (tensor_type_ == TensorType_INT8) {
        EXPECT_TRUE(
            tensor->type == TensorType_RESOURCE ||  // resource
            tensor->type == TensorType_INT32 ||     // bias and gather indices
            tensor->type == TensorType_INT8);       // weights and activations
      } else if (tensor_type_ == TensorType_INT16) {
        EXPECT_TRUE(tensor->type == TensorType_RESOURCE ||  // resource
                    tensor->type == TensorType_INT64 ||     // bias
                    tensor->type == TensorType_INT32 ||     // gather indices
                    tensor->type == TensorType_INT16 ||     // activations
                    tensor->type == TensorType_INT8);       // weights
      }
    }
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); ++op_idx) {
      OperatorT* op = subgraph->operators[op_idx].get();
      const BuiltinOperator op_code =
          GetBuiltinCode(model_.operator_codes[op->opcode_index].get());
      if (op_code == BuiltinOperator_ASSIGN_VARIABLE) {
        TensorT* var_tensor = subgraph->tensors[op->inputs[1]].get();
        quant_params.push_back(var_tensor->quantization.get());
        if (model_.buffers[var_tensor->buffer] &&
            !model_.buffers[var_tensor->buffer]->data.empty()) {
          const BufferT* buffer = model_.buffers[var_tensor->buffer].get();
          const int num_elements = 25;
          const int expected_buffer_size = tensor_type_ == TensorType_INT8
                                               ? num_elements * sizeof(int8_t)
                                               : num_elements * sizeof(int16_t);
          EXPECT_EQ(buffer->data.size(), expected_buffer_size);
          for (int i = 0; i < num_elements; ++i) {
            float dequantized = 0;
            if (tensor_type_ == TensorType_INT8) {
              auto data = reinterpret_cast<const int8_t*>(buffer->data.data());
              const int zero_point = var_tensor->quantization->zero_point[0];
              dequantized =
                  (data[i] - zero_point) * var_tensor->quantization->scale[0];
            } else if (tensor_type_ == TensorType_INT16) {
              auto data = reinterpret_cast<const int16_t*>(buffer->data.data());
              dequantized = data[i] * var_tensor->quantization->scale[0];
            }
            EXPECT_NEAR(dequantized, 25.0 - i, quant_eps);
          }
        }
      } else if (op_code == BuiltinOperator_READ_VARIABLE) {
        TensorT* var_tensor = subgraph->tensors[op->outputs[0]].get();
        quant_params.push_back(var_tensor->quantization.get());
      }

      // Test that the bias was duplicated.
      if (op_code == BuiltinOperator_FULLY_CONNECTED) {
        TensorT* bias = subgraph->tensors[op->inputs[2]].get();
        EXPECT_EQ(bias->name, "Const_duplicate_1");
        if (tensor_type_ == TensorType_INT8) {
          EXPECT_EQ(bias->type, TensorType_INT32);
        } else if (tensor_type_ == TensorType_INT8) {
          EXPECT_EQ(bias->type, TensorType_INT64);
        }
      }
    }
  }
  EXPECT_EQ(quant_params.size(), 4);
  QuantizationParametersT* expected_quant_param = quant_params[0];
  EXPECT_EQ(expected_quant_param->scale.size(), 1);
  float expected_scale =
      tensor_type_ == TensorType_INT8 ? 0.1960605f : 0.0015258f;
  if (modify_range_ == ModifyRangeType::kAll) {
    expected_scale = tensor_type_ == TensorType_INT8 ? 0.0980392f : 0.0007629f;
  }
  const float eps = 1e-7;
  EXPECT_NEAR(expected_quant_param->scale[0], expected_scale, eps);
  for (int i = 1; i < quant_params.size(); ++i) {
    QuantizationParametersT* test_param = quant_params[i];
    EXPECT_EQ(test_param->scale, expected_quant_param->scale);
    EXPECT_EQ(test_param->zero_point, expected_quant_param->zero_point);
    EXPECT_EQ(test_param->min, expected_quant_param->min);
    EXPECT_EQ(test_param->max, expected_quant_param->max);
  }
}

class QuantizeConcatConstModelTest
    : public QuantizeModelTest,
      public testing::WithParamInterface<TensorType> {
 protected:
  QuantizeConcatConstModelTest() {
    input_model_ = ReadModel(internal::kFloatConcatMax5Max10Max10);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
    // Make one of the values constant.
    MakeInputConstant(&model_);
  }

  void SetUp() override {
    tensor_type_ = GetParam();
    bias_type_ = GetBiasTensorType(tensor_type_);
  }

  void MakeInputConstant(tflite::ModelT* model) {
    auto& subgraph = model->subgraphs[0];
    const int tensor_id = subgraph->inputs.back();
    int replace_tensor_id = subgraph->inputs[0];
    subgraph->inputs[0] = tensor_id;
    subgraph->inputs.pop_back();
    auto& tensor = subgraph->tensors[replace_tensor_id];
    tensor->name = "const_input0";
    model->buffers.emplace_back(new tflite::BufferT());
    tensor->buffer = model->buffers.size() - 1;
    auto& buffer = model->buffers[tensor->buffer];
    std::vector<float> tensor_buffer = {0.0, 5.0};
    uint8_t* uint8_data = reinterpret_cast<uint8_t*>(tensor_buffer.data());
    buffer->data = std::vector<uint8_t>(
        uint8_data, uint8_data + (sizeof(float) * tensor_buffer.size()));
  }

  TensorType tensor_type_;
  TensorType bias_type_;
};

TEST_P(QuantizeConcatConstModelTest, AddRequantBeforeConcat) {
  auto status = QuantizeModelAllOperators(&builder_, &model_, tensor_type_,
                                          tensor_type_, false, tensor_type_,
                                          bias_type_, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);

  // There is only one subgraph.
  const int32_t subgraph_idx = 0;
  const auto& subgraph = model_.subgraphs[subgraph_idx];
  const auto& readonly_subgraph =
      readonly_model_->subgraphs()->Get(subgraph_idx);

  // There should be 1 op: concat.
  EXPECT_EQ(readonly_subgraph->operators()->size(), 1);
  EXPECT_EQ(subgraph->operators.size(), 1);
  const auto& concat = subgraph->operators[0];
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[concat->opcode_index].get()),
            BuiltinOperator_CONCATENATION);

  auto zero_point_control = tensor_type_ == TensorType_INT8 ? -128 : 0;

  auto input0_scale_control =
      tensor_type_ == TensorType_INT8 ? 0.039215688 : 0.00030518509;
  auto input1_scale =
      tensor_type_ == TensorType_INT8 ? 0.039215688 : 0.00030518509;

  // There should be 3 tensors: const_input0, input1, output.
  EXPECT_EQ(subgraph->tensors.size(), 3);
  EXPECT_EQ(subgraph->tensors[0]->type, tensor_type_);
  EXPECT_EQ(subgraph->tensors[0]->name, "const_input0");
  EXPECT_EQ(subgraph->tensors[0]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[0]->quantization->zero_point.size(), 1);
  EXPECT_FLOAT_EQ(subgraph->tensors[0]->quantization->scale[0],
                  input0_scale_control);
  EXPECT_FLOAT_EQ(subgraph->tensors[0]->quantization->zero_point[0],
                  zero_point_control);

  EXPECT_EQ(subgraph->tensors[1]->type, tensor_type_);
  EXPECT_EQ(subgraph->tensors[1]->name, "input1");
  EXPECT_EQ(subgraph->tensors[1]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[1]->quantization->zero_point.size(), 1);
  EXPECT_FLOAT_EQ(subgraph->tensors[1]->quantization->scale[0], input1_scale);
  EXPECT_FLOAT_EQ(subgraph->tensors[1]->quantization->zero_point[0],
                  zero_point_control);
  EXPECT_EQ(subgraph->tensors[2]->type, tensor_type_);
  EXPECT_EQ(subgraph->tensors[2]->name, "output");
  EXPECT_EQ(subgraph->tensors[2]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[2]->quantization->zero_point.size(), 1);
  EXPECT_FLOAT_EQ(subgraph->tensors[2]->quantization->scale[0], input1_scale);
  EXPECT_FLOAT_EQ(subgraph->tensors[2]->quantization->zero_point[0],
                  zero_point_control);

  EXPECT_EQ(concat->inputs.size(), 2);
  EXPECT_EQ(concat->outputs.size(), 1);
  EXPECT_EQ(concat->inputs[0], 0);
  EXPECT_EQ(concat->inputs[1], 1);
  EXPECT_EQ(concat->outputs[0], 2);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_CONCATENATION);
  EXPECT_EQ(model_.operator_codes[0]->version, 2);
}

INSTANTIATE_TEST_SUITE_P(QuantizeConcatConstModelTest,
                         QuantizeConcatConstModelTest,
                         testing::ValuesIn({TensorType_INT8,
                                            TensorType_INT16}));

class BiasInputTest : public QuantizeModelTest,
                      public testing::WithParamInterface<BiasTestType> {
 protected:
  BiasInputTest() {
    BiasTestType obj = GetParam();
    tensor_type_ = obj.tensor_type;
    bias_type_ = obj.bias_type;
    is_valid_bias_type_ = obj.is_valid_bias_type;
    input_model_ = ReadModel(internal::kConvModelWith0Plus10Weights);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
  TensorType tensor_type_;
  TensorType bias_type_;
  bool is_valid_bias_type_;
  tflite::TestErrorReporter test_error_reporter_;
};

INSTANTIATE_TEST_SUITE_P(BiasInputTestInst, BiasInputTest,
                         testing::ValuesIn<BiasTestType>(
                             {{TensorType_INT8, TensorType_INT32, true},
                              {TensorType_INT8, TensorType_FLOAT32, false},
                              {TensorType_INT16, TensorType_INT32, true},
                              {TensorType_INT16, TensorType_INT64, true},
                              {TensorType_INT16, TensorType_FLOAT32, false}}));

TEST_P(BiasInputTest, QuantizationSucceeds) {
  auto status = QuantizeModelAllOperators(&builder_, &model_, tensor_type_,
                                          tensor_type_, false, tensor_type_,
                                          bias_type_, &test_error_reporter_);
  if (is_valid_bias_type_) {
    EXPECT_EQ(status, kTfLiteOk);
    const uint8_t* buffer = builder_.GetBufferPointer();
    const Model* output_model = GetModel(buffer);
    ASSERT_TRUE(output_model);
  } else {
    EXPECT_EQ(status, kTfLiteError);
  }
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
