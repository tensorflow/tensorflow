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
#include <cstddef>
#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // TF:flatbuffers
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/optimize/quantize_model.h"
#include "tensorflow/lite/tools/optimize/test_util.h"

// Note: More rigorous model tests can be found in subgraph_quantizer_test.cc

namespace {
tensorflow::string* g_test_model_dir = nullptr;
}  // namespace

namespace tflite {
namespace optimize {
namespace {

std::unique_ptr<FlatBufferModel> ReadTestModel() {
  auto model_path = tensorflow::io::JoinPath(
      *g_test_model_dir, internal::kConvModelWith0Plus10Weights);
  return FlatBufferModel::BuildFromFile(model_path.c_str());
}

template <typename T>
std::vector<T> GetAsVector(const flatbuffers::Vector<T>* vec) {
  return std::vector<T>(vec->begin(), vec->end());
}

class QuantizeModelTest : public testing::Test {
 protected:
  QuantizeModelTest() {
    input_model_ = ReadTestModel();
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }

  std::unique_ptr<FlatBufferModel> input_model_;
  const Model* readonly_model_;
  tflite::ModelT model_;
  flatbuffers::FlatBufferBuilder builder_;
  internal::FailOnErrorReporter error_reporter_;
};

TEST_F(QuantizeModelTest, QuantizationSucceeds) {
  auto status = QuantizeModel(&builder_, &model_, TensorType_INT8,
                              TensorType_INT8, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);
  const uint8_t* buffer = builder_.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);
}

TEST_F(QuantizeModelTest, TensorShapesAndStructureIsUnchanged) {
  auto status = QuantizeModel(&builder_, &model_, TensorType_INT8,
                              TensorType_INT8, &error_reporter_);
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
}

TEST_F(QuantizeModelTest, OperatorsAreUnchanged) {
  auto status = QuantizeModel(&builder_, &model_, TensorType_INT8,
                              TensorType_INT8, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);
  ASSERT_EQ(model_.operator_codes.size(),
            readonly_model_->operator_codes()->size());
  for (size_t i = 0; i < model_.operator_codes.size(); i++) {
    const auto float_model_op = readonly_model_->operator_codes()->Get(i);
    EXPECT_EQ(model_.operator_codes[i]->builtin_code,
              float_model_op->builtin_code());
    EXPECT_EQ(model_.operator_codes[i]->version, float_model_op->version());
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

TEST_F(QuantizeModelTest, GraphIsFullyQuantized) {
  auto status = QuantizeModel(&builder_, &model_, TensorType_INT8,
                              TensorType_INT8, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);
  for (const auto& subgraph : model_.subgraphs) {
    for (const auto& tensor : subgraph->tensors) {
      EXPECT_TRUE(tensor->type == TensorType_INT32 ||
                  tensor->type == TensorType_INT8);
    }
  }
}

TEST_F(QuantizeModelTest, FloatInputAndOutput) {
  auto status = QuantizeModel(&builder_, &model_, &error_reporter_);
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
    EXPECT_EQ(model_.operator_codes[quant_idx]->builtin_code,
              BuiltinOperator_QUANTIZE);
    EXPECT_EQ(model_.operator_codes[dequant_idx]->builtin_code,
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
    EXPECT_EQ(subgraph->tensors[quant_op->outputs[0]]->name, "input_int8");
    EXPECT_EQ(subgraph->tensors[dequant_op->inputs[0]]->name, "output_int8");
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

TEST_F(QuantizeModelTest, Uint8InputAndOutput) {
  auto status = QuantizeModel(&builder_, &model_, TensorType_UINT8,
                              TensorType_UINT8, &error_reporter_);
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
    EXPECT_EQ(model_.operator_codes[quant_op_uint8_int8_idx]->builtin_code,
              BuiltinOperator_QUANTIZE);
    EXPECT_EQ(model_.operator_codes[quant_op_int8_uint8_idx]->builtin_code,
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
