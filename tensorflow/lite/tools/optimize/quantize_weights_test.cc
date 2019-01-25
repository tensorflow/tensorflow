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
#include "tensorflow/lite/tools/optimize/quantize_weights.h"
#include "tensorflow/lite/tools/optimize/test_util.h"

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

std::unique_ptr<FlatBufferModel> ReadSharedWeightsTestModel() {
  auto model_path = tensorflow::io::JoinPath(*g_test_model_dir,
                                             internal::kModelWithSharedWeights);
  return FlatBufferModel::BuildFromFile(model_path.c_str());
}

template <typename T>
std::vector<T> GetAsVector(const flatbuffers::Vector<T>* vec) {
  return std::vector<T>(vec->begin(), vec->end());
}

class QuantizeWeightsTest : public testing::Test {
 protected:
  QuantizeWeightsTest() {}

  void LoadBasicModel() {
    input_model_ = ReadTestModel();
    model_ = input_model_->GetModel();
  }

  void LoadSharedWeightsModel() {
    input_model_ = ReadSharedWeightsTestModel();
    model_ = input_model_->GetModel();
  }

  std::unique_ptr<FlatBufferModel> input_model_;
  const Model* model_;

  bool IsModelInputOrOutput(const Model* model, uint32_t tensor_idx) {
    for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
         ++subgraph_idx) {
      const auto subgraph = model->subgraphs()->Get(subgraph_idx);
      for (size_t i = 0; i < subgraph->inputs()->size(); ++i) {
        if (subgraph->inputs()->Get(i) == tensor_idx) {
          return true;
        }
      }
      for (size_t i = 0; i < subgraph->outputs()->size(); ++i) {
        if (subgraph->outputs()->Get(i) == tensor_idx) {
          return true;
        }
      }
    }
    return false;
  }

  // Returns the producer op code of the specified tensor_idx.
  BuiltinOperator GetProducerOpCode(const Model* model, uint32_t subgraph_idx,
                                    uint32_t tensor_idx) {
    const auto subgraph = model->subgraphs()->Get(subgraph_idx);
    for (size_t op_idx = 0; op_idx < subgraph->operators()->size(); ++op_idx) {
      const auto op = subgraph->operators()->Get(op_idx);
      for (size_t i = 0; i < op->outputs()->size(); ++i) {
        if (op->outputs()->Get(i) == tensor_idx) {
          const uint32_t op_code_idx = op->opcode_index();
          return model->operator_codes()->Get(op_code_idx)->builtin_code();
        }
      }
    }

    LOG(FATAL) << "tensor_idx " << tensor_idx
               << " not produced by op in model.";
  }
};

TEST_F(QuantizeWeightsTest, QuantizationSucceeds) {
  LoadBasicModel();
  flatbuffers::FlatBufferBuilder builder;
  auto status = QuantizeWeights(&builder, model_, 0);
  EXPECT_EQ(status, kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);
}

TEST_F(QuantizeWeightsTest, WeightsMinNumElements) {
  LoadBasicModel();
  // Make weights_min_size sufficiently large such that no quantization should
  // happen, i.e. the original model is the same size as the old one.
  flatbuffers::FlatBufferBuilder builder;
  const uint64_t kWeightsMinNumElements = 1000000;
  EXPECT_EQ(QuantizeWeights(&builder, model_, kWeightsMinNumElements),
            kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
       subgraph_idx++) {
    const auto quantized_graph = output_model->subgraphs()->Get(subgraph_idx);
    const auto float_graph = model_->subgraphs()->Get(subgraph_idx);
    ASSERT_EQ(quantized_graph->tensors()->size(),
              float_graph->tensors()->size());
    for (size_t i = 0; i < quantized_graph->tensors()->size(); i++) {
      const auto quant_tensor = quantized_graph->tensors()->Get(i);
      const auto float_tensor = float_graph->tensors()->Get(i);
      // Everything should remain equal between the two graphs.
      EXPECT_EQ(quant_tensor->buffer(), float_tensor->buffer());
      EXPECT_EQ(quant_tensor->is_variable(), float_tensor->is_variable());
      EXPECT_EQ(GetAsVector(quant_tensor->shape()),
                GetAsVector(float_tensor->shape()));
      EXPECT_EQ(quant_tensor->name()->str(), float_tensor->name()->str());
      EXPECT_EQ(quant_tensor->type(), float_tensor->type());
    }
  }
}

TEST_F(QuantizeWeightsTest, HybridConv) {
  LoadBasicModel();
  flatbuffers::FlatBufferBuilder builder;
  auto status = QuantizeWeights(&builder, model_, 0);
  EXPECT_EQ(status, kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  // Nothing should change.
  ASSERT_EQ(output_model->subgraphs()->size(), model_->subgraphs()->size());
  for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
       subgraph_idx++) {
    const auto quantized_graph = output_model->subgraphs()->Get(subgraph_idx);
    const auto float_graph = model_->subgraphs()->Get(subgraph_idx);
    ASSERT_EQ(quantized_graph->tensors()->size(),
              float_graph->tensors()->size());
    // Make sure the graph only has one Conv operation.
    ASSERT_EQ(quantized_graph->operators()->size(), 1);
    const auto op = quantized_graph->operators()->Get(0);
    const uint32_t op_code_idx = op->opcode_index();
    ASSERT_EQ(output_model->operator_codes()->Get(op_code_idx)->builtin_code(),
              BuiltinOperator_CONV_2D);
    for (size_t i = 0; i < quantized_graph->tensors()->size(); i++) {
      const auto quant_tensor = quantized_graph->tensors()->Get(i);
      const auto float_tensor = float_graph->tensors()->Get(i);
      EXPECT_EQ(quant_tensor->buffer(), float_tensor->buffer());
      EXPECT_EQ(quant_tensor->is_variable(), float_tensor->is_variable());
      EXPECT_EQ(GetAsVector(quant_tensor->shape()),
                GetAsVector(float_tensor->shape()));
      EXPECT_EQ(quant_tensor->name()->str(), float_tensor->name()->str());
      // If the tensor is a weight, it should have type INT8, otherwise it
      // should stay with type FLOAT32.
      // If the tensor is a bias, it should have type FLOAT32.
      if (quant_tensor->name()->str() == "conv_bias") {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (IsModelInputOrOutput(output_model, i)) {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (quant_tensor->buffer() != 0) {
        EXPECT_EQ(quant_tensor->type(), TensorType_INT8)
            << quant_tensor->name()->str();
      } else {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      }
    }
  }
}

TEST_F(QuantizeWeightsTest, DequantizeConv) {
  LoadBasicModel();
  flatbuffers::FlatBufferBuilder builder;
  auto status = internal::QuantizeWeights(&builder, model_, 0,
                                          /*use_hybrid_evaluation=*/false);
  EXPECT_EQ(status, kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  ASSERT_EQ(output_model->subgraphs()->size(), model_->subgraphs()->size());
  for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
       ++subgraph_idx) {
    const auto quantized_graph = output_model->subgraphs()->Get(subgraph_idx);
    const auto float_graph = model_->subgraphs()->Get(subgraph_idx);
    // The output graph should have an extra tensor from the added dequantize
    // op.
    ASSERT_EQ(quantized_graph->tensors()->size(),
              float_graph->tensors()->size() + 1);
    // Check that a dequantize op exists.
    int32_t dequant_input_idx = -1;
    int32_t dequant_output_idx = -1;
    for (size_t i = 0; i < quantized_graph->operators()->size(); ++i) {
      const auto op = quantized_graph->operators()->Get(i);
      const uint32_t op_code_idx = op->opcode_index();
      if (output_model->operator_codes()->Get(op_code_idx)->builtin_code() ==
          BuiltinOperator_DEQUANTIZE) {
        dequant_input_idx = op->inputs()->Get(0);
        dequant_output_idx = op->outputs()->Get(0);
      }
    }
    ASSERT_GT(dequant_input_idx, -1);
    ASSERT_GT(dequant_output_idx, -1);
    for (size_t i = 0; i < quantized_graph->tensors()->size(); ++i) {
      const auto quant_tensor = quantized_graph->tensors()->Get(i);
      // If the tensor is a weight, it should have type INT8.
      // If the tensor is a bias, it should have type FLOAT32.
      // If the tensor is an input or output it should have type FLOAT32.
      // The input to dequantize should be INT8, and all other tensors should be
      // FLOAT32.
      if (i == dequant_input_idx) {
        EXPECT_EQ(quant_tensor->type(), TensorType_INT8);
      } else if (i == dequant_output_idx) {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (IsModelInputOrOutput(output_model, i)) {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (quant_tensor->name()->str() == "conv_bias") {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (quant_tensor->buffer() != 0) {
        // If its a non-bias constant tensor, is must be the weight.
        EXPECT_EQ(quant_tensor->type(), TensorType_INT8);
      } else {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      }
    }
  }
}

TEST_F(QuantizeWeightsTest, SharedWeights_Hybrid) {
  LoadSharedWeightsModel();
  flatbuffers::FlatBufferBuilder builder;
  auto status = QuantizeWeights(&builder, model_, 0);
  EXPECT_EQ(status, kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  ASSERT_EQ(output_model->subgraphs()->size(), model_->subgraphs()->size());
  uint32_t num_conv_ops = 0;
  for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
       ++subgraph_idx) {
    const auto quantized_graph = output_model->subgraphs()->Get(subgraph_idx);
    for (size_t i = 0; i < quantized_graph->operators()->size(); ++i) {
      const auto op = quantized_graph->operators()->Get(i);
      const uint32_t op_code_idx = op->opcode_index();
      const auto op_code =
          output_model->operator_codes()->Get(op_code_idx)->builtin_code();
      if (op_code == BuiltinOperator_CONV_2D) {
        num_conv_ops++;
        // Ensure that each convolution's weights tensor is now INT8.
        const auto weights_tensor =
            quantized_graph->tensors()->Get(op->inputs()->Get(1));
        EXPECT_EQ(weights_tensor->type(), TensorType_INT8);
      }
    }
  }
  // Ensure that there were exactly two convolutions in the model.
  EXPECT_EQ(num_conv_ops, 2);
}

TEST_F(QuantizeWeightsTest, SharedWeights_Dequantize) {
  LoadSharedWeightsModel();
  flatbuffers::FlatBufferBuilder builder;
  auto status = internal::QuantizeWeights(&builder, model_, 0,
                                          /*use_hybrid_evaluation*/ false);
  EXPECT_EQ(status, kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  ASSERT_EQ(output_model->subgraphs()->size(), model_->subgraphs()->size());
  uint32_t num_conv_ops = 0;
  for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
       ++subgraph_idx) {
    const auto quantized_graph = output_model->subgraphs()->Get(subgraph_idx);
    for (size_t i = 0; i < quantized_graph->operators()->size(); ++i) {
      const auto op = quantized_graph->operators()->Get(i);
      const uint32_t op_code_idx = op->opcode_index();
      const auto op_code =
          output_model->operator_codes()->Get(op_code_idx)->builtin_code();
      if (op_code == BuiltinOperator_CONV_2D) {
        num_conv_ops++;
        // Ensure that each convolution's weights tensor is still FLOAT
        // (the output of the dequantize).
        uint32_t weights_tensor_index = op->inputs()->Get(1);
        const auto weights_tensor =
            quantized_graph->tensors()->Get(weights_tensor_index);
        EXPECT_EQ(weights_tensor->type(), TensorType_FLOAT32);

        // Check that it comes from a dequantize operation.
        const auto producer_op_code =
            GetProducerOpCode(output_model, subgraph_idx, weights_tensor_index);
        EXPECT_EQ(producer_op_code, BuiltinOperator_DEQUANTIZE);
      }
    }
  }
  // Ensure that there were exactly two convolutions in the model.
  EXPECT_EQ(num_conv_ops, 2);
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
  CHECK(parse_result) << "Required test_model_file";
  g_test_model_dir =
      new tensorflow::string(tensorflow::io::Dirname(model_file));
  ::tensorflow::port::InitMain(argv[0], &argc, &argv);
  return RUN_ALL_TESTS();
}
