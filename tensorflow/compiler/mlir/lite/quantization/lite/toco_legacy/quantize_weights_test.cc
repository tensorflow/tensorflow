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
#include "tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/quantize_weights.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/core/absl_error_model_builder.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/test_util.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_utils.h"
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
using tflite::BuiltinOperator_CUSTOM;
using tflite::BuiltinOperator_DEQUANTIZE;
using tflite::GetModel;
using tflite::Model;
using tflite::TensorType_FLOAT16;
using tflite::TensorType_FLOAT32;
using tflite::TensorType_INT8;

std::unique_ptr<FlatBufferModelAbslError> ReadTestModel() {
  auto model_path = tsl::io::JoinPath(
      *g_test_model_dir, ::mlir::lite::internal::kConvModelWith0Plus10Weights);
  return FlatBufferModelAbslError::BuildFromFile(model_path.c_str());
}

std::unique_ptr<FlatBufferModelAbslError> ReadSharedWeightsTestModel() {
  auto model_path = tsl::io::JoinPath(
      *g_test_model_dir, ::mlir::lite::internal::kModelWithSharedWeights);
  return FlatBufferModelAbslError::BuildFromFile(model_path.c_str());
}

std::unique_ptr<FlatBufferModelAbslError> ReadGatherTestModel() {
  auto model_path = tsl::io::JoinPath(
      *g_test_model_dir, ::mlir::lite::internal::kQuantizedWithGather);
  return FlatBufferModelAbslError::BuildFromFile(model_path.c_str());
}

std::unique_ptr<FlatBufferModelAbslError> ReadCustomOpTestModel() {
  auto model_path = tsl::io::JoinPath(
      *g_test_model_dir, ::mlir::lite::internal::kModelWithCustomOp);
  return FlatBufferModelAbslError::BuildFromFile(model_path.c_str());
}

template <typename T>
std::vector<T> GetAsVector(const flatbuffers::Vector<T>* vec) {
  return std::vector<T>(vec->begin(), vec->end());
}

class QuantizeWeightsTest : public testing::Test {
 protected:
  QuantizeWeightsTest() = default;

  void LoadBasicModel() {
    input_model_ = ReadTestModel();
    model_ = input_model_->GetModel();
  }

  void LoadSharedWeightsModel() {
    input_model_ = ReadSharedWeightsTestModel();
    model_ = input_model_->GetModel();
  }

  void LoadGatherTestModel() {
    input_model_ = ReadGatherTestModel();
    model_ = input_model_->GetModel();
  }

  void LoadCustomOpTestModel() {
    input_model_ = ReadCustomOpTestModel();
    model_ = input_model_->GetModel();
  }

  std::unique_ptr<FlatBufferModelAbslError> input_model_;
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
  bool GetProducerOpCode(const Model* model, uint32_t subgraph_idx,
                         uint32_t tensor_idx,
                         tflite::BuiltinOperator* op_code) {
    const auto subgraph = model->subgraphs()->Get(subgraph_idx);
    for (size_t op_idx = 0; op_idx < subgraph->operators()->size(); ++op_idx) {
      const auto op = subgraph->operators()->Get(op_idx);
      for (size_t i = 0; i < op->outputs()->size(); ++i) {
        if (op->outputs()->Get(i) == tensor_idx) {
          const uint32_t op_code_idx = op->opcode_index();
          *op_code = GetBuiltinCode(model->operator_codes()->Get(op_code_idx));
          return true;
        }
      }
    }
    return false;
  }
};

TEST_F(QuantizeWeightsTest, QuantizationSucceeds) {
  LoadBasicModel();
  flatbuffers::FlatBufferBuilder builder;
  ASSERT_TRUE(
      QuantizeWeights(&builder, model_, 0, QuantizerType::OLD_QUANTIZER).ok());

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
  ASSERT_TRUE(QuantizeWeights(&builder, model_, kWeightsMinNumElements,
                              QuantizerType::OLD_QUANTIZER)
                  .ok());

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
  ASSERT_TRUE(
      QuantizeWeights(&builder, model_, 0, QuantizerType::OLD_QUANTIZER).ok());

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
    ASSERT_EQ(GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx)),
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
        auto shape = GetAsVector(quant_tensor->shape());
        if (kUseUpdatedHybridSchemeDefault) {
          EXPECT_EQ(quant_tensor->quantization()->scale()->size(), shape[0]);
        } else {
          EXPECT_EQ(quant_tensor->quantization()->scale()->size(), 1);
        }
      } else {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      }
    }
  }
}

TEST_F(QuantizeWeightsTest, DequantizeConv) {
  LoadBasicModel();
  flatbuffers::FlatBufferBuilder builder;
  ASSERT_TRUE(internal::QuantizeWeights(&builder, model_, 0,
                                        /*use_hybrid_evaluation=*/false,
                                        QuantizerType::OLD_QUANTIZER)
                  .ok());

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
      if (GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx)) ==
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
        // If it's a non-bias constant tensor, it must be the weight.
        EXPECT_EQ(quant_tensor->type(), TensorType_INT8);
      } else {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      }
    }
  }
}

TEST_F(QuantizeWeightsTest, DequantizeConvFloat16) {
  LoadBasicModel();
  flatbuffers::FlatBufferBuilder builder;
  ASSERT_TRUE(QuantizeWeights(&builder, model_, BufferType::QUANTIZED_FLOAT16,
                              kUseUpdatedHybridSchemeDefault,
                              QuantizerType::OLD_QUANTIZER)
                  .ok());

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  ASSERT_EQ(output_model->subgraphs()->size(), model_->subgraphs()->size());
  for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
       ++subgraph_idx) {
    const auto quantized_graph = output_model->subgraphs()->Get(subgraph_idx);
    const auto float_graph = model_->subgraphs()->Get(subgraph_idx);
    // The output graph should have two extra tensors from the added dequantize
    // op.
    ASSERT_EQ(quantized_graph->tensors()->size(),
              float_graph->tensors()->size() + 2);
    // Check that a dequantize op exists.
    int32_t dequant_input_idx = -1;
    int32_t dequant_output_idx = -1;
    for (size_t i = 0; i < quantized_graph->operators()->size(); ++i) {
      const auto op = quantized_graph->operators()->Get(i);
      const uint32_t op_code_idx = op->opcode_index();
      if (GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx)) ==
          BuiltinOperator_DEQUANTIZE) {
        dequant_input_idx = op->inputs()->Get(0);
        dequant_output_idx = op->outputs()->Get(0);
      }
    }
    ASSERT_GT(dequant_input_idx, -1);
    ASSERT_GT(dequant_output_idx, -1);
    for (size_t i = 0; i < quantized_graph->tensors()->size(); ++i) {
      const auto quant_tensor = quantized_graph->tensors()->Get(i);
      // If the tensor is a weight, it should have type FLOAT16.
      // If the tensor is a bias, it should have type FLOAT16.
      // If the tensor is an input or output it should have type FLOAT32.
      // The input to dequantize should be FLOAT16, and all other tensors should
      // be FLOAT32.
      if (i == dequant_input_idx) {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT16);
      } else if (i == dequant_output_idx) {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (IsModelInputOrOutput(output_model, i)) {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (quant_tensor->name()->str() == "conv_bias") {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT16);
      } else if (quant_tensor->buffer() != 0) {
        // If it's a non-bias constant tensor, it must be the weight.
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT16);
      } else {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      }
    }
  }
}

TEST_F(QuantizeWeightsTest, SharedWeights_Hybrid) {
  LoadSharedWeightsModel();
  flatbuffers::FlatBufferBuilder builder;
  ASSERT_TRUE(
      QuantizeWeights(&builder, model_, 0, QuantizerType::OLD_QUANTIZER).ok());

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
          GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx));
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
  ASSERT_TRUE(internal::QuantizeWeights(&builder, model_, 0,
                                        /*use_hybrid_evaluation*/ false,
                                        QuantizerType::OLD_QUANTIZER)
                  .ok());

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
          GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx));
      if (op_code == BuiltinOperator_CONV_2D) {
        num_conv_ops++;
        // Ensure that each convolution's weights tensor is still FLOAT
        // (the output of the dequantize).
        uint32_t weights_tensor_index = op->inputs()->Get(1);
        const auto weights_tensor =
            quantized_graph->tensors()->Get(weights_tensor_index);
        EXPECT_EQ(weights_tensor->type(), TensorType_FLOAT32);

        // Check that it comes from a dequantize operation.
        BuiltinOperator producer_op_code;
        ASSERT_TRUE(GetProducerOpCode(output_model, subgraph_idx,
                                      weights_tensor_index, &producer_op_code));
        EXPECT_EQ(producer_op_code, BuiltinOperator_DEQUANTIZE);
      }
    }
  }
  // Ensure that there were exactly two convolutions in the model.
  EXPECT_EQ(num_conv_ops, 2);
}

TEST_F(QuantizeWeightsTest, VerifyGatherQuantization) {
  LoadGatherTestModel();
  flatbuffers::FlatBufferBuilder builder;
  ASSERT_TRUE(
      QuantizeWeights(&builder, model_, 0, QuantizerType::OLD_QUANTIZER).ok());

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  ASSERT_EQ(output_model->subgraphs()->size(), model_->subgraphs()->size());
  for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
       ++subgraph_idx) {
    const auto quantized_graph = output_model->subgraphs()->Get(subgraph_idx);
    for (size_t i = 0; i < quantized_graph->operators()->size(); ++i) {
      const auto op = quantized_graph->operators()->Get(i);
      const uint32_t op_code_idx = op->opcode_index();
      const auto op_code =
          GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx));
      if (op_code == tflite::BuiltinOperator_GATHER) {
        uint32_t input_tensor_index = op->inputs()->Get(0);
        const auto weights_tensor =
            quantized_graph->tensors()->Get(input_tensor_index);
        EXPECT_EQ(weights_tensor->type(), TensorType_INT8);
      }
    }
  }
}

TEST_F(QuantizeWeightsTest, VerifyCustomOpQuantizationDequantize) {
  LoadCustomOpTestModel();

  // The custom op is not hybrid, and the second input is a constant that can
  // be quantized.
  CustomOpMap custom_op_map;
  custom_op_map["CustomTestOp"] = {
      .quantizable_input_indices = {1},
      .is_hybrid = false,
  };

  flatbuffers::FlatBufferBuilder builder;
  ASSERT_TRUE(QuantizeWeights(&builder, model_, 0, custom_op_map,
                              QuantizerType::OLD_QUANTIZER)
                  .ok());

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  ASSERT_EQ(output_model->subgraphs()->size(), model_->subgraphs()->size());
  const auto quantized_graph = output_model->subgraphs()->Get(0);
  // A dequantize op should be added.
  ASSERT_EQ(quantized_graph->operators()->size(),
            model_->subgraphs()->Get(0)->operators()->size() + 1);
  int num_custom_ops_found = 0;
  for (size_t i = 0; i < quantized_graph->operators()->size(); ++i) {
    const auto op = quantized_graph->operators()->Get(i);
    const uint32_t op_code_idx = op->opcode_index();
    const auto op_code =
        GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx));
    if (op_code == BuiltinOperator_CUSTOM) {
      uint32_t weights_tensor_index = op->inputs()->Get(1);
      const auto weights_tensor =
          quantized_graph->tensors()->Get(weights_tensor_index);
      EXPECT_EQ(weights_tensor->type(), TensorType_FLOAT32);

      // Check that it comes from a dequantize operation.
      BuiltinOperator producer_op_code;
      ASSERT_TRUE(GetProducerOpCode(output_model, 0, weights_tensor_index,
                                    &producer_op_code));
      EXPECT_EQ(producer_op_code, BuiltinOperator_DEQUANTIZE);
      num_custom_ops_found++;
    }
  }
  EXPECT_EQ(num_custom_ops_found, 1);
}

TEST_F(QuantizeWeightsTest, VerifyCustomOpQuantizationHybrid) {
  LoadCustomOpTestModel();

  // The custom op is hybrid, and the second input is a constant that can
  // be quantized.
  CustomOpMap custom_op_map;
  custom_op_map["CustomTestOp"] = {
      .quantizable_input_indices = {1},
      .is_hybrid = true,
  };

  flatbuffers::FlatBufferBuilder builder;
  ASSERT_TRUE(QuantizeWeights(&builder, model_, 0, custom_op_map,
                              QuantizerType::OLD_QUANTIZER)
                  .ok());

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  ASSERT_EQ(output_model->subgraphs()->size(), model_->subgraphs()->size());
  const auto quantized_graph = output_model->subgraphs()->Get(0);
  ASSERT_EQ(quantized_graph->operators()->size(),
            model_->subgraphs()->Get(0)->operators()->size());
  int num_custom_ops_found = 0;
  for (size_t i = 0; i < quantized_graph->operators()->size(); ++i) {
    const auto op = quantized_graph->operators()->Get(i);
    const uint32_t op_code_idx = op->opcode_index();
    const auto op_code =
        GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx));
    if (op_code == BuiltinOperator_CUSTOM) {
      uint32_t weights_tensor_index = op->inputs()->Get(1);
      const auto weights_tensor =
          quantized_graph->tensors()->Get(weights_tensor_index);
      EXPECT_EQ(weights_tensor->type(), TensorType_INT8);
      num_custom_ops_found++;
    }
  }
  EXPECT_EQ(num_custom_ops_found, 1);
}

TEST_F(QuantizeWeightsTest, VerifyUpdatedHybridSchemeFalseQuantizationHybrid) {
  LoadBasicModel();
  flatbuffers::FlatBufferBuilder builder;
  const CustomOpMap custom_op_map;
  ASSERT_TRUE(QuantizeWeights(&builder, model_, 0, custom_op_map,
                              /*use_updated_hybrid_scheme=*/false,
                              /*op_denylist=*/{}, QuantizerType::OLD_QUANTIZER)
                  .ok());

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
    ASSERT_EQ(GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx)),
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
        auto shape = GetAsVector(quant_tensor->shape());
        EXPECT_EQ(quant_tensor->quantization()->scale()->size(), 1);
      } else {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      }
    }
  }
}

TEST_F(QuantizeWeightsTest, DequantizeConvBlocklisted) {
  LoadBasicModel();
  flatbuffers::FlatBufferBuilder builder;
  const CustomOpMap custom_op_map;
  ASSERT_TRUE(QuantizeWeights(&builder, model_, 0, custom_op_map,
                              /*use_updated_hybrid_scheme=*/true,
                              /*op_denylist*/ {BuiltinOperator_CONV_2D},
                              QuantizerType::OLD_QUANTIZER)
                  .ok());

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
      if (GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx)) ==
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
        // The dequantize should still be quantized per-channel
        EXPECT_EQ(quant_tensor->quantization()->scale()->size(), 5);
        EXPECT_EQ(quant_tensor->quantization()->quantized_dimension(), 0);
      } else if (i == dequant_output_idx) {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (IsModelInputOrOutput(output_model, i)) {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (quant_tensor->name()->str() == "conv_bias") {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (quant_tensor->buffer() != 0) {
        // If it's a non-bias constant tensor, it must be the weight.
        EXPECT_EQ(quant_tensor->type(), TensorType_INT8);
      } else {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      }
    }
  }
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
