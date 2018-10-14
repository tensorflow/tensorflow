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
#include "tensorflow/contrib/lite/tools/optimize/quantize_weights.h"

#include <memory>

#include "flatbuffers/flexbuffers.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {
namespace {

class QuantizeWeightsTest : public ::testing::Test {
 protected:
  int GetElementsNum(const TensorT* tensor) {
    int tensor_size = 1;
    for (const int dim : tensor->shape) {
      tensor_size *= dim;
    }
    return tensor_size;
  }

  const OperatorT* GetOpWithOutput(const SubGraphT* subgraph,
                                   int32_t output_tensor_idx) {
    for (int i = 0; i < subgraph->operators.size(); ++i) {
      OperatorT* op = subgraph->operators[i].get();
      if (std::find(op->outputs.begin(), op->outputs.end(),
                    output_tensor_idx) != op->outputs.end()) {
        return op;
      }
    }
    return nullptr;
  }

  void SymmetricDequantizeAndCompare(const BufferT* input_buffer,
                                     const BufferT* output_buffer,
                                     float scale) {
    const float* input_buffer_data =
        reinterpret_cast<const float*>(input_buffer->data.data());
    const int8_t* output_buffer_data =
        reinterpret_cast<const int8_t*>(output_buffer->data.data());
    for (int i = 0; i < output_buffer->data.size(); i++) {
      float diff = input_buffer_data[i] - (output_buffer_data[i] * scale);
      ASSERT_TRUE(std::abs(diff) <= scale);
    }
  }

  void AsymmetricDequantizeAndCompare(const BufferT* input_buffer,
                                      const BufferT* output_buffer, float scale,
                                      int64_t zero_point) {
    const float* input_buffer_data =
        reinterpret_cast<const float*>(input_buffer->data.data());
    const uint8_t* output_buffer_data = output_buffer->data.data();
    for (int i = 0; i < output_buffer->data.size(); i++) {
      float diff =
          input_buffer_data[i] - ((output_buffer_data[i] - zero_point) * scale);
      ASSERT_TRUE(std::abs(diff) <= scale);
    }
  }

  void CheckWeights(const Model* input_model_packed,
                    const Model* output_model_packed,
                    bool use_hybrid_evaluation,
                    uint64_t weights_min_num_elements = 1024) {
    std::unique_ptr<ModelT> input_model;
    input_model.reset(input_model_packed->UnPack());

    std::unique_ptr<ModelT> output_model;
    output_model.reset(output_model_packed->UnPack());

    SubGraphT* subgraph = output_model->subgraphs.at(0).get();

    for (int i = 0; i < subgraph->operators.size(); ++i) {
      OperatorT* op = subgraph->operators[i].get();
      const BuiltinOperator op_code =
          output_model->operator_codes[op->opcode_index]->builtin_code;

      // These are the operations that should be quantized.
      // TODO(suharshs): Right now this test only checks the relevant operations
      // for the mobilenet v1 model used in the tests below.
      int32_t tensor_idx;
      if (op_code == BuiltinOperator_CONV_2D ||
          op_code == BuiltinOperator_DEPTHWISE_CONV_2D ||
          op_code == BuiltinOperator_FULLY_CONNECTED) {
        tensor_idx = op->inputs[1];
      } else {
        continue;
      }

      bool eval_hybrid = false;
      // These are the ops that support hybrid evaluation.
      if (op_code == BuiltinOperator_FULLY_CONNECTED ||
          op_code == BuiltinOperator_CONV_2D) {
        eval_hybrid = true;
      }

      const TensorT* tensor = subgraph->tensors[tensor_idx].get();
      int tensor_size = GetElementsNum(tensor);
      // If the tensor_size is less than 1024 we expect the tensor to remain
      // unquantized.
      if (tensor_size < weights_min_num_elements) {
        ASSERT_TRUE(tensor->type == TensorType_FLOAT32)
            << tensor->name << " of type " << tensor->type;
        const OperatorT* preceding_op = GetOpWithOutput(subgraph, tensor_idx);
        // The weight tensor should not come from a dequantize op.
        ASSERT_TRUE(preceding_op == nullptr);
      } else if (use_hybrid_evaluation && eval_hybrid) {
        // The input to the op should still be uint8.
        ASSERT_TRUE(tensor->type == TensorType_UINT8) << tensor->name;
        // The weight tensor should not come from a dequantize op.
        const OperatorT* preceding_op = GetOpWithOutput(subgraph, tensor_idx);
        ASSERT_TRUE(preceding_op == nullptr);

        // Test symmetric quantization.
        SymmetricDequantizeAndCompare(
            input_model->buffers[tensor->buffer].get(),
            output_model->buffers[tensor->buffer].get(),
            tensor->quantization->scale[0]);

      } else {
        // The input to the op should still be float.
        ASSERT_TRUE(tensor->type == TensorType_FLOAT32) << tensor->name;
        const OperatorT* preceding_op = GetOpWithOutput(subgraph, tensor_idx);
        ASSERT_TRUE(preceding_op != nullptr);
        // The float input should be the dequantize output.
        ASSERT_TRUE(output_model->operator_codes[preceding_op->opcode_index]
                        ->builtin_code == BuiltinOperator_DEQUANTIZE);
        // Finally, ensure that the input to the dequantize operation is
        // quantized.
        const TensorT* quantized_tensor =
            subgraph->tensors[preceding_op->inputs[0]].get();
        ASSERT_TRUE(quantized_tensor->type == TensorType_UINT8);

        // Test the assymetric quantization.
        AsymmetricDequantizeAndCompare(
            input_model->buffers[quantized_tensor->buffer].get(),
            output_model->buffers[quantized_tensor->buffer].get(),
            quantized_tensor->quantization->scale[0],
            quantized_tensor->quantization->zero_point[0]);
      }
    }
  }
};

TEST_F(QuantizeWeightsTest, SimpleTestWithHybrid) {
  string model_path =
      "third_party/tensorflow/contrib/lite/tools/optimize/testdata/"
      "mobilenet_v1_0.25_128.tflite";
  std::unique_ptr<FlatBufferModel> input_fb =
      FlatBufferModel::BuildFromFile(model_path.data());
  const Model* input_model = input_fb->GetModel();

  flatbuffers::FlatBufferBuilder builder;
  EXPECT_EQ(QuantizeWeights(&builder, input_model), kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);

  CheckWeights(input_model, output_model, true);
}

TEST_F(QuantizeWeightsTest, SimpleTestWithoutHybrid) {
  string model_path =
      "third_party/tensorflow/contrib/lite/tools/optimize/testdata/"
      "mobilenet_v1_0.25_128.tflite";
  std::unique_ptr<FlatBufferModel> input_fb =
      FlatBufferModel::BuildFromFile(model_path.data());
  const Model* input_model = input_fb->GetModel();

  flatbuffers::FlatBufferBuilder builder;
  // Disable hybrid evaluation.
  EXPECT_EQ(internal::QuantizeWeights(&builder, input_model, false), kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);

  CheckWeights(input_model, output_model, false);
}

TEST_F(QuantizeWeightsTest, SimpleTestWithWeightsMinNumElements) {
  string model_path =
      "third_party/tensorflow/contrib/lite/tools/optimize/testdata/"
      "mobilenet_v1_0.25_128.tflite";
  std::unique_ptr<FlatBufferModel> input_fb =
      FlatBufferModel::BuildFromFile(model_path.data());
  const Model* input_model = input_fb->GetModel();

  flatbuffers::FlatBufferBuilder builder;
  // Make weights_min_size sufficiently large such that no quantization should
  // happen, i.e. the original model is the same size as the old one.
  const uint64_t kWeightsMinNumElements = 1000000;
  EXPECT_EQ(QuantizeWeights(&builder, input_model, kWeightsMinNumElements),
            kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  CheckWeights(input_model, output_model, true, kWeightsMinNumElements);
}

// TODO(suharshs): Add tests that run the resulting model.

}  // namespace
}  // namespace optimize
}  // namespace tflite

int main(int argc, char** argv) {
  // On Linux, add: FLAGS_logtostderr = true;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
