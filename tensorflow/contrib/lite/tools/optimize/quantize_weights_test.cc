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

  void CheckWeights(const Model* model_packed) {
    std::unique_ptr<ModelT> model;
    model.reset(model_packed->UnPack());

    SubGraphT* subgraph = model->subgraphs.at(0).get();

    for (int i = 0; i < subgraph->operators.size(); ++i) {
      OperatorT* op = subgraph->operators[i].get();
      const BuiltinOperator op_code =
          model->operator_codes[op->opcode_index]->builtin_code;

      // These are the operations that should be quantized.
      int32_t tensor_idx;
      if (op_code == BuiltinOperator_CONV_2D ||
          op_code == BuiltinOperator_DEPTHWISE_CONV_2D ||
          op_code == BuiltinOperator_FULLY_CONNECTED) {
        tensor_idx = op->inputs[1];
      } else if (op_code == BuiltinOperator_LSTM) {
        // TODO(suharshs): Add tests for LSTMs.
        tensor_idx = op->inputs[1];
      } else {
        continue;
      }
      const TensorT* tensor = subgraph->tensors[tensor_idx].get();
      int tensor_size = GetElementsNum(tensor);
      // If the tensor_size is less than 1024 we expect the tensor to remain
      // unquantized.
      if (tensor_size < 1024) {
        ASSERT_TRUE(tensor->type == TensorType_FLOAT32) << tensor->name;
        const OperatorT* preceding_op = GetOpWithOutput(subgraph, tensor_idx);
        // The weight tensor should not come from a dequantize op.
        ASSERT_TRUE(preceding_op == nullptr);
      } else {
        // The input to the op should still be float.
        ASSERT_TRUE(tensor->type == TensorType_FLOAT32) << tensor->name;
        const OperatorT* preceding_op = GetOpWithOutput(subgraph, tensor_idx);
        ASSERT_TRUE(preceding_op != nullptr);
        // The float input should be the dequantize output.
        ASSERT_TRUE(
            model->operator_codes[preceding_op->opcode_index]->builtin_code ==
            BuiltinOperator_DEQUANTIZE);
        // Finally, ensure that the input to the dequantize operation is
        // quantized.
        ASSERT_TRUE(subgraph->tensors[preceding_op->inputs[0]]->type ==
                    TensorType_UINT8);
        // TODO(suharshs): Add more rigorous testing for the numerical values in
        // the tensors.
      }
    }
  }
};

TEST_F(QuantizeWeightsTest, SimpleTest) {
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

  CheckWeights(output_model);
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
