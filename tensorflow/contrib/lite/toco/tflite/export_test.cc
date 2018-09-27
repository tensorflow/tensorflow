/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/contrib/lite/toco/tflite/export.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/schema/schema_generated.h"
#include "tensorflow/contrib/lite/toco/tflite/builtin_operator.h"
#include "tensorflow/contrib/lite/toco/tflite/operator.h"
#include "tensorflow/contrib/lite/toco/tflite/types.h"

namespace toco {
namespace tflite {
namespace {

using ::testing::ElementsAre;

class ExportTest : public ::testing::Test {
 protected:
  // This is a very simplistic model. We are not interested in testing all the
  // details here, since tf.mini's testing framework will be exercising all the
  // conversions multiple times, and the conversion of operators is tested by
  // separate unittests.
  void BuildTestModel() {
    input_model_.GetOrCreateArray("tensor_one");
    input_model_.GetOrCreateArray("tensor_two");
    {
      auto* op = new ConvOperator;
      op->padding.type = PaddingType::kSame;
      input_model_.operators.emplace_back(op);
    }
    input_model_.operators.emplace_back(new AddOperator);
    {
      auto* op = new TensorFlowUnsupportedOperator;
      op->tensorflow_op = "MyCrazyOp";
      input_model_.operators.emplace_back(op);
    }
    // Note that Sub is not know to TF Lite, so it gets exported as a custom
    // op (and no options).
    input_model_.operators.emplace_back(new SubOperator);
  }

  void BuildQuantizableTestModel() {
    input_model_.GetOrCreateArray("inputs");
    Array& weight_array = input_model_.GetOrCreateArray("weights");

    // Make the buffer large enough for QuantizeWeights transformation to take
    // effect.
    int buf_size = 1296;
    auto weight_buf = absl::make_unique<float[]>(buf_size);
    for (int i = 0; i < buf_size; i++) {
      // Fill the array with some garbage values.
      weight_buf[i] = static_cast<float>(i % 128);
    }

    weight_array.data_type = ArrayDataType::kFloat;

    // Initialize shape for the input array.
    Shape* weight_array_shape = weight_array.mutable_shape();
    std::vector<int>* weight_array_shape_dim =
        weight_array_shape->mutable_dims();
    weight_array_shape_dim->resize(4, 6);
    auto& weight_array_buffer =
        weight_array.GetMutableBuffer<ArrayDataType::kFloat>();
    weight_array_buffer.data.resize(buf_size);
    float* buf_ptr =
        weight_array.GetMutableBuffer<ArrayDataType::kFloat>().data.data();
    std::copy(weight_buf.get(), weight_buf.get() + buf_size, buf_ptr);

    {
      auto* op = new ConvOperator;
      op->padding.type = PaddingType::kSame;
      op->inputs = {"inputs", "weights"};
      input_model_.operators.emplace_back(op);
    }
    input_model_.operators.emplace_back(new AddOperator);
  }

  Model input_model_;
};

TEST_F(ExportTest, LoadTensorsMap) {
  BuildTestModel();

  details::TensorsMap tensors;
  details::LoadTensorsMap(input_model_, &tensors);
  EXPECT_EQ(0, tensors["tensor_one"]);
  EXPECT_EQ(1, tensors["tensor_two"]);
}

TEST_F(ExportTest, LoadOperatorsMap) {
  BuildTestModel();

  details::OperatorsMap operators;
  const auto ops_by_type = BuildOperatorByTypeMap();
  // TODO(ycling): Add a test for allow_eager_ops.
  details::LoadOperatorsMap(input_model_, &operators, ops_by_type, false);
  EXPECT_EQ(0, operators[details::OperatorKey(OperatorType::kAdd, "", 1)]);
  EXPECT_EQ(1, operators[details::OperatorKey(OperatorType::kConv, "", 1)]);
  EXPECT_EQ(2, operators[details::OperatorKey(OperatorType::kSub, "", 1)]);
  EXPECT_EQ(3, operators[details::OperatorKey(OperatorType::kUnsupported,
                                              "MyCrazyOp", 1)]);
}

TEST_F(ExportTest, Export) {
  BuildTestModel();

  string result;
  Export(input_model_, true, false, &result);

  auto* model = ::tflite::GetModel(result.data());

  std::vector<string> names;
  for (const ::tflite::OperatorCode* opcode : *model->operator_codes()) {
    if (opcode->builtin_code() != ::tflite::BuiltinOperator_CUSTOM) {
      names.push_back(string("builtin:") + ::tflite::EnumNameBuiltinOperator(
                                               opcode->builtin_code()));
    } else {
      names.push_back(string("custom:") + opcode->custom_code()->c_str());
    }
  }

  EXPECT_THAT(names, ElementsAre("builtin:ADD", "builtin:CONV_2D",
                                 "builtin:SUB", "custom:MyCrazyOp"));

  std::vector<uint32_t> indices;
  auto operators = (*model->subgraphs())[0]->operators();
  EXPECT_EQ(operators->Length(), 4);
  for (const auto* op : *operators) {
    indices.push_back(op->opcode_index());
  }

  EXPECT_THAT(indices, ElementsAre(1, 0, 3, 2));
}

TEST_F(ExportTest, QuantizeWeights) {
  // Sanity check for quantize_weights parameter.
  BuildQuantizableTestModel();
  string unquantized_result;
  Export(input_model_, true, /*quantize_weights*/ false, &unquantized_result);

  BuildQuantizableTestModel();
  string quantized_result;
  Export(input_model_, true, /*quantize_weights*/ true, &quantized_result);

  // The quantized models should be smaller.
  EXPECT_LT(quantized_result.size(), unquantized_result.size());
}

// This test is based on a hypothetical scenario that dilation is supported
// only in Conv version 2. So Toco populates version=1 when dialation
// parameters are all 1, and version=2 otehrwise.
class FakeConvolutionOperator
    : public BuiltinOperator<ConvOperator, ::tflite::Conv2DOptions,
                             ::tflite::BuiltinOptions_Conv2DOptions> {
 public:
  FakeConvolutionOperator()
      : BuiltinOperator(::tflite::BuiltinOperator_CONV_2D,
                        OperatorType::kConv) {}

  // Returning the op version according to the op parameters.
  int GetVersion(const Operator& op) const override {
    const TocoOperator& conv_op = static_cast<const TocoOperator&>(op);
    if (conv_op.dilation_width_factor != 1 ||
        conv_op.dilation_height_factor != 1) {
      // Version 2 if dilation is used.
      return 2;
    }
    return 1;
  }

  // Note: The read / write code doesn't need to be changed if we stick with
  // the restrictions:
  // * Only adding parameters at the bottom of the Flatbuffer tables.
  // * When the default value of parameters are used, the op works consistently
  //   with the previous version.
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    auto padding = Padding::Serialize(op.padding.type);
    auto activation_function =
        ActivationFunction::Serialize(op.fused_activation_function);
    return ::tflite::CreateConv2DOptions(*builder, padding, op.stride_width,
                                         op.stride_height, activation_function,
                                         op.dilation_width_factor,
                                         op.dilation_height_factor);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->padding.type = Padding::Deserialize(options.padding());
    op->stride_width = options.stride_w();
    op->stride_height = options.stride_h();
    op->dilation_width_factor = options.dilation_w_factor();
    op->dilation_height_factor = options.dilation_h_factor();
    op->fused_activation_function =
        ActivationFunction::Deserialize(options.fused_activation_function());
  }
};

class VersionedOpExportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    input_model_.GetOrCreateArray("input");
    input_model_.GetOrCreateArray("filter");
    input_model_.GetOrCreateArray("output");
  }
  void AddConvOp(bool use_dialation) {
    {
      auto* op = new ConvOperator;
      op->inputs.push_back("input");
      op->inputs.push_back("filter");
      op->inputs.push_back("output");

      op->padding.type = PaddingType::kSame;
      op->stride_width = 1;
      op->stride_height = 1;
      if (use_dialation) {
        op->dilation_width_factor = 2;
        op->dilation_height_factor = 2;
      } else {
        op->dilation_width_factor = 1;
        op->dilation_height_factor = 1;
      }
      input_model_.operators.emplace_back(op);
    }
  }

  std::map<OperatorType, std::unique_ptr<BaseOperator>>
  BuildFakeOperatorByTypeMap() {
    std::map<OperatorType, std::unique_ptr<BaseOperator>> result;
    result[OperatorType::kConv] =
        std::unique_ptr<BaseOperator>(new FakeConvolutionOperator);
    return result;
  }

  Model input_model_;
};

TEST_F(VersionedOpExportTest, LoadOperatorsMapWithOpV1) {
  AddConvOp(false);

  details::OperatorsMap operators;
  const auto ops_by_type = BuildFakeOperatorByTypeMap();
  details::LoadOperatorsMap(input_model_, &operators, ops_by_type, false);

  EXPECT_EQ(1, operators.size());
  EXPECT_EQ(0, operators.at(details::OperatorKey(OperatorType::kConv, "", 1)));
}

TEST_F(VersionedOpExportTest, LoadOperatorsMapWithOpV2) {
  AddConvOp(true);

  details::OperatorsMap operators;
  const auto ops_by_type = BuildFakeOperatorByTypeMap();
  details::LoadOperatorsMap(input_model_, &operators, ops_by_type, false);

  EXPECT_EQ(1, operators.size());
  EXPECT_EQ(0, operators.at(details::OperatorKey(OperatorType::kConv, "", 2)));
}

TEST_F(VersionedOpExportTest, LoadOperatorsMapWithBothVersions) {
  AddConvOp(false);
  AddConvOp(true);

  details::OperatorsMap operators;
  const auto ops_by_type = BuildFakeOperatorByTypeMap();
  details::LoadOperatorsMap(input_model_, &operators, ops_by_type, false);

  EXPECT_EQ(2, operators.size());
  EXPECT_EQ(0, operators.at(details::OperatorKey(OperatorType::kConv, "", 1)));
  EXPECT_EQ(1, operators.at(details::OperatorKey(OperatorType::kConv, "", 2)));
}

TEST_F(VersionedOpExportTest, Export) {
  AddConvOp(false);
  AddConvOp(true);

  string result;
  const auto ops_by_type = BuildFakeOperatorByTypeMap();
  Export(input_model_, true, false, &result, ops_by_type);

  auto* model = ::tflite::GetModel(result.data());
  auto operator_codes = model->operator_codes();

  // Verify that 2 operator codes are populdated. Both are CONV_2D but with
  // different versions.
  EXPECT_EQ(2, operator_codes->size());
  EXPECT_EQ(::tflite::BuiltinOperator_CONV_2D,
            (*operator_codes)[0]->builtin_code());
  EXPECT_EQ(1, (*operator_codes)[0]->version());
  EXPECT_EQ(::tflite::BuiltinOperator_CONV_2D,
            (*operator_codes)[1]->builtin_code());
  EXPECT_EQ(2, (*operator_codes)[1]->version());

  // Verify that the 2 operators points to the correct indices of the operation
  // codes.
  auto operators = (*model->subgraphs())[0]->operators();
  EXPECT_EQ(2, operators->size());
  EXPECT_EQ(0, (*operators)[0]->opcode_index());
  EXPECT_EQ(1, (*operators)[1]->opcode_index());
}

// TODO(ahentz): tests for tensors, inputs, outputs, opcodes and operators.

}  // namespace
}  // namespace tflite
}  // namespace toco
