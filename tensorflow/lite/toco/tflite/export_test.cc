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
#include "tensorflow/lite/toco/tflite/export.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/toco/tflite/builtin_operator.h"
#include "tensorflow/lite/toco/tflite/operator.h"
#include "tensorflow/lite/toco/tflite/types.h"
#include "tensorflow/core/framework/node_def.pb.h"

namespace toco {
namespace tflite {
namespace {

using ::testing::ElementsAre;

class ExportTest : public ::testing::Test {
 protected:
  void ResetOperators() { input_model_.operators.clear(); }
  void AddTensorsByName(std::initializer_list<string> names) {
    for (const string& name : names) {
      input_model_.GetOrCreateArray(name);
    }
  }
  void AddOperatorsByName(std::initializer_list<string> names) {
    for (const string& name : names) {
      if (name == "Conv") {
        auto* op = new ConvOperator;
        op->padding.type = PaddingType::kSame;
        input_model_.operators.emplace_back(op);
      } else if (name == "Add") {
        input_model_.operators.emplace_back(new AddOperator);
      } else if (name == "Sub") {
        input_model_.operators.emplace_back(new SubOperator);
      } else if (name == "Assert") {
        auto* op = new TensorFlowAssertOperator;

        // Even though assert is known to TOCO, it doesn't have a tflite
        // serializer, so it has to be exported as a custom op. If we attach a
        // NodeDef to it, however, it will be exported as a flex op instead.
        ::tensorflow::NodeDef node_def;
        node_def.set_name("Assert");
        node_def.set_op("Assert");
        node_def.SerializeToString(&op->tensorflow_node_def);

        input_model_.operators.emplace_back(op);
      } else {
        auto* op = new TensorFlowUnsupportedOperator;
        op->tensorflow_op = name;
        input_model_.operators.emplace_back(op);
      }
    }
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

  std::vector<string> ExportAndSummarizeOperators(const ExportParams& params) {
    std::vector<string> names;

    string result;
    auto status = Export(input_model_, &result, params);
    if (!status.ok()) {
      LOG(INFO) << status.error_message();
      return names;
    }

    auto* model = ::tflite::GetModel(result.data());

    for (const ::tflite::OperatorCode* opcode : *model->operator_codes()) {
      if (opcode->builtin_code() != ::tflite::BuiltinOperator_CUSTOM) {
        names.push_back(string("builtin:") + ::tflite::EnumNameBuiltinOperator(
                                                 opcode->builtin_code()));
      } else {
        names.push_back(string("custom:") + opcode->custom_code()->c_str());
      }
    }

    return names;
  }

  std::vector<uint32_t> ExportAndGetOperatorIndices(
      const ExportParams& params) {
    std::vector<uint32_t> indices;

    string result;
    if (!Export(input_model_, &result, params).ok()) return indices;
    auto* model = ::tflite::GetModel(result.data());

    auto operators = (*model->subgraphs())[0]->operators();
    for (const auto* op : *operators) {
      indices.push_back(op->opcode_index());
    }
    return indices;
  }

  Model input_model_;
};

TEST_F(ExportTest, LoadTensorsMap) {
  AddTensorsByName({"tensor_one", "tensor_two"});

  details::TensorsMap tensors;
  details::LoadTensorsMap(input_model_, &tensors);
  EXPECT_EQ(0, tensors["tensor_one"]);
  EXPECT_EQ(1, tensors["tensor_two"]);
}

TEST_F(ExportTest, LoadOperatorsMap) {
  AddOperatorsByName({"Conv", "Add", "MyCrazyOp", "Sub"});

  details::OperatorsMap operators;
  const auto ops_by_type = BuildOperatorByTypeMap();
  details::LoadOperatorsMap(input_model_, &operators, ops_by_type, false);
  EXPECT_EQ(
      0, operators[details::OperatorKey(::tflite::BuiltinOperator_ADD, "", 1)]);
  EXPECT_EQ(1, operators[details::OperatorKey(::tflite::BuiltinOperator_CONV_2D,
                                              "", 1)]);
  EXPECT_EQ(2, operators[details::OperatorKey(::tflite::BuiltinOperator_CUSTOM,
                                              "MyCrazyOp", 1)]);
  EXPECT_EQ(
      3, operators[details::OperatorKey(::tflite::BuiltinOperator_SUB, "", 1)]);
}

TEST_F(ExportTest, Export) {
  AddOperatorsByName({"Conv", "Add", "MyCrazyOp", "Sub"});

  ExportParams params;
  params.allow_custom_ops = true;
  params.enable_select_tf_ops = false;
  params.quantize_weights = false;

  EXPECT_THAT(ExportAndSummarizeOperators(params),
              ElementsAre("builtin:ADD", "builtin:CONV_2D", "custom:MyCrazyOp",
                          "builtin:SUB"));
  EXPECT_THAT(ExportAndGetOperatorIndices(params), ElementsAre(1, 0, 2, 3));
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

class OpSetsTest : public ExportTest {
 public:
  enum OpSet { kTfLiteBuiltins, kSelectTfOps, kCustomOps };

  void SetAllowedOpSets(std::initializer_list<OpSet> sets) {
    import_all_ops_as_unsupported_ = true;
    params_.allow_custom_ops = false;
    params_.enable_select_tf_ops = false;
    params_.quantize_weights = false;

    for (OpSet i : sets) {
      switch (i) {
        case kTfLiteBuiltins:
          import_all_ops_as_unsupported_ = false;
          break;
        case kSelectTfOps:
          params_.enable_select_tf_ops = true;
          break;
        case kCustomOps:
          params_.allow_custom_ops = true;
          break;
      }
    }
  }

  std::vector<string> ImportExport(std::initializer_list<string> op_names) {
    ResetOperators();
    if (!import_all_ops_as_unsupported_) {
      AddOperatorsByName(op_names);
    } else {
      for (const string& name : op_names) {
        auto* op = new TensorFlowUnsupportedOperator;
        op->tensorflow_op = name;
        input_model_.operators.emplace_back(op);
      }
    }
    return ExportAndSummarizeOperators(params_);
  }

 private:
  bool import_all_ops_as_unsupported_;
  ExportParams params_;
};

TEST_F(OpSetsTest, BuiltinsOnly) {
  // --target_op_set=TFLITE_BUILTINS
  SetAllowedOpSets({kTfLiteBuiltins});
  EXPECT_THAT(ImportExport({"Add", "AdjustHue", "UnrollAndFold", "Assert"}),
              ElementsAre());
  EXPECT_THAT(ImportExport({"Add"}), ElementsAre("builtin:ADD"));

  // --target_op_set=TFLITE_BUILTINS --allow_custom_ops
  SetAllowedOpSets({kTfLiteBuiltins, kCustomOps});
  EXPECT_THAT(ImportExport({"Add", "AdjustHue", "UnrollAndFold", "Assert"}),
              ElementsAre("builtin:ADD", "custom:AdjustHue", "custom:Assert",
                          "custom:UnrollAndFold"));
}

TEST_F(OpSetsTest, TfSelectOnly) {
  // --target_op_set=SELECT_TF_OPS
  SetAllowedOpSets({kSelectTfOps});
  EXPECT_THAT(ImportExport({"Add", "AdjustHue", "RandomUniform",
                            "UnrollAndFold", "Assert"}),
              ElementsAre());
  EXPECT_THAT(ImportExport({"Add"}), ElementsAre("custom:FlexAdd"));

  // --target_op_set=SELECT_TF_OPS --allow_custom_ops
  SetAllowedOpSets({kSelectTfOps, kCustomOps});
  EXPECT_THAT(
      ImportExport(
          {"Add", "AdjustHue", "RandomUniform", "UnrollAndFold", "Assert"}),
      ElementsAre("custom:AdjustHue", "custom:FlexAdd", "custom:FlexAssert",
                  "custom:FlexRandomUniform", "custom:UnrollAndFold"));
}

TEST_F(OpSetsTest, BuiltinsAndTfSelect) {
  // --target_op_set=TFLITE_BUILTINS,SELECT_TF_OPS
  SetAllowedOpSets({kTfLiteBuiltins, kSelectTfOps});
  EXPECT_THAT(ImportExport({"Add", "AdjustHue", "UnrollAndFold", "Assert"}),
              ElementsAre());
  EXPECT_THAT(ImportExport({"Add", "RandomUniform"}),
              ElementsAre("builtin:ADD", "custom:FlexRandomUniform"));

  // --target_op_set=TFLITE_BUILTINS,SELECT_TF_OPS --allow_custom_ops
  SetAllowedOpSets({kTfLiteBuiltins, kSelectTfOps, kCustomOps});
  EXPECT_THAT(
      ImportExport(
          {"Add", "AdjustHue", "RandomUniform", "UnrollAndFold", "Assert"}),
      ElementsAre("builtin:ADD", "custom:AdjustHue", "custom:FlexAssert",
                  "custom:FlexRandomUniform", "custom:UnrollAndFold"));
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
  int GetVersion(const OperatorSignature& op_signature) const override {
    const TocoOperator& conv_op =
        static_cast<const TocoOperator&>(*op_signature.op);
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
  EXPECT_EQ(0, operators.at(details::OperatorKey(
                   ::tflite::BuiltinOperator_CONV_2D, "", 1)));
}

TEST_F(VersionedOpExportTest, LoadOperatorsMapWithOpV2) {
  AddConvOp(true);

  details::OperatorsMap operators;
  const auto ops_by_type = BuildFakeOperatorByTypeMap();
  details::LoadOperatorsMap(input_model_, &operators, ops_by_type, false);

  EXPECT_EQ(1, operators.size());
  EXPECT_EQ(0, operators.at(details::OperatorKey(
                   ::tflite::BuiltinOperator_CONV_2D, "", 2)));
}

TEST_F(VersionedOpExportTest, LoadOperatorsMapWithBothVersions) {
  AddConvOp(false);
  AddConvOp(true);

  details::OperatorsMap operators;
  const auto ops_by_type = BuildFakeOperatorByTypeMap();
  details::LoadOperatorsMap(input_model_, &operators, ops_by_type, false);

  EXPECT_EQ(2, operators.size());
  EXPECT_EQ(0, operators.at(details::OperatorKey(
                   ::tflite::BuiltinOperator_CONV_2D, "", 1)));
  EXPECT_EQ(1, operators.at(details::OperatorKey(
                   ::tflite::BuiltinOperator_CONV_2D, "", 2)));
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

TEST(OperatorKeyTest, TestBuiltinOp) {
  Model model;
  auto op = absl::make_unique<ConvOperator>();

  // Test a normal float operation.
  op->inputs = {"input", "filter"};
  op->outputs = {"output"};
  Array& input_array = model.GetOrCreateArray(op->inputs[0]);
  Array& filter_array = model.GetOrCreateArray(op->inputs[1]);
  Array& output_array = model.GetOrCreateArray(op->outputs[0]);
  input_array.data_type = ArrayDataType::kFloat;
  filter_array.data_type = ArrayDataType::kFloat;
  output_array.data_type = ArrayDataType::kFloat;

  const auto ops_by_type = BuildOperatorByTypeMap();
  const toco::OperatorSignature op_signature = {op.get(), &model};
  const auto key = details::OperatorKey(op_signature, ops_by_type, false);

  EXPECT_EQ(key.type(), ::tflite::BuiltinOperator_CONV_2D);
  EXPECT_EQ(key.custom_code(), "");
  EXPECT_EQ(key.version(), 1);
}

TEST(OperatorKeyTest, TestBuiltinOpWithVersionedInputTypes) {
  Model model;
  auto op = absl::make_unique<DequantizeOperator>();

  op->inputs = {"input"};
  op->outputs = {"output"};
  Array& input_array = model.GetOrCreateArray(op->inputs[0]);
  Array& output_array = model.GetOrCreateArray(op->outputs[0]);
  input_array.data_type = ArrayDataType::kInt8;
  output_array.data_type = ArrayDataType::kFloat;

  const auto ops_by_type = BuildOperatorByTypeMap();

  // Test a signed int8 dequantize operation.
  const toco::OperatorSignature op_signature = {op.get(), &model};
  const auto key = details::OperatorKey(op_signature, ops_by_type, false);

  EXPECT_EQ(key.type(), ::tflite::BuiltinOperator_DEQUANTIZE);
  EXPECT_EQ(key.custom_code(), "");
  EXPECT_EQ(key.version(), 2);
}

TEST(OperatorKeyTest, TestCustomOp) {
  Model model;
  auto op = absl::make_unique<TensorFlowUnsupportedOperator>();
  op->tensorflow_op = "MyCrazyCustomOp";

  const auto ops_by_type = BuildOperatorByTypeMap();
  const toco::OperatorSignature op_signature = {op.get(), &model};
  const auto key = details::OperatorKey(op_signature, ops_by_type, false);

  EXPECT_EQ(key.type(), ::tflite::BuiltinOperator_CUSTOM);
  EXPECT_EQ(key.custom_code(), "MyCrazyCustomOp");
  EXPECT_EQ(key.version(), 1);
}

TEST(OperatorKeyTest, TestFlexOp) {
  Model model;
  auto op = absl::make_unique<TensorFlowUnsupportedOperator>();
  op->tensorflow_op = "BatchMatMul";

  const auto ops_by_type = BuildOperatorByTypeMap();
  {
    const toco::OperatorSignature op_signature = {op.get(), &model};
    const auto key = details::OperatorKey(op_signature, ops_by_type, false);
    // It shouldn't be converted to Flex op if `allow_flex_op` is false.
    EXPECT_EQ(key.type(), ::tflite::BuiltinOperator_CUSTOM);
    EXPECT_EQ(key.custom_code(), "BatchMatMul");
    EXPECT_EQ(key.version(), 1);
    EXPECT_TRUE(key.is_custom_op());
    EXPECT_FALSE(key.is_flex_op());
  }

  {
    // Verify that the custom op name is prefixed by "Flex" and `is_flex_op`
    // is true.
    const toco::OperatorSignature op_signature = {op.get(), &model};
    const auto key = details::OperatorKey(op_signature, ops_by_type, true);
    EXPECT_EQ(key.type(), ::tflite::BuiltinOperator_CUSTOM);
    EXPECT_EQ(key.custom_code(), "FlexBatchMatMul");
    EXPECT_EQ(key.version(), 1);
    EXPECT_FALSE(key.is_custom_op());
    EXPECT_TRUE(key.is_flex_op());
  }
}

TEST(OperatorKeyTest, TestFlexWithControlFlowOp) {
  Model model;
  auto op = absl::make_unique<TensorFlowUnsupportedOperator>();
  op->tensorflow_op = "Merge";

  const auto ops_by_type = BuildOperatorByTypeMap();
  const toco::OperatorSignature op_signature = {op.get(), &model};
  const auto key = details::OperatorKey(op_signature, ops_by_type, true);

  EXPECT_EQ(key.type(), ::tflite::BuiltinOperator_CUSTOM);
  EXPECT_EQ(key.custom_code(), "FlexMerge");
  EXPECT_EQ(key.version(), 1);
  EXPECT_FALSE(key.is_custom_op());
  EXPECT_TRUE(key.is_flex_op());
  // The control flow ops should be marked as unsupported.
  EXPECT_TRUE(key.is_unsupported_flex_op());
}

TEST(OperatorKeyTest, TestFlexWithUnsupportedOp) {
  Model model;
  auto op = absl::make_unique<TensorFlowUnsupportedOperator>();
  op->tensorflow_op = "HashTableV2";

  const auto ops_by_type = BuildOperatorByTypeMap();
  const toco::OperatorSignature op_signature = {op.get(), &model};
  const auto key = details::OperatorKey(op_signature, ops_by_type, true);

  EXPECT_EQ(key.type(), ::tflite::BuiltinOperator_CUSTOM);
  EXPECT_EQ(key.custom_code(), "HashTableV2");
  EXPECT_EQ(key.version(), 1);
  // While HashTableV2 is excluded from the whitelisted flex op list, eventually
  // it won't be, and the following expectations will need to change as the op
  // is explicitly blacklisted due to lack of asset support.
  EXPECT_FALSE(key.is_flex_op());
  EXPECT_FALSE(key.is_unsupported_flex_op());
}

TEST(OperatorKeyTest, TestFlexWithPartiallySupportedOps) {
  // Test Toco-supported/TFLite-unsupported operators.
  Model model;
  // TODO(ycling): The test will be broken if TensorFlowAssert is implemented in
  // TFLite. Find a more robust way to test the fallback logic.
  auto op = absl::make_unique<TensorFlowAssertOperator>();

  const auto ops_by_type = BuildOperatorByTypeMap();

  {
    // If NodeDef isn't retained in the Toco op, a regular custom op
    // will be exported.
    const toco::OperatorSignature op_signature = {op.get(), &model};
    const auto key = details::OperatorKey(op_signature, ops_by_type, true);
    EXPECT_EQ(key.type(), ::tflite::BuiltinOperator_CUSTOM);
    EXPECT_EQ(key.custom_code(), "Assert");
    EXPECT_EQ(key.version(), 1);
    EXPECT_TRUE(key.is_custom_op());
    EXPECT_FALSE(key.is_flex_op());
  }

  ::tensorflow::NodeDef node_def;
  node_def.set_name("TensorFlowAssert");
  node_def.set_op("TensorFlowAssert");
  node_def.SerializeToString(&op->tensorflow_node_def);

  {
    // If NodeDef is retained in the Toco op, a Flex op will be exported.
    const toco::OperatorSignature op_signature = {op.get(), &model};
    const auto key = details::OperatorKey(op_signature, ops_by_type, true);
    EXPECT_EQ(key.type(), ::tflite::BuiltinOperator_CUSTOM);
    EXPECT_EQ(key.custom_code(), "FlexAssert");
    EXPECT_EQ(key.version(), 1);
    EXPECT_FALSE(key.is_custom_op());
    EXPECT_TRUE(key.is_flex_op());
  }
}

// TODO(ahentz): tests for tensors, inputs, outputs, opcodes and operators.

}  // namespace
}  // namespace tflite
}  // namespace toco
