/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tools/list_flex_ops.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {
namespace flex {

class FlexOpsListTest : public ::testing::Test {
 protected:
  FlexOpsListTest() {}

  void ReadOps(const string& model_path) {
    auto model = FlatBufferModel::BuildFromFile(model_path.data());
    AddFlexOpsFromModel(model->GetModel(), &flex_ops_);
    output_text_ = OpListToJSONString(flex_ops_);
  }

  void ReadOps(const tflite::Model* model) {
    AddFlexOpsFromModel(model, &flex_ops_);
    output_text_ = OpListToJSONString(flex_ops_);
  }

  std::string output_text_;
  OpKernelSet flex_ops_;
};

TfLiteRegistration* Register_TEST() {
  static TfLiteRegistration r = {nullptr, nullptr, nullptr, nullptr};
  return &r;
}

std::vector<uint8_t> CreateFlexCustomOptions(std::string nodedef_raw_string) {
  tensorflow::NodeDef node_def;
  tensorflow::protobuf::TextFormat::ParseFromString(nodedef_raw_string,
                                                    &node_def);
  std::string node_def_str = node_def.SerializeAsString();
  auto flex_builder = std::make_unique<flexbuffers::Builder>();
  flex_builder->Vector([&]() {
    flex_builder->String(node_def.op());
    flex_builder->String(node_def_str);
  });
  flex_builder->Finish();
  return flex_builder->GetBuffer();
}

class FlexOpModel : public SingleOpModel {
 public:
  FlexOpModel(const std::string& op_name, const TensorData& input1,
              const TensorData& input2, const TensorType& output,
              const std::vector<uint8_t>& custom_options) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetCustomOp(op_name, custom_options, Register_TEST);
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

 protected:
  int input1_;
  int input2_;
  int output_;
};

TEST_F(FlexOpsListTest, TestModelsNoFlex) {
  ReadOps("third_party/tensorflow/lite/testdata/test_model.bin");
  EXPECT_EQ(output_text_, "[]");
}

TEST_F(FlexOpsListTest, TestBrokenModel) {
  EXPECT_DEATH_IF_SUPPORTED(
      ReadOps("third_party/tensorflow/lite/testdata/test_model_broken.bin"),
      "");
}

TEST_F(FlexOpsListTest, TestZeroSubgraphs) {
  ReadOps("third_party/tensorflow/lite/testdata/0_subgraphs.bin");
  EXPECT_EQ(output_text_, "[]");
}

TEST_F(FlexOpsListTest, TestFlexAdd) {
  ReadOps("third_party/tensorflow/lite/testdata/multi_add_flex.bin");
  EXPECT_EQ(output_text_,
            "[[\"Add\", \"BinaryOp<CPUDevice, functor::add<float>>\"]]");
}

TEST_F(FlexOpsListTest, TestTwoModel) {
  ReadOps("third_party/tensorflow/lite/testdata/multi_add_flex.bin");
  ReadOps("third_party/tensorflow/lite/testdata/softplus_flex.bin");
  EXPECT_EQ(output_text_,
            "[[\"Add\", \"BinaryOp<CPUDevice, "
            "functor::add<float>>\"],\n[\"Softplus\", \"SoftplusOp<CPUDevice, "
            "float>\"]]");
}

TEST_F(FlexOpsListTest, TestDuplicatedOp) {
  ReadOps("third_party/tensorflow/lite/testdata/multi_add_flex.bin");
  ReadOps("third_party/tensorflow/lite/testdata/multi_add_flex.bin");
  EXPECT_EQ(output_text_,
            "[[\"Add\", \"BinaryOp<CPUDevice, functor::add<float>>\"]]");
}

TEST_F(FlexOpsListTest, TestInvalidCustomOptions) {
  // Using a invalid custom options, expected to fail.
  std::vector<uint8_t> random_custom_options(20);
  FlexOpModel max_model("FlexAdd", {TensorType_FLOAT32, {3, 1, 2, 2}},
                        {TensorType_FLOAT32, {3, 1, 2, 1}}, TensorType_FLOAT32,
                        random_custom_options);
  EXPECT_DEATH_IF_SUPPORTED(
      ReadOps(tflite::GetModel(max_model.GetModelBuffer())),
      "Failed to parse data into a valid NodeDef");
}

TEST_F(FlexOpsListTest, TestOpNameEmpty) {
  // NodeDef with empty opname.
  std::string nodedef_raw_str =
      "name: \"node_1\""
      "op: \"\""
      "input: [ \"b\", \"c\" ]"
      "attr: { key: \"T\" value: { type: DT_FLOAT } }";
  std::string random_fieldname = "random string";
  FlexOpModel max_model("FlexAdd", {TensorType_FLOAT32, {3, 1, 2, 2}},
                        {TensorType_FLOAT32, {3, 1, 2, 1}}, TensorType_FLOAT32,
                        CreateFlexCustomOptions(nodedef_raw_str));
  EXPECT_DEATH_IF_SUPPORTED(
      ReadOps(tflite::GetModel(max_model.GetModelBuffer())), "Invalid NodeDef");
}

TEST_F(FlexOpsListTest, TestOpNotFound) {
  // NodeDef with invalid opname.
  std::string nodedef_raw_str =
      "name: \"node_1\""
      "op: \"FlexInvalidOp\""
      "input: [ \"b\", \"c\" ]"
      "attr: { key: \"T\" value: { type: DT_FLOAT } }";

  FlexOpModel max_model("FlexAdd", {TensorType_FLOAT32, {3, 1, 2, 2}},
                        {TensorType_FLOAT32, {3, 1, 2, 1}}, TensorType_FLOAT32,
                        CreateFlexCustomOptions(nodedef_raw_str));
  EXPECT_DEATH_IF_SUPPORTED(
      ReadOps(tflite::GetModel(max_model.GetModelBuffer())),
      "Op FlexInvalidOp not found");
}

TEST_F(FlexOpsListTest, TestKernelNotFound) {
  // NodeDef with non-supported type.
  std::string nodedef_raw_str =
      "name: \"node_1\""
      "op: \"Add\""
      "input: [ \"b\", \"c\" ]"
      "attr: { key: \"T\" value: { type: DT_BOOL } }";

  FlexOpModel max_model("FlexAdd", {TensorType_FLOAT32, {3, 1, 2, 2}},
                        {TensorType_FLOAT32, {3, 1, 2, 1}}, TensorType_FLOAT32,
                        CreateFlexCustomOptions(nodedef_raw_str));
  EXPECT_DEATH_IF_SUPPORTED(
      ReadOps(tflite::GetModel(max_model.GetModelBuffer())),
      "Failed to find kernel class for op: Add");
}

TEST_F(FlexOpsListTest, TestFlexAddWithSingleOpModel) {
  std::string nodedef_raw_str =
      "name: \"node_1\""
      "op: \"Add\""
      "input: [ \"b\", \"c\" ]"
      "attr: { key: \"T\" value: { type: DT_FLOAT } }";

  FlexOpModel max_model("FlexAdd", {TensorType_FLOAT32, {3, 1, 2, 2}},
                        {TensorType_FLOAT32, {3, 1, 2, 1}}, TensorType_FLOAT32,
                        CreateFlexCustomOptions(nodedef_raw_str));
  ReadOps(tflite::GetModel(max_model.GetModelBuffer()));
  EXPECT_EQ(output_text_,
            "[[\"Add\", \"BinaryOp<CPUDevice, functor::add<float>>\"]]");
}
}  // namespace flex
}  // namespace tflite

int main(int argc, char** argv) {
  // On Linux, add: FLAGS_logtostderr = true;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
