/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tools/versioning/gpu_compatibility.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/versioning/op_signature.h"

namespace tflite {

namespace {

absl::Status CheckGpuDelegateCompatibility(const tflite::Model* model) {
  auto subgraphs = model->subgraphs();

  for (int i = 0; i < subgraphs->Length(); ++i) {
    const SubGraph* subgraph = subgraphs->Get(i);
    for (int j = 0; j < subgraph->operators()->Length(); ++j) {
      const Operator* op = subgraph->operators()->Get(j);
      const OperatorCode* op_code =
          model->operator_codes()->Get(op->opcode_index());
      auto status = CheckGpuDelegateCompatibility(op_code, op, subgraph, model);
      if (!status.ok()) {
        return status;
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

// FYI, CheckGpuDelegateCompatibility() will be validated by
// third_party/tensorflow/lite/delegates/gpu/common:model_builder_test

TEST(CheckGpuDelegateCompatibility, Conv2DModel) {
  const std::string& full_path = tensorflow::GetDataDependencyFilepath(
      "tensorflow/lite/testdata/conv_huge_im2col.bin");
  auto model = FlatBufferModel::BuildFromFile(full_path.data());
  ASSERT_TRUE(model);
  EXPECT_TRUE(CheckGpuDelegateCompatibility(model->GetModel()).ok());
}

TEST(CheckGpuDelegateCompatibility, Conv3DModel) {
  const std::string& full_path = tensorflow::GetDataDependencyFilepath(
      "tensorflow/lite/testdata/conv3d_huge_im2col.bin");
  auto model = FlatBufferModel::BuildFromFile(full_path.data());
  ASSERT_TRUE(model);
  EXPECT_EQ(CheckGpuDelegateCompatibility(model->GetModel()).message(),
            "Not supported op CONV_3D");
}

TEST(CheckGpuDelegateCompatibility, FlexModel) {
  const std::string& full_path = tensorflow::GetDataDependencyFilepath(
      "tensorflow/lite/testdata/multi_add_flex.bin");
  auto model = FlatBufferModel::BuildFromFile(full_path.data());
  ASSERT_TRUE(model);
  EXPECT_EQ(CheckGpuDelegateCompatibility(model->GetModel()).message(),
            "Not supported custom op FlexAddV2");
}

TEST(CheckGpuDelegateCompatibility, FCConstInput) {
  OpSignature op_sig = OpSignature();
  op_sig.op = BuiltinOperator_FULLY_CONNECTED;
  auto params = std::make_unique<TfLiteFullyConnectedParams>();
  params->weights_format = kTfLiteFullyConnectedWeightsFormatDefault;
  op_sig.builtin_data = static_cast<void*>(params.get());
  op_sig.inputs = std::vector<OpSignatureTensorSpec>(1);
  op_sig.inputs[0] = OpSignatureTensorSpec();
  op_sig.inputs[0].is_const = true;
  EXPECT_EQ(CheckGpuDelegateCompatibility(op_sig).message(),
            "FullyConnected doesn't support constant input.");
}

TEST(CheckGpuDelegateCompatibility, Add1Dto3DBroadcastSuccess) {
  OpSignature op_sig = OpSignature();
  op_sig.op = BuiltinOperator_ADD;
  auto params = std::make_unique<TfLiteAddParams>();
  op_sig.builtin_data = static_cast<void*>(params.get());
  op_sig.inputs = std::vector<OpSignatureTensorSpec>(2);
  op_sig.inputs[0] = OpSignatureTensorSpec();
  op_sig.inputs[0].dims = {4, 1, 2};
  op_sig.inputs[1] = OpSignatureTensorSpec();
  op_sig.inputs[1].dims = {2};

  EXPECT_TRUE(CheckGpuDelegateCompatibility(op_sig).message().empty());
}

TEST(CheckGpuDelegateCompatibility, Add2Dto3DBroadcastFail) {
  OpSignature op_sig = OpSignature();
  op_sig.op = BuiltinOperator_ADD;
  auto params = std::make_unique<TfLiteAddParams>();
  op_sig.builtin_data = static_cast<void*>(params.get());
  op_sig.inputs = std::vector<OpSignatureTensorSpec>(2);
  op_sig.inputs[0] = OpSignatureTensorSpec();
  op_sig.inputs[0].dims = {1, 100, 256};
  op_sig.inputs[1] = OpSignatureTensorSpec();
  op_sig.inputs[1].dims = {100, 256};

  EXPECT_EQ(CheckGpuDelegateCompatibility(op_sig).message(),
            "Doesn't support broadcasting - input0: [1,100,256], input1: "
            "[100,256]");
}

TEST(CheckGpuDelegateCompatibility, Add3Dto4DBroadcastFail) {
  OpSignature op_sig = OpSignature();
  op_sig.op = BuiltinOperator_ADD;
  auto params = std::make_unique<TfLiteAddParams>();
  op_sig.builtin_data = static_cast<void*>(params.get());
  op_sig.inputs = std::vector<OpSignatureTensorSpec>(2);
  op_sig.inputs[0] = OpSignatureTensorSpec();
  op_sig.inputs[0].dims = {4, 1, 1, 2};
  op_sig.inputs[1] = OpSignatureTensorSpec();
  // Can't broadcast using batch of 4
  op_sig.inputs[1].dims = {1, 1, 2};

  EXPECT_EQ(
      CheckGpuDelegateCompatibility(op_sig).message(),
      "Doesn't support broadcasting - input0: [4,1,1,2], input1: [1,1,2]");
}

TEST(CheckGpuDelegateCompatibility, Add3Dto4DBroadcastSuccess) {
  OpSignature op_sig = OpSignature();
  op_sig.op = BuiltinOperator_ADD;
  auto params = std::make_unique<TfLiteAddParams>();
  op_sig.builtin_data = static_cast<void*>(params.get());
  op_sig.inputs = std::vector<OpSignatureTensorSpec>(2);
  op_sig.inputs[0] = OpSignatureTensorSpec();
  op_sig.inputs[0].dims = {1, 128, 513, 3};
  op_sig.inputs[1] = OpSignatureTensorSpec();
  // Can be broadcasted to {1, 128, 513, 3}
  op_sig.inputs[1].dims = {128, 513, 3};

  EXPECT_TRUE(CheckGpuDelegateCompatibility(op_sig).message().empty());
}

TEST(CheckGpuDelegateCompatibility, Add2Dto4DBroadcastSuccess) {
  OpSignature op_sig = OpSignature();
  op_sig.op = BuiltinOperator_ADD;
  auto params = std::make_unique<TfLiteAddParams>();
  op_sig.builtin_data = static_cast<void*>(params.get());
  op_sig.inputs = std::vector<OpSignatureTensorSpec>(2);
  op_sig.inputs[0] = OpSignatureTensorSpec();
  op_sig.inputs[0].dims = {1, 512, 512, 1};
  op_sig.inputs[1] = OpSignatureTensorSpec();
  // Can be broadcasted to {1, 1, 1, 1}
  op_sig.inputs[1].dims = {1, 1};

  EXPECT_TRUE(CheckGpuDelegateCompatibility(op_sig).message().empty());
}

TEST(CheckGpuDelegateCompatibility, Add2Dto4DBroadcastSuccess2) {
  OpSignature op_sig = OpSignature();
  op_sig.op = BuiltinOperator_ADD;
  auto params = std::make_unique<TfLiteAddParams>();
  op_sig.builtin_data = static_cast<void*>(params.get());
  op_sig.inputs = std::vector<OpSignatureTensorSpec>(2);
  op_sig.inputs[0] = OpSignatureTensorSpec();
  op_sig.inputs[0].dims = {1, 384, 384, 3};
  op_sig.inputs[1] = OpSignatureTensorSpec();
  // Can be broadcasted to {1, 1, 1, 1}
  op_sig.inputs[1].dims = {1, 1};

  EXPECT_TRUE(CheckGpuDelegateCompatibility(op_sig).message().empty());
}

}  // namespace tflite
