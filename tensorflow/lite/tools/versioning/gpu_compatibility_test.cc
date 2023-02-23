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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/lite/core/model_builder.h"

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

}  // namespace tflite
