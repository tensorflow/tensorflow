/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/sparsity/sparsify_model.h"

#include <stdint.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/core/absl_error_model_builder.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/tools/optimize/reduced_precision_metadata.h"

namespace mlir {
namespace lite {
namespace {


TEST(SparsifyModelTest, MetadataIsAddedToOutputModel) {
  std::string expected_key = tflite::optimize::kTfLiteReducedPrecisionKey;
  std::string expected_value = "test_data";

  // Load input model
  auto input_fbm = mlir::TFL::FlatBufferModelAbslError::BuildFromFile(
      "tensorflow/compiler/mlir/lite/sparsity/testdata/"
      "sparse_tensor.bin");
  tflite::ModelT input_model;
  input_fbm->GetModel()->UnPackTo(&input_model);

  // Populate input metadata
  auto model_metadata_buffer = std::make_unique<tflite::BufferT>();
  model_metadata_buffer->data =
      std::vector<uint8_t>(expected_value.begin(), expected_value.end());
  input_model.buffers.push_back(std::move(model_metadata_buffer));
  auto metadata_t = std::make_unique<tflite::MetadataT>();
  metadata_t->name = tflite::optimize::kTfLiteReducedPrecisionKey;
  metadata_t->buffer = input_model.buffers.size() - 1;
  input_model.metadata.push_back(std::move(metadata_t));

  // Sparsify and create output model
  flatbuffers::FlatBufferBuilder output_builder;
  ASSERT_TRUE(SparsifyModel(input_model, &output_builder).ok());
  auto output_fbm = mlir::TFL::FlatBufferModelAbslError::BuildFromBuffer(
      reinterpret_cast<const char*>(output_builder.GetCurrentBufferPointer()),
      output_builder.GetSize());
  tflite::ModelT output_model;
  output_fbm->GetModel()->UnPackTo(&output_model);

  // Extract output metadata
  std::map<std::string, std::string> output_metadata;
  for (const auto& metadata : output_model.metadata) {
    const auto& data = output_model.buffers[metadata->buffer]->data;
    output_metadata[metadata->name] = std::string(data.begin(), data.end());
  }

  EXPECT_THAT(output_metadata,
              testing::Contains(testing::Pair(expected_key, expected_value)));
}

}  // namespace
}  // namespace lite
}  // namespace mlir
