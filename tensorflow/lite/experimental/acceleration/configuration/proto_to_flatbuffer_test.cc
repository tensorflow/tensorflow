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
#include "tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace tflite {
namespace {

// Tests converting EdgeTpuSettings from proto to flatbuffer format.
TEST(ConversionTest, EdgeTpuSettings) {
  // Define the fields to be tested.
  const std::vector<int32_t> kHardwareClusterIds{1};
  const std::string kPublicModelId = "public_model_id";

  // Create the proto settings.
  proto::ComputeSettings input_settings;
  auto* edgetpu_settings =
      input_settings.mutable_tflite_settings()->mutable_edgetpu_settings();
  edgetpu_settings->set_public_model_id(kPublicModelId);
  flatbuffers::FlatBufferBuilder flatbuffers_builder;
  *edgetpu_settings->mutable_hardware_cluster_ids() = {
      kHardwareClusterIds.begin(), kHardwareClusterIds.end()};

  // Convert.
  auto output_settings = ConvertFromProto(input_settings, &flatbuffers_builder)
                             ->tflite_settings()
                             ->edgetpu_settings();

  // Verify the conversion results.
  EXPECT_EQ(output_settings->hardware_cluster_ids()->size(), 1);
  EXPECT_EQ(output_settings->hardware_cluster_ids()->Get(0),
            kHardwareClusterIds[0]);
  EXPECT_EQ(output_settings->public_model_id()->str(), kPublicModelId);
}

// Tests converting TFLiteSettings from proto to flatbuffer format.
TEST(ConversionTest, TFLiteSettings) {
  // Define the fields to be tested.
  const std::vector<int32_t> kHardwareClusterIds{1};
  const std::string kPublicModelId = "public_model_id";

  // Create the proto settings.
  proto::TFLiteSettings input_settings;
  input_settings.set_delegate(::tflite::proto::EDGETPU);
  auto* edgetpu_settings = input_settings.mutable_edgetpu_settings();
  edgetpu_settings->set_public_model_id(kPublicModelId);
  flatbuffers::FlatBufferBuilder flatbuffers_builder;
  *edgetpu_settings->mutable_hardware_cluster_ids() = {
      kHardwareClusterIds.begin(), kHardwareClusterIds.end()};

  // Convert.
  auto output_settings = ConvertFromProto(input_settings, &flatbuffers_builder);

  // Verify the conversion results.
  EXPECT_EQ(output_settings->delegate(), ::tflite::Delegate_EDGETPU);
  const auto* output_edgetpu_settings = output_settings->edgetpu_settings();
  EXPECT_EQ(output_edgetpu_settings->hardware_cluster_ids()->size(), 1);
  EXPECT_EQ(output_edgetpu_settings->hardware_cluster_ids()->Get(0),
            kHardwareClusterIds[0]);
  EXPECT_EQ(output_edgetpu_settings->public_model_id()->str(), kPublicModelId);
}

}  // namespace
}  // namespace tflite
