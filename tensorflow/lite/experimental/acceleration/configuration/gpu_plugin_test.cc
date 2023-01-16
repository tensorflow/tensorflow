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
#include "tensorflow/lite/experimental/acceleration/configuration/gpu_plugin.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/experimental/acceleration/configuration/delegate_registry.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration.pb.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.h"

namespace tflite {
namespace delegates {
namespace {

#if TFLITE_SUPPORTS_GPU_DELEGATE || defined(REAL_IPHONE_DEVICE)

// While we could easily move the preprocessor lines around the expect part, we
// prefer to have two separate tests so it's clear which test is being
// exercised.
TEST(GpuPluginTest, GpuIsSupported) {
  flatbuffers::FlatBufferBuilder fbb;
  tflite::proto::ComputeSettings compute_settings;
  compute_settings.mutable_tflite_settings()->set_delegate(
      tflite::proto::Delegate::GPU);

  const tflite::ComputeSettings* compute_settings_fb =
      tflite::ConvertFromProto(compute_settings, &fbb);
  TfLiteDelegatePtr gpu_delegate =
      DelegatePluginRegistry::CreateByName(
          "GpuPlugin", *compute_settings_fb->tflite_settings())
          ->Create();
  EXPECT_NE(gpu_delegate, nullptr);
}

#else

TEST(GpuPluginTest, GpuNotSupported) {
  flatbuffers::FlatBufferBuilder fbb;
  tflite::proto::ComputeSettings compute_settings;
  compute_settings.mutable_tflite_settings()->set_delegate(
      tflite::proto::Delegate::GPU);

  const tflite::ComputeSettings* compute_settings_fb =
      tflite::ConvertFromProto(compute_settings, &fbb);
  TfLiteDelegatePtr gpu_delegate =
      DelegatePluginRegistry::CreateByName(
          "GpuPlugin", *compute_settings_fb->tflite_settings())
          ->Create();
  EXPECT_EQ(gpu_delegate, nullptr);
}

#endif

}  // namespace
}  // namespace delegates
}  // namespace tflite
