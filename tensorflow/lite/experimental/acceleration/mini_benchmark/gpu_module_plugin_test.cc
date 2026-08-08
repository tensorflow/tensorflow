/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/gpu_module_plugin.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/acceleration/configuration/configuration.pb.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/acceleration/configuration/proto_to_flatbuffer.h"

namespace tflite {
namespace acceleration {
namespace {

// Helper to create ComputeSettings flatbuffer with the given GPU plugin path.
std::unique_ptr<flatbuffers::FlatBufferBuilder> CreateGpuSettings(
    const std::string& delegate_path, const ComputeSettings** settings_out) {
  auto fbb = std::make_unique<flatbuffers::FlatBufferBuilder>();
  proto::ComputeSettings settings_proto;
  settings_proto.mutable_tflite_settings()->set_delegate(proto::GPU);
  settings_proto.mutable_tflite_settings()
      ->mutable_stable_delegate_loader_settings()
      ->set_delegate_path(delegate_path);
  *settings_out = ConvertFromProto(settings_proto, fbb.get());
  return fbb;
}

// Helper to verify that the plugin can load and create a delegate.
void VerifyPluginCanLoadAndCreateDelegate(const TFLiteSettings& settings) {
  auto plugin = GpuModulePlugin::New(settings);
  ASSERT_NE(plugin.get(), nullptr);
  auto delegate = plugin->Create();
  EXPECT_NE(delegate, nullptr);
}

// Verifies that the plugin initialization handles dlopen failures gracefully
// when provided with an invalid path (i.e. it doesn't crash).
TEST(GpuModulePluginTest, DlopenFlags) {
  const ComputeSettings* settings = nullptr;
  auto fbb =
      CreateGpuSettings("invalid_path_to_force_dlopen_fail.so", &settings);
  ASSERT_NE(settings, nullptr);
  ASSERT_NE(settings->tflite_settings(), nullptr);

  auto plugin = GpuModulePlugin::New(*settings->tflite_settings());
  ASSERT_NE(plugin.get(), nullptr);
}

// Verifies that the plugin can be successfully loaded from a shared library
// and that it can create a delegate.
// This test uses a mock GPU delegate plugin to ensure hermeticity and avoid
// dependencies on real GPU hardware or libraries that may not be available
// in the test environment.
TEST(GpuModulePluginTest, LoadSucceeds) {
  const ComputeSettings* settings = nullptr;
  auto fbb = CreateGpuSettings(
      "tensorflow/lite/experimental/acceleration/"
      "mini_benchmark/mock_gpu_delegate_plugin.so",
      &settings);
  ASSERT_NE(settings, nullptr);
  ASSERT_NE(settings->tflite_settings(), nullptr);

  VerifyPluginCanLoadAndCreateDelegate(*settings->tflite_settings());
}

// Verifies that the plugin can be loaded, unloaded (when the plugin object
// is destroyed), and then loaded again successfully.
// This ensures that the dynamic loading/unloading mechanism works correctly
// across multiple lifecycles in the same process.
// This test also uses the mock GPU delegate plugin for hermeticity.
TEST(GpuModulePluginTest, LoadUnloadLoad) {
  const ComputeSettings* settings = nullptr;
  auto fbb = CreateGpuSettings(
      "tensorflow/lite/experimental/acceleration/"
      "mini_benchmark/mock_gpu_delegate_plugin.so",
      &settings);
  ASSERT_NE(settings, nullptr);
  ASSERT_NE(settings->tflite_settings(), nullptr);

  VerifyPluginCanLoadAndCreateDelegate(*settings->tflite_settings());
  VerifyPluginCanLoadAndCreateDelegate(*settings->tflite_settings());
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite
