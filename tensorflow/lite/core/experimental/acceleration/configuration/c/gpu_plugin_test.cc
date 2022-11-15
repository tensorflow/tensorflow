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

// Some very simple unit tests of the C API Delegate Plugin for the
// GPU Delegate.

#include "tensorflow/lite/core/experimental/acceleration/configuration/c/gpu_plugin.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"

namespace tflite {

class GpuTest : public testing::Test {
 public:
  void SetUp() override {
    // Construct a FlatBuffer that contains
    // TFLiteSettings { GpuSettings { foo1 : bar1, foo2 : bar2,  ...} }.
    GPUSettingsBuilder gpu_settings_builder(flatbuffer_builder_);
    flatbuffers::Offset<GPUSettings> gpu_settings =
        gpu_settings_builder.Finish();
    // gpu_settings_builder.add_foo1(bar1);
    // gpu_settings_builder.add_foo2(bar2);
    TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder_);
    tflite_settings_builder.add_gpu_settings(gpu_settings);
    flatbuffers::Offset<TFLiteSettings> tflite_settings =
        tflite_settings_builder.Finish();
    flatbuffer_builder_.Finish(tflite_settings);
    settings_ = flatbuffers::GetRoot<TFLiteSettings>(
        flatbuffer_builder_.GetBufferPointer());
  }
  ~GpuTest() override {}

 protected:
  // settings_ points into storage owned by flatbuffer_builder_.
  flatbuffers::FlatBufferBuilder flatbuffer_builder_;
  const TFLiteSettings *settings_;
};

TEST_F(GpuTest, CanCreateAndDestroyDelegate) {
  TfLiteDelegate *delegate = TfLiteGpuDelegatePluginCApi()->create(settings_);
  EXPECT_NE(delegate, nullptr);
  TfLiteGpuDelegatePluginCApi()->destroy(delegate);
}

TEST_F(GpuTest, CanGetDelegateErrno) {
  TfLiteDelegate *delegate = TfLiteGpuDelegatePluginCApi()->create(settings_);
  int error_number =
      TfLiteGpuDelegatePluginCApi()->get_delegate_errno(delegate);
  EXPECT_EQ(error_number, 0);
  TfLiteGpuDelegatePluginCApi()->destroy(delegate);
}

}  // namespace tflite
