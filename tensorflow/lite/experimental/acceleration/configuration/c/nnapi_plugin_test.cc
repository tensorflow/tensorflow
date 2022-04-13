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
// NNAPI Delegate.

#include "tensorflow/lite/experimental/acceleration/configuration/c/nnapi_plugin.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"

namespace tflite {

class NnapiTest : public testing::Test {
 public:
  void SetUp() override {
    // Construct a FlatBuffer that contains
    // TFLiteSettings { NnapiSettings { foo1 : bar1, foo2 : bar2,  ...} }.
    NNAPISettingsBuilder nnapi_settings_builder(flatbuffer_builder_);
    flatbuffers::Offset<NNAPISettings> nnapi_settings =
        nnapi_settings_builder.Finish();
    // nnapi_settings_builder.add_foo1(bar1);
    // nnapi_settings_builder.add_foo2(bar2);
    TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder_);
    tflite_settings_builder.add_nnapi_settings(nnapi_settings);
    flatbuffers::Offset<TFLiteSettings> tflite_settings =
        tflite_settings_builder.Finish();
    flatbuffer_builder_.Finish(tflite_settings);
    settings_ = flatbuffers::GetRoot<TFLiteSettings>(
        flatbuffer_builder_.GetBufferPointer());
  }
  ~NnapiTest() override {}

 protected:
  // settings_ points into storage owned by flatbuffer_builder_.
  flatbuffers::FlatBufferBuilder flatbuffer_builder_;
  const TFLiteSettings *settings_;
};

TEST_F(NnapiTest, CanCreateAndDestroyDelegate) {
  TfLiteDelegate *delegate = TfLiteNnapiDelegatePluginCApi()->create(settings_);
  EXPECT_NE(delegate, nullptr);
  TfLiteNnapiDelegatePluginCApi()->destroy(delegate);
}

TEST_F(NnapiTest, CanGetDelegateErrno) {
  TfLiteDelegate *delegate = TfLiteNnapiDelegatePluginCApi()->create(settings_);
  int error_number =
      TfLiteNnapiDelegatePluginCApi()->get_delegate_errno(delegate);
  EXPECT_EQ(error_number, 0);
  TfLiteNnapiDelegatePluginCApi()->destroy(delegate);
}

}  // namespace tflite
