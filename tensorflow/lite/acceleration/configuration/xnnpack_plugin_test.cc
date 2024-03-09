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

// Some very simple unit tests of the (C++) XNNPack Delegate Plugin.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "pthreadpool.h"  // from @pthreadpool
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/acceleration/configuration/delegate_registry.h"
#include "tensorflow/lite/test_util.h"

namespace tflite {

class XnnpackPluginTest : public tflite::testing::Test {
 public:
  static constexpr int kNumThreadsForTest = 7;
  static constexpr tflite::XNNPackFlags kFlagsForTest =
      tflite::XNNPackFlags::XNNPackFlags_TFLITE_XNNPACK_DELEGATE_FLAG_QS8_QU8;
  void SetUp() override {
    // Construct a FlatBuffer that contains
    //   TFLiteSettings {
    //     delegate: Delegate.XNNPACK,
    //     XNNPackSettings { num_threads: kNumThreadsForTest
    //                       flags: TFLITE_XNNPACK_DELEGATE_FLAG_QS8 |
    //                           TFLITE_XNNPACK_DELEGATE_FLAG_QU8
    //     }
    //   }.
    XNNPackSettingsBuilder xnnpack_settings_builder(flatbuffer_builder_);
    xnnpack_settings_builder.add_num_threads(kNumThreadsForTest);
    xnnpack_settings_builder.add_flags(kFlagsForTest);
    flatbuffers::Offset<XNNPackSettings> xnnpack_settings =
        xnnpack_settings_builder.Finish();
    TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder_);
    tflite_settings_builder.add_xnnpack_settings(xnnpack_settings);
    tflite_settings_builder.add_delegate(Delegate_XNNPACK);
    flatbuffers::Offset<TFLiteSettings> tflite_settings =
        tflite_settings_builder.Finish();
    flatbuffer_builder_.Finish(tflite_settings);
    tflite_settings_ = flatbuffers::GetRoot<TFLiteSettings>(
        flatbuffer_builder_.GetBufferPointer());
    // Create an XNNPack delegate plugin using the settings from the flatbuffer.
    delegate_plugin_ = delegates::DelegatePluginRegistry::CreateByName(
        "XNNPackPlugin", *tflite_settings_);
    ASSERT_NE(delegate_plugin_, nullptr);
  }
  void TearDown() override { delegate_plugin_.reset(); }
  ~XnnpackPluginTest() override {}

 protected:
  // settings_ points into storage owned by flatbuffer_builder_.
  flatbuffers::FlatBufferBuilder flatbuffer_builder_;
  const TFLiteSettings *tflite_settings_;
  std::unique_ptr<delegates::DelegatePluginInterface> delegate_plugin_;
};

constexpr int XnnpackPluginTest::kNumThreadsForTest;

TEST_F(XnnpackPluginTest, CanCreateAndDestroyDelegate) {
  delegates::TfLiteOpaqueDelegatePtr delegate = delegate_plugin_->Create();
  EXPECT_NE(delegate, nullptr);
}

TEST_F(XnnpackPluginTest, CanGetDelegateErrno) {
  delegates::TfLiteOpaqueDelegatePtr delegate = delegate_plugin_->Create();
  int error_number = delegate_plugin_->GetDelegateErrno(delegate.get());
  EXPECT_EQ(error_number, 0);
}

}  // namespace tflite
