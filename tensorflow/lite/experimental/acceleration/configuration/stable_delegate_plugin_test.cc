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

// Some very simple unit tests of the (C++) XNNPack Delegate Plugin.

#include <memory>

#include <gtest/gtest.h>
#include "pthreadpool.h"  // from @pthreadpool
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/configuration/delegate_registry.h"

namespace tflite {

class StableDelegatePluginTest : public testing::Test {
 public:
  static constexpr int kNumThreadsForTest = 7;
  static constexpr tflite::XNNPackFlags kFlagsForTest =
      tflite::XNNPackFlags::XNNPackFlags_TFLITE_XNNPACK_DELEGATE_FLAG_QS8_QU8;
  static constexpr char kDelegateBinaryPath[] =
      "tensorflow/lite/delegates/utils/experimental/"
      "stable_delegate/libtensorflowlite_stable_xnnpack_delegate.so";

  void SetUp() override {
    // Construct a FlatBuffer that contains
    //   TFLiteSettings {
    //     delegate: Delegate.XNNPACK,
    //     XNNPackSettings { num_threads: kNumThreadsForTest
    //                       flags: TFLITE_XNNPACK_DELEGATE_FLAG_QS8 |
    //                           TFLITE_XNNPACK_DELEGATE_FLAG_QU8
    //     },
    //     StableDelegateLoaderSettings { delegate_path: kDelegateBinaryPath }
    //   }.
    // We use the stable XNNPack delegate binary for testing stable delegate
    // provider.
    flatbuffers::Offset<flatbuffers::String> stable_delegate_path_offset =
        flatbuffer_builder_.CreateString(kDelegateBinaryPath);
    StableDelegateLoaderSettingsBuilder stable_delegate_loader_settings_builder(
        flatbuffer_builder_);
    stable_delegate_loader_settings_builder.add_delegate_path(
        stable_delegate_path_offset);
    flatbuffers::Offset<StableDelegateLoaderSettings>
        stable_delegate_loader_settings =
            stable_delegate_loader_settings_builder.Finish();
    XNNPackSettingsBuilder xnnpack_settings_builder(flatbuffer_builder_);
    xnnpack_settings_builder.add_num_threads(kNumThreadsForTest);
    xnnpack_settings_builder.add_flags(kFlagsForTest);
    flatbuffers::Offset<XNNPackSettings> xnnpack_settings =
        xnnpack_settings_builder.Finish();
    TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder_);
    tflite_settings_builder.add_stable_delegate_loader_settings(
        stable_delegate_loader_settings);
    tflite_settings_builder.add_xnnpack_settings(xnnpack_settings);
    // Stable delegate plugin doesn't rely on the delegate specified in the
    // TFLiteSettings provided.
    tflite_settings_builder.add_delegate(Delegate_XNNPACK);
    flatbuffers::Offset<TFLiteSettings> tflite_settings =
        tflite_settings_builder.Finish();
    flatbuffer_builder_.Finish(tflite_settings);
    tflite_settings_ = flatbuffers::GetRoot<TFLiteSettings>(
        flatbuffer_builder_.GetBufferPointer());
    // Create a stable delegate plugin for an XNNPack delegate using the
    // settings from the flatbuffer.
    delegate_plugin_ = delegates::DelegatePluginRegistry::CreateByName(
        "StableDelegatePlugin", *tflite_settings_);
    ASSERT_NE(delegate_plugin_, nullptr);
  }
  void TearDown() override { delegate_plugin_.reset(); }

 protected:
  // settings_ points into storage owned by flatbuffer_builder_.
  flatbuffers::FlatBufferBuilder flatbuffer_builder_;
  const TFLiteSettings *tflite_settings_;
  std::unique_ptr<delegates::DelegatePluginInterface> delegate_plugin_;
};

TEST_F(StableDelegatePluginTest, CanCreateAndDestroyDelegate) {
  delegates::TfLiteDelegatePtr delegate = delegate_plugin_->Create();
  EXPECT_NE(delegate, nullptr);
}

TEST_F(StableDelegatePluginTest, CanGetDelegateErrno) {
  delegates::TfLiteDelegatePtr delegate = delegate_plugin_->Create();

  EXPECT_EQ(delegate_plugin_->GetDelegateErrno(delegate.get()), 0);
}

TEST_F(StableDelegatePluginTest, SetsCorrectThreadCount) {
  delegates::TfLiteDelegatePtr delegate = delegate_plugin_->Create();
  pthreadpool_t threadpool = static_cast<pthreadpool_t>(
      TfLiteXNNPackDelegateGetThreadPool(delegate.get()));

  EXPECT_EQ(pthreadpool_get_threads_count(threadpool), kNumThreadsForTest);
}

}  // namespace tflite
