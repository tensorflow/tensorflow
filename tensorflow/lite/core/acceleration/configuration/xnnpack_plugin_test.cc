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

#include <memory>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "pthreadpool.h"  // from @pthreadpool
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/core/acceleration/configuration/delegate_registry.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {

class XnnpackPluginTest : public testing::Test {
 public:
  static constexpr int kNumThreadsForTest = 7;
  void SetUp() override {
    // Construct a FlatBuffer that contains
    //   TFLiteSettings {
    //     delegate: Delegate.XNNPACK,
    //     XNNPackSettings {
    //       num_threads: kNumThreadsForTest
    //     }
    //   }.
    XNNPackSettingsBuilder xnnpack_settings_builder(flatbuffer_builder_);
    xnnpack_settings_builder.add_num_threads(kNumThreadsForTest);
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
  ~XnnpackPluginTest() override = default;

 protected:
  // settings_ points into storage owned by flatbuffer_builder_.
  flatbuffers::FlatBufferBuilder flatbuffer_builder_;
  const TFLiteSettings *tflite_settings_;
  std::unique_ptr<delegates::DelegatePluginInterface> delegate_plugin_;
};

constexpr int XnnpackPluginTest::kNumThreadsForTest;

TEST_F(XnnpackPluginTest, CanCreateAndDestroyDelegate) {
  delegates::TfLiteDelegatePtr delegate = delegate_plugin_->Create();
  EXPECT_NE(delegate, nullptr);
}

TEST_F(XnnpackPluginTest, CanGetDelegateErrno) {
  delegates::TfLiteDelegatePtr delegate = delegate_plugin_->Create();
  int error_number = delegate_plugin_->GetDelegateErrno(delegate.get());
  EXPECT_EQ(error_number, 0);
}

TEST_F(XnnpackPluginTest, SetsCorrectThreadCount) {
  delegates::TfLiteDelegatePtr delegate = delegate_plugin_->Create();
  pthreadpool_t threadpool = static_cast<pthreadpool_t>(
      TfLiteXNNPackDelegateGetThreadPool(delegate.get()));
  int thread_count = pthreadpool_get_threads_count(threadpool);
  EXPECT_EQ(thread_count, kNumThreadsForTest);
}

TEST_F(XnnpackPluginTest, UsesDefaultFlagsByDefault) {
  delegates::TfLiteDelegatePtr delegate = delegate_plugin_->Create();
  int flags = TfLiteXNNPackDelegateGetFlags(delegate.get());
  EXPECT_EQ(flags, TfLiteXNNPackDelegateOptionsDefault().flags);
}

TEST_F(XnnpackPluginTest, UsesSpecifiedFlagsWhenNonzero) {
  XNNPackSettingsBuilder xnnpack_settings_builder(flatbuffer_builder_);
  xnnpack_settings_builder.add_flags(
      tflite::XNNPackFlags_TFLITE_XNNPACK_DELEGATE_FLAG_QS8);
  flatbuffers::Offset<XNNPackSettings> xnnpack_settings =
      xnnpack_settings_builder.Finish();
  TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder_);
  tflite_settings_builder.add_xnnpack_settings(xnnpack_settings);
  flatbuffers::Offset<TFLiteSettings> tflite_settings =
      tflite_settings_builder.Finish();
  flatbuffer_builder_.Finish(tflite_settings);
  tflite_settings_ = flatbuffers::GetRoot<TFLiteSettings>(
      flatbuffer_builder_.GetBufferPointer());
  delegate_plugin_ = delegates::DelegatePluginRegistry::CreateByName(
      "XNNPackPlugin", *tflite_settings_);

  delegates::TfLiteDelegatePtr delegate = delegate_plugin_->Create();
  int flags = TfLiteXNNPackDelegateGetFlags(delegate.get());
  EXPECT_EQ(flags, tflite::XNNPackFlags_TFLITE_XNNPACK_DELEGATE_FLAG_QS8);
}

// Settings flags to XNNPackFlags_TFLITE_XNNPACK_DELEGATE_NO_FLAGS (zero)
// causes flags to be set to their default values, not zero.
// This is potentially confusing behaviour, but we can't distinguish
// the case when flags isn't set from the case when flags is set to zero.
TEST_F(XnnpackPluginTest, UsesDefaultFlagsWhenZero) {
  XNNPackSettingsBuilder xnnpack_settings_builder(flatbuffer_builder_);
  xnnpack_settings_builder.add_flags(
      tflite::XNNPackFlags_TFLITE_XNNPACK_DELEGATE_NO_FLAGS);
  flatbuffers::Offset<XNNPackSettings> xnnpack_settings =
      xnnpack_settings_builder.Finish();
  TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder_);
  tflite_settings_builder.add_xnnpack_settings(xnnpack_settings);
  flatbuffers::Offset<TFLiteSettings> tflite_settings =
      tflite_settings_builder.Finish();
  flatbuffer_builder_.Finish(tflite_settings);
  tflite_settings_ = flatbuffers::GetRoot<TFLiteSettings>(
      flatbuffer_builder_.GetBufferPointer());
  delegate_plugin_ = delegates::DelegatePluginRegistry::CreateByName(
      "XNNPackPlugin", *tflite_settings_);

  delegates::TfLiteDelegatePtr delegate = delegate_plugin_->Create();
  int flags = TfLiteXNNPackDelegateGetFlags(delegate.get());
  EXPECT_EQ(flags, TfLiteXNNPackDelegateOptionsDefault().flags);
}

TEST_F(XnnpackPluginTest, DoesNotSetWeightCacheFilePathByDefault) {
  delegates::TfLiteDelegatePtr delegate = delegate_plugin_->Create();
  const TfLiteXNNPackDelegateOptions *options =
      TfLiteXNNPackDelegateGetOptions(delegate.get());
  EXPECT_EQ(options->weight_cache_file_path, nullptr);
}

TEST_F(XnnpackPluginTest, HonoursWeightCacheFilePathSetting) {
  const char *const kWeightCachePath = "/tmp/wcfp";
  const auto weight_cache_file_path_string =
      flatbuffer_builder_.CreateString(kWeightCachePath);
  XNNPackSettingsBuilder xnnpack_settings_builder(flatbuffer_builder_);
  xnnpack_settings_builder.add_weight_cache_file_path(
      weight_cache_file_path_string);
  flatbuffers::Offset<XNNPackSettings> xnnpack_settings =
      xnnpack_settings_builder.Finish();
  TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder_);
  tflite_settings_builder.add_xnnpack_settings(xnnpack_settings);
  flatbuffers::Offset<TFLiteSettings> tflite_settings =
      tflite_settings_builder.Finish();
  flatbuffer_builder_.Finish(tflite_settings);
  tflite_settings_ = flatbuffers::GetRoot<TFLiteSettings>(
      flatbuffer_builder_.GetBufferPointer());
  delegate_plugin_ = delegates::DelegatePluginRegistry::CreateByName(
      "XNNPackPlugin", *tflite_settings_);

  delegates::TfLiteDelegatePtr delegate = delegate_plugin_->Create();
  const TfLiteXNNPackDelegateOptions *options =
      TfLiteXNNPackDelegateGetOptions(delegate.get());
  EXPECT_STREQ(options->weight_cache_file_path, kWeightCachePath);
}

}  // namespace tflite
