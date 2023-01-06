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
// XNNPACK Delegate.

#include "tensorflow/lite/core/experimental/acceleration/configuration/c/xnnpack_plugin.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "pthreadpool.h"  // from @pthreadpool
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"

namespace tflite {

class XnnpackTest : public testing::Test {
 public:
  static constexpr int kNumThreadsForTest = 7;
  void SetUp() override {
    // Construct a FlatBuffer that contains
    // TFLiteSettings { XNNPackSettings { num_threads: kNumThreadsForTest } }.
    XNNPackSettingsBuilder xnnpack_settings_builder(flatbuffer_builder_);
    xnnpack_settings_builder.add_num_threads(kNumThreadsForTest);
    flatbuffers::Offset<XNNPackSettings> xnnpack_settings =
        xnnpack_settings_builder.Finish();
    TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder_);
    tflite_settings_builder.add_xnnpack_settings(xnnpack_settings);
    flatbuffers::Offset<TFLiteSettings> tflite_settings =
        tflite_settings_builder.Finish();
    flatbuffer_builder_.Finish(tflite_settings);
    settings_ = flatbuffers::GetRoot<TFLiteSettings>(
        flatbuffer_builder_.GetBufferPointer());
  }
  ~XnnpackTest() override {}

 protected:
  // settings_ points into storage owned by flatbuffer_builder_.
  flatbuffers::FlatBufferBuilder flatbuffer_builder_;
  const TFLiteSettings *settings_;
};

constexpr int XnnpackTest::kNumThreadsForTest;

TEST_F(XnnpackTest, CanCreateAndDestroyDelegate) {
  TfLiteDelegate *delegate =
      TfLiteXnnpackDelegatePluginCApi()->create(settings_);
  EXPECT_NE(delegate, nullptr);
  TfLiteXnnpackDelegatePluginCApi()->destroy(delegate);
}

TEST_F(XnnpackTest, CanGetDelegateErrno) {
  TfLiteDelegate *delegate =
      TfLiteXnnpackDelegatePluginCApi()->create(settings_);
  int error_number =
      TfLiteXnnpackDelegatePluginCApi()->get_delegate_errno(delegate);
  EXPECT_EQ(error_number, 0);
  TfLiteXnnpackDelegatePluginCApi()->destroy(delegate);
}

TEST_F(XnnpackTest, SetsCorrectThreadCount) {
  TfLiteDelegate *delegate =
      TfLiteXnnpackDelegatePluginCApi()->create(settings_);
  pthreadpool_t threadpool =
      static_cast<pthreadpool_t>(TfLiteXNNPackDelegateGetThreadPool(delegate));
  int thread_count = pthreadpool_get_threads_count(threadpool);
  EXPECT_EQ(thread_count, kNumThreadsForTest);
  TfLiteXnnpackDelegatePluginCApi()->destroy(delegate);
}
}  // namespace tflite
