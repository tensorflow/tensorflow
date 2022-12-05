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
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/delegate_loader.h"

#include <cstddef>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/sample_stable_delegate.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"

namespace {

using tflite::TFLiteSettings;
using tflite::TFLiteSettingsBuilder;
using tflite::delegates::utils::LoadDelegateFromSharedLibrary;

TEST(TfLiteDelegateLoaderUtilsTest, Simple) {
  const TfLiteStableDelegate* stable_delegate_handle =
      LoadDelegateFromSharedLibrary(
          "tensorflow/lite/delegates/utils/experimental/"
          "sample_stable_delegate/libtensorflowlite_sample_stable_delegate.so");

  EXPECT_NE(stable_delegate_handle, nullptr);
  EXPECT_STREQ(stable_delegate_handle->delegate_abi_version,
               TFL_STABLE_DELEGATE_ABI_VERSION);
  EXPECT_STREQ(stable_delegate_handle->delegate_name,
               tflite::example::kSampleStableDelegateName);
  EXPECT_STREQ(stable_delegate_handle->delegate_version,
               tflite::example::kSampleStableDelegateVersion);
  EXPECT_NE(stable_delegate_handle->delegate_plugin, nullptr);

  // Builds TFLiteSettings flatbuffer and passes into delegate plugin create
  // method.
  flatbuffers::FlatBufferBuilder flatbuffer_builder;
  TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder);
  flatbuffers::Offset<TFLiteSettings> tflite_settings =
      tflite_settings_builder.Finish();
  flatbuffer_builder.Finish(tflite_settings);
  const TFLiteSettings* settings = flatbuffers::GetRoot<TFLiteSettings>(
      flatbuffer_builder.GetBufferPointer());
  auto delegate = stable_delegate_handle->delegate_plugin->create(settings);

  EXPECT_NE(delegate, nullptr);
  EXPECT_EQ(
      stable_delegate_handle->delegate_plugin->get_delegate_errno(delegate), 0);
  stable_delegate_handle->delegate_plugin->destroy(delegate);
}

TEST(TfLiteDelegateLoaderUtilsTest, WrongSymbolReturnsNullptr) {
  const TfLiteStableDelegate* stable_delegate_handle =
      LoadDelegateFromSharedLibrary(
          "tensorflow/lite/delegates/utils/experimental/"
          "sample_stable_delegate/libtensorflowlite_sample_stable_delegate.so",
          "NOT_REAL_SYMBOL");
  EXPECT_EQ(stable_delegate_handle, nullptr);
}

TEST(TfLiteDelegateLoaderUtilsTest, MissingLibReturnsNullptr) {
  const TfLiteStableDelegate* stable_delegate_handle =
      LoadDelegateFromSharedLibrary("not_real_delegate.so");
  EXPECT_EQ(stable_delegate_handle, nullptr);
}

}  // namespace
