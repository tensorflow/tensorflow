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

TEST(GpuPluginTest, CachingFieldsMissing) {
  flatbuffers::FlatBufferBuilder fbb;
  tflite::proto::ComputeSettings compute_settings;
  compute_settings.mutable_tflite_settings()->set_delegate(
      tflite::proto::Delegate::GPU);
  const tflite::ComputeSettings* compute_settings_fb =
      tflite::ConvertFromProto(compute_settings, &fbb);

  GpuPlugin gpu_plugin(*compute_settings_fb->tflite_settings());
  EXPECT_TRUE(gpu_plugin.GetCacheDir().empty());
  EXPECT_TRUE(gpu_plugin.GetModelToken().empty());
}

TEST(GpuPluginTest, CompilationCachingFieldsSourcedFromGpuSettings) {
  flatbuffers::FlatBufferBuilder fbb;
  auto gpu_settings_cache_dir = fbb.CreateString("gpu_settings_cache_dir");
  auto gpu_settings_model_token = fbb.CreateString("gpu_settings_model_token");
  GPUSettingsBuilder gpu_settings_builder(fbb);
  gpu_settings_builder.add_cache_directory(gpu_settings_cache_dir);
  gpu_settings_builder.add_model_token(gpu_settings_model_token);
  auto gpu_settings = gpu_settings_builder.Finish();

  TFLiteSettingsBuilder tflite_settings_builder(fbb);
  tflite_settings_builder.add_gpu_settings(gpu_settings);
  auto tflite_settings = tflite_settings_builder.Finish();

  fbb.Finish(tflite_settings);
  auto tflite_settings_root =
      flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer());

  GpuPlugin gpu_plugin(*tflite_settings_root);
  EXPECT_STREQ("gpu_settings_cache_dir", gpu_plugin.GetCacheDir().c_str());
  EXPECT_STREQ("gpu_settings_model_token", gpu_plugin.GetModelToken().c_str());
}

TEST(GpuPluginTest,
     CacheDirFromCompilationSettingsAndModelTokenFromGpuSettings) {
  flatbuffers::FlatBufferBuilder fbb;
  auto gpu_settings_cache_dir = fbb.CreateString("gpu_settings_cache_dir");
  auto gpu_settings_model_token = fbb.CreateString("gpu_settings_model_token");
  GPUSettingsBuilder gpu_settings_builder(fbb);
  gpu_settings_builder.add_cache_directory(gpu_settings_cache_dir);
  gpu_settings_builder.add_model_token(gpu_settings_model_token);
  auto gpu_settings = gpu_settings_builder.Finish();

  auto compilation_caching_dir =
      fbb.CreateString("top_level_compilation_caching_dir");
  CompilationCachingSettingsBuilder compilation_caching_settings_builder(fbb);
  compilation_caching_settings_builder.add_cache_dir(compilation_caching_dir);
  auto compilation_caching_settings =
      compilation_caching_settings_builder.Finish();

  TFLiteSettingsBuilder tflite_settings_builder(fbb);
  tflite_settings_builder.add_gpu_settings(gpu_settings);
  tflite_settings_builder.add_compilation_caching_settings(
      compilation_caching_settings);
  auto tflite_settings = tflite_settings_builder.Finish();

  fbb.Finish(tflite_settings);
  auto tflite_settings_root =
      flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer());

  GpuPlugin gpu_plugin(*tflite_settings_root);
  EXPECT_STREQ("top_level_compilation_caching_dir",
               gpu_plugin.GetCacheDir().c_str());
  EXPECT_STREQ("gpu_settings_model_token", gpu_plugin.GetModelToken().c_str());
}

TEST(GpuPluginTest,
     ModelTokenFromCompilationSettingsAndCacheDirFromGpuSettings) {
  flatbuffers::FlatBufferBuilder fbb;
  auto gpu_settings_cache_dir = fbb.CreateString("gpu_settings_cache_dir");
  auto gpu_settings_model_token = fbb.CreateString("gpu_settings_model_token");
  GPUSettingsBuilder gpu_settings_builder(fbb);
  gpu_settings_builder.add_cache_directory(gpu_settings_cache_dir);
  gpu_settings_builder.add_model_token(gpu_settings_model_token);
  auto gpu_settings = gpu_settings_builder.Finish();

  auto top_level_compilation_model_token =
      fbb.CreateString("top_level_compilation_model_token");
  CompilationCachingSettingsBuilder compilation_caching_settings_builder(fbb);
  compilation_caching_settings_builder.add_model_token(
      top_level_compilation_model_token);
  auto compilation_caching_settings =
      compilation_caching_settings_builder.Finish();

  TFLiteSettingsBuilder tflite_settings_builder(fbb);
  tflite_settings_builder.add_gpu_settings(gpu_settings);
  tflite_settings_builder.add_compilation_caching_settings(
      compilation_caching_settings);
  auto tflite_settings = tflite_settings_builder.Finish();

  fbb.Finish(tflite_settings);
  auto tflite_settings_root =
      flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer());

  GpuPlugin gpu_plugin(*tflite_settings_root);
  EXPECT_STREQ("gpu_settings_cache_dir", gpu_plugin.GetCacheDir().c_str());
  EXPECT_STREQ("top_level_compilation_model_token",
               gpu_plugin.GetModelToken().c_str());
}

TEST(GpuPluginTest, CacheDirAndModelTokenCompilationSettings) {
  flatbuffers::FlatBufferBuilder fbb;
  auto gpu_settings_cache_dir = fbb.CreateString("gpu_settings_cache_dir");
  auto gpu_settings_model_token = fbb.CreateString("gpu_settings_model_token");
  GPUSettingsBuilder gpu_settings_builder(fbb);
  gpu_settings_builder.add_cache_directory(gpu_settings_cache_dir);
  gpu_settings_builder.add_model_token(gpu_settings_model_token);
  auto gpu_settings = gpu_settings_builder.Finish();

  auto top_level_compilation_model_token =
      fbb.CreateString("top_level_compilation_model_token");
  auto compilation_caching_dir =
      fbb.CreateString("top_level_compilation_caching_dir");
  CompilationCachingSettingsBuilder compilation_caching_settings_builder(fbb);
  compilation_caching_settings_builder.add_cache_dir(compilation_caching_dir);
  compilation_caching_settings_builder.add_model_token(
      top_level_compilation_model_token);
  auto compilation_caching_settings =
      compilation_caching_settings_builder.Finish();

  TFLiteSettingsBuilder tflite_settings_builder(fbb);
  tflite_settings_builder.add_gpu_settings(gpu_settings);
  tflite_settings_builder.add_compilation_caching_settings(
      compilation_caching_settings);
  auto tflite_settings = tflite_settings_builder.Finish();

  fbb.Finish(tflite_settings);
  auto tflite_settings_root =
      flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer());

  GpuPlugin gpu_plugin(*tflite_settings_root);
  EXPECT_STREQ("top_level_compilation_caching_dir",
               gpu_plugin.GetCacheDir().c_str());
  EXPECT_STREQ("top_level_compilation_model_token",
               gpu_plugin.GetModelToken().c_str());
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
