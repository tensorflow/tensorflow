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
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "pthreadpool.h"  // from @pthreadpool
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/tool_params.h"

namespace tflite {
namespace tools {
namespace {

static constexpr char kTestSettingsSrcDir[] =
    "tensorflow/lite/tools/delegates/experimental/stable_delegate/";
static constexpr char kGoodStableDelegateSettings[] =
    "test_sample_stable_delegate_settings.json";
static constexpr char kGoodXNNPackDelegateSettings[] =
    "test_stable_xnnpack_settings.json";
static constexpr char kBadMissingFile[] = "missing.json";
static constexpr char kBadInvalidSettings[] = "test_invalid_settings.json";
static constexpr char kBadMissingStableDelegateSettings[] =
    "test_missing_stable_delegate_settings.json";
static constexpr char kBadMissingDelegatePathSettings[] =
    "test_missing_delegate_path_settings.json";

std::vector<ProvidedDelegateList::ProvidedDelegate> CreateDelegates(
    const std::string& settings_file_path) {
  ToolParams params;
  ProvidedDelegateList providers(&params);
  providers.AddAllDelegateParams();
  params.Set<std::string>("stable_delegate_settings_file", settings_file_path,
                          /*position=*/1);

  return providers.CreateAllRankedDelegates();
}

TEST(StableAbiDelegateProviderTest, CreateDelegate) {
  auto delegates = CreateDelegates(std::string(kTestSettingsSrcDir) +
                                   kGoodStableDelegateSettings);

  // Only the stable ABI delegate is registered.
  EXPECT_EQ(1, delegates.size());
  EXPECT_EQ("STABLE_DELEGATE", delegates.front().provider->GetName());
  EXPECT_NE(nullptr, delegates.front().delegate.get());
  EXPECT_EQ(1, delegates.front().rank);
}

TEST(StableAbiDelegateProviderTest, CreateDelegateWithStableXNNPack) {
  auto delegates = CreateDelegates(std::string(kTestSettingsSrcDir) +
                                   kGoodXNNPackDelegateSettings);

  EXPECT_EQ(1, delegates.size());
  EXPECT_EQ("STABLE_DELEGATE", delegates.front().provider->GetName());
  EXPECT_NE(nullptr, delegates.front().delegate.get());
  EXPECT_EQ(1, delegates.front().rank);
  pthreadpool_t threadpool = static_cast<pthreadpool_t>(
      TfLiteXNNPackDelegateGetThreadPool(delegates.front().delegate.get()));
  EXPECT_EQ(5, pthreadpool_get_threads_count(threadpool));
}

TEST(StableAbiDelegateProviderTest, CreateDelegateFailedWithInvalidSettings) {
  std::vector<std::string> invalid_settings_names = {
      kBadMissingFile, kBadInvalidSettings, kBadMissingStableDelegateSettings,
      kBadMissingDelegatePathSettings};

  for (const std::string& name : invalid_settings_names) {
    auto delegates = CreateDelegates(std::string(kTestSettingsSrcDir) + name);

    EXPECT_EQ(0, delegates.size());
  }
}

TEST(StableAbiDelegateProviderTest, CreateDelegateFailedWithBlankSettingsPath) {
  auto delegates = CreateDelegates("");

  EXPECT_EQ(0, delegates.size());
}

}  // namespace
}  // namespace tools
}  // namespace tflite
