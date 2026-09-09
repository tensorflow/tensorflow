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
#include <atomic>
#include <string>
#include <thread>
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

TEST(StableAbiDelegateProviderTest, ConcurrentCreateDelegate) {
  constexpr int kNumThreads = 40;
  constexpr int kNumIterations = 500;
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([t, &start]() {
      while (!start.load()) {
        std::this_thread::yield();
      }
      for (int i = 0; i < kNumIterations; ++i) {
        std::string path;
        if (t % 2 == 0) {
          path = "shared_settings_" + std::to_string(i % 5) + ".json";
        } else {
          path = "unique_settings_" +
                 std::to_string(t * kNumIterations + i) + ".json";
        }
        auto delegates = CreateDelegates(path);
      }
    });
  }

  start.store(true);
  for (auto& t : threads) {
    t.join();
  }
}

}  // namespace
}  // namespace tools
}  // namespace tflite
