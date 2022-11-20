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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h"

#include <algorithm>
#include <string>
#include <thread>  // NOLINT - only production use is on Android, where std::thread is allowed

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"

namespace tflite {
namespace acceleration {
namespace {

std::string GetTemporaryDirectory() {
#ifdef __ANDROID__
  return "/data/local/tmp";
#else
  if (getenv("TEST_TMPDIR")) {
    return getenv("TEST_TMPDIR");
  }
  if (getenv("TEMP")) {
    return getenv("TEMP");
  }
  return ".";
#endif
}

std::string GetStoragePath() {
  std::string path = GetTemporaryDirectory() + "/storage.fb";
  unlink(path.c_str());
  return path;
}

TEST(FlatbufferStorageTest, AppendAndReadOneItem) {
  std::string path = GetStoragePath();
  flatbuffers::FlatBufferBuilder fbb;
  flatbuffers::Offset<BenchmarkEvent> o =
      CreateBenchmarkEvent(fbb, 0, BenchmarkEventType_START);

  FlatbufferStorage<BenchmarkEvent> storage(path);
  EXPECT_EQ(storage.Read(), kMinibenchmarkSuccess);
  EXPECT_EQ(storage.Count(), 0);

  EXPECT_EQ(storage.Append(&fbb, o), kMinibenchmarkSuccess);
  ASSERT_EQ(storage.Count(), 1);
  EXPECT_EQ(storage.Get(0)->event_type(), BenchmarkEventType_START);

  storage = FlatbufferStorage<BenchmarkEvent>(path);
  EXPECT_EQ(storage.Read(), kMinibenchmarkSuccess);
  ASSERT_EQ(storage.Count(), 1);
  EXPECT_EQ(storage.Get(0)->event_type(), BenchmarkEventType_START);
}

TEST(FlatbufferStorageTest, AppendAndReadThreeItems) {
  std::string path = GetStoragePath();
  FlatbufferStorage<BenchmarkEvent> storage(path);
  EXPECT_EQ(storage.Read(), kMinibenchmarkSuccess);
  EXPECT_EQ(storage.Count(), 0);

  for (auto event : {BenchmarkEventType_START, BenchmarkEventType_ERROR,
                     BenchmarkEventType_END}) {
    flatbuffers::FlatBufferBuilder fbb;
    flatbuffers::Offset<BenchmarkEvent> object =
        CreateBenchmarkEvent(fbb, 0, event);
    EXPECT_EQ(storage.Append(&fbb, object), kMinibenchmarkSuccess);
  }

  ASSERT_EQ(storage.Count(), 3);
  EXPECT_EQ(storage.Get(0)->event_type(), BenchmarkEventType_START);
  EXPECT_EQ(storage.Get(1)->event_type(), BenchmarkEventType_ERROR);
  EXPECT_EQ(storage.Get(2)->event_type(), BenchmarkEventType_END);

  storage = FlatbufferStorage<BenchmarkEvent>(path);
  EXPECT_EQ(storage.Read(), kMinibenchmarkSuccess);
  ASSERT_EQ(storage.Count(), 3);
  EXPECT_EQ(storage.Get(0)->event_type(), BenchmarkEventType_START);
  EXPECT_EQ(storage.Get(1)->event_type(), BenchmarkEventType_ERROR);
  EXPECT_EQ(storage.Get(2)->event_type(), BenchmarkEventType_END);
}

TEST(FlatbufferStorageTest, PathDoesntExist) {
  std::string path = GetTemporaryDirectory() + "/nosuchdirectory/storage.pb";
  FlatbufferStorage<BenchmarkEvent> storage(path);
  EXPECT_EQ(storage.Read(), kMinibenchmarkCantCreateStorageFile);
}

#ifndef __ANDROID__
// chmod(0444) doesn't block writing on Android.
TEST(FlatbufferStorageTest, WriteFailureResetsStorage) {
  std::string path = GetStoragePath();
  flatbuffers::FlatBufferBuilder fbb;
  flatbuffers::Offset<BenchmarkEvent> o =
      CreateBenchmarkEvent(fbb, 0, BenchmarkEventType_START);

  FlatbufferStorage<BenchmarkEvent> storage(path);
  EXPECT_EQ(storage.Append(&fbb, o), kMinibenchmarkSuccess);
  ASSERT_EQ(storage.Count(), 1);

  chmod(path.c_str(), 0444);
  EXPECT_EQ(storage.Append(&fbb, o),
            kMinibenchmarkFailedToOpenStorageFileForWriting);
  ASSERT_EQ(storage.Count(), 0);
}
#endif  // !__ANDROID__

TEST(FlatbufferStorageTest, Locking) {
  std::string path = GetStoragePath();

  std::vector<std::thread> threads;
  const int kNumThreads = 4;
  const int kIterations = 10;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; i++) {
    threads.push_back(std::thread([path]() {
      for (int j = 0; j < kIterations; j++) {
        FlatbufferStorage<BenchmarkEvent> storage(path);
        flatbuffers::FlatBufferBuilder fbb;
        flatbuffers::Offset<BenchmarkEvent> o =
            CreateBenchmarkEvent(fbb, 0, BenchmarkEventType_START);
        EXPECT_EQ(storage.Append(&fbb, o), kMinibenchmarkSuccess);
      }
    }));
  }
  std::for_each(threads.begin(), threads.end(),
                [](std::thread& t) { t.join(); });
  FlatbufferStorage<BenchmarkEvent> storage(path);
  EXPECT_EQ(storage.Read(), kMinibenchmarkSuccess);
  EXPECT_EQ(storage.Count(), kNumThreads * kIterations);
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite
