/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/xnnpack/mmap_handle.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>

#if defined(_MSC_VER)
#include <io.h>
#define ftruncate64 _chsize_s
#else
#include <unistd.h>
#endif
#if defined(__APPLE__)
#define ftruncate64 ftruncate
#endif

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/xnnpack/weight_cache_test_helpers.h"

namespace tflite::xnnpack {
namespace {

using testing::ElementsAreArray;
using testing::Ge;
using testing::Gt;

TEST(MMapHandleTest, DefaultConstructs) {
  MMapHandle handle;
  EXPECT_FALSE(handle.IsMapped());
  EXPECT_EQ(handle.data(), nullptr);
  EXPECT_EQ(handle.size(), 0);
}

TEST(MMapHandleTest, MapNonExistingFileFails) {
  // This path is unlikely to exist.
  const char* file_path = "sdbgfd";
  MMapHandle handle;
  EXPECT_FALSE(handle.Map(file_path));
}

TEST(MMapHandleTest, MapExistingFileWorks) {
  using std::size;

  const std::string payload = "This is some data in the file.";

  TempFileDesc tmp_file;
  ASSERT_TRUE(tmp_file.IsValid());
  ASSERT_TRUE(tmp_file.Write(payload.c_str(), size(payload)));
  tmp_file.Close();

  MMapHandle handle;
  ASSERT_TRUE(handle.Map(tmp_file.GetCPath()));
  EXPECT_TRUE(handle.IsMapped());
  EXPECT_NE(handle.data(), nullptr);
  EXPECT_THAT(handle.size(), Ge(size(payload)));
  EXPECT_THAT(handle, ElementsAreArray(payload));

  handle.UnMap();
  EXPECT_FALSE(handle.IsMapped());
  EXPECT_EQ(handle.data(), nullptr);
  EXPECT_EQ(handle.size(), 0);
}

TEST(MMapHandleTest, MoveConstructs) {
  const std::string payload = "This is some data in the file.";

  TempFileDesc tmp_file;
  ASSERT_TRUE(tmp_file.IsValid());
  ASSERT_TRUE(tmp_file.Write(payload.c_str(), size(payload)));
  tmp_file.Close();

  MMapHandle handle;
  ASSERT_TRUE(handle.Map(tmp_file.GetCPath()));

  MMapHandle handle2(std::move(handle));

  // We are checking that the moved from handle has lost control over the data.
  // NOLINTBEGIN(bugprone-use-after-move)
  EXPECT_FALSE(handle.IsMapped());
  EXPECT_EQ(handle.data(), nullptr);
  EXPECT_EQ(handle.size(), 0);
  // NOLINTEND(bugprone-use-after-move)

  EXPECT_TRUE(handle2.IsMapped());
  EXPECT_NE(handle2.data(), nullptr);
  EXPECT_THAT(handle2.size(), Ge(size(payload)));
  EXPECT_THAT(handle2, ElementsAreArray(payload));
}

TEST(MMapHandleTest, Resize) {
  const std::string payload = "This is some data in the file.";

  TempFileDesc tmp_file;
  ASSERT_TRUE(tmp_file.IsValid());
  ASSERT_TRUE(tmp_file.Write(payload.c_str(), size(payload)));
  tmp_file.Close();

  MMapHandle handle;
  ASSERT_TRUE(handle.Map(tmp_file.GetCPath()));

#if defined(__linux__) || defined(__ANDROID__)
  const size_t kMaxResizeTestCount = 20;
  bool was_resized = true;
  for (size_t i = 0; i < kMaxResizeTestCount && was_resized; ++i) {
    was_resized = handle.Resize(payload.size() * 2);
    EXPECT_TRUE(was_resized || errno == ENOMEM);
  }
#else
  EXPECT_FALSE(handle.Resize(payload.size()));
#endif
}

TEST(MMapHandleTest, MapWithOffset) {
  const std::string payload = "This is some data in the file.";
  const std::string payload2 = "Some other data appended to the the offset.";

  TempFileDesc tmp_file;
  // We want to create a file that is bigger than the mapping granularity to
  // test how alignment behaves.
  const int64_t min_mapping_granularity = [] {
#ifdef _MSC_VER
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    return sys_info.dwAllocationGranularity;
#else
    return getpagesize();
#endif
  }();
  ASSERT_TRUE(tmp_file.IsValid());
  ASSERT_EQ(ftruncate64(tmp_file.Value(), min_mapping_granularity + 1), 0);
  tmp_file.SetPos(0);
  ASSERT_TRUE(tmp_file.Write(payload.c_str(), size(payload)));
  tmp_file.SetPosFromEnd(0);
  ASSERT_THAT(tmp_file.GetPos(), Gt(min_mapping_granularity));
  ASSERT_TRUE(tmp_file.Write(payload2.c_str(), size(payload2)));
  tmp_file.Close();

  MMapHandle handle;
  ASSERT_TRUE(
      handle.Map(tmp_file.GetCPath(), /*offset=*/min_mapping_granularity + 1));
  EXPECT_EQ(handle.size(), size(payload2));
  EXPECT_THAT(std::string((const char*)handle.data(), handle.size()),
              testing::StrEq(payload2));
}

// This test case is geared towards Windows that supports both forward slashes
// and backslashes as path separators.
TEST(MMapHandleTest, MapWorksWithBackslashInPath) {
  const std::string payload = "This is some data in the file.";

  TempFileDesc tmp_file;
  ASSERT_TRUE(tmp_file.IsValid());
  ASSERT_TRUE(tmp_file.Write(payload.c_str(), size(payload)));

  MMapHandle handle;
  ASSERT_TRUE(
      handle.Map(tmp_file, /*offset=*/0, "C:\\\\A\\path\\with\\backslashes"));
}

TEST(MMapHandleTest, MapWorksWithUnspecifiedFilePath) {
  const std::string payload = "This is some data in the file.";

  TempFileDesc tmp_file;
  ASSERT_TRUE(tmp_file.IsValid());
  ASSERT_TRUE(tmp_file.Write(payload.c_str(), size(payload)));

  MMapHandle handle;
  ASSERT_TRUE(handle.Map(tmp_file, /*offset=*/0));
}

// This test case is geared towards Windows that supports both forward slashes
// and backslashes as path separators.
TEST(MMapHandleTest, MapWorksWithForwardSlashInPath) {
  const std::string payload = "This is some data in the file.";

  TempFileDesc tmp_file;
  ASSERT_TRUE(tmp_file.IsValid());
  ASSERT_TRUE(tmp_file.Write(payload.c_str(), size(payload)));

  MMapHandle handle;
  ASSERT_TRUE(
      handle.Map(tmp_file, /*offset=*/0, "C:/A/path/with/forward/slashes"));
}

TEST(MMapHandleTest, ResizeMapWithOffset) {
  const std::string payload = "This is some data in the file.";
  const std::string payload2 = "Some other data appended to the the offset.";
  const std::string payload3 =
      "Yet some other data written after the initial mapping.";

  TempFileDesc tmp_file;
  ASSERT_TRUE(tmp_file.IsValid());
  ASSERT_TRUE(tmp_file.Write(payload.c_str(), size(payload)));
  ASSERT_TRUE(tmp_file.Write(payload2.c_str(), size(payload2)));

  MMapHandle handle;
  ASSERT_TRUE(handle.Map(tmp_file.GetCPath(), /*offset=*/size(payload)));

  ASSERT_TRUE(tmp_file.Write(payload3.c_str(), size(payload3)));
  tmp_file.Close();
#if defined(__linux__) || defined(__ANDROID__)
  bool was_resized = handle.Resize(payload2.size() + payload3.size());
  if (was_resized) {
    EXPECT_THAT(std::string((const char*)handle.data(), handle.size()),
                testing::StrEq(payload2 + payload3));
  } else {
    GTEST_SKIP()
        << "This run did not end up in a resize of the mmaped interval.";
  }
#else
  GTEST_SKIP() << "Resize is not supported for this build.";
#endif
}

TEST(MMapHandleTest, WorksWithHugeFiles) {
#if INTPTR_MAX == INT32_MAX
  GTEST_SKIP()
      << "Files bigger than 2GiB cannot be mapped on 32 bit architectures.";
#endif
  TempFileDesc tmp_file;
  int64_t huge_file_size = 2254857830;  // More than 2 GiB.
  ASSERT_EQ(ftruncate64(tmp_file.Value(), huge_file_size), 0);
  tmp_file.Close();
  MMapHandle handle;
  ASSERT_TRUE(handle.Map(tmp_file.GetCPath(), /*offset=*/0));
}

}  // namespace
}  // namespace tflite::xnnpack
