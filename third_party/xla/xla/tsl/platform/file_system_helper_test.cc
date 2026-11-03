/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/tsl/platform/file_system_helper.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "tsl/platform/null_file_system.h"

namespace tsl {
namespace {

TEST(WritableFileCopyingOutputStreamTest, Write) {
  Env* env = Env::Default();
  std::unique_ptr<WritableFile> file;
  std::string filename = ::testing::TempDir() + "/write_using_stream.txt";

  TF_ASSERT_OK(env->NewWritableFile(filename, &file));

  WritableFileCopyingOutputStream output(file.get());
  output.Write("hello\n", 6);
  TF_ASSERT_OK(file->Close());
  std::string contents;
  TF_ASSERT_OK(ReadFileToString(env, filename, &contents));
  EXPECT_EQ(contents, "hello\n");
}

TEST(RandomAccessFileCopyingInputStreamTest, Read) {
  Env* env = Env::Default();
  std::unique_ptr<WritableFile> file;
  std::string filename = ::testing::TempDir() + "/read_using_stream.txt";

  TF_ASSERT_OK(env->NewWritableFile(filename, &file));
  TF_ASSERT_OK(file->Append("hello\n"));
  TF_ASSERT_OK(file->Close());

  std::unique_ptr<RandomAccessFile> random_file;
  TF_ASSERT_OK(env->NewRandomAccessFile(filename, &random_file));

  RandomAccessFileCopyingInputStream input(random_file.get());
  char buffer[10];
  EXPECT_EQ(input.Read(buffer, 1), 1);
  EXPECT_EQ(absl::string_view(buffer, 1), "h");

  EXPECT_EQ(input.Read(buffer, 8), 5);
  EXPECT_EQ(absl::string_view(buffer, 5), "ello\n");

  EXPECT_EQ(input.Read(buffer, 1), 0);
}

TEST(GetMatchingPathsTest, MatchesSingleAndMultiLevelGlobPatterns) {
  Env* env = Env::Default();
  const std::string base_dir =
      ::testing::TempDir() + "/get_matching_paths_test";
  TF_ASSERT_OK(env->RecursivelyCreateDir(base_dir + "/dir_a/sub"));
  TF_ASSERT_OK(env->RecursivelyCreateDir(base_dir + "/dir_b"));

  const std::string file_a = base_dir + "/file_a.txt";
  const std::string file_sub_1 = base_dir + "/dir_a/file_1.txt";
  const std::string file_sub_2 = base_dir + "/dir_b/file_2.txt";
  TF_ASSERT_OK(WriteStringToFile(env, file_a, "a"));
  TF_ASSERT_OK(WriteStringToFile(env, file_sub_1, "1"));
  TF_ASSERT_OK(WriteStringToFile(env, file_sub_2, "2"));

  FileSystem* fs = nullptr;
  TF_ASSERT_OK(env->GetFileSystemForFile(base_dir, &fs));

  for (bool multithreaded : {false, true}) {
    std::vector<std::string> results;
    TF_ASSERT_OK(
        internal::GetMatchingPaths(fs, env, "", &results, multithreaded));
    EXPECT_TRUE(results.empty());

    TF_ASSERT_OK(
        internal::GetMatchingPaths(fs, env, file_a, &results, multithreaded));
    EXPECT_EQ(results, std::vector<std::string>{file_a});

    TF_ASSERT_OK(internal::GetMatchingPaths(fs, env, base_dir + "/*", &results,
                                            multithreaded));
    EXPECT_EQ(results, (std::vector<std::string>{base_dir + "/dir_a",
                                                 base_dir + "/dir_b", file_a}));

    TF_ASSERT_OK(internal::GetMatchingPaths(
        fs, env, base_dir + "/dir_*/file_*.txt", &results, multithreaded));
    EXPECT_EQ(results, (std::vector<std::string>{file_sub_1, file_sub_2}));
  }
}

// Delegates to `base`, but lists directory entries in reverse lexicographic
// order.
class ReverseListingFileSystem : public NullFileSystem {
 public:
  explicit ReverseListingFileSystem(FileSystem* base) : base_(base) {}

  absl::Status FileExists(absl::string_view fname) override {
    return base_->FileExists(fname);
  }

  absl::Status GetChildren(const std::string& dir,
                           std::vector<std::string>* result) override {
    absl::Status status = base_->GetChildren(dir, result);
    std::sort(result->begin(), result->end(), std::greater<>());
    return status;
  }

  absl::Status IsDirectory(const std::string& fname) override {
    return base_->IsDirectory(fname);
  }

 private:
  FileSystem* base_;
};

TEST(GetMatchingPathsTest, SortsResultsRegardlessOfListingOrder) {
  Env* env = Env::Default();
  const std::string base_dir =
      ::testing::TempDir() + "/get_matching_paths_sort_test";
  TF_ASSERT_OK(env->RecursivelyCreateDir(base_dir + "/dir_a"));
  TF_ASSERT_OK(env->RecursivelyCreateDir(base_dir + "/dir_b"));
  const std::vector<std::string> files = {
      base_dir + "/dir_a/file_1.txt", base_dir + "/dir_a/file_2.txt",
      base_dir + "/dir_b/file_1.txt", base_dir + "/dir_b/file_2.txt"};
  for (const std::string& file : files) {
    TF_ASSERT_OK(WriteStringToFile(env, file, "content"));
  }

  FileSystem* base_fs = nullptr;
  TF_ASSERT_OK(env->GetFileSystemForFile(base_dir, &base_fs));
  ReverseListingFileSystem fs(base_fs);

  for (bool multithreaded : {false, true}) {
    std::vector<std::string> results;
    TF_ASSERT_OK(internal::GetMatchingPaths(&fs, env, base_dir + "/*", &results,
                                            multithreaded));
    EXPECT_EQ(results, (std::vector<std::string>{base_dir + "/dir_a",
                                                 base_dir + "/dir_b"}));

    TF_ASSERT_OK(internal::GetMatchingPaths(&fs, env, base_dir + "/*/*.txt",
                                            &results, multithreaded));
    EXPECT_EQ(results, files);
  }
}

}  // namespace
}  // namespace tsl
