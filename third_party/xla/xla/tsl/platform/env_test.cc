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

#include "xla/tsl/platform/env.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "tsl/platform/null_file_system.h"
#include "tsl/platform/path.h"

namespace tsl {
namespace {

using absl_testing::IsOk;
using ::testing::ElementsAre;

TEST(EnvTest, StartDetachedThread) {
  Env* env = Env::Default();
  const int num_threads = 10;
  absl::BlockingCounter counter(num_threads);

  ThreadOptions thread_options;
  for (int i = 0; i < num_threads; ++i) {
    env->StartDetachedThread(thread_options, "MyDetachedThread", [&]() {
      absl::SleepFor(absl::Milliseconds(50));
      counter.DecrementCount();
    });
  }

  counter.Wait();
}

TEST(EnvTest, FileOperations) {
  Env* env = Env::Default();
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  std::string file_path = tsl::io::JoinPath(temp_dir.path(), "test.txt");
  EXPECT_THAT(WriteStringToFile(env, file_path, "test1"), IsOk());
  EXPECT_THAT(AppendStringToFile(env, file_path, "test2"), IsOk());
  std::string content;
  EXPECT_THAT(ReadFileToString(env, file_path, &content), IsOk());
  EXPECT_EQ(content, "test1test2");
}

TEST(EnvTest, ReadFileToStringEmptyFile) {
  Env* env = Env::Default();
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  std::string file_path = tsl::io::JoinPath(temp_dir.path(), "empty.txt");
  EXPECT_THAT(WriteStringToFile(env, file_path, ""), IsOk());
  std::string content = "pre_populated_garbage";
  EXPECT_THAT(ReadFileToString(env, file_path, &content), IsOk());
  EXPECT_TRUE(content.empty());
}

TEST(EnvTest, SimpleFileSystemConformance) {
  std::vector<std::string> schemes;
  Env* env = Env::Default();
  ASSERT_OK(env->GetRegisteredFileSystemSchemes(&schemes));

  ASSERT_OK_AND_ASSIGN(testing::TemporaryDirectory temp_dir,
                       testing::TemporaryDirectory::CreateForCurrentTestcase());
  for (const absl::string_view scheme : schemes) {
    std::string contents = "data";
    std::string file_path;
    if (!scheme.empty()) {
      file_path = io::JoinPath(
          absl::StrCat(scheme, "://", temp_dir.path(), "data.txt"));
    } else {
      file_path = io::JoinPath(temp_dir.path(), "data.txt");
    }
    EXPECT_OK(WriteStringToFile(env, file_path, contents));
    EXPECT_OK(WriteStringToFile(env, file_path, contents));
    std::string content;
    EXPECT_OK(ReadFileToString(env, file_path, &content));
    EXPECT_EQ(content, contents) << file_path;
  }
}

TEST(EnvTest, RenameFile) {
  Env* env = Env::Default();
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  std::string src_path = tsl::io::JoinPath(temp_dir.path(), "src.txt");
  std::string target_path = tsl::io::JoinPath(temp_dir.path(), "target.txt");
  EXPECT_THAT(WriteStringToFile(env, src_path, "source content"), IsOk());
  EXPECT_THAT(env->FileExists(src_path), IsOk());

  EXPECT_THAT(WriteStringToFile(env, target_path, "target content"), IsOk());
  EXPECT_THAT(env->RenameFile(src_path, target_path, /*overwrite=*/true),
              IsOk());
  EXPECT_TRUE(absl::IsNotFound(env->FileExists(src_path)));
  std::string target_content;
  EXPECT_THAT(ReadFileToString(env, target_path, &target_content), IsOk());
  EXPECT_EQ(target_content, "source content");
}

TEST(EnvTest, GetMatchingPaths) {
  Env* env = Env::Default();
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  std::string dir_a = tsl::io::JoinPath(temp_dir.path(), "dir_a");
  std::string dir_b = tsl::io::JoinPath(temp_dir.path(), "dir_b");
  ASSERT_THAT(env->RecursivelyCreateDir(dir_a), IsOk());
  ASSERT_THAT(env->RecursivelyCreateDir(dir_b), IsOk());

  std::string file_1 = tsl::io::JoinPath(dir_a, "file_1.txt");
  std::string file_2 = tsl::io::JoinPath(dir_b, "file_2.txt");
  ASSERT_THAT(WriteStringToFile(env, file_1, "1"), IsOk());
  ASSERT_THAT(WriteStringToFile(env, file_2, "2"), IsOk());

  std::string pattern =
      tsl::io::JoinPath(temp_dir.path(), "dir_*", "file_*.txt");
  absl::string_view pattern_sv = pattern;
  for (bool multithreaded : {false, true}) {
    std::vector<std::string> results;
    EXPECT_THAT(env->GetMatchingPaths(pattern, &results, multithreaded),
                IsOk());
    EXPECT_THAT(results, ElementsAre(file_1, file_2));

    results.clear();
    EXPECT_THAT(env->GetMatchingPaths(pattern_sv, &results, multithreaded),
                IsOk());
    EXPECT_THAT(results, ElementsAre(file_1, file_2));
  }
}

// Returns the matches in reverse lexicographic order.
class ReverseOrderFileSystem : public NullFileSystem {
 public:
  absl::Status GetMatchingPaths(const std::string& pattern,
                                std::vector<std::string>* results) override {
    *results = {"c", "b", "a"};
    return absl::OkStatus();
  }
};

// Uses `fs` for all paths.
class SingleFileSystemEnv : public EnvWrapper {
 public:
  explicit SingleFileSystemEnv(FileSystem* fs)
      : EnvWrapper(Env::Default()), fs_(fs) {}

  absl::Status GetFileSystemForFile(absl::string_view fname,
                                    FileSystem** result) override {
    *result = fs_;
    return absl::OkStatus();
  }

 private:
  FileSystem* fs_;
};

TEST(EnvTest, GetMatchingPathsSortsResultsOfAnyFileSystem) {
  ReverseOrderFileSystem fs;
  SingleFileSystemEnv env(&fs);

  std::vector<std::string> results;
  ASSERT_OK(env.GetMatchingPaths("*", &results));
  EXPECT_THAT(results, ElementsAre("a", "b", "c"));
}

#if defined(__linux__)
TEST(EnvTest, ReadFileToStringAllowsShortReadsForSystemFiles) {
  std::string content;
  absl::Status s = ReadFileToString(
      Env::Default(), "/sys/devices/system/cpu/kernel_max", &content);
  if (absl::IsNotFound(s)) {
    GTEST_SKIP() << "/sys/devices/system/cpu/kernel_max not available: " << s;
  }
  EXPECT_OK(s);
  EXPECT_FALSE(content.empty());
}
#endif

}  // namespace
}  // namespace tsl
