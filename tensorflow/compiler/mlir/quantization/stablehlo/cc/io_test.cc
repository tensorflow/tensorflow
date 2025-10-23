/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/io.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/types.h"

namespace stablehlo::quantization::io {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

// A test-only derived class of `tsl::Env` which is broken. Used to cause
// failure for the `CreateTmpDir` function. Each of the overridden member
// functions implements a dummy functionality just to be able to create an
// instance of this class.
class TestEnvBrokenFileSystem : public tsl::Env {
 public:
  TestEnvBrokenFileSystem() = default;

  bool MatchPath(const tsl::string& path, const tsl::string& pattern) override {
    return false;
  }

  void SleepForMicroseconds(int64_t micros) override {}

  tsl::string GetRunfilesDir() override { return tsl::string("dummy_path"); }

  int64_t GetCurrentThreadId() override { return 0; }

  tsl::Thread* StartThread(const tsl::ThreadOptions& thread_options,
                           const tsl::string& name,
                           absl::AnyInvocable<void()> fn) override {
    return nullptr;
  }

  bool GetCurrentThreadName(tsl::string* name) override { return false; }

  void SchedClosure(absl::AnyInvocable<void()> closure) override {}

  void SchedClosureAfter(int64_t micros,
                         absl::AnyInvocable<void()> closure) override {}

  absl::Status LoadDynamicLibrary(const char* library_filename,
                                  void** handle) override {
    return absl::OkStatus();
  }

  absl::Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                                    void** symbol) override {
    return absl::OkStatus();
  }

  tsl::string FormatLibraryFileName(const tsl::string& name,
                                    const tsl::string& version) override {
    return tsl::string("dummy_path");
  }

  // This is the part that would break the `CreateTmpDir` function because it
  // fails to provide a valid file system.
  absl::Status GetFileSystemForFile(const std::string& fname,
                                    tsl::FileSystem** result) override {
    return absl::InternalError("Broken file system");
  }

 private:
  void GetLocalTempDirectories(std::vector<tsl::string>* list) override {
    list->push_back("/tmp");
  }
};

// Represents an environment with broken file system and no available local tmp
// directories.
class TestEnvBrokenFileSystemAndNoLocalTempDirs
    : public TestEnvBrokenFileSystem {
 private:
  // This is the part that essentially breaks the `GetLocalTmpFileName` function
  // because it doesn't provide any available temp dirs.
  void GetLocalTempDirectories(std::vector<tsl::string>* list) override {}
};

TEST(IoTest, GetLocalTmpFileNameGivesValidFileName) {
  absl::StatusOr<std::string> tmp_file_name = GetLocalTmpFileName();

  ASSERT_THAT(tmp_file_name, absl_testing::IsOk());
  EXPECT_THAT(*tmp_file_name, Not(IsEmpty()));
}

TEST(IoTest, GetLocalTmpFileNameWhenNoTempDirsReturnsInternalError) {
  TestEnvBrokenFileSystemAndNoLocalTempDirs broken_env;
  absl::StatusOr<std::string> tmp_file_name = GetLocalTmpFileName(&broken_env);

  EXPECT_THAT(tmp_file_name, absl_testing::StatusIs(
                                 absl::StatusCode::kInternal,
                                 HasSubstr("Failed to create tmp file name")));
}

TEST(IoTest, CreateTmpDirReturnsValidTmpPath) {
  absl::StatusOr<std::string> tmp_dir = CreateTmpDir();

  ASSERT_THAT(tmp_dir, absl_testing::IsOk());

  auto* const env = tsl::Env::Default();
  EXPECT_THAT(env->FileExists(*tmp_dir), absl_testing::IsOk());
}

TEST(IoTest, CreateTmpDirWhenInvalidPathReturnsInternalError) {
  TestEnvBrokenFileSystem test_env{};
  absl::StatusOr<std::string> tmp_dir = CreateTmpDir(&test_env);

  EXPECT_THAT(tmp_dir,
              absl_testing::StatusIs(absl::StatusCode::kInternal,
                                     HasSubstr("Failed to create tmp dir")));
}

TEST(IoTest, WriteStringToFile) {
  const std::string dst_file_path =
      absl::StrCat(testing::TempDir(), "/tmp_file");

  const absl::Status write_status =
      WriteStringToFile(dst_file_path, "test_string");
  ASSERT_THAT(write_status, absl_testing::IsOk());

  auto* const env = tsl::Env::Default();
  ASSERT_THAT(env->FileExists(dst_file_path), absl_testing::IsOk());

  std::string data{};
  ASSERT_THAT(tsl::ReadFileToString(env, dst_file_path, &data),
              absl_testing::IsOk());

  EXPECT_THAT(data, Eq("test_string"));
}

TEST(IoTest, ReadFileToString) {
  // Prepare a temp file and write some string to it.
  const std::string src_file_path =
      absl::StrCat(testing::TempDir(), "/tmp_file");

  {
    std::ofstream ofs(src_file_path);
    ofs << "test_string";
  }

  // Test that the contents match.
  const absl::StatusOr<std::string> read_status =
      ReadFileToString(src_file_path);
  ASSERT_THAT(read_status, absl_testing::IsOk());
  EXPECT_THAT(*read_status, Eq("test_string"));
}

TEST(IoTest, ListChildrenInDirectory) {
  absl::StatusOr<std::string> tmp_dir = CreateTmpDir();

  ASSERT_THAT(tmp_dir, absl_testing::IsOk());

  auto* const env = tsl::Env::Default();
  EXPECT_THAT(env->FileExists(*tmp_dir), absl_testing::IsOk());

  ASSERT_THAT(
      WriteStringToFile(absl::StrCat(*tmp_dir, "/tmp_file1"), "test_string"),
      absl_testing::IsOk());
  ASSERT_THAT(
      WriteStringToFile(absl::StrCat(*tmp_dir, "/tmp_file2"), "test_string"),
      absl_testing::IsOk());
  ASSERT_THAT(env->RecursivelyCreateDir(absl::StrCat(*tmp_dir, "/subdir")),
              absl_testing::IsOk());

  absl::StatusOr<std::vector<std::string>> children = ListDirectory(*tmp_dir);
  EXPECT_THAT(children, absl_testing::IsOk());
  EXPECT_THAT(children.value(), SizeIs(3));
  EXPECT_THAT(children.value(),
              UnorderedElementsAre("subdir", "tmp_file1", "tmp_file2"));
}

}  // namespace
}  // namespace stablehlo::quantization::io
