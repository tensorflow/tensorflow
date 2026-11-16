/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/utils/file_utils.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::TFL {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::Eq;
using ::testing::HasSubstr;

std::string TestDir() {
  const auto* const test_info =
      testing::UnitTest::GetInstance()->current_test_info();
  return JoinPath(testing::TempDir(), absl::StrCat(test_info->test_suite_name(),
                                                   ".", test_info->name()));
}

std::string ReadFile(const std::string& path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) return "";
  return (*buffer)->getBuffer().str();
}

TEST(JoinPathTest, Joins) {
  EXPECT_THAT(JoinPath("a", "b"), Eq("a/b"));
  EXPECT_THAT(JoinPath("a/", "b"), Eq("a/b"));
  EXPECT_THAT(JoinPath("a", "/b"), Eq("a/b"));
  EXPECT_THAT(JoinPath("a/", "/b"), Eq("a/b"));
  EXPECT_THAT(JoinPath("/a/b", "c/d"), Eq("/a/b/c/d"));
}

TEST(JoinPathTest, EmptyComponents) {
  EXPECT_THAT(JoinPath("", "b"), Eq("b"));
  EXPECT_THAT(JoinPath("a", ""), Eq("a"));
  EXPECT_THAT(JoinPath("", ""), Eq(""));
}

TEST(FileUtilsTest, CreateDirectoriesCreatesParents) {
  const std::string dir = JoinPath(TestDir(), "x/y/z");
  ASSERT_THAT(CreateDirectories(dir), IsOk());
  EXPECT_TRUE(llvm::sys::fs::is_directory(dir));
  // Creating an existing directory succeeds.
  EXPECT_THAT(CreateDirectories(dir), IsOk());
}

TEST(FileUtilsTest, CreateDirectoriesFailsUnderAFile) {
  const std::string dir = TestDir();
  ASSERT_THAT(CreateDirectories(dir), IsOk());
  const std::string file = JoinPath(dir, "file");
  ASSERT_THAT(WriteStringToFile(file, "x"), IsOk());
  EXPECT_THAT(CreateDirectories(JoinPath(file, "sub")),
              StatusIs(absl::StatusCode::kUnknown,
                       HasSubstr("Failed to create directory")));
}

TEST(FileUtilsTest, WriteStringToFileReplacesContents) {
  const std::string dir = TestDir();
  ASSERT_THAT(CreateDirectories(dir), IsOk());
  const std::string file = JoinPath(dir, "file.txt");
  ASSERT_THAT(WriteStringToFile(file, "first contents"), IsOk());
  EXPECT_THAT(ReadFile(file), Eq("first contents"));
  ASSERT_THAT(WriteStringToFile(file, std::string("a\0b", 3)), IsOk());
  EXPECT_THAT(ReadFile(file), Eq(std::string("a\0b", 3)));
}

TEST(FileUtilsTest, WriteStringToFileFailsInMissingDirectory) {
  EXPECT_THAT(
      WriteStringToFile(JoinPath(TestDir(), "missing/file.txt"), "x"),
      StatusIs(absl::StatusCode::kUnknown, HasSubstr("Failed to open")));
}

TEST(FileUtilsTest, WriteStringToFileReportsWriteErrors) {
  if (!llvm::sys::fs::exists("/dev/full")) {
    GTEST_SKIP() << "/dev/full is not available";
  }
  EXPECT_THAT(WriteStringToFile("/dev/full", "x"),
              StatusIs(absl::StatusCode::kUnknown,
                       HasSubstr("Failed to write '/dev/full'")));
}

TEST(FileUtilsTest, RejectsStdoutPath) {
  EXPECT_THAT(OpenFileForWriting("-"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(WriteStringToFile("-", "x"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(FileUtilsTest, OpenFileForWriting) {
  const std::string dir = TestDir();
  ASSERT_THAT(CreateDirectories(dir), IsOk());
  const std::string file = JoinPath(dir, "stream.txt");
  {
    absl::StatusOr<std::unique_ptr<llvm::raw_fd_ostream>> os =
        OpenFileForWriting(file);
    ASSERT_THAT(os, IsOk());
    **os << "hello " << 42;
  }
  EXPECT_THAT(ReadFile(file), Eq("hello 42"));
}

}  // namespace
}  // namespace mlir::TFL
