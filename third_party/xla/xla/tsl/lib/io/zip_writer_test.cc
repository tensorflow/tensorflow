/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/tsl/lib/io/zip_writer.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/test.h"

namespace tsl {
namespace io {
namespace {

using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

constexpr absl::string_view kZipFileMagic = "\x50\x4b\x03\x04";
constexpr absl::string_view kEndOfCentralDirectoryRecordMagic =
    "\x50\x4b\x05\x06";

std::string GetTestTempDir() { return testing::TmpDir(); }

TEST(ZipWriterTest, DestructorSafetyForEmptyWriter) {
  std::string path = absl::StrCat(GetTestTempDir(), "/empty_safety.zip");
  std::unique_ptr<WritableFile> file;
  ASSERT_OK(Env::Default()->NewWritableFile(path, &file));

  {
    ASSERT_OK_AND_ASSIGN(ZipWriter writer, ZipWriter::Create(std::move(file)));
    // Goes out of scope without calling Finish().
    // Destructor should run cleanly and attempt auto-finish.
  }
}

TEST(ZipWriterTest, DestructorSafetyForMovedFromWriter) {
  std::string path = absl::StrCat(GetTestTempDir(), "/move_safety.zip");
  std::unique_ptr<WritableFile> file;
  ASSERT_OK(Env::Default()->NewWritableFile(path, &file));

  {
    ASSERT_OK_AND_ASSIGN(ZipWriter writer, ZipWriter::Create(std::move(file)));
    ZipWriter moved_writer = std::move(writer);
    // writer is now in a moved-from state (null file_, finished_ is false).
    // Both writer and moved_writer will go out of scope.
    // Destructor must not crash on the moved-from writer.
  }
}

TEST(ZipWriterTest, AddEmptyFileSuccess) {
  std::string path = absl::StrCat(GetTestTempDir(), "/empty_file.zip");
  std::unique_ptr<WritableFile> file;
  ASSERT_OK(Env::Default()->NewWritableFile(path, &file));

  ASSERT_OK_AND_ASSIGN(ZipWriter writer, ZipWriter::Create(std::move(file)));
  EXPECT_OK(writer.AddFile("empty.txt", ""));
  EXPECT_OK(std::move(writer).Finish());
}

TEST(ZipWriterTest, AddSingleFileSuccess) {
  std::string path = absl::StrCat(GetTestTempDir(), "/single_file.zip");
  std::unique_ptr<WritableFile> file;
  ASSERT_OK(Env::Default()->NewWritableFile(path, &file));

  ASSERT_OK_AND_ASSIGN(ZipWriter writer, ZipWriter::Create(std::move(file)));
  EXPECT_OK(writer.AddFile("hello.txt", "Hello, world!"));
  EXPECT_OK(std::move(writer).Finish());
}

TEST(ZipWriterTest, FileCountLimitEnforced) {
  std::string path = absl::StrCat(GetTestTempDir(), "/limit_test.zip");
  std::unique_ptr<WritableFile> file;
  ASSERT_OK(Env::Default()->NewWritableFile(path, &file));

  ASSERT_OK_AND_ASSIGN(ZipWriter writer, ZipWriter::Create(std::move(file)));

  // 65535 is the max number of files possible to store in a single zip file.
  // We name them dynamically to avoid duplicate names.
  for (int i = 0; i < 65535; ++i) {
    std::string name = absl::StrCat("file_", i, ".txt");
    ASSERT_OK(writer.AddFile(name, ""));
  }

  // Adding the 65,536th file must fail with ResourceExhaustedError.
  EXPECT_THAT(writer.AddFile("overflow.txt", ""),
              absl_testing::StatusIs(
                  absl::StatusCode::kResourceExhausted,
                  HasSubstr("ZIP archive exceeds maximum of 65535 files")));
}

TEST(ZipWriterTest, MoveConstructorDoesNotDoubleFinalize) {
  std::string path =
      absl::StrCat(GetTestTempDir(), "/move_double_finalize.zip");
  std::unique_ptr<WritableFile> file;
  ASSERT_OK(Env::Default()->NewWritableFile(path, &file));

  {
    ASSERT_OK_AND_ASSIGN(ZipWriter writer, ZipWriter::Create(std::move(file)));
    ASSERT_OK(writer.AddFile("test.txt", "content"));

    // Move the writer. The moved-from 'writer' is now marked as finished.
    ZipWriter moved_writer = std::move(writer);

    // Destructing the moved-from `writer` should not crash.
  }

  // At the end of the block, 'moved_writer' is also destroyed, which
  // finalizes the archive once.
  std::string content;
  ASSERT_OK(ReadFileToString(Env::Default(), path, &content));
  EXPECT_TRUE(absl::StartsWith(content, kZipFileMagic));
  EXPECT_TRUE(absl::StrContains(content, kEndOfCentralDirectoryRecordMagic));

  // Verify that EOCD signature occurs exactly once.
  int eocd_count = 0;
  size_t pos = 0;
  while ((pos = content.find(kEndOfCentralDirectoryRecordMagic, pos)) !=
         std::string::npos) {
    eocd_count++;
    pos += kEndOfCentralDirectoryRecordMagic.size();
  }
  EXPECT_EQ(eocd_count, 1);
}

TEST(ZipWriterTest, LargeFilenameFails) {
  std::string path = absl::StrCat(GetTestTempDir(), "/large_filename.zip");
  std::unique_ptr<WritableFile> file;
  ASSERT_OK(Env::Default()->NewWritableFile(path, &file));

  ASSERT_OK_AND_ASSIGN(ZipWriter writer, ZipWriter::Create(std::move(file)));

  std::string long_name(0xFFFF + 1, 'a');
  EXPECT_THAT(writer.AddFile(long_name, ""),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Filename too long")));
}

}  // namespace
}  // namespace io
}  // namespace tsl
