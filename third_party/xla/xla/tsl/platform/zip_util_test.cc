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

#include "xla/tsl/platform/zip_util.h"

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/libzip/lib/zip.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/zip.h"
#include "tsl/platform/path.h"

namespace tsl {
namespace zip {
namespace {

using ::testing::UnorderedElementsAre;
using ::testing::status::IsOk;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

class ZipUtilTest : public ::testing::Test {
 protected:
  void SetUp() override {
    zip_path_ = tsl::io::JoinPath(::testing::TempDir(), "test_zip_util.zip");
    int err = 0;
    zip_t* za = zip_open(zip_path_.c_str(), ZIP_CREATE | ZIP_TRUNCATE, &err);
    ASSERT_NE(za, nullptr);

    const std::string file1_content = "hello world";
    zip_source_t* s1 =
        zip_source_buffer(za, file1_content.data(), file1_content.size(), 0);
    ASSERT_GE(zip_file_add(za, "file1.txt", s1, ZIP_FL_OVERWRITE), 0);

    const std::string file2_content = "foo bar";
    zip_source_t* s2 =
        zip_source_buffer(za, file2_content.data(), file2_content.size(), 0);
    ASSERT_GE(zip_file_add(za, "dir/file2.txt", s2, ZIP_FL_OVERWRITE), 0);

    const std::string file3_content = "bad content";
    zip_source_t* s3 =
        zip_source_buffer(za, file3_content.data(), file3_content.size(), 0);
    ASSERT_GE(zip_file_add(za, "../file3.txt", s3, ZIP_FL_OVERWRITE), 0);

    zip_source_t* s4 =
        zip_source_buffer(za, file3_content.data(), file3_content.size(), 0);
    ASSERT_GE(zip_file_add(za, "dir/../../file4.txt", s4, ZIP_FL_OVERWRITE), 0);

    ASSERT_EQ(zip_close(za), 0);
  }

  std::string zip_path_;
};

TEST_F(ZipUtilTest, OpenNonExistentFile) {
  EXPECT_THAT(OpenArchiveWithTsl("nonexistent.zip").status(),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(ZipUtilTest, OpenInvalidZipFile) {
  std::string invalid_zip_path =
      tsl::io::JoinPath(::testing::TempDir(), "invalid.zip");
  ASSERT_THAT(tsl::WriteStringToFile(tsl::Env::Default(), invalid_zip_path,
                                     "this is not a zip file"),
              IsOk());
  EXPECT_THAT(OpenArchiveWithTsl(invalid_zip_path).status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ZipUtilTest, GetEntries) {
  auto zip_archive_or = OpenArchiveWithTsl(zip_path_);
  ASSERT_THAT(zip_archive_or.status(), IsOk());
  std::unique_ptr<ZipArchive> zip_archive = *std::move(zip_archive_or);
  EXPECT_THAT(zip_archive->GetEntries(),
              IsOkAndHolds(UnorderedElementsAre("file1.txt", "dir/file2.txt",
                                                "../file3.txt",
                                                "dir/../../file4.txt")));
}

TEST_F(ZipUtilTest, GetContents) {
  auto zip_archive_or = OpenArchiveWithTsl(zip_path_);
  ASSERT_THAT(zip_archive_or.status(), IsOk());
  std::unique_ptr<ZipArchive> zip_archive = *std::move(zip_archive_or);
  EXPECT_THAT(zip_archive->GetContents("file1.txt"),
              IsOkAndHolds("hello world"));
  EXPECT_THAT(zip_archive->GetContents("dir/file2.txt"),
              IsOkAndHolds("foo bar"));
  EXPECT_THAT(zip_archive->GetContents("nonexistent.txt").status(),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(ZipUtilTest, PathTraversal) {
  auto zip_archive_or = OpenArchiveWithTsl(zip_path_);
  ASSERT_THAT(zip_archive_or.status(), IsOk());
  std::unique_ptr<ZipArchive> zip_archive = *std::move(zip_archive_or);
  EXPECT_THAT(zip_archive->GetContents("../file3.txt").status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(zip_archive->Open("../file3.txt").status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(zip_archive->GetZeroCopyInputStream("../file3.txt").status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(zip_archive->GetContents("dir/../../file4.txt").status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(zip_archive->Open("dir/../../file4.txt").status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      zip_archive->GetZeroCopyInputStream("dir/../../file4.txt").status(),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ZipUtilTest, OpenRandomAccessFile) {
  auto zip_archive_or = OpenArchiveWithTsl(zip_path_);
  ASSERT_THAT(zip_archive_or.status(), IsOk());
  std::unique_ptr<ZipArchive> zip_archive = *std::move(zip_archive_or);
  auto raf_or = zip_archive->Open("file1.txt");
  ASSERT_THAT(raf_or.status(), IsOk());
  std::unique_ptr<RandomAccessFile> raf = *std::move(raf_or);
  char scratch[12];
  absl::string_view result;
  ASSERT_THAT(raf->Read(0, result, absl::MakeSpan(scratch, 11)), IsOk());
  EXPECT_EQ(result, "hello world");

  ASSERT_THAT(raf->Read(6, result, absl::MakeSpan(scratch, 5)), IsOk());
  EXPECT_EQ(result, "world");

  // Read past EOF.
  ASSERT_THAT(raf->Read(0, result, absl::MakeSpan(scratch, 12)),
              StatusIs(absl::StatusCode::kOutOfRange));
  EXPECT_EQ(result.size(), 11);
}

TEST_F(ZipUtilTest, GetZeroCopyInputStream) {
  auto zip_archive_or = OpenArchiveWithTsl(zip_path_);
  ASSERT_THAT(zip_archive_or.status(), IsOk());
  std::unique_ptr<ZipArchive> zip_archive = *std::move(zip_archive_or);
  auto stream_or = zip_archive->GetZeroCopyInputStream("dir/file2.txt");
  ASSERT_THAT(stream_or.status(), IsOk());
  std::unique_ptr<google::protobuf::io::ZeroCopyInputStream> stream =
      *std::move(stream_or);
  const void* data;
  int size;
  std::string content;
  while (stream->Next(&data, &size)) {
    content.append(static_cast<const char*>(data), size);
  }
  EXPECT_EQ(content, "foo bar");
  EXPECT_EQ(stream->ByteCount(), 7);
}

}  // namespace
}  // namespace zip
}  // namespace tsl
