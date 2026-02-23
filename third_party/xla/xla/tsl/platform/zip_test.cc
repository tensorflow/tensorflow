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

#include "xla/tsl/platform/zip.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "third_party/libzip/lib/zip.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/path.h"

namespace tsl {
namespace zip {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

TEST(ZipTest, OpenNonExistentFile) {
  EXPECT_THAT(Open("nonexistent.zip"), StatusIs(absl::StatusCode::kInternal));
}

TEST(ZipTest, ReadArchive) {
  const std::string zip_path = io::JoinPath(::testing::TempDir(), "test.zip");
  int error = 0;
  zip_t* archive_writer =
      zip_open(zip_path.c_str(), ZIP_CREATE | ZIP_TRUNCATE, &error);
  ASSERT_NE(archive_writer, nullptr);
  const std::string a_content = "a content";
  zip_source_t* s_a = zip_source_buffer(archive_writer, a_content.data(),
                                        a_content.length(), 0);
  ASSERT_NE(s_a, nullptr);
  ASSERT_GE(zip_file_add(archive_writer, "a.txt", s_a, 0), 0);
  const std::string c_content = "c content";
  zip_source_t* s_c = zip_source_buffer(archive_writer, c_content.data(),
                                        c_content.length(), 0);
  ASSERT_NE(s_c, nullptr);
  ASSERT_GE(zip_file_add(archive_writer, "b/c.txt", s_c, 0), 0);
  ASSERT_EQ(zip_close(archive_writer), 0);

  TF_ASSERT_OK_AND_ASSIGN(auto archive, Open(zip_path));

  EXPECT_THAT(archive->GetEntries(),
              IsOkAndHolds(UnorderedElementsAre(
                  io::JoinPath("/zip", zip_path, "a.txt"),
                  io::JoinPath("/zip", zip_path, "b/c.txt"))));

  EXPECT_THAT(archive->GetContents("a.txt"), IsOkAndHolds("a content"));
  EXPECT_THAT(archive->GetContents("b/c.txt"), IsOkAndHolds("c content"));
  EXPECT_THAT(archive->GetContents("d.txt"),
              StatusIs(absl::StatusCode::kNotFound));

  TF_ASSERT_OK_AND_ASSIGN(auto raf, archive->Open("a.txt"));
  absl::string_view read_result;
  std::vector<char> scratch(9);
  EXPECT_THAT(raf->Read(0, 9, &read_result, scratch.data()), IsOk());
  EXPECT_EQ(read_result, "a content");

  EXPECT_THAT(archive->Open("d.txt"), StatusIs(absl::StatusCode::kNotFound));
}

TEST(ZipTest, ReadArchiveUsingZeroCopyInputStream) {
  const std::string zip_path =
      io::JoinPath(::testing::TempDir(), "test_zerocopy.zip");
  int error = 0;
  zip_t* archive_writer =
      zip_open(zip_path.c_str(), ZIP_CREATE | ZIP_TRUNCATE, &error);
  ASSERT_NE(archive_writer, nullptr);
  const std::string a_content = "a content";
  zip_source_t* s_a = zip_source_buffer(archive_writer, a_content.data(),
                                        a_content.length(), 0);
  ASSERT_NE(s_a, nullptr);
  ASSERT_GE(zip_file_add(archive_writer, "a.txt", s_a, 0), 0);
  const std::string c_content = "c content";
  zip_source_t* s_c = zip_source_buffer(archive_writer, c_content.data(),
                                        c_content.length(), 0);
  ASSERT_NE(s_c, nullptr);
  ASSERT_GE(zip_file_add(archive_writer, "b/c.txt", s_c, 0), 0);
  ASSERT_EQ(zip_close(archive_writer), 0);

  TF_ASSERT_OK_AND_ASSIGN(auto archive, Open(zip_path));

  TF_ASSERT_OK_AND_ASSIGN(auto stream,
                          archive->GetZeroCopyInputStream("a.txt"));
  const void* data;
  int size;
  std::string content;
  while (stream->Next(&data, &size)) {
    content.append(static_cast<const char*>(data), size);
  }
  EXPECT_EQ(content, "a content");

  EXPECT_THAT(archive->GetZeroCopyInputStream("d.txt"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(ZipTest, EmptyArchive) {
  const std::string zip_path = io::JoinPath(::testing::TempDir(), "empty.zip");
  std::unique_ptr<WritableFile> file;
  ASSERT_OK(tsl::Env::Default()->NewWritableFile(zip_path, &file));
  ASSERT_OK(file->Append(
      absl::string_view("PK\x05\x06\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", 22)));
  ASSERT_OK(file->Close());

  TF_ASSERT_OK_AND_ASSIGN(auto archive, Open(zip_path));
  EXPECT_THAT(archive->GetEntries(), IsOkAndHolds(IsEmpty()));
}

}  // namespace zip
}  // namespace tsl
