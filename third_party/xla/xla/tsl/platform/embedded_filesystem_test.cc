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

#include "xla/tsl/platform/embedded_filesystem.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"

namespace tsl {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Not;

class EmbedFileSystemTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    fs_ = new EmbedFileSystem();
    FileSystem* fs_ptr = fs_;
    QCHECK_OK(Env::Default()->RegisterFileSystem(
        "embed", [fs_ptr]() -> FileSystem* { return fs_ptr; }));
  }

  static void TearDownTestCase() { delete fs_; }

  static inline EmbedFileSystem* fs_ = nullptr;
};

TEST_F(EmbedFileSystemTest, Exists) {
  std::string filename = "embed://exists";
  std::string contents = "";

  EXPECT_THAT(Env::Default()->FileExists(filename), Not(IsOk()));

  ASSERT_OK(fs_->EmbedFile(filename, contents));

  EXPECT_OK(Env::Default()->FileExists(filename));
}

TEST_F(EmbedFileSystemTest, Read) {
  std::string filename = "embed://foo";
  std::string contents = "foo";

  ASSERT_OK(fs_->EmbedFile(filename, contents));

  std::string read_contents;
  ASSERT_OK(ReadFileToString(Env::Default(), filename, &read_contents));
  EXPECT_EQ(read_contents, contents);
}

TEST_F(EmbedFileSystemTest, ReadConformance) {
  std::string filename = "embed://read_conformance";
  std::string contents = "abcdef";

  ASSERT_OK(fs_->EmbedFile(filename, contents));

  std::unique_ptr<RandomAccessFile> f;
  ASSERT_OK(fs_->NewRandomAccessFile(filename, &f));

  absl::string_view result;
  std::vector<char> scratch(10);

  {
    // 1. Full read.
    result = absl::string_view();
    ASSERT_OK(f->Read(0, result,
                      absl::MakeSpan(scratch).subspan(0, contents.size())));
    EXPECT_EQ(result, contents);
  }

  {
    // 2. Subset read.
    result = absl::string_view();
    ASSERT_OK(f->Read(1, result, absl::MakeSpan(scratch).subspan(0, 4)));
    EXPECT_EQ(result, "bcde");
  }

  {
    // 3. Read past EOF (offset + n > file_size).
    result = absl::string_view();
    EXPECT_THAT(f->Read(4, result, absl::MakeSpan(scratch).subspan(0, 3)),
                StatusIs(absl::StatusCode::kOutOfRange));
    EXPECT_EQ(result, "ef");
  }

  {
    // 4. Read starting at EOF.
    result = absl::string_view();
    EXPECT_THAT(
        f->Read(contents.size(), result, absl::MakeSpan(scratch).subspan(0, 1)),
        StatusIs(absl::StatusCode::kOutOfRange));
    EXPECT_THAT(result, IsEmpty());
  }

  {
    // 5. Zero-length read at valid offset.
    result = absl::string_view();
    ASSERT_OK(f->Read(2, result, absl::MakeSpan(scratch).subspan(0, 0)));
    EXPECT_THAT(result, IsEmpty());
  }
}

TEST_F(EmbedFileSystemTest, UnsupportedWrites) {
  std::string filename = "embed://write";

  std::unique_ptr<WritableFile> f;
  EXPECT_THAT(fs_->NewWritableFile(filename, &f),
              StatusIs(absl::StatusCode::kUnimplemented));

  EXPECT_THAT(fs_->NewAppendableFile(filename, &f),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(EmbedFileSystemTest, FileSizeSuccess) {
  std::string filename = "embed://bar";
  std::string contents = "bar";

  ASSERT_OK(fs_->EmbedFile(filename, contents));

  uint64_t file_size;
  ASSERT_OK(Env::Default()->GetFileSize(filename, &file_size));
  EXPECT_EQ(file_size, contents.size());
}

TEST_F(EmbedFileSystemTest, FileSizeFailure) {
  std::string filename = "embed://goodbye";
  std::string contents = "bar";

  uint64_t file_size;
  ASSERT_THAT(Env::Default()->GetFileSize(filename, &file_size),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(EmbedFileSystemTest, IsDirectory) {
  std::string filename = "embed://dir/";

  EXPECT_THAT(Env::Default()->IsDirectory(filename),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(EmbedFileSystemTest, GetMatchingPaths) {
  const std::string filename = "embed://paths";
  const std::string contents = "contents";

  ASSERT_OK(fs_->EmbedFile(filename, contents));

  std::vector<std::string> paths;
  ASSERT_OK(fs_->GetMatchingPaths("embed://pat*", &paths));
  EXPECT_THAT(paths, ElementsAre("embed://paths"));
}

}  // namespace
}  // namespace tsl
