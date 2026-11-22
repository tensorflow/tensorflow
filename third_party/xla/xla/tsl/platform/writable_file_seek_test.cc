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

#include <sys/types.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/ram_file_system.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "tsl/platform/platform.h"

namespace tsl {
namespace {

using ::absl_testing::StatusIs;

TEST(WritableFileSeekTest, PosixSeekOnEmptyFile) {
  Env* env = Env::Default();
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  const std::string file_path = temp_dir.path() + "/seek.txt";

  std::unique_ptr<WritableFile> f;
  ASSERT_OK(env->NewWritableFile(file_path, &f));
  ASSERT_OK(f->Seek(0));
  ASSERT_OK(f->Close());
}

// Seeking on a write-only file allows buffered data to be overwritten.
TEST(WritableFileSeekTest, PosixSeekOverwritesWrittenData) {
  Env* env = Env::Default();
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  const std::string file_path = temp_dir.path() + "/seek.txt";

  std::unique_ptr<WritableFile> f;
  ASSERT_OK(env->NewWritableFile(file_path, &f));
  ASSERT_OK(f->Append("hello"));
  ASSERT_OK(f->Seek(0));
  int64_t pos = -1;
  ASSERT_OK(f->Tell(&pos));
  EXPECT_EQ(pos, 0);
  ASSERT_OK(f->Append("X"));
  ASSERT_OK(f->Tell(&pos));
  EXPECT_EQ(pos, 1);
  ASSERT_OK(f->Close());

  std::string content;
  ASSERT_OK(ReadFileToString(env, file_path, &content));
  EXPECT_EQ(content, "Xello");
}

TEST(WritableFileSeekTest, PosixSeekWithinWrittenData) {
  Env* env = Env::Default();
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  const std::string file_path = temp_dir.path() + "/seek.txt";

  std::unique_ptr<WritableFile> f;
  ASSERT_OK(env->NewWritableFile(file_path, &f));
  ASSERT_OK(f->Append("abc"));
  ASSERT_OK(f->Seek(2));
  ASSERT_OK(f->Append("Z"));
  ASSERT_OK(f->Close());

  std::string content;
  ASSERT_OK(ReadFileToString(env, file_path, &content));
  EXPECT_EQ(content, "abZ");
}

TEST(WritableFileSeekTest, PosixSeekNegativePositionFails) {
  Env* env = Env::Default();
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  const std::string file_path = temp_dir.path() + "/seek.txt";

  std::unique_ptr<WritableFile> f;
  ASSERT_OK(env->NewWritableFile(file_path, &f));
  ASSERT_OK(f->Append("hello"));
  EXPECT_THAT(f->Seek(-1), StatusIs(absl::StatusCode::kInvalidArgument));
  ASSERT_OK(f->Close());
}

TEST(WritableFileSeekTest, PosixSeekAfterCloseFails) {
  Env* env = Env::Default();
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  const std::string file_path = temp_dir.path() + "/seek.txt";

  std::unique_ptr<WritableFile> f;
  ASSERT_OK(env->NewWritableFile(file_path, &f));
  ASSERT_OK(f->Close());
  EXPECT_THAT(f->Seek(0), StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(WritableFileSeekTest, PosixSeekLargePosition) {
  if (!tsl::kIsOpenSource) {
    GTEST_SKIP() << "Google file API does not support seeking past EOF.";
  }
  Env* env = Env::Default();
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  const std::string file_path = temp_dir.path() + "/seek.txt";

  std::unique_ptr<WritableFile> f;
  ASSERT_OK(env->NewWritableFile(file_path, &f));
  constexpr int64_t kPosition = (int64_t{1} << 31) + 17;
  if (kPosition > static_cast<int64_t>(std::numeric_limits<off_t>::max())) {
    EXPECT_THAT(f->Seek(kPosition), StatusIs(absl::StatusCode::kOutOfRange));
    ASSERT_OK(f->Close());
    return;
  }
  ASSERT_OK(f->Seek(kPosition));
  ASSERT_OK(f->Append("X"));
  int64_t position = -1;
  ASSERT_OK(f->Tell(&position));
  EXPECT_EQ(position, kPosition + 1);
  ASSERT_OK(f->Close());

  // Check the actual write offset without reading the entire sparse file.
  uint64_t size = 0;
  ASSERT_OK(env->GetFileSize(file_path, &size));
  EXPECT_EQ(size, kPosition + 1);
}

TEST(WritableFileSeekTest, PosixSeekInAppendModeFails) {
  Env* env = Env::Default();
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  const std::string file_path = temp_dir.path() + "/seek.txt";

  ASSERT_OK(WriteStringToFile(env, file_path, "first"));
  std::unique_ptr<WritableFile> f;
  ASSERT_OK(env->NewAppendableFile(file_path, &f));
  EXPECT_THAT(f->Seek(0), StatusIs(absl::StatusCode::kFailedPrecondition,
                                   "Cannot seek in append mode"));
  ASSERT_OK(f->Append("second"));
  EXPECT_THAT(f->Seek(0), StatusIs(absl::StatusCode::kFailedPrecondition,
                                   "Cannot seek in append mode"));
  ASSERT_OK(f->Append("X"));
  ASSERT_OK(f->Close());

  std::string content;
  ASSERT_OK(ReadFileToString(env, file_path, &content));
  EXPECT_EQ(content, "firstsecondX");
}

TEST(WritableFileSeekTest, DefaultImplementationIsUnimplemented) {
  RamFileSystem fs;
  std::unique_ptr<WritableFile> f;
  ASSERT_OK(fs.NewWritableFile("foo.txt", &f));
  ASSERT_OK(f->Append("data"));
  EXPECT_THAT(f->Seek(0), StatusIs(absl::StatusCode::kUnimplemented));
  ASSERT_OK(f->Close());
}

}  // namespace
}  // namespace tsl
