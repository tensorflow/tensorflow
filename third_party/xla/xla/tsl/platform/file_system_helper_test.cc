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

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"

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

}  // namespace
}  // namespace tsl
