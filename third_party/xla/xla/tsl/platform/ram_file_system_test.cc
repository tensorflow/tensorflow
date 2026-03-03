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

#include "xla/tsl/platform/ram_file_system.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/file_system.h"

namespace tsl {
namespace {

using ::absl_testing::StatusIs;

TEST(RamFileSystemTest, Basic) {
  RamFileSystem fs;

  std::string data = "data";
  {
    std::unique_ptr<WritableFile> f;
    TF_ASSERT_OK(fs.NewWritableFile("foo.txt", nullptr, &f));

    TF_ASSERT_OK(f->Append(data));
  }
  TF_ASSERT_OK(fs.FileExists("foo.txt"));

  {
    std::unique_ptr<RandomAccessFile> f;
    TF_ASSERT_OK(fs.NewRandomAccessFile("foo.txt", nullptr, &f));

    absl::string_view contents;
    std::vector<char> scratch(data.size());
    TF_ASSERT_OK(f->Read(0, contents, absl::MakeSpan(scratch)));

    EXPECT_EQ(contents, data);
  }
}

TEST(RamFileSystemTest, CustomScheme) {
  RamFileSystem fs("embed://");

  std::string data = "data";
  {
    std::unique_ptr<WritableFile> f;
    TF_ASSERT_OK(fs.NewWritableFile("foo.txt", nullptr, &f));

    TF_ASSERT_OK(f->Append(data));
  }
  TF_ASSERT_OK(fs.FileExists("embed://foo.txt"));
  TF_ASSERT_OK(fs.FileExists("foo.txt"));
  ASSERT_THAT(fs.FileExists("ram://foo.txt"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(RamFileSystemTest, ReadAfterWrite) {
  RamFileSystem fs;

  std::string data = "data";
  std::string filename = "file.txt";

  // Write.
  {
    std::unique_ptr<WritableFile> f;
    TF_ASSERT_OK(fs.NewWritableFile(filename, &f));

    TF_ASSERT_OK(f->Append(data));
    TF_ASSERT_OK(f->Close());
  }

  // Now write again.
  {
    std::unique_ptr<WritableFile> f;
    TF_ASSERT_OK(fs.NewWritableFile(filename, &f));

    TF_ASSERT_OK(f->Append(data));
    TF_ASSERT_OK(f->Close());
  }

  // Now read, and we should see only the second write.
  {
    std::unique_ptr<RandomAccessFile> f;
    TF_ASSERT_OK(fs.NewRandomAccessFile(filename, &f));

    absl::string_view contents;
    std::vector<char> scratch(data.size());
    TF_ASSERT_OK(f->Read(0, contents, absl::MakeSpan(scratch)));

    EXPECT_EQ(contents, data);
  }
}

}  // namespace
}  // namespace tsl
