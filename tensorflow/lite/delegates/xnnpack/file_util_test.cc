/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/xnnpack/file_util.h"

#include <fcntl.h>

#include <atomic>
#include <string>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

namespace tflite::xnnpack {
namespace {

// Returns a path for a temporary file.
//
// Each call will return a new path.
std::string NewTempFilePath() {
  static std::atomic<int> i = 0;
  return testing::TempDir() + "test_file_" + std::to_string(i++);
}

TEST(FileDescriptorTest, DefaultConstructedIsInvalid) {
  FileDescriptor fd;
  EXPECT_FALSE(fd.IsValid());
}

TEST(FileDescriptorTest, ConstructAndRelease) {
  const int kFd = 53;
  // Construct from int.
  FileDescriptor fd(kFd);
  EXPECT_TRUE(fd.IsValid());
  EXPECT_EQ(fd.Value(), kFd);
  // Move construction
  FileDescriptor fd2(std::move(fd));
  EXPECT_FALSE(fd.IsValid());
  EXPECT_TRUE(fd2.IsValid());
  EXPECT_EQ(fd2.Value(), kFd);
  // We release because we don't own kFd.
  EXPECT_EQ(fd2.Release(), kFd);
  EXPECT_FALSE(fd2.IsValid());
  EXPECT_FALSE(std::is_copy_constructible_v<FileDescriptor>);
}

TEST(FileDescriptorTest, OpenNullFileFails) {
  FileDescriptor fd =
      FileDescriptor::Open(nullptr, O_CREAT | O_TRUNC | O_RDWR, 0644);
  EXPECT_FALSE(fd.IsValid());
}

TEST(FileDescriptorTest, OpenWriteRewindAndReadWorks) {
  const std::string tmp_file = NewTempFilePath();
  FileDescriptor fd =
      FileDescriptor::Open(tmp_file.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
  ASSERT_TRUE(fd.IsValid());
  const std::string src_data = "The quick brown fox jumps over the lazy dog.";
  EXPECT_TRUE(fd.Write(src_data.data(), src_data.size()));
  EXPECT_EQ(fd.SetPos(0), 0);
  std::string dst_data(src_data.size(), ' ');
  EXPECT_TRUE(fd.Read(dst_data.data(), src_data.size()));
  EXPECT_EQ(dst_data, src_data);
}

TEST(FileDescriptorTest, WriteFailureReturnsFalse) {
  const std::string tmp_file = NewTempFilePath();
  FileDescriptor fd = FileDescriptor::Open(tmp_file.c_str(),
                                           O_CREAT | O_TRUNC | O_RDONLY, 0644);
  ASSERT_TRUE(fd.IsValid());
  const std::string src_data = "The quick brown fox jumps over the lazy dog.";
  EXPECT_FALSE(fd.Write(src_data.data(), src_data.size()));
}

TEST(FileDescriptorTest, ReadFailureReturnsFalse) {
  const std::string tmp_file = NewTempFilePath();
  FileDescriptor fd = FileDescriptor::Open(tmp_file.c_str(),
                                           O_CREAT | O_TRUNC | O_WRONLY, 0644);
  ASSERT_TRUE(fd.IsValid());
  std::string dst_data(5, ' ');
  EXPECT_FALSE(fd.Read(dst_data.data(), dst_data.size()));
}

TEST(FileDescriptorTest, IsFileEmptyReturnTrueForAnEmptyFileThatExists) {
  const std::string tmp_file = NewTempFilePath();
  FileDescriptor fd = FileDescriptor::Open(tmp_file.c_str(),
                                           O_CREAT | O_TRUNC | O_WRONLY, 0644);
  fd.Close();
  EXPECT_TRUE(IsFileEmpty(tmp_file.c_str(), FileDescriptor()));
}

TEST(FileDescriptorTest, IsFileEmptyReturnTrueForAnNonExistingFile) {
  const std::string tmp_file = NewTempFilePath();
  EXPECT_TRUE(IsFileEmpty(tmp_file.c_str(), FileDescriptor()));
}

TEST(FileDescriptorTest,
     IsFileEmptyReturnTrueForAnNonExistingFileWithFileDescriptor) {
  const std::string tmp_file = NewTempFilePath();
  FileDescriptor fd = FileDescriptor::Open(tmp_file.c_str(),
                                           O_CREAT | O_TRUNC | O_WRONLY, 0644);
  EXPECT_TRUE(IsFileEmpty("asdfasdf", FileDescriptor()));
}

TEST(FileDescriptorTest, IsFileEmptyReturnFalseForAFileThatHasContents) {
  const std::string tmp_file = NewTempFilePath();
  FileDescriptor fd = FileDescriptor::Open(tmp_file.c_str(),
                                           O_CREAT | O_TRUNC | O_WRONLY, 0644);
  const std::string src_data = "The quick brown fox jumps over the lazy dog.";
  EXPECT_TRUE(fd.Write(src_data.data(), src_data.size()));
  EXPECT_FALSE(IsFileEmpty(tmp_file.c_str(), fd));
}

TEST(FileDescriptorTest, IsFileEmptyPrioritizesTheFileDescriptor) {
  // We open 2 files, put some data only in one and then pass the file name of
  // the one that has data and the file descriptor of the empty one.
  const std::string tmp_file = NewTempFilePath();
  const std::string tmp_file2 = NewTempFilePath();
  FileDescriptor fd = FileDescriptor::Open(tmp_file.c_str(),
                                           O_CREAT | O_TRUNC | O_WRONLY, 0644);
  FileDescriptor fd2 = FileDescriptor::Open(tmp_file2.c_str(),
                                            O_CREAT | O_TRUNC | O_WRONLY, 0644);
  const std::string src_data = "The quick brown fox jumps over the lazy dog.";
  EXPECT_TRUE(fd.Write(src_data.data(), src_data.size()));
  fd.Close();
  EXPECT_TRUE(IsFileEmpty(tmp_file.c_str(), fd2));
}

}  // namespace
}  // namespace tflite::xnnpack
