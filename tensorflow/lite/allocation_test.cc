/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/allocation.h"

#include <cstddef>

#if defined(__linux__)
#include <fcntl.h>
#endif

#include <sys/stat.h>

#include <gtest/gtest.h>
#include "tensorflow/lite/testing/util.h"

namespace tflite {

TEST(MMAPAllocation, TestInvalidFile) {
  if (!MMAPAllocation::IsSupported()) {
    return;
  }

  TestErrorReporter error_reporter;
  MMAPAllocation allocation("/tmp/tflite_model_1234", &error_reporter);
  EXPECT_FALSE(allocation.valid());
}

TEST(MMAPAllocation, TestValidFile) {
  if (!MMAPAllocation::IsSupported()) {
    return;
  }

  TestErrorReporter error_reporter;
  MMAPAllocation allocation(
      "tensorflow/lite/testdata/empty_model.bin", &error_reporter);

  ASSERT_TRUE(allocation.valid());
  EXPECT_GT(allocation.fd(), 0);
  EXPECT_GT(allocation.bytes(), 0);
  EXPECT_NE(allocation.base(), nullptr);
}

#if defined(__linux__)
TEST(MMAPAllocation, TestInvalidFileDescriptor) {
  if (!MMAPAllocation::IsSupported()) {
    return;
  }

  TestErrorReporter error_reporter;
  MMAPAllocation allocation(-1, &error_reporter);
  EXPECT_FALSE(allocation.valid());
}

TEST(MMAPAllocation, TestInvalidSizeAndOffset) {
  if (!MMAPAllocation::IsSupported()) {
    return;
  }

  int fd =
      open("tensorflow/lite/testdata/empty_model.bin", O_RDONLY);
  ASSERT_GT(fd, 0);

  struct stat fd_stat;
  ASSERT_EQ(fstat(fd, &fd_stat), 0);

  size_t file_size = fd_stat.st_size;

  TestErrorReporter error_reporter;
  MMAPAllocation allocation_invalid_offset(fd, /*offset=*/file_size + 100,
                                           /*length=*/1, &error_reporter);
  EXPECT_FALSE(allocation_invalid_offset.valid());

  MMAPAllocation allocation_invalid_length(fd, /*offset=*/0, /*length=*/0,
                                           &error_reporter);
  EXPECT_FALSE(allocation_invalid_length.valid());

  MMAPAllocation allocation_excessive_length(fd, /*offset=*/0,
                                             /*length=*/file_size + 1,
                                             &error_reporter);
  EXPECT_FALSE(allocation_excessive_length.valid());

  MMAPAllocation allocation_excessive_length_with_offset(
      fd, /*offset=*/10, /*length=*/file_size, &error_reporter);
  EXPECT_FALSE(allocation_excessive_length_with_offset.valid());

  close(fd);
}

TEST(MMAPAllocation, TestValidFileDescriptor) {
  if (!MMAPAllocation::IsSupported()) {
    return;
  }

  int fd =
      open("tensorflow/lite/testdata/empty_model.bin", O_RDONLY);
  ASSERT_GT(fd, 0);

  TestErrorReporter error_reporter;
  MMAPAllocation allocation(fd, &error_reporter);
  EXPECT_TRUE(allocation.valid());
  EXPECT_GT(allocation.fd(), 0);
  EXPECT_GT(allocation.bytes(), 0);
  EXPECT_NE(allocation.base(), nullptr);

  close(fd);
}

TEST(MMAPAllocation, TestValidFileDescriptorWithOffset) {
  if (!MMAPAllocation::IsSupported()) {
    return;
  }

  int fd =
      open("tensorflow/lite/testdata/empty_model.bin", O_RDONLY);
  ASSERT_GT(fd, 0);

  struct stat fd_stat;
  ASSERT_EQ(fstat(fd, &fd_stat), 0);
  size_t file_size = fd_stat.st_size;

  TestErrorReporter error_reporter;
  MMAPAllocation allocation(fd, /*offset=*/10, /*length=*/file_size - 10,
                            &error_reporter);
  EXPECT_TRUE(allocation.valid());
  EXPECT_GT(allocation.fd(), 0);
  EXPECT_GT(allocation.bytes(), 0);
  EXPECT_NE(allocation.base(), nullptr);

  close(fd);
}
#endif  // defined(__linux__)

}  // namespace tflite
