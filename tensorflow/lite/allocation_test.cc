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
#include <cstdint>

#if defined(__linux__) || defined(_WIN32)
#include <fcntl.h>
#endif
#if defined(_WIN32)
#include <io.h>
#endif

#include <sys/stat.h>

#if defined(_WIN32)
#define sys_open _open
#define SYS_O_RDONLY (_O_RDONLY | _O_BINARY)
#define sys_close _close
#define sys_stat _stat64
#define sys_fstat _fstat64
#else
#define sys_open open
#define SYS_O_RDONLY O_RDONLY
#define sys_close close
#define sys_stat stat
#define sys_fstat fstat
#endif

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

#if defined(__linux__) || defined(_WIN32)
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

  int fd = sys_open("tensorflow/lite/testdata/empty_model.bin",
                    SYS_O_RDONLY);
  ASSERT_GT(fd, 0);

  struct sys_stat fd_stat;
  ASSERT_EQ(sys_fstat(fd, &fd_stat), 0);

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

  MMAPAllocation allocation_integer_overflow(
      fd, /*offset=*/10, /*length=*/SIZE_MAX - 5, &error_reporter);
  EXPECT_FALSE(allocation_integer_overflow.valid());

  sys_close(fd);
}

TEST(MMAPAllocation, TestValidFileDescriptor) {
  if (!MMAPAllocation::IsSupported()) {
    return;
  }

  int fd = sys_open("tensorflow/lite/testdata/empty_model.bin",
                    SYS_O_RDONLY);
  ASSERT_GT(fd, 0);

  TestErrorReporter error_reporter;
  MMAPAllocation allocation(fd, &error_reporter);
  EXPECT_TRUE(allocation.valid());
  EXPECT_GT(allocation.fd(), 0);
  EXPECT_GT(allocation.bytes(), 0);
  EXPECT_NE(allocation.base(), nullptr);

  sys_close(fd);
}

TEST(MMAPAllocation, TestValidFileDescriptorWithOffset) {
  if (!MMAPAllocation::IsSupported()) {
    return;
  }

  int fd = sys_open("tensorflow/lite/testdata/empty_model.bin",
                    SYS_O_RDONLY);
  ASSERT_GT(fd, 0);

  struct sys_stat fd_stat;
  ASSERT_EQ(sys_fstat(fd, &fd_stat), 0);
  size_t file_size = fd_stat.st_size;

  TestErrorReporter error_reporter;
  MMAPAllocation allocation(fd, /*offset=*/10, /*length=*/file_size - 10,
                            &error_reporter);
  EXPECT_TRUE(allocation.valid());
  EXPECT_GT(allocation.fd(), 0);
  EXPECT_GT(allocation.bytes(), 0);
  EXPECT_NE(allocation.base(), nullptr);

  sys_close(fd);
}
#endif  // defined(__linux__) || defined(_WIN32)

}  // namespace tflite
