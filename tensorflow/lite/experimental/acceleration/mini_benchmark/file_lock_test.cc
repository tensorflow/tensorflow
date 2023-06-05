/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/file_lock.h"

#include <csignal>
#include <iostream>
#include <string>
#include <utility>

#include <gtest/gtest.h>

namespace tflite {
namespace acceleration {
namespace {

class FileLockTest : public ::testing::Test {
 protected:
  void SetUp() override { file_path_ = ::testing::TempDir() + "/file_lock"; }

  std::string file_path_;
};

TEST_F(FileLockTest, CanLock) { EXPECT_TRUE(FileLock(file_path_).TryLock()); }

TEST_F(FileLockTest, FailIfLockMoreThanOnce) {
  FileLock lock_one(file_path_);
  FileLock lock_two(file_path_);
  ASSERT_TRUE(lock_one.TryLock());
  EXPECT_FALSE(lock_two.TryLock());
}

TEST_F(FileLockTest, LockReleasedWhenThreadCrash) {
  pid_t pid = fork();
  if (pid == 0) {
    // Child process crashed after TryLock().
    FileLock lock(file_path_);
    if (!lock.TryLock()) {
      _exit(1);
    }
    std::cout << "Lock acquired successfully.";
    kill(getpid(), SIGKILL);
  }
  int wstatus;
  int w = waitpid(pid, &wstatus, WUNTRACED);
  ASSERT_NE(w, -1);

  // Lock again from main process.
  FileLock lock_two(file_path_);
  EXPECT_TRUE(lock_two.TryLock());
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite
