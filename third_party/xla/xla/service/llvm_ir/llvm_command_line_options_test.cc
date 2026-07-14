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

#include "xla/service/llvm_ir/llvm_command_line_options.h"

#include <gtest/gtest.h>

namespace xla::llvm_ir {
namespace {

TEST(LLVMCommandLineOptionsReleasableLockTest, Basic) {
  LLVMCommandLineOptionsReleasableLock lock(
      /*client_options=*/{});
  EXPECT_TRUE(lock.IsLocked());
  {
    auto cleanup = lock.TemporarilyReleaseLock();
    EXPECT_FALSE(lock.IsLocked());
  }
  EXPECT_TRUE(lock.IsLocked());
}

TEST(LLVMCommandLineOptionsReleasableLockTest, NestedReleases) {
  LLVMCommandLineOptionsReleasableLock lock(
      /*client_options=*/{});
  EXPECT_TRUE(lock.IsLocked());
  {
    auto keeper = lock.TemporarilyReleaseLock();
    EXPECT_FALSE(lock.IsLocked());
    {
      auto keeper = lock.TemporarilyReleaseLock();
      EXPECT_FALSE(lock.IsLocked());
    }
    EXPECT_FALSE(lock.IsLocked());
  }
  EXPECT_TRUE(lock.IsLocked());
}

}  // namespace
}  // namespace xla::llvm_ir
