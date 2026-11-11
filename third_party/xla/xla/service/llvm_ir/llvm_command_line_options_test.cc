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

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"

namespace xla::llvm_ir {
namespace {

llvm::cl::opt<int>& GetTestIntOption() {
  static llvm::cl::opt<int>* option = new llvm::cl::opt<int>(
      "xla-test-llvm-cl-int-option", llvm::cl::init(0), llvm::cl::Hidden);
  return *option;
}

TEST(LLVMCommandLineOptionsLockTest, ApplyLibraryOptions) {
  LLVMCommandLineOptionsLock lock(/*client_options=*/{});
  llvm::LLVMContext context;
  LLVMCommandLineOptionsLock::ApplyLibraryOptions(context);
  EXPECT_TRUE(context.getLibraryOptions().args().empty());
}

TEST(LLVMCommandLineOptionsLockTest, ReappliesOptionsAfterExternalReset) {
  llvm::cl::opt<int>& test_option = GetTestIntOption();
  const std::vector<std::string> client_options = {
      "-xla-test-llvm-cl-int-option=42"};

  {
    LLVMCommandLineOptionsLock lock(client_options);
    EXPECT_EQ(test_option.getValue(), 42);
  }

  // Simulate an external in-process LLVM user (such as Triton's
  // ScopedLLVMOptions or LLD) resetting global LLVM command-line options.
  llvm::cl::ResetAllOptionOccurrences();
  ASSERT_EQ(test_option.getValue(), 0);

  {
    LLVMCommandLineOptionsLock lock(client_options);
    EXPECT_EQ(test_option.getValue(), 42);
  }
}

TEST(LLVMCommandLineOptionsLockTest,
     NestedLockAfterInProcessResetDoesNotDeadlock) {
  llvm::cl::opt<int>& test_option = GetTestIntOption();
  const std::vector<std::string> client_options = {
      "-xla-test-llvm-cl-int-option=42"};

  {
    LLVMCommandLineOptionsReleasableLock outer_lock(client_options);
    EXPECT_EQ(test_option.getValue(), 42);

    {
      LLVMCommandLineOptionsLock inner_lock_1(client_options);
      // Simulate in-process lld::lldMain calling ResetAllOptionOccurrences()
      // while the outer lock is held (as happens in AMDGPU CompileToHsaco).
      llvm::cl::ResetAllOptionOccurrences();
    }

    // Acquiring a second nested lock with the same options while outer_lock is
    // still held must not deadlock.
    {
      LLVMCommandLineOptionsLock inner_lock_2(client_options);
    }
  }

  // Once all locks are released, the next outermost lock must detect the
  // earlier reset and re-apply client_options.
  {
    LLVMCommandLineOptionsLock lock(client_options);
    EXPECT_EQ(test_option.getValue(), 42);
  }
}

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
