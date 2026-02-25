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

#include "xla/tsl/platform/env.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "tsl/platform/path.h"

namespace tsl {
namespace {

using absl_testing::IsOk;

TEST(EnvTest, StartDetachedThread) {
  Env* env = Env::Default();
  const int num_threads = 10;
  absl::BlockingCounter counter(num_threads);

  ThreadOptions thread_options;
  for (int i = 0; i < num_threads; ++i) {
    env->StartDetachedThread(thread_options, "MyDetachedThread", [&]() {
      absl::SleepFor(absl::Milliseconds(50));
      counter.DecrementCount();
    });
  }

  counter.Wait();
}

TEST(EnvTest, FileOperations) {
  Env* env = Env::Default();
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  std::string file_path = tsl::io::JoinPath(temp_dir.path(), "test.txt");
  EXPECT_THAT(WriteStringToFile(env, file_path, "test1"), IsOk());
  EXPECT_THAT(AppendStringToFile(env, file_path, "test2"), IsOk());
  std::string content;
  EXPECT_THAT(ReadFileToString(env, file_path, &content), IsOk());
  EXPECT_EQ(content, "test1test2");
}

}  // namespace
}  // namespace tsl
