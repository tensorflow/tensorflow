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

#include "xla/backends/profiler/subprocess/subprocess_registry.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

namespace xla {
namespace profiler {
namespace subprocess {
namespace {

using absl_testing::IsOk;
using absl_testing::StatusIs;
using ::testing::Not;

class SubprocessRegistryTest : public ::testing::Test {
 public:
  void SetUp() override {
    std::string subprocess_main_path =
        std::getenv("XPROF_TEST_SUBPROCESS_MAIN_PATH");
    ASSERT_FALSE(subprocess_main_path.empty())
        << "The env variable XPROF_TEST_SUBPROCESS_MAIN_PATH is required.";
    const char* srcdir = std::getenv("TEST_SRCDIR");
    ASSERT_NE(srcdir, nullptr) << "Environment variable TEST_SRCDIR unset!";
    subprocess_main_path_ = tsl::io::JoinPath(srcdir, subprocess_main_path);
  }

  std::string subprocess_main_path_;
};

TEST_F(SubprocessRegistryTest, FailsToRegisterCurrentProcessAsSubprocess) {
  EXPECT_THAT(RegisterSubprocess(tsl::Env::Default()->GetProcessId(), 1234,
                                 std::nullopt),
              Not(IsOk()));
}

TEST_F(SubprocessRegistryTest, FailsDueToTimeout) {
  EXPECT_THAT(RegisterSubprocess(1234, tsl::testing::PickUnusedPortOrDie(),
                                 std::nullopt),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
}

TEST_F(SubprocessRegistryTest, FailsWithNoPortOrUnixDomainSocket) {
  EXPECT_THAT(RegisterSubprocess(1234, std::nullopt, std::nullopt),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(SubprocessRegistryTest, RegistersSubprocessWithPort) {
  int port = tsl::testing::PickUnusedPortOrDie();
  std::unique_ptr<tsl::SubProcess> subprocess = tsl::CreateSubProcess(
      {subprocess_main_path_, absl::StrCat("--port=", port)});
  ASSERT_TRUE(subprocess->Start());
  ASSERT_OK_AND_ASSIGN(auto unregister_fn,
                       RegisterSubprocess(1234, port, std::nullopt));
  auto registered_subprocesses = GetRegisteredSubprocesses();
  ASSERT_EQ(registered_subprocesses.size(), 1);
  EXPECT_THAT(registered_subprocesses[0].pid, 1234);
  EXPECT_EQ(registered_subprocesses[0].address,
            absl::StrCat("localhost:", port));
  EXPECT_THAT(registered_subprocesses[0].profiler_stub, testing::NotNull());
}

TEST_F(SubprocessRegistryTest, FailsToRegisterSubprocessTwice) {
  int port = tsl::testing::PickUnusedPortOrDie();
  std::unique_ptr<tsl::SubProcess> subprocess = tsl::CreateSubProcess(
      {subprocess_main_path_, absl::StrCat("--port=", port)});
  ASSERT_TRUE(subprocess->Start());
  ASSERT_OK_AND_ASSIGN(auto unregister_fn,
                       RegisterSubprocess(123456, port, std::nullopt));
  EXPECT_THAT(RegisterSubprocess(123456, port, std::nullopt), Not(IsOk()));
  auto registered_subprocesses = GetRegisteredSubprocesses();
  ASSERT_EQ(registered_subprocesses.size(), 1);
  EXPECT_THAT(registered_subprocesses[0].pid, 123456);
  EXPECT_EQ(registered_subprocesses[0].address,
            absl::StrCat("localhost:", port));
  EXPECT_THAT(registered_subprocesses[0].profiler_stub, testing::NotNull());
}

TEST_F(SubprocessRegistryTest, TestUnregisterSubprocess) {
  int port = tsl::testing::PickUnusedPortOrDie();
  std::unique_ptr<tsl::SubProcess> subprocess = tsl::CreateSubProcess(
      {subprocess_main_path_, absl::StrCat("--port=", port)});
  ASSERT_TRUE(subprocess->Start());
  ASSERT_OK_AND_ASSIGN(auto unregister_fn,
                       RegisterSubprocess(123456, port, std::nullopt));
  unregister_fn.Invoke();
  auto registered_subprocesses = GetRegisteredSubprocesses();
  EXPECT_EQ(registered_subprocesses.size(), 0);
}

}  // namespace
}  // namespace subprocess
}  // namespace profiler
}  // namespace xla
