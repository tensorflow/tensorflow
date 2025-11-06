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

#include "absl/status/status_matchers.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace profiler {
namespace subprocess {
namespace {

using absl_testing::IsOk;
using ::testing::Not;

TEST(SubprocessRegistryTest, FailsToRegisterCurrentProcessAsSubprocess) {
  EXPECT_THAT(RegisterSubprocess(tsl::Env::Default()->GetProcessId(), 1234),
              Not(IsOk()));
}

TEST(SubprocessRegistryTest, RegistersSubprocessWithPort) {
  EXPECT_THAT(RegisterSubprocess(1234, tsl::testing::PickUnusedPortOrDie()),
              IsOk());
}

TEST(SubprocessRegistryTest, RegistersSubprocessWithUnixDomainSocket) {
  EXPECT_THAT(RegisterSubprocess(12345, "/tmp/valid_socket"), IsOk());
}

TEST(SubprocessRegistryTest, FailsToRegisterSubprocessTwice) {
  int port = tsl::testing::PickUnusedPortOrDie();
  EXPECT_THAT(RegisterSubprocess(123456, port), IsOk());
  EXPECT_THAT(RegisterSubprocess(123456, port), Not(IsOk()));
}

}  // namespace
}  // namespace subprocess
}  // namespace profiler
}  // namespace xla
