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

#include "xla/backends/profiler/subprocess/subprocess_profiling_session.h"  // IWYU pragma: keep

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/backends/profiler/subprocess/subprocess_registry.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace subprocess {
namespace {

using ::absl_testing::IsOk;
using ::testing::IsEmpty;
using ::testing::Not;

std::unique_ptr<tsl::SubProcess> CreateSubProcess(
    const std::vector<std::string>& args) {
  auto subprocess = std::make_unique<tsl::SubProcess>();
  subprocess->SetProgram(args[0], args);
  subprocess->SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_DUPPARENT);
  subprocess->SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_DUPPARENT);
  return subprocess;
}

class SubprocessProfilingSessionTest : public ::testing::Test {
 public:
  void SetUpSubprocesses(int num_subprocesses) {
    std::string subprocess_main_path =
        std::getenv("XPROF_TEST_SUBPROCESS_MAIN_PATH");
    ASSERT_FALSE(subprocess_main_path.empty())
        << "The env variable XPROF_TEST_SUBPROCESS_MAIN_PATH is required.";
    const char* srcdir = std::getenv("TEST_SRCDIR");
    ASSERT_NE(srcdir, nullptr) << "Environment variable TEST_SRCDIR unset!";
    subprocess_main_path = tsl::io::JoinPath(srcdir, subprocess_main_path);
    ASSERT_TRUE(subprocesses_.empty());
    subprocesses_.resize(num_subprocesses);
    for (int i = 0; i < num_subprocesses; ++i) {
      std::vector<std::string> args;
      args.push_back(subprocess_main_path);
      int port = tsl::testing::PickUnusedPortOrDie();
      args.push_back(absl::StrCat("--port=", port));

      SubProcessRuntime& subprocess_runtime = subprocesses_[i];
      subprocess_runtime.port = port;
      subprocess_runtime.subprocess = CreateSubProcess(args);
      ASSERT_TRUE(subprocess_runtime.subprocess->Start());
      ASSERT_THAT(
          RegisterSubprocess(subprocess_runtime.port, subprocess_runtime.port),
          IsOk());
    }
    // Wait for connections to be established.
    absl::SleepFor(absl::Seconds(num_subprocesses + 1));
  }

  void TearDown() override {
    for (auto& subprocess_runtime : subprocesses_) {
      ASSERT_NE(subprocess_runtime.subprocess, nullptr);
      ASSERT_TRUE(subprocess_runtime.subprocess->Kill(SIGKILL));
      ASSERT_THAT(UnregisterSubprocess(subprocess_runtime.port), IsOk());
    }
  }

  struct SubProcessRuntime {
    std::unique_ptr<tsl::SubProcess> subprocess;
    int port;
  };
  std::vector<SubProcessRuntime> subprocesses_;
};

TEST_F(SubprocessProfilingSessionTest, MultipleSubprocessesIntegrationTest) {
  SetUpSubprocesses(3);

  auto session =
      tsl::ProfilerSession::Create(tsl::ProfilerSession::DefaultOptions());
  auto deadline = absl::Now() + absl::Seconds(1);
  while (absl::Now() < deadline) {
    tsl::profiler::TraceMe trace_me("main_process");
    absl::SleepFor(absl::Milliseconds(10));
  }
  tensorflow::profiler::XSpace space;
  ASSERT_THAT(session->CollectData(&space), IsOk());

  ASSERT_THAT(space.planes(), Not(IsEmpty()));

  absl::flat_hash_set<std::string> trace_me_names;
  for (const auto& plane : space.planes()) {
    const auto& event_metadata = plane.event_metadata();
    for (const auto& line : plane.lines()) {
      if (!line.events().empty()) {
        const auto& event = line.events(0);
        std::string trace_me_name =
            event_metadata.at(event.metadata_id()).name();
        if (absl::StartsWith(trace_me_name, "subprocess_test_")) {
          std::vector<std::string> parts = absl::StrSplit(
              event_metadata.at(event.metadata_id()).name(), '_');
          ASSERT_EQ(parts.size(), 4)
              << "Invalid trace me name: " << trace_me_name;
          trace_me_name = absl::StrJoin(parts.begin(), parts.end() - 1, "_");
        }
        trace_me_names.insert(trace_me_name);
      }
    }
  }
  absl::flat_hash_set<std::string> expected_trace_me_names;
  expected_trace_me_names.insert("main_process");
  for (const auto& subproc : subprocesses_) {
    // The pid is actually the port number since tsl::SubProcess does not
    // expose the pid.
    expected_trace_me_names.insert(
        absl::StrCat("subprocess_test_", subproc.port));
  }
  EXPECT_EQ(trace_me_names, expected_trace_me_names);

  // Emit the collected XSpace as an undeclared output file.
  const char* outputs_dir = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  if (outputs_dir != nullptr) {
    std::string output_path = tsl::io::JoinPath(outputs_dir, "xspace.pb");
    ASSERT_THAT(tsl::WriteBinaryProto(tsl::Env::Default(), output_path, space),
                IsOk());
    LOG(INFO) << "Wrote XSpace proto to: " << output_path;
  } else {
    LOG(WARNING)
        << "TEST_UNDECLARED_OUTPUTS_DIR not set, skipping writing xspace.pb";
  }
}

}  // namespace
}  // namespace subprocess
}  // namespace profiler
}  // namespace xla
