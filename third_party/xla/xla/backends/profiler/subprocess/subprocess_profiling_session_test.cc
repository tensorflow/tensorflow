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
#include "xla/backends/profiler/subprocess/subprocess_profiling_session.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "xla/backends/profiler/subprocess/subprocess_registry.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/profiler_service.grpc.pb.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace subprocess {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::IsEmpty;
using ::testing::Not;

absl::Status SaveXSpaceToUndeclaredOutputs(
    const tensorflow::profiler::XSpace& space) {
  // Emit the collected XSpace as an undeclared output file.
  const char* outputs_dir = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  LOG_IF(WARNING, outputs_dir == nullptr)
      << "TEST_UNDECLARED_OUTPUTS_DIR not set, skipping writing xspace.pb";
  if (outputs_dir != nullptr) {
    std::string output_path = tsl::io::JoinPath(outputs_dir, "xspace.pb");
    return tsl::WriteBinaryProto(tsl::Env::Default(), output_path, space);
  }
  return absl::OkStatus();
}

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
      ASSERT_OK_AND_ASSIGN(
          subprocess_runtime.unregister_fn,
          RegisterSubprocess(subprocess_runtime.port, subprocess_runtime.port,
                             std::nullopt));
    }
  }

  void TearDown() override {
    for (auto& subprocess_runtime : subprocesses_) {
      ASSERT_NE(subprocess_runtime.subprocess, nullptr);
      ASSERT_TRUE(subprocess_runtime.subprocess->Kill(/*sig=SIGKILL*/ 9));
      subprocess_runtime.unregister_fn.Invoke();
    }
  }

  struct SubProcessRuntime {
    std::unique_ptr<tsl::SubProcess> subprocess;
    int port;
    SubprocessCleanup unregister_fn;
  };
  std::vector<SubProcessRuntime> subprocesses_;
};

TEST_F(SubprocessProfilingSessionTest, SubprocessCollectionTest) {
  SetUpSubprocesses(1);
  std::vector<SubprocessInfo> subprocesses = GetRegisteredSubprocesses();
  ASSERT_EQ(subprocesses.size(), 1);
  SubprocessInfo subprocess_info = subprocesses[0];

  tensorflow::ProfileOptions options = tsl::ProfilerSession::DefaultOptions();
  TF_ASSERT_OK_AND_ASSIGN(auto session, SubprocessProfilingSession::Create(
                                            subprocess_info, options));

  ASSERT_THAT(session->Start(), IsOk());
  absl::SleepFor(absl::Seconds(2));
  ASSERT_THAT(session->Stop(), IsOk());
  tensorflow::profiler::XSpace space;
  ASSERT_THAT(session->CollectData(&space), IsOk());
  // For debugging purposes.
  EXPECT_THAT(SaveXSpaceToUndeclaredOutputs(space), IsOk());

  ASSERT_THAT(space.planes(), Not(IsEmpty()));
  tsl::profiler::XPlaneVisitor visitor =
      tsl::profiler::CreateTfXPlaneVisitor(&space.planes()[0]);
  std::optional<tsl::profiler::XStatVisitor> pid_stat =
      visitor.GetStat(tsl::profiler::StatType::kProcessId);
  ASSERT_TRUE(pid_stat.has_value());
  EXPECT_THAT(visitor.Name(), ::testing::HasSubstr(absl::StrCat(
                                  "[", pid_stat->IntOrUintValue(), "]")));
}

// In-process ProfilerService whose Terminate RPC fails. Its Profile RPC only
// ends when the client cancels it.
class TerminateFailsProfilerService
    : public tensorflow::grpc::ProfilerService::Service {
 public:
  ::grpc::Status Profile(::grpc::ServerContext* context,
                         const tensorflow::ProfileRequest* request,
                         tensorflow::ProfileResponse* response) override {
    while (!context->IsCancelled()) {
      absl::SleepFor(absl::Milliseconds(10));
    }
    return ::grpc::Status::CANCELLED;
  }

  ::grpc::Status Terminate(::grpc::ServerContext* context,
                           const tensorflow::TerminateRequest* request,
                           tensorflow::TerminateResponse* response) override {
    return ::grpc::Status(::grpc::StatusCode::UNAVAILABLE, "terminate failed");
  }
};

TEST(SubprocessProfilingSessionTerminateTest, TerminateFailureDoesNotHang) {
  TerminateFailsProfilerService service;
  ::grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(&service);
  std::unique_ptr<::grpc::Server> server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  SubprocessInfo subprocess_info;
  subprocess_info.pid = 4242;
  subprocess_info.address = absl::StrCat("localhost:", port);
  subprocess_info.profiler_stub =
      tensorflow::grpc::ProfilerService::NewStub(::grpc::CreateChannel(
          subprocess_info.address, ::grpc::InsecureChannelCredentials()));

  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto session,
        SubprocessProfilingSession::Create(
            subprocess_info, tsl::ProfilerSession::DefaultOptions()));
    ASSERT_THAT(session->Start(), IsOk());
    EXPECT_THAT(session->Stop(), StatusIs(absl::StatusCode::kUnavailable));
    tensorflow::profiler::XSpace space;
    EXPECT_THAT(session->CollectData(&space), IsOk());
    EXPECT_THAT(space.planes(), IsEmpty());
  }
  server->Shutdown(absl::ToChronoTime(absl::Now() + absl::Seconds(5)));
}

}  // namespace
}  // namespace subprocess
}  // namespace profiler
}  // namespace xla
