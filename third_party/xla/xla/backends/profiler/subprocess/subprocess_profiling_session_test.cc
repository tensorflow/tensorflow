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

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
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
#include "xla/tsl/profiler/utils/timestamp_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
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
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

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

// In-process ProfilerService that records requests and returns a fixed XSpace.
// Profile blocks until Terminate releases it or the RPC is cancelled.
class FakeProfilerService : public tensorflow::grpc::ProfilerService::Service {
 public:
  using Metadata = absl::flat_hash_map<std::string, std::string>;

  ::grpc::Status Profile(::grpc::ServerContext* context,
                         const tensorflow::ProfileRequest* request,
                         tensorflow::ProfileResponse* response) override {
    {
      absl::MutexLock lock(mu_);
      profile_request_ = *request;
      profile_metadata_ = ToMap(*context);
      if (!profile_status_.ok()) {
        return profile_status_;
      }
    }
    while (true) {
      if (context->IsCancelled()) {
        return ::grpc::Status::CANCELLED;
      }
      absl::MutexLock lock(mu_);
      if (mu_.AwaitWithTimeout(absl::Condition(&release_profile_),
                               absl::Milliseconds(10))) {
        break;
      }
    }
    absl::MutexLock lock(mu_);
    *response->mutable_xspace() = xspace_;
    response->set_empty_trace(false);
    return ::grpc::Status::OK;
  }

  ::grpc::Status Terminate(::grpc::ServerContext* context,
                           const tensorflow::TerminateRequest* request,
                           tensorflow::TerminateResponse* response) override {
    absl::MutexLock lock(mu_);
    terminate_request_ = *request;
    terminate_metadata_ = ToMap(*context);
    if (!terminate_status_.ok()) {
      return terminate_status_;
    }
    if (release_on_terminate_) {
      release_profile_ = true;
    }
    return ::grpc::Status::OK;
  }

  void set_xspace(const tensorflow::profiler::XSpace& xspace) {
    absl::MutexLock lock(mu_);
    xspace_ = xspace;
  }
  // Makes Profile return `status` right away instead of waiting.
  void set_profile_status(const ::grpc::Status& status) {
    absl::MutexLock lock(mu_);
    profile_status_ = status;
  }
  void set_terminate_status(const ::grpc::Status& status) {
    absl::MutexLock lock(mu_);
    terminate_status_ = status;
  }
  void set_release_on_terminate(bool release) {
    absl::MutexLock lock(mu_);
    release_on_terminate_ = release;
  }
  // Lets Profile return its XSpace without waiting for Terminate.
  void release_profile() {
    absl::MutexLock lock(mu_);
    release_profile_ = true;
  }
  tensorflow::ProfileRequest profile_request() {
    absl::MutexLock lock(mu_);
    return profile_request_;
  }
  tensorflow::TerminateRequest terminate_request() {
    absl::MutexLock lock(mu_);
    return terminate_request_;
  }
  Metadata profile_metadata() {
    absl::MutexLock lock(mu_);
    return profile_metadata_;
  }
  Metadata terminate_metadata() {
    absl::MutexLock lock(mu_);
    return terminate_metadata_;
  }

 private:
  static Metadata ToMap(const ::grpc::ServerContext& context) {
    Metadata metadata;
    for (const auto& [key, value] : context.client_metadata()) {
      metadata[std::string(key.data(), key.size())] =
          std::string(value.data(), value.size());
    }
    return metadata;
  }

  absl::Mutex mu_;
  tensorflow::profiler::XSpace xspace_ ABSL_GUARDED_BY(mu_);
  ::grpc::Status profile_status_ ABSL_GUARDED_BY(mu_);
  ::grpc::Status terminate_status_ ABSL_GUARDED_BY(mu_);
  bool release_on_terminate_ ABSL_GUARDED_BY(mu_) = true;
  bool release_profile_ ABSL_GUARDED_BY(mu_) = false;
  tensorflow::ProfileRequest profile_request_ ABSL_GUARDED_BY(mu_);
  tensorflow::TerminateRequest terminate_request_ ABSL_GUARDED_BY(mu_);
  Metadata profile_metadata_ ABSL_GUARDED_BY(mu_);
  Metadata terminate_metadata_ ABSL_GUARDED_BY(mu_);
};

constexpr int32_t kFakeSubprocessPid = 4242;

// Adds a plane with one event and, if `pid` is set, a kProcessId stat.
void AddPlane(tensorflow::profiler::XSpace& space, absl::string_view name,
              std::optional<int32_t> pid = std::nullopt) {
  tensorflow::profiler::XPlane* plane = space.add_planes();
  tsl::profiler::XPlaneBuilder builder(plane);
  builder.SetName(name);
  tsl::profiler::XLineBuilder line = builder.GetOrCreateLine(0);
  tsl::profiler::XEventBuilder event =
      line.AddEvent(*builder.GetOrCreateEventMetadata("event"));
  event.SetTimestampNs(1000);
  event.SetDurationNs(10);
  if (pid.has_value()) {
    builder.AddStatValue(
        *builder.GetOrCreateStatMetadata(
            tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kProcessId)),
        *pid);
  }
}

std::vector<std::string> PlaneNames(const tensorflow::profiler::XSpace& space) {
  std::vector<std::string> names;
  for (const auto& plane : space.planes()) {
    names.push_back(plane.name());
  }
  return names;
}

class FakeSubprocessProfilingSessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ::grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(),
                             &port);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    ASSERT_NE(server_, nullptr);
    ASSERT_GT(port, 0);
    subprocess_info_.pid = kFakeSubprocessPid;
    subprocess_info_.address = absl::StrCat("localhost:", port);
    subprocess_info_.profiler_stub =
        tensorflow::grpc::ProfilerService::NewStub(::grpc::CreateChannel(
            subprocess_info_.address, ::grpc::InsecureChannelCredentials()));
  }

  void TearDown() override {
    if (server_ != nullptr) {
      server_->Shutdown(absl::ToChronoTime(absl::Now() + absl::Seconds(5)));
    }
  }

  // Runs Start, Stop and CollectData and returns the merged XSpace.
  tensorflow::profiler::XSpace RunSession(
      const tensorflow::ProfileOptions& options =
          tsl::ProfilerSession::DefaultOptions()) {
    tensorflow::profiler::XSpace space;
    absl::StatusOr<std::unique_ptr<SubprocessProfilingSession>> session =
        SubprocessProfilingSession::Create(subprocess_info_, options);
    EXPECT_THAT(session, IsOk());
    if (!session.ok()) {
      return space;
    }
    EXPECT_THAT((*session)->Start(), IsOk());
    EXPECT_THAT((*session)->Stop(), IsOk());
    EXPECT_THAT((*session)->CollectData(&space), IsOk());
    // A session that completed must not report a collection error.
    EXPECT_THAT(space.errors(), IsEmpty());
    return space;
  }

  FakeProfilerService service_;
  std::unique_ptr<::grpc::Server> server_;
  SubprocessInfo subprocess_info_;
};

TEST_F(FakeSubprocessProfilingSessionTest, SendsMetadataOnProfileAndTerminate) {
  subprocess_info_.grpc_metadata = {{"x-test-client-id", "7"}};
  RunSession();

  EXPECT_THAT(service_.profile_metadata(),
              Contains(Pair("x-test-client-id", "7")));
  EXPECT_THAT(service_.terminate_metadata(),
              Contains(Pair("x-test-client-id", "7")));
  EXPECT_THAT(service_.terminate_request().session_id(),
              Eq(service_.profile_request().session_id()));
}

TEST_F(FakeSubprocessProfilingSessionTest, ForcesCpuDeviceTypeByDefault) {
  tensorflow::ProfileOptions options = tsl::ProfilerSession::DefaultOptions();
  options.set_device_type(tensorflow::ProfileOptions::TPU);
  RunSession(options);

  EXPECT_EQ(service_.profile_request().opts().device_type(),
            tensorflow::ProfileOptions::CPU);
}

TEST_F(FakeSubprocessProfilingSessionTest, DeviceOwnerKeepsDeviceType) {
  subprocess_info_.device_owner = true;
  tensorflow::ProfileOptions options = tsl::ProfilerSession::DefaultOptions();
  options.set_device_type(tensorflow::ProfileOptions::TPU);
  RunSession(options);

  EXPECT_EQ(service_.profile_request().opts().device_type(),
            tensorflow::ProfileOptions::TPU);
}

TEST_F(FakeSubprocessProfilingSessionTest, DropsPlanesWithoutPidByDefault) {
  tensorflow::profiler::XSpace subprocess_space;
  AddPlane(subprocess_space, "/host:CPU");
  AddPlane(subprocess_space, "/device:TPU:0");
  AddPlane(subprocess_space, "Task Environment", /*pid=*/99);
  service_.set_xspace(subprocess_space);

  tensorflow::profiler::XSpace space = RunSession();

  EXPECT_THAT(PlaneNames(space), ElementsAre("Task Environment [99]"));
}

TEST_F(FakeSubprocessProfilingSessionTest,
       AddsPidSuffixToDevicePlanesByDefault) {
  tensorflow::profiler::XSpace subprocess_space;
  AddPlane(subprocess_space, "/device:TPU:0", /*pid=*/99);
  AddPlane(subprocess_space, "/host:metadata", /*pid=*/99);
  service_.set_xspace(subprocess_space);

  tensorflow::profiler::XSpace space = RunSession();

  EXPECT_THAT(PlaneNames(space),
              ElementsAre("/device:TPU:0 [99]", "/host:metadata [99]"));
}

TEST_F(FakeSubprocessProfilingSessionTest, KeepPlanesWithoutPidTagsThem) {
  subprocess_info_.keep_planes_without_pid = true;
  tensorflow::profiler::XSpace subprocess_space;
  AddPlane(subprocess_space, "/host:CPU");
  AddPlane(subprocess_space, "/device:TPU:0");
  service_.set_xspace(subprocess_space);

  tensorflow::profiler::XSpace space = RunSession();

  // Without device_owner every plane gets the pid suffix.
  EXPECT_THAT(PlaneNames(space),
              UnorderedElementsAre(
                  absl::StrCat("/host:CPU [", kFakeSubprocessPid, "]"),
                  absl::StrCat("/device:TPU:0 [", kFakeSubprocessPid, "]")));
}

TEST_F(FakeSubprocessProfilingSessionTest,
       KeepPlanesWithoutPidKeepsExistingPid) {
  subprocess_info_.keep_planes_without_pid = true;
  tensorflow::profiler::XSpace subprocess_space;
  AddPlane(subprocess_space, "/host:CPU", /*pid=*/99);
  service_.set_xspace(subprocess_space);

  tensorflow::profiler::XSpace space = RunSession();

  EXPECT_THAT(PlaneNames(space), ElementsAre("/host:CPU [99]"));
}

TEST_F(FakeSubprocessProfilingSessionTest,
       DeviceOwnerMovesDevicePlanesVerbatim) {
  subprocess_info_.device_owner = true;
  subprocess_info_.keep_planes_without_pid = true;
  tensorflow::profiler::XSpace subprocess_space;
  AddPlane(subprocess_space, "/host:CPU");
  AddPlane(subprocess_space, "/device:TPU:0");
  AddPlane(subprocess_space, "#Chip0 Core0");
  AddPlane(subprocess_space, "/host:metadata");
  AddPlane(subprocess_space, "Task Environment");
  subprocess_space.add_warnings("subprocess warning");
  service_.set_xspace(subprocess_space);

  tensorflow::profiler::XSpace space;
  AddPlane(space, "/host:CPU", /*pid=*/1);
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SubprocessProfilingSession::Create(
          subprocess_info_, tsl::ProfilerSession::DefaultOptions()));
  ASSERT_THAT(session->Start(), IsOk());
  ASSERT_THAT(session->Stop(), IsOk());
  ASSERT_THAT(session->CollectData(&space), IsOk());

  EXPECT_THAT(
      PlaneNames(space),
      UnorderedElementsAre(
          "/host:CPU", "/device:TPU:0", "#Chip0 Core0", "/host:metadata",
          absl::StrCat("/host:CPU [", kFakeSubprocessPid, "]"),
          absl::StrCat("Task Environment [", kFakeSubprocessPid, "]")));
  EXPECT_THAT(space.warnings(), Contains("subprocess warning"));
  EXPECT_THAT(space.errors(), IsEmpty());
  for (const auto& plane : space.planes()) {
    if (plane.name() == "/device:TPU:0") {
      EXPECT_EQ(plane.lines_size(), 1);
      EXPECT_EQ(plane.lines(0).events_size(), 1);
    }
  }
}

TEST_F(FakeSubprocessProfilingSessionTest,
       DenormalizesTimestampsIncludingVerbatimPlanes) {
  subprocess_info_.device_owner = true;
  subprocess_info_.keep_planes_without_pid = true;
  // Small enough that XEventVisitor's floating-point timestamp math is exact.
  constexpr uint64_t kStartNs = 1'000'000'000'000;
  constexpr uint64_t kStopNs = kStartNs + 5'000'000'000;
  tensorflow::profiler::XSpace subprocess_space;
  // AddPlane's line starts at 0 and its event at 1000 ns, relative to the
  // subprocess session start.
  AddPlane(subprocess_space, "/device:TPU:0");
  AddPlane(subprocess_space, "/host:CPU");
  tsl::profiler::SetSessionTimestamps(kStartNs, kStopNs, subprocess_space);
  service_.set_xspace(subprocess_space);

  tensorflow::profiler::XSpace space = RunSession();

  const std::string suffixed_host_plane =
      absl::StrCat("/host:CPU [", kFakeSubprocessPid, "]");
  int checked_planes = 0;
  for (const auto& plane : space.planes()) {
    if (plane.name() != "/device:TPU:0" &&
        plane.name() != suffixed_host_plane) {
      continue;
    }
    ++checked_planes;
    tsl::profiler::XPlaneVisitor visitor =
        tsl::profiler::CreateTfXPlaneVisitor(&plane);
    int events = 0;
    visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
      EXPECT_EQ(line.TimestampNs(), kStartNs) << plane.name();
      line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
        ++events;
        EXPECT_EQ(event.TimestampNs(), kStartNs + 1000) << plane.name();
      });
    });
    EXPECT_EQ(events, 1) << plane.name();
  }
  // The verbatim device plane must be denormalized too, not only the ones that
  // go through MergeSubprocessXSpace.
  EXPECT_EQ(checked_planes, 2);
}

TEST_F(FakeSubprocessProfilingSessionTest, TerminateFailureDoesNotHang) {
  service_.set_terminate_status(
      ::grpc::Status(::grpc::StatusCode::UNAVAILABLE, "terminate failed"));
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SubprocessProfilingSession::Create(
          subprocess_info_, tsl::ProfilerSession::DefaultOptions()));
  ASSERT_THAT(session->Start(), IsOk());

  EXPECT_THAT(session->Stop(), StatusIs(absl::StatusCode::kUnavailable));
  EXPECT_THAT(session->Stop(), StatusIs(absl::StatusCode::kFailedPrecondition));
  tensorflow::profiler::XSpace space;
  EXPECT_THAT(session->CollectData(&space), IsOk());
  EXPECT_THAT(space.planes(), IsEmpty());
  // The Profile RPC ends as CANCELLED, so the Terminate error is what explains
  // it.
  EXPECT_THAT(
      space.errors(),
      ElementsAre(
          AllOf(HasSubstr("Failed to collect profile from subprocess"),
                HasSubstr("CANCELLED"),
                HasSubstr("Terminate failed: UNAVAILABLE: terminate failed"))));
  EXPECT_THAT(space.warnings(), IsEmpty());
}

TEST_F(FakeSubprocessProfilingSessionTest, ProfileRpcTimeoutBoundsStop) {
  service_.set_release_on_terminate(false);
  subprocess_info_.profile_rpc_timeout = absl::Milliseconds(500);
  subprocess_info_.terminate_rpc_timeout = absl::Seconds(10);
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SubprocessProfilingSession::Create(
          subprocess_info_, tsl::ProfilerSession::DefaultOptions()));
  ASSERT_THAT(session->Start(), IsOk());

  EXPECT_THAT(session->Stop(), StatusIs(absl::StatusCode::kDeadlineExceeded));
  tensorflow::profiler::XSpace space;
  EXPECT_THAT(session->CollectData(&space), IsOk());
  EXPECT_THAT(space.planes(), IsEmpty());
  EXPECT_THAT(
      space.errors(),
      ElementsAre(AllOf(HasSubstr("Failed to collect profile from subprocess"),
                        HasSubstr("DEADLINE_EXCEEDED"),
                        Not(HasSubstr("Terminate")))));
}

TEST_F(FakeSubprocessProfilingSessionTest, ProfileRpcErrorIsReported) {
  service_.set_profile_status(
      ::grpc::Status(::grpc::StatusCode::INTERNAL, "ingestion failed"));
  tensorflow::profiler::XSpace subprocess_space;
  AddPlane(subprocess_space, "/host:CPU", /*pid=*/99);
  service_.set_xspace(subprocess_space);
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SubprocessProfilingSession::Create(
          subprocess_info_, tsl::ProfilerSession::DefaultOptions()));
  ASSERT_THAT(session->Start(), IsOk());

  EXPECT_THAT(session->Stop(), StatusIs(absl::StatusCode::kInternal));
  tensorflow::profiler::XSpace space;
  EXPECT_THAT(session->CollectData(&space), IsOk());
  EXPECT_THAT(space.planes(), IsEmpty());
  EXPECT_THAT(
      space.errors(),
      ElementsAre(AllOf(HasSubstr("Failed to collect profile from subprocess"),
                        HasSubstr(absl::StrCat("pid: ", kFakeSubprocessPid)),
                        HasSubstr("INTERNAL: ingestion failed"))));
  EXPECT_THAT(space.warnings(), IsEmpty());
}

TEST_F(FakeSubprocessProfilingSessionTest,
       CollectDataWithoutStartReportsError) {
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SubprocessProfilingSession::Create(
          subprocess_info_, tsl::ProfilerSession::DefaultOptions()));
  tensorflow::profiler::XSpace space;
  EXPECT_THAT(session->CollectData(&space), IsOk());
  EXPECT_THAT(space.errors(), ElementsAre(HasSubstr("not started")));
}

TEST_F(FakeSubprocessProfilingSessionTest,
       CollectDataBeforeStopReportsErrorAndMergesNothing) {
  tensorflow::profiler::XSpace subprocess_space;
  AddPlane(subprocess_space, "/host:CPU", /*pid=*/99);
  service_.set_xspace(subprocess_space);
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SubprocessProfilingSession::Create(
          subprocess_info_, tsl::ProfilerSession::DefaultOptions()));
  ASSERT_THAT(session->Start(), IsOk());

  tensorflow::profiler::XSpace space;
  EXPECT_THAT(session->CollectData(&space), IsOk());
  EXPECT_THAT(space.planes(), IsEmpty());
  EXPECT_THAT(space.errors(), ElementsAre(HasSubstr("Stop() was not called")));
  EXPECT_THAT(space.warnings(), IsEmpty());
}

TEST_F(FakeSubprocessProfilingSessionTest,
       TerminateFailureAfterProfileReturnedKeepsDataWithoutError) {
  tensorflow::profiler::XSpace subprocess_space;
  AddPlane(subprocess_space, "/host:CPU", /*pid=*/99);
  service_.set_xspace(subprocess_space);
  service_.set_terminate_status(
      ::grpc::Status(::grpc::StatusCode::UNAVAILABLE, "terminate failed"));
  // Profile returns its data as soon as it arrives.
  service_.release_profile();
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SubprocessProfilingSession::Create(
          subprocess_info_, tsl::ProfilerSession::DefaultOptions()));
  ASSERT_THAT(session->Start(), IsOk());
  // Give the Profile response time to reach the client before Terminate fails
  // and Stop() cancels the call. The response travels on the same connection
  // ahead of the Terminate response, so the cancel finds the call complete.
  absl::SleepFor(absl::Milliseconds(500));

  EXPECT_THAT(session->Stop(), StatusIs(absl::StatusCode::kUnavailable));
  tensorflow::profiler::XSpace space;
  EXPECT_THAT(session->CollectData(&space), IsOk());
  // The Profile RPC succeeded, so its data is merged and nothing is reported.
  EXPECT_THAT(PlaneNames(space), ElementsAre("/host:CPU [99]"));
  EXPECT_THAT(space.errors(), IsEmpty());
}

TEST_F(FakeSubprocessProfilingSessionTest, DestroyWithoutStop) {
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SubprocessProfilingSession::Create(
          subprocess_info_, tsl::ProfilerSession::DefaultOptions()));
  ASSERT_THAT(session->Start(), IsOk());
  // The destructor cancels and drains the pending Profile RPC.
  session.reset();
}

}  // namespace
}  // namespace subprocess
}  // namespace profiler
}  // namespace xla
