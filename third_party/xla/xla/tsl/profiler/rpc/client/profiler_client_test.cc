/* Copyright 2020 The TensorFlow Authors All Rights Reserved.

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
#include "xla/tsl/profiler/rpc/client/profiler_client.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/rpc/client/profiler_client_test_util.h"
#include "xla/tsl/profiler/rpc/profiler_server.h"
#include "xla/tsl/profiler/utils/file_system_utils.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"

namespace tsl {
namespace profiler {
namespace {

using tensorflow::ProfileRequest;
using tensorflow::ProfileResponse;
using ::tsl::profiler::test::DurationApproxLess;
using ::tsl::profiler::test::DurationNear;
using ::tsl::profiler::test::StartServer;

TEST(RemoteProfilerSession, Simple) {
  absl::Duration duration = absl::Milliseconds(10);
  ProfileRequest request;
  std::string service_addr;
  auto server = StartServer(duration, &service_addr, &request);
  absl::Duration grace = absl::Seconds(1);
  absl::Duration max_duration = duration + grace;
  absl::Time approx_start = absl::Now();
  absl::Time deadline = approx_start + max_duration;

  auto remote_session =
      RemoteProfilerSession::Create(service_addr, deadline, request);

  absl::Status status;
  auto response = remote_session->WaitForCompletion(status);
  absl::Duration elapsed = absl::Now() - approx_start;
  // At end of session this evaluates to true still.
  EXPECT_TRUE(status.ok());
  // True because there was no workload traced and subsequently no XEvents.
  EXPECT_TRUE(response->empty_trace());
  // XSpaces are serialized and not returned as tools in ProfileResponse.
  EXPECT_EQ(response->tool_data_size(), 0);
  EXPECT_THAT(elapsed, DurationApproxLess(max_duration));
}

TEST(RemoteProfilerSession, WaitNotCalled) {
  absl::Duration duration = absl::Milliseconds(10);
  ProfileRequest request;
  std::string service_addr;
  auto server = StartServer(duration, &service_addr, &request);
  absl::Duration grace = absl::Seconds(1);
  absl::Duration max_duration = duration + grace;
  absl::Time approx_start = absl::Now();
  absl::Time deadline = approx_start + max_duration;

  auto remote_session =
      RemoteProfilerSession::Create(service_addr, deadline, request);
  absl::Duration elapsed = absl::Now() - approx_start;

  EXPECT_THAT(elapsed, DurationApproxLess(max_duration));
}

TEST(RemoteProfilerSession, Timeout) {
  absl::Duration duration = absl::Milliseconds(10);
  ProfileRequest request;
  std::string service_addr;
  auto server = StartServer(duration, &service_addr, &request);
  // Expect this to fail immediately since deadline was set to the past,
  auto remote_session =
      RemoteProfilerSession::Create(service_addr, absl::Now(), request);
  absl::Status status;
  auto response = remote_session->WaitForCompletion(status);
  // At end of session we will have a timeout error.
  EXPECT_TRUE(absl::IsDeadlineExceeded(status));
  // True because there was no workload traced and subsequently no XEvents.
  EXPECT_TRUE(response->empty_trace());
  // XSpaces are serialized and not returned as tools in ProfileResponse.
  EXPECT_EQ(response->tool_data_size(), 0);
}

TEST(RemoteProfilerSession, LongDeadline) {
  absl::Duration duration = absl::Milliseconds(10);
  ProfileRequest request;
  std::string service_addr;
  auto server = StartServer(duration, &service_addr, &request);

  absl::Time approx_start = absl::Now();
  absl::Duration grace = absl::Seconds(1000);
  absl::Duration max_duration = duration + grace;
  const absl::Time deadline = approx_start + max_duration;

  auto remote_session =
      RemoteProfilerSession::Create(service_addr, deadline, request);
  absl::Status status;
  auto response = remote_session->WaitForCompletion(status);
  absl::Duration elapsed = absl::Now() - approx_start;
  // At end of session this evaluates to true still.
  EXPECT_TRUE(status.ok());
  // True because there was no workload traced and subsequently no XEvents.
  EXPECT_TRUE(response->empty_trace());
  // XSpaces are serialized and not returned as tools in ProfileResponse.
  EXPECT_EQ(response->tool_data_size(), 0);
  // Elapsed time is near profiling duration despite long grace period.
  EXPECT_THAT(elapsed, DurationNear(duration));
}

TEST(RemoteProfilerSession, LongDuration) {
  absl::Duration duration = absl::Seconds(3);
  ProfileRequest request;
  std::string service_addr;
  auto server = StartServer(duration, &service_addr, &request);

  absl::Time approx_start = absl::Now();
  // Empirically determined value.
  absl::Duration grace = absl::Seconds(1);
  absl::Duration max_duration = duration + grace;
  const absl::Time deadline = approx_start + max_duration;

  auto remote_session =
      RemoteProfilerSession::Create(service_addr, deadline, request);
  absl::Status status;
  auto response = remote_session->WaitForCompletion(status);
  absl::Duration elapsed = absl::Now() - approx_start;
  // At end of session this evaluates to true still.
  EXPECT_TRUE(status.ok());
  // True because there was no workload traced and subsequently no XEvents.
  EXPECT_TRUE(response->empty_trace());
  // XSpaces are serialized and not returned as tools in ProfileResponse.
  EXPECT_EQ(response->tool_data_size(), 0);
  // Elapsed time takes longer to complete for larger traces.
  EXPECT_THAT(elapsed, DurationApproxLess(max_duration));
}

TEST(ProfileGrpcTest, ProfileWithOverrideHostname) {
  absl::Duration duration = absl::Milliseconds(100);
  ProfileRequest request;
  std::string service_addr;
  std::unique_ptr<ProfilerServer> server =
      StartServer(duration, &service_addr, &request);

  request.mutable_opts()->set_override_hostname("testhost");

  tensorflow::ProfileResponse response;
  absl::Status status = ProfileGrpc(service_addr, request, &response);
  EXPECT_TRUE(status.ok());

  std::string expected_filepath = ProfilerJoinPath(
      request.repository_root(), request.session_id(), "testhost.xplane.pb");

  EXPECT_TRUE(Env::Default()->FileExists(expected_filepath).ok());
}

TEST(ContinuousProfiler, GetSnapshot) {
  std::string service_addr;
  auto server =
      StartServer(/*duration=*/absl::Milliseconds(100), &service_addr);

  // Start a continuous profiling session.
  ProfileRequest request;
  request.set_session_id("continuous_profiling_session");
  *request.mutable_opts() = ProfilerSession::DefaultOptions();
  tensorflow::ContinuousProfilingResponse response;
  absl::Status status =
      ContinuousProfilingGrpc(service_addr, request, &response);
  ASSERT_TRUE(status.ok());

  // Generate a more substantial CPU workload for the profiler to capture.
  auto start_time = absl::Now();
  while (absl::Now() - start_time < absl::Seconds(1)) {
    volatile int x = 0;
    for (int i = 0; i < 10000; ++i) {
      x++;
    }
  }

  // Get a snapshot.
  tensorflow::GetSnapshotRequest snapshot_request;
  tensorflow::ProfileResponse snapshot_response;
  status = GetSnapshotGrpc(service_addr, snapshot_request, &snapshot_response);
  ASSERT_TRUE(status.ok());
  EXPECT_GT(snapshot_response.xspace().planes_size(), 0);
  snapshot_response.clear_xspace();
  status = GetSnapshotGrpc(service_addr, snapshot_request, &snapshot_response);
  ASSERT_TRUE(status.ok());
  EXPECT_GT(snapshot_response.xspace().planes_size(), 0);
}

TEST(ContinuousProfiler, GetSnapshotBeforeContinuousProfiling) {
  std::string service_addr;
  auto server =
      StartServer(/*duration=*/absl::Milliseconds(100), &service_addr);

  // Get a snapshot before continuous profiling starts.
  tensorflow::GetSnapshotRequest snapshot_request;
  tensorflow::ProfileResponse snapshot_response;
  absl::Status status =
      GetSnapshotGrpc(service_addr, snapshot_request, &snapshot_response);
  EXPECT_TRUE(absl::IsNotFound(status));
  EXPECT_EQ(status.message(), "No continuous profiling session found.");
}

TEST(ContinuousProfiler, ContinuousProfilingTwice) {
  std::string service_addr;
  auto server =
      StartServer(/*duration=*/absl::Milliseconds(100), &service_addr);

  // Start a continuous profiling session.
  ProfileRequest request;
  request.set_session_id("continuous_profiling_session");
  *request.mutable_opts() = ProfilerSession::DefaultOptions();
  tensorflow::ContinuousProfilingResponse response;
  absl::Status status =
      ContinuousProfilingGrpc(service_addr, request, &response);
  ASSERT_TRUE(status.ok());

  // Calling ContinuousProfilingGrpc again should fail.
  status = ContinuousProfilingGrpc(service_addr, request, &response);
  EXPECT_TRUE(absl::IsAlreadyExists(status));
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
