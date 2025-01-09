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
#include "xla/tsl/profiler/rpc/client/remote_profiler_session_manager.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/profiler/rpc/client/profiler_client_test_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/test.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"

namespace tsl {
namespace profiler {
namespace {

using tensorflow::ProfileRequest;
using tensorflow::RemoteProfilerSessionManagerOptions;
using ::tsl::profiler::test::DurationApproxLess;
using ::tsl::profiler::test::DurationNear;
using ::tsl::profiler::test::StartServer;
using ::tsl::testing::TmpDir;
using Response = tsl::profiler::RemoteProfilerSessionManager::Response;

// Tests have intemittently failed with 2s grace period, so setting this to
// a large enough value.
constexpr double kGracePeriodSeconds = 10.0;

// Copied from capture_profile to not introduce a dependency.
ProfileRequest PopulateProfileRequest(
    absl::string_view repository_root, absl::string_view session_id,
    absl::string_view host_name,
    const RemoteProfilerSessionManagerOptions& options) {
  constexpr uint64 kMaxEvents = 1000000;
  const absl::string_view kXPlanePb = "xplane.pb";
  ProfileRequest request;
  // TODO(b/169976117) Remove duration from request.
  request.set_duration_ms(options.profiler_options().duration_ms());
  request.set_max_events(kMaxEvents);
  request.set_repository_root(repository_root.data(), repository_root.size());
  request.set_session_id(session_id.data(), session_id.size());
  request.set_host_name(host_name.data(), host_name.size());
  // XPlane tool is only used by OSS profiler and safely ignored by TPU
  // profiler.
  request.add_tools(kXPlanePb.data(), kXPlanePb.size());
  *request.mutable_opts() = options.profiler_options();
  return request;
}

TEST(RemoteProfilerSessionManagerTest, Simple) {
  absl::Duration duration = absl::Milliseconds(30);
  RemoteProfilerSessionManagerOptions options;
  *options.mutable_profiler_options() = tsl::ProfilerSession::DefaultOptions();
  options.mutable_profiler_options()->set_duration_ms(
      absl::ToInt64Milliseconds(duration));

  std::string service_address;
  auto server = StartServer(duration, &service_address);
  options.add_service_addresses(service_address);
  absl::Time approx_start = absl::Now();
  absl::Duration grace = absl::Seconds(kGracePeriodSeconds);
  absl::Duration max_duration = duration + grace;
  options.set_max_session_duration_ms(absl::ToInt64Milliseconds(max_duration));
  options.set_session_creation_timestamp_ns(absl::ToUnixNanos(approx_start));

  ProfileRequest request =
      PopulateProfileRequest(TmpDir(), "session_id", service_address, options);
  absl::Status status;
  auto sessions =
      RemoteProfilerSessionManager::Create(options, request, status);
  EXPECT_TRUE(status.ok());
  std::vector<Response> responses = sessions->WaitForCompletion();
  absl::Duration elapsed = absl::Now() - approx_start;
  ASSERT_EQ(responses.size(), 1);
  EXPECT_TRUE(responses.back().status.ok());
  EXPECT_TRUE(responses.back().profile_response->empty_trace());
  EXPECT_EQ(responses.back().profile_response->tool_data_size(), 0);
  EXPECT_THAT(elapsed, DurationApproxLess(max_duration));
}

TEST(RemoteProfilerSessionManagerTest, ExpiredDeadline) {
  absl::Duration duration = absl::Milliseconds(30);
  RemoteProfilerSessionManagerOptions options;
  *options.mutable_profiler_options() = tsl::ProfilerSession::DefaultOptions();
  options.mutable_profiler_options()->set_duration_ms(
      absl::ToInt64Milliseconds(duration));

  std::string service_address;
  auto server = StartServer(duration, &service_address);
  options.add_service_addresses(service_address);
  absl::Duration grace = absl::Seconds(kGracePeriodSeconds);
  absl::Duration max_duration = duration + grace;
  options.set_max_session_duration_ms(absl::ToInt64Milliseconds(max_duration));
  // This will create a deadline in the past.
  options.set_session_creation_timestamp_ns(0);

  absl::Time approx_start = absl::Now();
  ProfileRequest request =
      PopulateProfileRequest(TmpDir(), "session_id", service_address, options);
  absl::Status status;
  auto sessions =
      RemoteProfilerSessionManager::Create(options, request, status);
  EXPECT_TRUE(status.ok());
  std::vector<Response> responses = sessions->WaitForCompletion();
  absl::Duration elapsed = absl::Now() - approx_start;
  EXPECT_THAT(elapsed, DurationNear(absl::Seconds(0)));
  ASSERT_EQ(responses.size(), 1);
  EXPECT_TRUE(absl::IsDeadlineExceeded(responses.back().status));
  EXPECT_TRUE(responses.back().profile_response->empty_trace());
  EXPECT_EQ(responses.back().profile_response->tool_data_size(), 0);
}

TEST(RemoteProfilerSessionManagerTest, LongSession) {
  absl::Duration duration = absl::Seconds(3);
  RemoteProfilerSessionManagerOptions options;
  *options.mutable_profiler_options() = tsl::ProfilerSession::DefaultOptions();
  options.mutable_profiler_options()->set_duration_ms(
      absl::ToInt64Milliseconds(duration));

  std::string service_address;
  auto server = StartServer(duration, &service_address);
  options.add_service_addresses(service_address);
  absl::Time approx_start = absl::Now();
  // Empirically determined value.
  absl::Duration grace = absl::Seconds(kGracePeriodSeconds);
  absl::Duration max_duration = duration + grace;
  options.set_max_session_duration_ms(absl::ToInt64Milliseconds(max_duration));
  options.set_session_creation_timestamp_ns(absl::ToUnixNanos(approx_start));

  ProfileRequest request =
      PopulateProfileRequest(TmpDir(), "session_id", service_address, options);
  absl::Status status;
  auto sessions =
      RemoteProfilerSessionManager::Create(options, request, status);
  EXPECT_TRUE(status.ok());
  std::vector<Response> responses = sessions->WaitForCompletion();
  absl::Duration elapsed = absl::Now() - approx_start;
  ASSERT_EQ(responses.size(), 1);
  EXPECT_TRUE(responses.back().status.ok());
  EXPECT_TRUE(responses.back().profile_response->empty_trace());
  EXPECT_EQ(responses.back().profile_response->tool_data_size(), 0);
  EXPECT_THAT(elapsed, DurationApproxLess(max_duration));
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
