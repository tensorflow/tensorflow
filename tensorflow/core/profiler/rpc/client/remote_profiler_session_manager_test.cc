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
#include "tensorflow/core/profiler/rpc/client/remote_profiler_session_manager.h"

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/profiler_service.pb.h"
#include "tensorflow/core/profiler/rpc/client/profiler_client_test_util.h"
#include "tensorflow/core/profiler/rpc/profiler_server.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::test::DurationApproxLess;
using ::tensorflow::profiler::test::DurationNear;
using ::tensorflow::profiler::test::StartServer;
using Response = tensorflow::profiler::RemoteProfilerSessionManager::Response;

TEST(RemoteProfilerSessionManagerTest, Simple) {
  absl::Duration duration = absl::Milliseconds(30);
  RemoteProfilerSessionManagerOptions options =
      RemoteProfilerSessionManager::DefaultOptions();
  options.mutable_profiler_options()->set_duration_ms(
      absl::ToInt64Milliseconds(duration));

  std::string service_addresses;
  auto server = StartServer(duration, &service_addresses);
  options.add_service_addresses(service_addresses);
  absl::Time approx_start = absl::Now();
  absl::Duration grace = absl::Seconds(1);
  absl::Duration max_duration = duration + grace;
  options.set_max_session_duration_ms(absl::ToInt64Milliseconds(max_duration));
  options.set_session_creation_timestamp_ns(absl::ToUnixNanos(approx_start));

  Status status;
  auto sessions = RemoteProfilerSessionManager::Create(options, status);
  EXPECT_TRUE(status.ok());
  std::vector<Response> responses = sessions->WaitForCompletion();
  absl::Duration elapsed = absl::Now() - approx_start;
  ASSERT_EQ(responses.size(), 1);
  EXPECT_TRUE(responses.back().status.ok());
  EXPECT_FALSE(responses.back().profile_response->empty_trace());
  EXPECT_GT(responses.back().profile_response->tool_data_size(), 0);
  EXPECT_THAT(elapsed, DurationApproxLess(max_duration));
}

TEST(RemoteProfilerSessionManagerTest, ExpiredDeadline) {
  absl::Duration duration = absl::Milliseconds(30);
  RemoteProfilerSessionManagerOptions options =
      RemoteProfilerSessionManager::DefaultOptions();
  options.mutable_profiler_options()->set_duration_ms(
      absl::ToInt64Milliseconds(duration));

  std::string service_addresses;
  auto server = StartServer(duration, &service_addresses);
  options.add_service_addresses(service_addresses);
  absl::Duration grace = absl::Seconds(1);
  absl::Duration max_duration = duration + grace;
  options.set_max_session_duration_ms(absl::ToInt64Milliseconds(max_duration));
  // This will create a deadline in the past.
  options.set_session_creation_timestamp_ns(0);

  absl::Time approx_start = absl::Now();
  Status status;
  auto sessions = RemoteProfilerSessionManager::Create(options, status);
  EXPECT_TRUE(status.ok());
  std::vector<Response> responses = sessions->WaitForCompletion();
  absl::Duration elapsed = absl::Now() - approx_start;
  EXPECT_THAT(elapsed, DurationNear(absl::Seconds(0)));
  ASSERT_EQ(responses.size(), 1);
  EXPECT_EQ(responses.back().status.code(), error::DEADLINE_EXCEEDED);
  EXPECT_FALSE(responses.back().profile_response->empty_trace());
  EXPECT_EQ(responses.back().profile_response->tool_data_size(), 0);
}

TEST(RemoteProfilerSessionManagerTest, LongSession) {
  absl::Duration duration = absl::Seconds(3);
  RemoteProfilerSessionManagerOptions options =
      RemoteProfilerSessionManager::DefaultOptions();
  options.mutable_profiler_options()->set_duration_ms(
      absl::ToInt64Milliseconds(duration));

  std::string service_addresses;
  auto server = StartServer(duration, &service_addresses);
  options.add_service_addresses(service_addresses);
  absl::Time approx_start = absl::Now();
  // Empirically determined value.
  absl::Duration grace = absl::Seconds(20);
  absl::Duration max_duration = duration + grace;
  options.set_max_session_duration_ms(absl::ToInt64Milliseconds(max_duration));
  options.set_session_creation_timestamp_ns(absl::ToUnixNanos(approx_start));

  Status status;
  auto sessions = RemoteProfilerSessionManager::Create(options, status);
  EXPECT_TRUE(status.ok());
  std::vector<Response> responses = sessions->WaitForCompletion();
  absl::Duration elapsed = absl::Now() - approx_start;
  ASSERT_EQ(responses.size(), 1);
  EXPECT_TRUE(responses.back().status.ok());
  EXPECT_FALSE(responses.back().profile_response->empty_trace());
  EXPECT_GT(responses.back().profile_response->tool_data_size(), 0);
  EXPECT_THAT(elapsed, DurationApproxLess(max_duration));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
