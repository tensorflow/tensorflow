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

#include "xla/tsl/profiler/rpc/profiler_service_impl.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/server_context.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/profiler_service.grpc.pb.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"

namespace tsl {
namespace profiler {
namespace {

using ::testing::ElementsAre;
using ::testing::Property;

// Registers a profiler factory that records the options of every profiler
// session created while the test runs.
class ProfilerServiceImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    RegisterProfilerFactory([this](const tensorflow::ProfileOptions& options)
                                -> std::unique_ptr<ProfilerInterface> {
      absl::MutexLock lock(mu_);
      session_options_.push_back(options);
      return nullptr;
    });
  }

  void TearDown() override { ClearRegisteredProfilersForTest(); }

  std::vector<tensorflow::ProfileOptions> session_options() {
    absl::MutexLock lock(mu_);
    return session_options_;
  }

 private:
  absl::Mutex mu_;
  std::vector<tensorflow::ProfileOptions> session_options_ ABSL_GUARDED_BY(mu_);
};

// Returns a request that is served immediately (`opts.duration_ms` is 0) and
// whose profile data is returned in the response instead of being saved.
tensorflow::ProfileRequest MakeProfileRequest(absl::string_view session_id) {
  tensorflow::ProfileRequest request;
  *request.mutable_opts() = ProfilerSession::DefaultOptions();
  request.mutable_opts()->set_duration_ms(0);
  request.set_session_id(session_id);
  request.set_emit_xspace(true);
  return request;
}

TEST_F(ProfilerServiceImplTest, ProfilePropagatesSessionIdToProfilers) {
  std::unique_ptr<tensorflow::grpc::ProfilerService::Service> service =
      CreateProfilerService();
  ::grpc::ServerContext context;
  tensorflow::ProfileRequest request = MakeProfileRequest("request_session");
  tensorflow::ProfileResponse response;

  EXPECT_TRUE(service->Profile(&context, &request, &response).ok());

  EXPECT_THAT(session_options(),
              ElementsAre(Property(&tensorflow::ProfileOptions::session_id,
                                   "request_session")));
}

TEST_F(ProfilerServiceImplTest, ProfileKeepsSessionIdOfOptions) {
  std::unique_ptr<tensorflow::grpc::ProfilerService::Service> service =
      CreateProfilerService();
  ::grpc::ServerContext context;
  tensorflow::ProfileRequest request = MakeProfileRequest("request_session");
  request.mutable_opts()->set_session_id("options_session");
  tensorflow::ProfileResponse response;

  EXPECT_TRUE(service->Profile(&context, &request, &response).ok());

  EXPECT_THAT(session_options(),
              ElementsAre(Property(&tensorflow::ProfileOptions::session_id,
                                   "options_session")));
}

TEST_F(ProfilerServiceImplTest,
       StartContinuousProfilingPropagatesSessionIdToProfilers) {
  std::unique_ptr<tensorflow::grpc::ProfilerService::Service> service =
      CreateProfilerService();
  ::grpc::ServerContext start_context;
  tensorflow::ProfileRequest request = MakeProfileRequest("request_session");
  tensorflow::ContinuousProfilingResponse start_response;

  EXPECT_TRUE(
      service
          ->StartContinuousProfiling(&start_context, &request, &start_response)
          .ok());

  EXPECT_THAT(session_options(),
              ElementsAre(Property(&tensorflow::ProfileOptions::session_id,
                                   "request_session")));

  ::grpc::ServerContext stop_context;
  tensorflow::StopContinuousProfilingRequest stop_request;
  tensorflow::StopContinuousProfilingResponse stop_response;
  EXPECT_TRUE(service
                  ->StopContinuousProfiling(&stop_context, &stop_request,
                                            &stop_response)
                  .ok());
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
