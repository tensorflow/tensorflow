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
// GRPC client to perform on-demand profiling

#ifndef TENSORFLOW_TSL_PROFILER_RPC_CLIENT_PROFILER_CLIENT_TEST_UTIL_H_
#define TENSORFLOW_TSL_PROFILER_RPC_CLIENT_PROFILER_CLIENT_TEST_UTIL_H_

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"
#include "tsl/profiler/rpc/profiler_server.h"

namespace tsl {
namespace profiler {
namespace test {

using tensorflow::ProfileRequest;

inline std::unique_ptr<ProfilerServer> StartServer(
    absl::Duration duration, std::string* service_address,
    ProfileRequest* request = nullptr) {
  auto profiler_server = absl::make_unique<ProfilerServer>();
  int port = testing::PickUnusedPortOrDie();
  profiler_server->StartProfilerServer(port);

  DCHECK(service_address);
  *service_address = absl::StrCat("localhost:", port);

  if (request) {
    request->set_duration_ms(absl::ToInt64Milliseconds(duration));
    request->set_max_events(10000);
    *request->mutable_opts() = ProfilerSession::DefaultOptions();
    request->mutable_opts()->set_duration_ms(
        absl::ToInt64Milliseconds(duration));
    request->set_session_id("test_session");
    request->set_host_name(*service_address);
    request->set_repository_root(testing::TmpDir());
  }

  LOG(INFO) << "Started " << *service_address << " at " << absl::Now();
  LOG(INFO) << "Duration: " << duration;

  return profiler_server;
}

inline ::testing::Matcher<absl::Duration> DurationNear(
    const absl::Duration duration, absl::Duration epsilon = absl::Seconds(1)) {
  return ::testing::AllOf(::testing::Ge(duration - epsilon),
                          ::testing::Le(duration + epsilon));
}

inline ::testing::Matcher<absl::Duration> DurationApproxLess(
    const absl::Duration duration, absl::Duration epsilon = absl::Seconds(1)) {
  return ::testing::Le(duration + epsilon);
}

}  // namespace test
}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_RPC_CLIENT_PROFILER_CLIENT_TEST_UTIL_H_
