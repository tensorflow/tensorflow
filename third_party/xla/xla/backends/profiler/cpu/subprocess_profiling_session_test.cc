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

#include "xla/backends/profiler/cpu/subprocess_profiling_session.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/channel_arguments.h"
#include "xla/backends/profiler/cpu/subprocess_registry.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/rpc/profiler_server.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/profiler_service.grpc.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace cpu {
namespace {

using ::testing::IsEmpty;
using ::testing::Not;

class SubprocessProfilingSessionTest : public ::testing::Test {};

TEST_F(SubprocessProfilingSessionTest, InProcessProfilingTest) {
  auto profiler_server = std::make_unique<tsl::profiler::ProfilerServer>();
  int port = tsl::testing::PickUnusedPortOrDie();
  profiler_server->StartProfilerServer(port);
  std::string server_address = absl::StrCat("localhost:", port);

  grpc::ChannelArguments channel_args;
  channel_args.SetMaxReceiveMessageSize(std::numeric_limits<int32_t>::max());
  std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(
      server_address, grpc::InsecureChannelCredentials(), channel_args);
  channel->WaitForConnected(absl::ToChronoTime(absl::Now() + absl::Seconds(1)));
  std::shared_ptr<tensorflow::grpc::ProfilerService::Stub> stub =
      tensorflow::grpc::ProfilerService::NewStub(channel);

  tensorflow::ProfileOptions options;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<SubprocessProfilingSession> session,
                       SubprocessProfilingSession::Create(
                           SubprocessInfo{tsl::Env::Default()->GetProcessId(),
                                          server_address, stub},
                           options));

  ASSERT_OK(session->Start());
  // HELPS TO REDUCE FLAKINESS; gives time for the subprocess profiler to start.
  absl::SleepFor(absl::Milliseconds(15));
  {
    tsl::profiler::TraceMe trace_me("test");
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_OK(session->Stop());
  tensorflow::profiler::XSpace space;
  ASSERT_OK(session->CollectData(&space));

  ASSERT_THAT(space.planes(), Not(IsEmpty()));
  ASSERT_THAT(space.planes(0).lines(), Not(IsEmpty()));
  ASSERT_THAT(space.planes(0).lines(0).events(), Not(IsEmpty()));
  const auto& event_metadata = space.planes(0).event_metadata();
  EXPECT_EQ(space.planes(0).lines(0).events_size(), 1);
  EXPECT_EQ(event_metadata.at(space.planes(0).lines(0).events(0).metadata_id())
                .name(),
            "test");
}

}  // namespace
}  // namespace cpu
}  // namespace profiler
}  // namespace xla
