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

#include <memory>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/profiler/rpc/profiler_server.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/traceme.h"

// The main method for the subprocess created by the
// subprocess_profiling_session_test.cc.
int main(int argc, char** argv) {
  int port = 0;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("port", &port, "Port to start the profiler server on.")};
  tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(argv[0], &argc, &argv);

  auto profiler_server = std::make_unique<tsl::profiler::ProfilerServer>();
  profiler_server->StartProfilerServer(port);

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "subprocess_test", 2);
  while (true) {
    tsl::profiler::TraceMe root_trace_me([&] {
      return tsl::profiler::TraceMeEncode(
          absl::StrCat("root_", port),
          {{"_r", 1}, {"_time", absl::ToUnixMillis(absl::Now())}});
    });
    tsl::profiler::TraceMeProducer producer([&] {
      return tsl::profiler::TraceMeEncode(
          absl::StrCat("producer_", port),
          {{"_time", absl::ToUnixMillis(absl::Now())}});
    });
    absl::SleepFor(absl::Milliseconds(5));
    thread_pool.Schedule([context_id = producer.GetContextId(), port = port] {
      tsl::profiler::TraceMeConsumer consumer(
          [&] {
            return tsl::profiler::TraceMeEncode(
                absl::StrCat("consumer_", port),
                {{"_time", absl::ToUnixMillis(absl::Now())}});
          },
          context_id);
      tsl::profiler::TraceMe trace_me([&] {
        return tsl::profiler::TraceMeEncode(
            absl::StrCat("child_", port),
            {{"_time", absl::ToUnixMillis(absl::Now())}});
      });
      absl::SleepFor(absl::Milliseconds(5));
    });
  }
  return 0;
}
