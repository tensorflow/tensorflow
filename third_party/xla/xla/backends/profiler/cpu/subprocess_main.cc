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
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/profiler/rpc/profiler_server.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"
#include "tsl/profiler/lib/traceme.h"

int main(int argc, char** argv) {
  int port = 0;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("port", &port, "Port to start the profiler server on.")};
  tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(argv[0], &argc, &argv);

  auto profiler_server = std::make_unique<tsl::profiler::ProfilerServer>();
  profiler_server->StartProfilerServer(port);
  std::string trace_me_name = absl::StrCat("subprocess_test_", port);
  // TraceMe requires an explicit string_view otherwise it will resolve to the
  // deleted std::string& constructor.
  absl::string_view trace_me_name_ref = trace_me_name;

  while (true) {
    tsl::profiler::TraceMe trace_me(trace_me_name_ref);
    absl::SleepFor(absl::Milliseconds(1));
  }
  return 0;
}
