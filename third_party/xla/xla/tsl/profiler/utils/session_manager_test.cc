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

#include "xla/tsl/profiler/utils/session_manager.h"

#include <string>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/test.h"

namespace tsl {
namespace profiler {
namespace {

using tensorflow::RemoteProfilerSessionManagerOptions;

TEST(SessionManagerTest, OptionsWithSessionIdTest) {
  absl::string_view logdir = "/tmp/logdir";
  absl::flat_hash_map<std::string, std::variant<bool, int, std::string>> opts;
  opts["session_id"] = std::string("test_session_id");
  RemoteProfilerSessionManagerOptions options =
      GetRemoteSessionManagerOptionsLocked(logdir, opts);
  EXPECT_EQ(options.profiler_options().session_id(), "test_session_id");
}

TEST(SessionManagerTest, OptionsWithoutSessionIdTest) {
  absl::string_view logdir = "/tmp/logdir";
  absl::flat_hash_map<std::string, std::variant<bool, int, std::string>> opts;
  RemoteProfilerSessionManagerOptions options =
      GetRemoteSessionManagerOptionsLocked(logdir, opts);
  EXPECT_EQ(options.profiler_options().session_id().empty(), true);
}

TEST(SessionManagerTest, MultiHostDefaultDelayTest) {
  absl::string_view service_addresses = "host1:123,host2:456";
  absl::string_view logdir = "/tmp/logdir";
  absl::flat_hash_map<std::string, std::variant<bool, int, std::string>> opts;
  bool is_cloud_tpu_session;

  RemoteProfilerSessionManagerOptions options =
      GetRemoteSessionManagerOptionsLocked(service_addresses, logdir,
                                           /*worker_list=*/"",
                                           /*include_dataset_ops=*/false,
                                           /*duration_ms=*/100, opts,
                                           &is_cloud_tpu_session);
  EXPECT_EQ(options.delay_ms(), 3000);
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
