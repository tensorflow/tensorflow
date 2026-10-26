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

#include <gmock/gmock.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/test.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace tsl {
namespace profiler {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::tensorflow::RemoteProfilerSessionManagerOptions;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;

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

TEST(SessionManagerTest, UseSystemHostnameTest) {
  absl::string_view logdir = "/tmp/logdir";
  absl::flat_hash_map<std::string, std::variant<bool, int, std::string>> opts =
      {{"use_system_hostname", true}};
  RemoteProfilerSessionManagerOptions options =
      GetRemoteSessionManagerOptionsLocked(logdir, opts);
  const auto& config = options.profiler_options().advanced_configuration();
  auto it = config.find("use_system_hostname");
  ASSERT_NE(it, config.end());
  EXPECT_TRUE(it->second.bool_value());
}

TEST(SessionManagerTest, UseSystemHostnameFalseTest) {
  absl::string_view logdir = "/tmp/logdir";
  absl::flat_hash_map<std::string, std::variant<bool, int, std::string>> opts =
      {{"use_system_hostname", false}};
  RemoteProfilerSessionManagerOptions options =
      GetRemoteSessionManagerOptionsLocked(logdir, opts);
  const auto& config = options.profiler_options().advanced_configuration();
  auto it = config.find("use_system_hostname");
  ASSERT_NE(it, config.end());
  EXPECT_FALSE(it->second.bool_value());
}

TEST(SessionManagerTest, ValidateHostPortPairValidAddresses) {
  EXPECT_THAT(ValidateHostPortPair("localhost:8466"), IsOk());
  EXPECT_THAT(ValidateHostPortPair("127.0.0.1:8466"), IsOk());
  EXPECT_THAT(ValidateHostPortPair("[::1]:8466"), IsOk());
  EXPECT_THAT(ValidateHostPortPair("[::]:8466"), IsOk());
  EXPECT_THAT(ValidateHostPortPair("[2001:db8:21da:7::b]:8466"), IsOk());
  EXPECT_THAT(ValidateHostPortPair("[2001:db8:85a3::8a2e:370:7334]:8466"),
              IsOk());
  EXPECT_THAT(
      ValidateHostPortPair("[2001:0db8:85a3:0000:0000:8a2e:0370:7334]:8466"),
      IsOk());
  EXPECT_THAT(ValidateHostPortPair("[::ffff:192.0.2.128]:8466"), IsOk());
  EXPECT_THAT(ValidateHostPortPair("[fe80::1%eth0]:8466"), IsOk());
}

TEST(SessionManagerTest, ValidateHostPortPairInvalidUnbracketedIpv6) {
  EXPECT_THAT(ValidateHostPortPair("2001:db8::1:8466"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("::1:8466"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
}

TEST(SessionManagerTest, ValidateHostPortPairInvalidBracketAndPortForms) {
  EXPECT_THAT(ValidateHostPortPair("[2001:db8::1]"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("[2001:db8::1]:"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("[2001:db8::1]:abc"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("[2001:db8::1]:-1"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("[2001:db8::1]:70000"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("[2001:db8::1]8466"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("[]"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("[]:8466"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("[localhost]:8466"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("[127.0.0.1]:8466"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("[[::1]:8466"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("[::1]]:8466"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("[[::1]]:8466"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("host[0]:8466"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("host]:8466"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("[2001:db8::1/64]:8466"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("host/path:8466"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("localhost"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("8466"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair(":8466"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair("localhost:"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
  EXPECT_THAT(ValidateHostPortPair(""),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not interpret")));
}

TEST(SessionManagerTest, MultiHostIpv6OptionsValidation) {
  absl::string_view service_addresses = "[2001:db8::1]:8466,[2001:db8::2]:8466";
  absl::string_view logdir = "/tmp/logdir";
  absl::flat_hash_map<std::string, std::variant<bool, int, std::string>> opts;
  bool is_cloud_tpu_session = false;

  RemoteProfilerSessionManagerOptions options =
      GetRemoteSessionManagerOptionsLocked(service_addresses, logdir,
                                           /*worker_list=*/"",
                                           /*include_dataset_ops=*/false,
                                           /*duration_ms=*/100, opts,
                                           &is_cloud_tpu_session);
  EXPECT_FALSE(is_cloud_tpu_session);
  EXPECT_THAT(options.service_addresses(),
              ElementsAre(Eq("[2001:db8::1]:8466"), Eq("[2001:db8::2]:8466")));
  EXPECT_THAT(ValidateRemoteProfilerSessionManagerOptions(options), IsOk());
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
