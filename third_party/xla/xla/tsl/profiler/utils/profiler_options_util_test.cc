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

#include "xla/tsl/profiler/utils/profiler_options_util.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/tsl/platform/test.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace tsl {
namespace profiler {
namespace {

using ::testing::Eq;
using ::testing::Optional;
using ::testing::VariantWith;

TEST(ProfilerOptionsUtilTest, GetConfigValueString) {
  tensorflow::ProfileOptions options;
  auto& advanced_config = *options.mutable_advanced_configuration();
  tensorflow::ProfileOptions::AdvancedConfigValue config_value;
  config_value.set_string_value("test_value");
  advanced_config["test_key"] = config_value;

  std::optional<std::variant<std::string, bool, int64_t>> result =
      GetConfigValue(options, "test_key");
  EXPECT_THAT(result, Optional(VariantWith<std::string>("test_value")));
}

TEST(ProfilerOptionsUtilTest, GetConfigValueBool) {
  tensorflow::ProfileOptions options;
  auto& advanced_config = *options.mutable_advanced_configuration();
  tensorflow::ProfileOptions::AdvancedConfigValue config_value;
  config_value.set_bool_value(true);
  advanced_config["test_key"] = config_value;

  std::optional<std::variant<std::string, bool, int64_t>> result =
      GetConfigValue(options, "test_key");
  EXPECT_THAT(result, Optional(VariantWith<bool>(true)));
}

TEST(ProfilerOptionsUtilTest, GetConfigValueInt64) {
  tensorflow::ProfileOptions options;
  auto& advanced_config = *options.mutable_advanced_configuration();
  tensorflow::ProfileOptions::AdvancedConfigValue config_value;
  config_value.set_int64_value(12345);
  advanced_config["test_key"] = config_value;

  std::optional<std::variant<std::string, bool, int64_t>> result =
      GetConfigValue(options, "test_key");
  EXPECT_THAT(result, Optional(VariantWith<int64_t>(12345)));
}

TEST(ProfilerOptionsUtilTest, GetConfigValueNotFound) {
  tensorflow::ProfileOptions options;

  std::optional<std::variant<std::string, bool, int64_t>> result =
      GetConfigValue(options, "test_key");
  EXPECT_FALSE(result.has_value());
  EXPECT_THAT(result, Eq(std::nullopt));
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
