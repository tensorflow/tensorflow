/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/acceleration_test_util_internal.h"

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {

using ::testing::Eq;
using ::testing::Not;
using ::testing::Test;

struct SimpleConfig {
 public:
  static constexpr char kAccelerationTestConfig[] =
      R"(
      #test-id,some-other-data
      test-1,data-1
      test-2,
      test-3,data-3
      test-4.*,data-4
      -test-5
      test-6
      test-7,data-7
      )";

  static const char* AccelerationTestConfig() {
    return kAccelerationTestConfig;
  }

  static SimpleConfig ParseConfigurationLine(const std::string& conf_line) {
    return {conf_line};
  }

  std::string value;
};

class ReadAccelerationConfigTest : public ::testing::Test {
 public:
  std::unordered_map<std::string, SimpleConfig> allowlist_;
  std::unordered_map<std::string, SimpleConfig> denylist_;
  std::function<void(std::string, std::string, bool)> consumer_ =
      [this](std::string key, std::string value, bool is_denylist) {
        if (is_denylist) {
          denylist_[key] = {value};
        } else {
          allowlist_[key] = {value};
        }
      };
};

TEST_F(ReadAccelerationConfigTest, ReadsAKeyOnlyLine) {
  ReadAccelerationConfig("key", consumer_);

  EXPECT_THAT(allowlist_.find("key"), Not(Eq(allowlist_.end())));
  EXPECT_TRUE(denylist_.empty());
}

TEST_F(ReadAccelerationConfigTest, ReadsADenylistKeyOnlyLine) {
  ReadAccelerationConfig("-key", consumer_);

  EXPECT_THAT(denylist_.find("key"), Not(Eq(allowlist_.end())));
  EXPECT_TRUE(allowlist_.empty());
}

TEST_F(ReadAccelerationConfigTest, ReadsAKeyValueLine) {
  ReadAccelerationConfig("key,value", consumer_);

  EXPECT_THAT(allowlist_["key"].value, Eq("value"));
  EXPECT_TRUE(denylist_.empty());
}

TEST_F(ReadAccelerationConfigTest, ReadsADenyListKeyValueLine) {
  ReadAccelerationConfig("-key,value", consumer_);

  EXPECT_THAT(denylist_["key"].value, Eq("value"));
  EXPECT_TRUE(allowlist_.empty());
}

TEST_F(ReadAccelerationConfigTest, KeysAreLeftTrimmed) {
  ReadAccelerationConfig("  key,value", consumer_);

  EXPECT_THAT(allowlist_["key"].value, Eq("value"));
  EXPECT_TRUE(denylist_.empty());
}

TEST_F(ReadAccelerationConfigTest, BlKeysAreLeftTrimmed) {
  ReadAccelerationConfig("  -key,value", consumer_);

  EXPECT_THAT(denylist_["key"].value, Eq("value"));
  EXPECT_TRUE(allowlist_.empty());
}

TEST_F(ReadAccelerationConfigTest, IgnoresCommentedLines) {
  ReadAccelerationConfig("#key,value", consumer_);

  EXPECT_TRUE(allowlist_.empty());
  EXPECT_TRUE(denylist_.empty());
}

TEST_F(ReadAccelerationConfigTest, CommentCanHaveTrailingBlanks) {
  ReadAccelerationConfig("  #key,value", consumer_);

  EXPECT_TRUE(allowlist_.empty());
  EXPECT_TRUE(denylist_.empty());
}

TEST_F(ReadAccelerationConfigTest, CommentsAreOnlyForTheFullLine) {
  ReadAccelerationConfig("key,value #comment", consumer_);

  EXPECT_THAT(allowlist_["key"].value, Eq("value #comment"));
}

TEST_F(ReadAccelerationConfigTest, IgnoresEmptyLines) {
  ReadAccelerationConfig("", consumer_);

  EXPECT_TRUE(allowlist_.empty());
  EXPECT_TRUE(denylist_.empty());
}

TEST_F(ReadAccelerationConfigTest, ParsesMultipleLines) {
  ReadAccelerationConfig("key1,value1\nkey2,value2\n-key3,value3", consumer_);

  EXPECT_THAT(allowlist_["key1"].value, Eq("value1"));
  EXPECT_THAT(allowlist_["key2"].value, Eq("value2"));
  EXPECT_THAT(denylist_["key3"].value, Eq("value3"));
}

TEST_F(ReadAccelerationConfigTest, ParsesMultipleLinesWithCommentsAndSpaces) {
  ReadAccelerationConfig("key1,value1\n#comment\n\nkey2,value2", consumer_);

  EXPECT_THAT(allowlist_["key1"].value, Eq("value1"));
  EXPECT_THAT(allowlist_["key2"].value, Eq("value2"));
}

TEST_F(ReadAccelerationConfigTest, ParsesMultipleLinesWithMissingConfigValues) {
  ReadAccelerationConfig("key1\nkey2,value2\nkey3\nkey4,value4", consumer_);

  EXPECT_THAT(allowlist_["key1"].value, Eq(""));
  EXPECT_THAT(allowlist_["key2"].value, Eq("value2"));
  EXPECT_THAT(allowlist_["key3"].value, Eq(""));
  EXPECT_THAT(allowlist_["key4"].value, Eq("value4"));
}

TEST(GetAccelerationTestParam, LoadsTestConfig) {
  const auto config_value_maybe =
      GetAccelerationTestParam<SimpleConfig>("test-3");
  ASSERT_TRUE(config_value_maybe.has_value());
  ASSERT_THAT(config_value_maybe.value().value, Eq("data-3"));
}

TEST(GetAccelerationTestParam, LoadsTestConfigWithEmptyValue) {
  const auto config_value_maybe =
      GetAccelerationTestParam<SimpleConfig>("test-2");
  ASSERT_TRUE(config_value_maybe.has_value());
  ASSERT_THAT(config_value_maybe.value().value, Eq(""));
}

TEST(GetAccelerationTestParam, SupportsWildcards) {
  const auto config_value_maybe =
      GetAccelerationTestParam<SimpleConfig>("test-41");
  ASSERT_TRUE(config_value_maybe.has_value());
  ASSERT_THAT(config_value_maybe.value().value, Eq("data-4"));
}

TEST(GetAccelerationTestParam, SupportDenylist) {
  const auto config_value_maybe =
      GetAccelerationTestParam<SimpleConfig>("test-5");
  ASSERT_FALSE(config_value_maybe.has_value());
}

struct UnmatchedSimpleConfig {
 public:
  static constexpr const char* kAccelerationTestConfig = nullptr;

  static const char* AccelerationTestConfig() {
    return kAccelerationTestConfig;
  }

  static UnmatchedSimpleConfig ParseConfigurationLine(
      const std::string& conf_line) {
    return {conf_line};
  }

  std::string value;
};

TEST(GetAccelerationTestParam, ReturnEmptyOptionalForNullConfig) {
  ASSERT_FALSE(
      GetAccelerationTestParam<UnmatchedSimpleConfig>("test-3").has_value());
}

}  // namespace tflite
