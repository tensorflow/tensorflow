/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/graph_executor/config.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/core/tfrt/graph_executor/config.pb.h"
#include "tensorflow/core/tfrt/graph_executor/test_config.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

TEST(ConfigTest, Basic) {
  TestConfig1 expected_test_config1;
  expected_test_config1.set_tag("test config1");
  TestConfig2 expected_test_config2;
  expected_test_config2.set_tag("test config2");

  RuntimeConfig runtime_config;

  TF_ASSERT_OK(runtime_config.Add(expected_test_config2));
  TF_ASSERT_OK(runtime_config.Add(expected_test_config1));

  auto test_config1 = runtime_config.Get<TestConfig1>();
  TF_ASSERT_OK(test_config1.status());
  auto test_config2 = runtime_config.Get<TestConfig2>();
  TF_ASSERT_OK(test_config2.status());

  EXPECT_EQ(test_config1->tag(), "test config1");
  EXPECT_EQ(test_config2->tag(), "test config2");
}

TEST(ConfigTest, Load) {
  TestConfig1 expected_test_config1;
  expected_test_config1.set_tag("test config1");
  TestConfig2 expected_test_config2;
  expected_test_config2.set_tag("test config2");

  RuntimeConfigProto runtime_config_proto;
  runtime_config_proto.add_config()->PackFrom(expected_test_config1);
  runtime_config_proto.add_config()->PackFrom(expected_test_config2);

  TF_ASSERT_OK_AND_ASSIGN(RuntimeConfig runtime_config,
                          RuntimeConfig::CreateFromProto(runtime_config_proto));

  auto test_config1 = runtime_config.Get<TestConfig1>();
  TF_ASSERT_OK(test_config1.status());
  auto test_config2 = runtime_config.Get<TestConfig2>();
  TF_ASSERT_OK(test_config2.status());

  EXPECT_EQ(test_config1->tag(), "test config1");
  EXPECT_EQ(test_config2->tag(), "test config2");
}

TEST(ConfigTest, NotFound) {
  TestConfig1 expected_test_config1;
  expected_test_config1.set_tag("test config1");

  RuntimeConfigProto runtime_config_proto;
  runtime_config_proto.add_config()->PackFrom(expected_test_config1);

  TF_ASSERT_OK_AND_ASSIGN(RuntimeConfig runtime_config,
                          RuntimeConfig::CreateFromProto(runtime_config_proto));

  EXPECT_THAT(runtime_config.Get<TestConfig2>(),
              ::tsl::testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST(ConfigTest, Duplicate) {
  TestConfig1 expected_test_config1;
  expected_test_config1.set_tag("test config1");

  RuntimeConfig runtime_config;

  TF_ASSERT_OK(runtime_config.Add(expected_test_config1));
  EXPECT_THAT(runtime_config.Add(expected_test_config1),
              ::tsl::testing::StatusIs(absl::StatusCode::kAlreadyExists));
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
