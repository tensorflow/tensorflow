/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/common.h"

#include <vector>

#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::testing::IsOkAndHolds;
using ::tensorflow::testing::StatusIs;
using ::testing::HasSubstr;

std::vector<ProcessingModeDef::ShardingPolicy> EnumerateShardingPolicies() {
  std::vector<ProcessingModeDef::ShardingPolicy> result;
  const ::tensorflow::protobuf::EnumDescriptor* enum_descriptor =
      ::tensorflow::protobuf::GetEnumDescriptor<
          ProcessingModeDef::ShardingPolicy>();
  for (int i = 0; i < enum_descriptor->value_count(); ++i) {
    result.push_back(static_cast<ProcessingModeDef::ShardingPolicy>(
        enum_descriptor->value(i)->number()));
  }
  return result;
}

TEST(CommonTest, NoShard) {
  ProcessingModeDef processing_mode;
  processing_mode.set_sharding_policy(ProcessingModeDef::OFF);
  EXPECT_TRUE(IsNoShard(processing_mode));
  EXPECT_FALSE(IsDynamicShard(processing_mode));
  EXPECT_FALSE(IsStaticShard(processing_mode));
}

TEST(CommonTest, DynamicShard) {
  ProcessingModeDef processing_mode;
  processing_mode.set_sharding_policy(ProcessingModeDef::DYNAMIC);
  EXPECT_FALSE(IsNoShard(processing_mode));
  EXPECT_TRUE(IsDynamicShard(processing_mode));
  EXPECT_FALSE(IsStaticShard(processing_mode));
}

TEST(CommonTest, StaticShard) {
  ProcessingModeDef processing_mode;
  std::vector<ProcessingModeDef::ShardingPolicy> policies = {
      ProcessingModeDef::FILE, ProcessingModeDef::DATA,
      ProcessingModeDef::FILE_OR_DATA, ProcessingModeDef::HINT};
  for (const ProcessingModeDef::ShardingPolicy policy : policies) {
    processing_mode.set_sharding_policy(policy);
    EXPECT_FALSE(IsNoShard(processing_mode));
    EXPECT_FALSE(IsDynamicShard(processing_mode));
    EXPECT_TRUE(IsStaticShard(processing_mode));
  }
}

TEST(CommonTest, DefaultShardingPolicyIsNoShard) {
  ProcessingModeDef processing_mode;
  EXPECT_TRUE(IsNoShard(processing_mode));
  EXPECT_FALSE(IsDynamicShard(processing_mode));
  EXPECT_FALSE(IsStaticShard(processing_mode));
}

TEST(CommonTest, ToAutoShardPolicy) {
  EXPECT_THAT(ToAutoShardPolicy(ProcessingModeDef::FILE_OR_DATA),
              IsOkAndHolds(AutoShardPolicy::AUTO));
  EXPECT_THAT(ToAutoShardPolicy(ProcessingModeDef::HINT),
              IsOkAndHolds(AutoShardPolicy::HINT));
  EXPECT_THAT(ToAutoShardPolicy(ProcessingModeDef::OFF),
              IsOkAndHolds(AutoShardPolicy::OFF));
  EXPECT_THAT(ToAutoShardPolicy(ProcessingModeDef::DYNAMIC),
              IsOkAndHolds(AutoShardPolicy::OFF));
}

TEST(CommonTest, ConvertValidShardingPolicyToAutoShardPolicy) {
  for (const ProcessingModeDef::ShardingPolicy sharding_policy :
       EnumerateShardingPolicies()) {
    TF_EXPECT_OK(ToAutoShardPolicy(sharding_policy).status());
  }
}

TEST(CommonTest, ConvertInvalidShardingPolicyToAutoShardPolicy) {
  const ProcessingModeDef::ShardingPolicy sharding_policy =
      static_cast<ProcessingModeDef::ShardingPolicy>(-100);
  EXPECT_THAT(ToAutoShardPolicy(sharding_policy),
              StatusIs(error::INTERNAL,
                       HasSubstr("please update the policy mapping.")));
}

TEST(CommonTest, ValidateProcessingMode) {
  for (const ProcessingModeDef::ShardingPolicy policy :
       EnumerateShardingPolicies()) {
    ProcessingModeDef processing_mode;
    processing_mode.set_sharding_policy(policy);
    TF_EXPECT_OK(ValidateProcessingMode(processing_mode));
  }
}

TEST(CommonTest, InvalidProcessingMode) {
  ProcessingModeDef processing_mode;
  processing_mode.set_sharding_policy(
      static_cast<ProcessingModeDef::ShardingPolicy>(100));
  EXPECT_THAT(ValidateProcessingMode(processing_mode),
              StatusIs(error::INTERNAL,
                       HasSubstr("does not specify a valid sharding policy.")));
}

TEST(CommonTest, ParseTargetWorkers) {
  EXPECT_THAT(ParseTargetWorkers("AUTO"), IsOkAndHolds(TARGET_WORKERS_AUTO));
  EXPECT_THAT(ParseTargetWorkers("Auto"), IsOkAndHolds(TARGET_WORKERS_AUTO));
  EXPECT_THAT(ParseTargetWorkers("ANY"), IsOkAndHolds(TARGET_WORKERS_ANY));
  EXPECT_THAT(ParseTargetWorkers("any"), IsOkAndHolds(TARGET_WORKERS_ANY));
  EXPECT_THAT(ParseTargetWorkers("LOCAL"), IsOkAndHolds(TARGET_WORKERS_LOCAL));
  EXPECT_THAT(ParseTargetWorkers("local"), IsOkAndHolds(TARGET_WORKERS_LOCAL));
  EXPECT_THAT(ParseTargetWorkers(""), IsOkAndHolds(TARGET_WORKERS_AUTO));
}

TEST(CommonTest, ParseInvalidTargetWorkers) {
  EXPECT_THAT(ParseTargetWorkers("TARGET_WORKERS_UNSPECIFIED"),
              testing::StatusIs(error::INVALID_ARGUMENT));
  EXPECT_THAT(ParseTargetWorkers("UNSET"),
              testing::StatusIs(error::INVALID_ARGUMENT));
}

TEST(CommonTest, TargetWorkersToString) {
  EXPECT_EQ(TargetWorkersToString(TARGET_WORKERS_AUTO), "AUTO");
  EXPECT_EQ(TargetWorkersToString(TARGET_WORKERS_ANY), "ANY");
  EXPECT_EQ(TargetWorkersToString(TARGET_WORKERS_LOCAL), "LOCAL");
}

TEST(CommonTest, ParseDeploymentMode) {
  EXPECT_THAT(ParseDeploymentMode("COLOCATED"),
              IsOkAndHolds(DeploymentMode::DEPLOYMENT_MODE_COLOCATED));
  EXPECT_THAT(ParseDeploymentMode("Colocated"),
              IsOkAndHolds(DeploymentMode::DEPLOYMENT_MODE_COLOCATED));
  EXPECT_THAT(ParseDeploymentMode("REMOTE"),
              IsOkAndHolds(DeploymentMode::DEPLOYMENT_MODE_REMOTE));
  EXPECT_THAT(ParseDeploymentMode("remote"),
              IsOkAndHolds(DeploymentMode::DEPLOYMENT_MODE_REMOTE));
  EXPECT_THAT(ParseDeploymentMode("HYBRID"),
              IsOkAndHolds(DeploymentMode::DEPLOYMENT_MODE_HYBRID));
  EXPECT_THAT(ParseDeploymentMode("hybrid"),
              IsOkAndHolds(DeploymentMode::DEPLOYMENT_MODE_HYBRID));
}

TEST(CommonTest, ParseInvalidDeploymentMode) {
  EXPECT_THAT(ParseDeploymentMode("DEPLOYMENT_MODE_UNSPECIFIED"),
              testing::StatusIs(error::INVALID_ARGUMENT));
}

TEST(CommonTest, IsPreemptedError) {
  EXPECT_TRUE(IsPreemptedError(errors::Aborted("Aborted")));
  EXPECT_TRUE(IsPreemptedError(errors::Cancelled("Cancelled")));
  EXPECT_TRUE(IsPreemptedError(errors::Unavailable("Unavailable")));
  EXPECT_FALSE(IsPreemptedError(absl::OkStatus()));
}

TEST(CommonTest, IsPermanentError) {
  EXPECT_FALSE(
      IsPreemptedError(errors::FailedPrecondition("Failed precondition")));
  EXPECT_FALSE(IsPreemptedError(errors::Internal("Internal")));
  EXPECT_FALSE(IsPreemptedError(errors::InvalidArgument("Invalid argument")));
  EXPECT_FALSE(IsPreemptedError(errors::NotFound("Not found")));
  EXPECT_FALSE(IsPreemptedError(errors::OutOfRange("Out of range")));
  EXPECT_FALSE(IsPreemptedError(errors::Unknown("Unknown")));
}
}  // namespace
}  // namespace data
}  // namespace tensorflow
