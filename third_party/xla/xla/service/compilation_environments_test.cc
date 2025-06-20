/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/compilation_environments.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/test_compilation_environment.pb.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/protobuf.h"

namespace xla {

using ::tsl::testing::StatusIs;

// In order to use TestCompilationEnvironment* with CompilationEnvironments, we
// must define ProcessNewEnv for them.
std::unique_ptr<tsl::protobuf::Message> ProcessNewEnv1(
    std::unique_ptr<tsl::protobuf::Message> msg) {
  std::unique_ptr<test::TestCompilationEnvironment1> env(
      tensorflow::down_cast<test::TestCompilationEnvironment1*>(msg.release()));
  if (!env) {
    env = std::make_unique<test::TestCompilationEnvironment1>();
  }
  if (env->some_flag() == 0 || env->some_flag() == 1) {
    env->set_some_flag(100);
  }
  return env;
}
std::unique_ptr<tsl::protobuf::Message> ProcessNewEnv2(
    std::unique_ptr<tsl::protobuf::Message> msg) {
  std::unique_ptr<test::TestCompilationEnvironment2> env(
      tensorflow::down_cast<test::TestCompilationEnvironment2*>(msg.release()));
  if (!env) {
    env = std::make_unique<test::TestCompilationEnvironment2>();
  }
  if (env->some_other_flag() == 0) {
    env->set_some_other_flag(200);
  }
  return env;
}
std::unique_ptr<tsl::protobuf::Message> ProcessNewEnv3(
    std::unique_ptr<tsl::protobuf::Message> msg) {
  std::unique_ptr<test::TestCompilationEnvironment3> env(
      tensorflow::down_cast<test::TestCompilationEnvironment3*>(msg.release()));
  if (!env) {
    env = std::make_unique<test::TestCompilationEnvironment3>();
  }
  if (env->a_third_flag() == 0) {
    env->set_a_third_flag(300);
  }
  return env;
}

std::unique_ptr<tsl::protobuf::Message> ProcessCustomDescInFallbackTest(
    std::unique_ptr<tsl::protobuf::Message> msg_dynamic) {
  auto new_generated_env =
      std::make_unique<xla::test::TestCompilationEnvironment1>();
  // This value is used to identify that the environment was processed via this
  // specific path for the custom descriptor. It should match the
  // kExpectedFallbackValue in the test that uses this function.
  auto kTestSpecificValue = 555;
  auto kDefaultValueIfInputIsUnexpected = 558;

  new_generated_env->set_some_flag(kDefaultValueIfInputIsUnexpected);

  if (msg_dynamic) {
    const tsl::protobuf::Reflection* refl = msg_dynamic->GetReflection();
    const tsl::protobuf::Descriptor* d = msg_dynamic->GetDescriptor();
    const tsl::protobuf::FieldDescriptor* f = d->FindFieldByName("some_flag");
    // Check if the incoming dynamic message has the flag set to
    // kTestSpecificValue
    if (refl && f && refl->HasField(*msg_dynamic, f) &&
        refl->GetUInt32(*msg_dynamic, f) == kTestSpecificValue) {
      new_generated_env->set_some_flag(kTestSpecificValue);
    }
  }

  return new_generated_env;
}

namespace test {
namespace {

class CompilationEnvironmentsTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    CompilationEnvironments::RegisterProcessNewEnvFn(
        test::TestCompilationEnvironment1::descriptor(), ProcessNewEnv1);
    CompilationEnvironments::RegisterProcessNewEnvFn(
        test::TestCompilationEnvironment2::descriptor(), ProcessNewEnv2);
    CompilationEnvironments::RegisterProcessNewEnvFn(
        test::TestCompilationEnvironment3::descriptor(), ProcessNewEnv3);
  }
};

TEST_F(CompilationEnvironmentsTest, GetDefaultEnv) {
  CompilationEnvironments envs;
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
}

TEST_F(CompilationEnvironmentsTest, GetDefaultMutableEnv) {
  CompilationEnvironments envs;
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 100);
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 100);
}

TEST_F(CompilationEnvironmentsTest, GetAddedEnvNotModifiedByProcessNewEnv) {
  CompilationEnvironments envs;
  auto env = std::make_unique<TestCompilationEnvironment1>();
  env->set_some_flag(5);
  TF_ASSERT_OK(envs.AddEnv(std::move(env)));
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 5);
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 5);
}

TEST_F(CompilationEnvironmentsTest, GetAddedEnvModifiedByProcessNewEnv) {
  CompilationEnvironments envs;
  auto env = std::make_unique<TestCompilationEnvironment1>();
  env->set_some_flag(1);
  TF_ASSERT_OK(envs.AddEnv(std::move(env)));
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 100);
}

TEST_F(CompilationEnvironmentsTest, MultipleEnvs) {
  CompilationEnvironments envs;
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment2>().some_other_flag(), 200);
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
}

TEST_F(CompilationEnvironmentsTest, MultipleMutableEnvs) {
  CompilationEnvironments envs;
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 100);
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment2>().some_other_flag(),
            200);
  envs.GetMutableEnv<TestCompilationEnvironment1>().set_some_flag(101);
  envs.GetMutableEnv<TestCompilationEnvironment2>().set_some_other_flag(201);
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 101);
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment2>().some_other_flag(),
            201);
}

TEST_F(CompilationEnvironmentsTest, ReplaceExistingEnv) {
  CompilationEnvironments envs;
  auto env1 = std::make_unique<TestCompilationEnvironment1>();
  env1->set_some_flag(5);
  TF_ASSERT_OK(envs.AddEnv(std::move(env1)));
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 5);
  {
    auto env2 = std::make_unique<TestCompilationEnvironment1>();
    env2->set_some_flag(6);
    ASSERT_THAT(envs.AddEnv(std::move(env2)),
                StatusIs(absl::StatusCode::kAlreadyExists));
  }
  envs.DeleteEnv<TestCompilationEnvironment1>();
  {
    auto env2 = std::make_unique<TestCompilationEnvironment1>();
    env2->set_some_flag(6);
    TF_ASSERT_OK(envs.AddEnv(std::move(env2)));
    EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 6);
  }
}

TEST_F(CompilationEnvironmentsTest, CopyConstructor) {
  // Setup envs with 2 environments
  auto envs = std::make_unique<CompilationEnvironments>();
  auto env1 = std::make_unique<TestCompilationEnvironment1>();
  env1->set_some_flag(10);
  TF_ASSERT_OK(envs->AddEnv(std::move(env1)));
  auto env2 = std::make_unique<TestCompilationEnvironment2>();
  TF_ASSERT_OK(envs->AddEnv(std::move(env2)));
  envs->GetMutableEnv<TestCompilationEnvironment2>().set_some_other_flag(20);

  // Call the copy constructor and delete the original CompilationEnvironments
  auto envs_copy = std::make_unique<CompilationEnvironments>(*envs);
  envs.reset();

  // Verify that envs_copy has the same values with which envs was initialized
  EXPECT_EQ(envs_copy->GetEnv<TestCompilationEnvironment1>().some_flag(), 10);
  EXPECT_EQ(envs_copy->GetEnv<TestCompilationEnvironment2>().some_other_flag(),
            20);
}

TEST_F(CompilationEnvironmentsTest, CopyAssignment) {
  // Setup envs1 with 2 environments
  auto envs1 = std::make_unique<CompilationEnvironments>();
  auto env1 = std::make_unique<TestCompilationEnvironment1>();
  env1->set_some_flag(10);
  TF_ASSERT_OK(envs1->AddEnv(std::move(env1)));
  auto env2 = std::make_unique<TestCompilationEnvironment2>();
  TF_ASSERT_OK(envs1->AddEnv(std::move(env2)));
  envs1->GetMutableEnv<TestCompilationEnvironment2>().set_some_other_flag(20);

  // Create envs2 with some environments that should be deleted on copy
  // assignment
  auto envs2 = std::make_unique<CompilationEnvironments>();
  auto env3 = std::make_unique<TestCompilationEnvironment1>();
  env3->set_some_flag(30);
  TF_ASSERT_OK(envs2->AddEnv(std::move(env3)));
  auto env4 = std::make_unique<TestCompilationEnvironment3>();
  env4->set_a_third_flag(40);
  TF_ASSERT_OK(envs2->AddEnv(std::move(env4)));

  // Assign envs1 to envs2, and delete envs1. After assignment, the environments
  // originaly added to envs2 should be deleted, and copies of the environments
  // in envs1 should be added to envs2.
  *envs2 = *envs1;
  envs1.reset();

  // Verify that envs2 has the same values with which envs1 was initialized
  EXPECT_EQ(envs2->GetEnv<TestCompilationEnvironment1>().some_flag(), 10);
  EXPECT_EQ(envs2->GetEnv<TestCompilationEnvironment2>().some_other_flag(), 20);

  // Since envs1 did not have TestCompilationEnvironment3, after copy
  // assignment, envs2 will not have one either. So, we should get the default
  // environment value.
  EXPECT_EQ(envs2->GetEnv<TestCompilationEnvironment3>().a_third_flag(), 300);
}

TEST_F(CompilationEnvironmentsTest, ProtoRoundTrip) {
  // Setup envs with 2 environments.
  auto envs = std::make_unique<CompilationEnvironments>();
  auto env1 = std::make_unique<TestCompilationEnvironment1>();
  env1->set_some_flag(10);
  TF_ASSERT_OK(envs->AddEnv(std::move(env1)));
  auto env2 = std::make_unique<TestCompilationEnvironment2>();
  TF_ASSERT_OK(envs->AddEnv(std::move(env2)));
  envs->GetMutableEnv<TestCompilationEnvironment2>().set_some_other_flag(20);

  auto proto = envs->ToProto();
  TF_ASSERT_OK_AND_ASSIGN(auto envs_deserialized,
                          CompilationEnvironments::CreateFromProto(proto));

  // Verify that envs_deserialized has the same values with which envs was
  // initialized.
  EXPECT_EQ(
      envs_deserialized->GetEnv<TestCompilationEnvironment1>().some_flag(), 10);
  EXPECT_EQ(envs_deserialized->GetEnv<TestCompilationEnvironment2>()
                .some_other_flag(),
            20);
}

TEST_F(CompilationEnvironmentsTest, EnvTypePresenceCheck) {
  CompilationEnvironments envs;
  EXPECT_FALSE(envs.HasEnv<TestCompilationEnvironment1>());
  envs.GetEnv<TestCompilationEnvironment1>();
  EXPECT_TRUE(envs.HasEnv<TestCompilationEnvironment1>());
}

TEST_F(CompilationEnvironmentsTest, InitializeAllKnownEnvs) {
  CompilationEnvironments envs;
  auto env1 = std::make_unique<TestCompilationEnvironment1>();
  env1->set_some_flag(400);
  TF_ASSERT_OK(envs.AddEnv(std::move(env1)));
  EXPECT_TRUE(envs.HasEnv<TestCompilationEnvironment1>());
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 400);
  TF_ASSERT_OK(envs.InitializeAllKnownEnvs());
  EXPECT_TRUE(envs.HasEnv<TestCompilationEnvironment1>());
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 400);
  EXPECT_TRUE(envs.HasEnv<TestCompilationEnvironment2>());
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment2>().some_other_flag(), 200);
  EXPECT_TRUE(envs.HasEnv<TestCompilationEnvironment3>());
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment3>().a_third_flag(), 300);
}

TEST_F(CompilationEnvironmentsTest, GetEnvTriggersFullNameFallback) {
  // Create a custom descriptor pool and load the proto into it.
  const tsl::protobuf::Descriptor* desc_generated =
      test::TestCompilationEnvironment1::descriptor();
  tsl::protobuf::DescriptorPool custom_pool(
      tsl::protobuf::DescriptorPool::generated_pool());
  tsl::protobuf::FileDescriptorProto file_proto;
  desc_generated->file()->CopyTo(&file_proto);
  custom_pool.BuildFile(file_proto);

  // Register a custom handler for the descriptor from the custom_pool.
  const tsl::protobuf::Descriptor* desc_custom =
      custom_pool.FindMessageTypeByName(desc_generated->full_name());
  CompilationEnvironments::RegisterProcessNewEnvFn(
      desc_custom, ProcessCustomDescInFallbackTest);

  // Create and populate a dynamic message instance using the custom descriptor.
  tsl::protobuf::DynamicMessageFactory factory(&custom_pool);
  std::unique_ptr<tsl::protobuf::Message> dynamic_env_instance(
      factory.GetPrototype(desc_custom)->New());
  const tsl::protobuf::FieldDescriptor* flag_field =
      desc_custom->FindFieldByName("some_flag");
  auto kExpectedFallbackValue = 555;
  dynamic_env_instance->GetReflection()->SetUInt32(
      dynamic_env_instance.get(), flag_field, kExpectedFallbackValue);

  // Add this dynamic instance to CompilationEnvironments.
  CompilationEnvironments envs;
  TF_ASSERT_OK(envs.AddEnv(std::move(dynamic_env_instance)));

  // Trigger lookup by full_name.
  const auto& retrieved_env = envs.GetEnv<test::TestCompilationEnvironment1>();

  // Verify that the fallback value was used.
  EXPECT_EQ(retrieved_env.some_flag(), kExpectedFallbackValue);
}

}  // namespace
}  // namespace test
}  // namespace xla
