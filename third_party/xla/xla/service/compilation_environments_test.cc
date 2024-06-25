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

#include "absl/status/statusor.h"
#include "xla/service/test_compilation_environment.pb.h"
#include "xla/test.h"
#include "xla/xla.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/protobuf.h"

namespace xla {

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

}  // namespace
}  // namespace test
}  // namespace xla
