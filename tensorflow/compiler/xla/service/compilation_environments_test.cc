/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/compilation_environments.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/service/test_compilation_environment.pb.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {

// In order to use TestCompilationEnvironment* with CompilationEnvironments, we
// must define CreateDefaultEnv for them.
template <>
std::unique_ptr<test::TestCompilationEnvironment1>
CompilationEnvironments::CreateDefaultEnv<test::TestCompilationEnvironment1>() {
  auto env = std::make_unique<test::TestCompilationEnvironment1>();
  env->set_some_flag(100);
  return env;
}
template <>
std::unique_ptr<test::TestCompilationEnvironment2>
CompilationEnvironments::CreateDefaultEnv<test::TestCompilationEnvironment2>() {
  auto env = std::make_unique<test::TestCompilationEnvironment2>();
  env->set_some_other_flag(200);
  return env;
}

namespace test {
namespace {

class EnvWrapperTest : public ::testing::Test {
 public:
  using CEnvWrapper = CompilationEnvironments::EnvWrapper;
};

TEST_F(EnvWrapperTest, FromUniquePtr) {
  auto env = std::make_unique<TestCompilationEnvironment1>();
  env->set_some_flag(10);
  auto env_ptr = env.get();
  auto wrapper = CEnvWrapper::FromUniquePtr(std::move(env));

  EXPECT_EQ(wrapper.EnvTypeid(), typeid(TestCompilationEnvironment1));
  EXPECT_EQ(&wrapper.Get(), env_ptr);
  EXPECT_EQ(
      tensorflow::down_cast<const TestCompilationEnvironment1&>(wrapper.Get())
          .some_flag(),
      10);
}

TEST_F(EnvWrapperTest, FromRef) {
  TestCompilationEnvironment1 env;
  env.set_some_flag(10);
  auto env_ptr = &env;
  auto wrapper = CEnvWrapper::FromRef(env);

  EXPECT_EQ(wrapper.EnvTypeid(), typeid(TestCompilationEnvironment1));
  EXPECT_EQ(&wrapper.Get(), env_ptr);
  EXPECT_EQ(
      tensorflow::down_cast<const TestCompilationEnvironment1&>(wrapper.Get())
          .some_flag(),
      10);
}

class CompilationEnvironmentsTest : public ::testing::Test {};

TEST_F(CompilationEnvironmentsTest, GetDefaultEnv) {
  CompilationEnvironments envs;
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
}

TEST_F(CompilationEnvironmentsTest, GetAddedEnv) {
  CompilationEnvironments envs;
  auto env = std::make_unique<TestCompilationEnvironment1>();
  env->set_some_flag(5);
  envs.AddEnv(
      CompilationEnvironments::EnvWrapper::FromUniquePtr(std::move(env)));
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 5);
}

TEST_F(CompilationEnvironmentsTest, MultipleEnvs) {
  CompilationEnvironments envs;
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment2>().some_other_flag(), 200);
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
}

}  // namespace
}  // namespace test
}  // namespace xla
