/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu_compilation_environment.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/parse_flags_from_env.h"
#include "xla/service/compilation_environments.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ::tsl::testing::StatusIs;

void set_xla_flags_env_var(const std::string& xla_flags) {
  int* pargc;
  std::vector<char*>* pargv;
  ResetFlagsFromEnvForTesting("XLA_FLAGS", &pargc, &pargv);
  tsl::setenv("XLA_FLAGS", xla_flags.c_str(), true /*overwrite*/);
}

TEST(CreateGpuCompEnvFromFlagStringsTest, ValidFlags) {
  std::vector<std::string> flags = {"--dummy_flag=2"};

  TF_ASSERT_OK_AND_ASSIGN(
      GpuCompilationEnvironment gpu_comp_env,
      CreateGpuCompEnvFromFlagStrings(flags, /*strict=*/true));

  ASSERT_EQ(gpu_comp_env.dummy_flag(), 2);
  ASSERT_TRUE(flags.empty());
}

TEST(CreateGpuCompEnvFromFlagStringsTest, EmptyFlags) {
  std::vector<std::string> flags;

  TF_ASSERT_OK_AND_ASSIGN(
      GpuCompilationEnvironment gpu_comp_env,
      CreateGpuCompEnvFromFlagStrings(flags, /*strict=*/true));
}

TEST(CreateGpuCompEnvFromFlagStringsTest, InvalidFlagName) {
  std::vector<std::string> flags = {"--xla_gpu_invalid_flag=2"};

  EXPECT_THAT(CreateGpuCompEnvFromFlagStrings(flags, /*strict=*/true),
              StatusIs(tsl::error::INVALID_ARGUMENT));

  TF_ASSERT_OK_AND_ASSIGN(
      GpuCompilationEnvironment gpu_comp_env,
      CreateGpuCompEnvFromFlagStrings(flags, /*strict=*/false));
  ASSERT_EQ(flags.size(), 1);
}

TEST(CreateGpuCompEnvFromEnvVarTest, ValidFlags) {
  set_xla_flags_env_var("--dummy_flag=4");

  TF_ASSERT_OK_AND_ASSIGN(GpuCompilationEnvironment gpu_comp_env,
                          CreateGpuCompEnvFromEnvVar());

  ASSERT_EQ(gpu_comp_env.dummy_flag(), 4);
}

TEST(InitializeMissingFieldsFromXLAFlagsTest, BothProtoAndEnvVarUnset) {
  set_xla_flags_env_var("");
  GpuCompilationEnvironment env;

  TF_ASSERT_OK(InitializeMissingFieldsFromXLAFlags(env));
  EXPECT_EQ(env.dummy_flag(), 1);
}

TEST(InitializeMissingFieldsFromXLAFlagsTest, ProtoSetButEnvVarUnset) {
  set_xla_flags_env_var("");
  GpuCompilationEnvironment env;
  env.set_dummy_flag(2);

  TF_ASSERT_OK(InitializeMissingFieldsFromXLAFlags(env));

  EXPECT_EQ(env.dummy_flag(), 2);
}

TEST(InitializeMissingFieldsFromXLAFlagsTest, ProtoUnsetButEnvVarSet) {
  set_xla_flags_env_var("--dummy_flag=4");

  GpuCompilationEnvironment env;
  TF_ASSERT_OK(InitializeMissingFieldsFromXLAFlags(env));

  EXPECT_EQ(env.dummy_flag(), 4);
}

TEST(InitializeMissingFieldsFromXLAFlagsTest,
     BothProtoAndEnvVarSetButNoConflict) {
  set_xla_flags_env_var("--dummy_flag=4");
  CompilationEnvironments envs;
  GpuCompilationEnvironment env;
  TF_ASSERT_OK(InitializeMissingFieldsFromXLAFlags(env));
  EXPECT_EQ(env.dummy_flag(), 4);
}

TEST(InitializeMissingFieldsFromXLAFlagsTest,
     BothProtoAndEnvVarSetWithConflict) {
  set_xla_flags_env_var("--dummy_flag=4");

  CompilationEnvironments envs;
  GpuCompilationEnvironment env;
  env.set_dummy_flag(2);
  EXPECT_THAT(InitializeMissingFieldsFromXLAFlags(env),
              StatusIs(tsl::error::INVALID_ARGUMENT));
}

}  // namespace
}  // namespace xla
