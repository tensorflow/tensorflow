/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/subprocess_compilation.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/stream_executor/semantic_version.h"
#include "tsl/platform/path.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor {
namespace {
using testing::Not;
using tsl::testing::IsOkAndHolds;
using tsl::testing::StatusIs;

TEST(SubprocessCompilationTest, GetToolVersion) {
  std::string cuda_dir;
  if (!tsl::io::GetTestWorkspaceDir(&cuda_dir)) {
    GTEST_SKIP() << "No test workspace directory found which means we can't "
                    "run this test. Was this called in a Bazel environment?";
  }

  TF_ASSERT_OK_AND_ASSIGN(
      SemanticVersion ptxas_version,
      GetToolVersion(tsl::io::JoinPath(cuda_dir, "bin", "ptxas")));
  EXPECT_EQ(ptxas_version.major(), 111);
  EXPECT_EQ(ptxas_version.minor(), 2);
  EXPECT_EQ(ptxas_version.patch(), 3);

  TF_ASSERT_OK_AND_ASSIGN(
      SemanticVersion nvlink_version,
      GetToolVersion(tsl::io::JoinPath(cuda_dir, "bin", "nvlink")));
  EXPECT_EQ(nvlink_version.major(), 444);
  EXPECT_EQ(nvlink_version.minor(), 5);
  EXPECT_EQ(nvlink_version.patch(), 6);

  TF_ASSERT_OK_AND_ASSIGN(
      SemanticVersion fatbinary_version,
      GetToolVersion(tsl::io::JoinPath(cuda_dir, "bin", "fatbinary")));
  EXPECT_EQ(fatbinary_version.major(), 777);
  EXPECT_EQ(fatbinary_version.minor(), 8);
  EXPECT_EQ(fatbinary_version.patch(), 9);
}

TEST(SubprocessCompilationTest, GetNvlinkVersion) {
  std::string cuda_dir;
  if (!tsl::io::GetTestWorkspaceDir(&cuda_dir)) {
    GTEST_SKIP() << "No test workspace directory found which means we can't "
                    "run this test. Was this called in a Bazel environment?";
  }

  TF_ASSERT_OK_AND_ASSIGN(SemanticVersion nvlink_version,
                          GetNvLinkVersion(cuda_dir));
  EXPECT_EQ(nvlink_version.major(), 444);
  EXPECT_EQ(nvlink_version.minor(), 5);
  EXPECT_EQ(nvlink_version.patch(), 6);
}

TEST(SubprocessCompilationTest, GetAsmCompilerVersion) {
  std::string cuda_dir;
  if (!tsl::io::GetTestWorkspaceDir(&cuda_dir)) {
    GTEST_SKIP() << "No test workspace directory found which means we can't "
                    "run this test. Was this called in a Bazel environment?";
  }

  TF_ASSERT_OK_AND_ASSIGN(SemanticVersion nvlink_version,
                          GetAsmCompilerVersion(cuda_dir));
  EXPECT_EQ(nvlink_version.major(), 111);
  EXPECT_EQ(nvlink_version.minor(), 2);
  EXPECT_EQ(nvlink_version.patch(), 3);
}

TEST(SubprocessCompilationTest, FindCudaExecutable) {
  std::string cuda_dir;
  if (!tsl::io::GetTestWorkspaceDir(&cuda_dir)) {
    GTEST_SKIP() << "No test workspace directory found which means we can't "
                    "run this test. Was this called in a Bazel environment?";
  }

  std::string ptxas_path = tsl::io::JoinPath(cuda_dir, "bin", "ptxas");

  EXPECT_THAT(FindCudaExecutable("ptxas", cuda_dir), IsOkAndHolds(ptxas_path));
  EXPECT_THAT(
      FindCudaExecutable("ptxas", cuda_dir, SemanticVersion{0, 0, 0}, {}),
      IsOkAndHolds(ptxas_path));
  EXPECT_THAT(
      FindCudaExecutable("ptxas", cuda_dir, SemanticVersion{111, 2, 3}, {}),
      IsOkAndHolds(ptxas_path));
  EXPECT_THAT(
      FindCudaExecutable("ptxas", cuda_dir, SemanticVersion{111, 2, 4}, {}),
      StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(FindCudaExecutable("ptxas", cuda_dir, SemanticVersion{0, 0, 0},
                                 {{111, 2, 3}}),
              Not(IsOkAndHolds(ptxas_path)));
  EXPECT_THAT(FindCudaExecutable("ptxas", cuda_dir, SemanticVersion{99, 0, 0},
                                 {{111, 2, 3}}),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace stream_executor
