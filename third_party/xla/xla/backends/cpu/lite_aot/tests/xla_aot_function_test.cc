/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/cpu/lite_aot/xla_aot_function.h"

#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/backends/cpu/lite_aot/tests/add_aot_example_lib.h"
#include "xla/tsl/platform/env.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/platform.h"

namespace xla::cpu {
namespace {

using ::absl_testing::StatusIs;

std::string GetRootDir() {
  std::string root_dir;
  CHECK(tsl::io::GetTestWorkspaceDir(&root_dir));

  std::string path_to_data_dir =
      tsl::kIsOpenSource
          ? "xla/backends/cpu/lite_aot/tests"
          : "third_party/tensorflow/compiler/xla/backends/cpu/lite_aot/tests";

  return tsl::io::JoinPath(root_dir, path_to_data_dir);
}

TEST(XlaAotFunctionTest, TestExampleLoadingModelFromLibrary) {
  ASSERT_OK_AND_ASSIGN(auto aot_function, xla::cpu::GetAddAotFunction());

  alignas(xla::cpu::Align()) float a = 1.0f;
  alignas(xla::cpu::Align()) float b = 2.0f;

  aot_function->set_arg_data(0, &a);
  aot_function->set_arg_data(1, &b);

  EXPECT_EQ(aot_function->arg_size(0), sizeof(float));
  EXPECT_EQ(aot_function->arg_size(1), sizeof(float));
  EXPECT_EQ(aot_function->result_size(0), sizeof(float));

  ASSERT_OK(aot_function->Execute());

  EXPECT_EQ(*static_cast<float*>(aot_function->result_data(0)), 3.0f);
}

TEST(XlaAotFunctionTest, TestManualModelLoading) {
  xla::cpu::CompilationResultProto proto;
  ASSERT_OK(tsl::ReadBinaryProto(
      tsl::Env::Default(), tsl::io::JoinPath(GetRootDir(), "dot_aot"), &proto));
  ASSERT_OK_AND_ASSIGN(auto aot_function,
                       xla::cpu::XlaAotFunction::Create(std::move(proto)));

  // 3x4 matrix
  alignas(xla::cpu::Align()) float a[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  // 4x5 matrix
  alignas(xla::cpu::Align()) float b[] = {
      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  aot_function->set_arg_data(0, &a);
  aot_function->set_arg_data(1, &b);

  EXPECT_EQ(aot_function->arg_size(0), 12 * sizeof(float));
  EXPECT_EQ(aot_function->arg_size(1), 20 * sizeof(float));
  EXPECT_EQ(aot_function->result_size(0), 15 * sizeof(float));

  ASSERT_OK(aot_function->Execute());

  for (int i = 0; i < 15; ++i) {
    EXPECT_EQ(static_cast<float*>(aot_function->result_data(0))[i], 4.0f);
  }
}

TEST(XlaAotFunctionTest, TestErrorPropagation) {
#ifdef NDEBUG
  GTEST_SKIP() << "Skipping test in optimized mode because XLA:CPU won't "
                  "check for the alignment error.";
  return;
#endif
  ASSERT_OK_AND_ASSIGN(auto aot_function, xla::cpu::GetAddAotFunction());

  // We don't align data which is incompatible with XLA:CPU and returns an error
  // in debug mode.
  float a = 1.0f;
  float b = 2.0f;

  aot_function->set_arg_data(0, &a);
  aot_function->set_arg_data(1, &b);

  EXPECT_THAT(aot_function->Execute(), StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace xla::cpu
