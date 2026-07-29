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

#include "xla/pjrt/triton.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"

namespace xla::triton {
namespace {

TEST(TritonCompileTest, EmptyNumCtasAttribute) {
  const char* mlir_module = R"(
    module attributes {"ttg.num-ctas" = array<i32>, ttg.shared = 0 : i32} {}
  )";

  auto result = Compile(mlir_module, "9.0", /*num_warps=*/4, /*num_ctas=*/2,
                        /*num_stages=*/3);
  if (result.status().code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Triton compilation not supported on this platform";
  }

  // We expect this to not crash. If it fails with a compilation error,
  // that's fine. But it should not crash due to OOB read.
  ASSERT_OK_AND_ASSIGN(auto compilation_result, result);
  EXPECT_EQ(compilation_result.cluster_dim_x, 2);
  EXPECT_EQ(compilation_result.cluster_dim_y, 1);
  EXPECT_EQ(compilation_result.cluster_dim_z, 1);
}

TEST(TritonCompileTest, ValidNumCtasAttribute) {
  const char* mlir_module = R"(
    module attributes {"ttg.num-ctas" = array<i32: 5>, ttg.shared = 0 : i32} {}
  )";

  auto result = Compile(mlir_module, "9.0", /*num_warps=*/4, /*num_ctas=*/2,
                        /*num_stages=*/3);
  if (result.status().code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Triton compilation not supported on this platform";
  }
  ASSERT_OK_AND_ASSIGN(auto compilation_result, result);
  EXPECT_EQ(compilation_result.cluster_dim_x, 2);
  EXPECT_EQ(compilation_result.cluster_dim_y, 1);
  EXPECT_EQ(compilation_result.cluster_dim_z, 1);
}

}  // namespace
}  // namespace xla::triton
