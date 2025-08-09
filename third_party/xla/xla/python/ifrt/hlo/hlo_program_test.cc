/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/python/ifrt/hlo/hlo_program.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::ifrt {
namespace {

std::unique_ptr<mlir::MLIRContext> CreateMlirContext() {
  auto context = std::make_unique<mlir::MLIRContext>(
      mlir::MLIRContext::Threading::DISABLED);

  mlir::DialectRegistry registry;
  xla::RegisterAllHloDialects(registry);
  context->appendDialectRegistry(registry);

  return context;
}

absl::StatusOr<std::unique_ptr<xla::ifrt::HloProgram>> ParseHloProgramString(
    absl::string_view str) {
  auto context = CreateMlirContext();
  TF_ASSIGN_OR_RETURN(auto module, xla::ParseMlirModuleString(str, *context));
  return std::make_unique<xla::ifrt::HloProgram>(std::move(context),
                                                 std::move(module));
}

TEST(HloProgramTest, Fingerprint) {
  static constexpr absl::string_view kModule1 = R"(
module attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = mhlo.add %arg0, %0 : tensor<f32>
    return %1 : tensor<f32>
  }
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto program1, ParseHloProgramString(kModule1));

  static constexpr absl::string_view kModule2 = R"(
module attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = mhlo.constant dense<2.000000e+00> : tensor<f32>
    %1 = mhlo.add %arg0, %0 : tensor<f32>
    return %1 : tensor<f32>
  }
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto program2, ParseHloProgramString(kModule2));

  EXPECT_EQ(program1->Fingerprint(), program1->Fingerprint());
  EXPECT_NE(program1->Fingerprint(), program2->Fingerprint());
}

TEST(HloProgramTest, FingerprintIgnoresDebugInfo) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<xla::ifrt::HloProgram> hlo_program1,
      ParseHloProgramString(R"(
module @foo {
  func.func @main(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
    return %arg0 : tensor<2x3xi32> loc("foo")
  }
})"));
  TF_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<xla::ifrt::HloProgram> hlo_program2,
      ParseHloProgramString(R"(
module @foo {
  func.func @main(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
    return %arg0 : tensor<2x3xi32> loc("bar")
  }
})"));

  EXPECT_EQ(hlo_program1->Fingerprint(), hlo_program2->Fingerprint());
}

TEST(HloProgramTest, BytesRoundTrip) {
  static constexpr absl::string_view kModule = R"(
module @hlo_module attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = mhlo.add %arg0, %0 : tensor<f32>
    return %1 : tensor<f32>
  }
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto program, ParseHloProgramString(kModule));
  TF_ASSERT_OK_AND_ASSIGN(auto serialized, program->ToBytes());
  TF_ASSERT_OK_AND_ASSIGN(auto deserialized, HloProgram::FromBytes(serialized));
  EXPECT_EQ(program->Fingerprint(), deserialized->Fingerprint());
}

}  // namespace
}  // namespace xla::ifrt
