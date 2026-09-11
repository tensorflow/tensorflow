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

#include "xla/python/ifrt/mlir/fingerprint_utils.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/pjrt/mlir_to_hlo.h"

namespace xla {
namespace ifrt {
namespace {

TEST(FingerprintUtilsTest, IdenticalModulesHaveSameFingerprint) {
  static constexpr absl::string_view kModule = R"(
module {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = stablehlo.add %arg0, %0 : tensor<f32>
    return %1 : tensor<f32>
  }
}
)";
  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       xla::ParseMlirModuleString(kModule, context));
  ASSERT_OK_AND_ASSIGN(uint64_t fp1, FingerprintModuleOp(*module));
  ASSERT_OK_AND_ASSIGN(uint64_t fp2, FingerprintModuleOp(*module));
  EXPECT_EQ(fp1, fp2);
}

TEST(FingerprintUtilsTest, DistinctStablehloModulesHaveDifferentFingerprints) {
  static constexpr absl::string_view kModule1 = R"(
module {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = stablehlo.add %arg0, %0 : tensor<f32>
    return %1 : tensor<f32>
  }
}
)";
  static constexpr absl::string_view kModule2 = R"(
module {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1 = stablehlo.add %arg0, %0 : tensor<f32>
    return %1 : tensor<f32>
  }
}
)";
  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module1,
                       xla::ParseMlirModuleString(kModule1, context));
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module2,
                       xla::ParseMlirModuleString(kModule2, context));
  ASSERT_OK_AND_ASSIGN(uint64_t fp1, FingerprintModuleOp(*module1));
  ASSERT_OK_AND_ASSIGN(uint64_t fp2, FingerprintModuleOp(*module2));
  EXPECT_NE(fp1, fp2);
}

TEST(FingerprintUtilsTest, IgnoresDebugLocations) {
  static constexpr absl::string_view kModule1 = R"(
module @foo {
  func.func @main(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
    return %arg0 : tensor<2x3xi32> loc("foo")
  }
}
)";
  static constexpr absl::string_view kModule2 = R"(
module @foo {
  func.func @main(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
    return %arg0 : tensor<2x3xi32> loc("bar")
  }
}
)";
  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module1,
                       xla::ParseMlirModuleString(kModule1, context));
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module2,
                       xla::ParseMlirModuleString(kModule2, context));
  ASSERT_OK_AND_ASSIGN(uint64_t fp1, FingerprintModuleOp(*module1));
  ASSERT_OK_AND_ASSIGN(uint64_t fp2, FingerprintModuleOp(*module2));
  EXPECT_EQ(fp1, fp2);
}

TEST(FingerprintUtilsTest, IgnoresDebugLocationStructure) {
  static constexpr absl::string_view kModule1 = R"(
module @foo {
  func.func @main(%arg0: tensor<2x3xi32> loc("foo")) -> tensor<2x3xi32> {
    return %arg0 : tensor<2x3xi32> loc("foo")
  } loc("foo")
} loc("foo")
)";
  static constexpr absl::string_view kModule2 = R"(
module @foo {
  func.func @main(%arg0: tensor<2x3xi32> loc("bar")) -> tensor<2x3xi32> {
    return %arg0 : tensor<2x3xi32> loc("baz")
  } loc("qux")
} loc("quux")
)";
  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module1,
                       xla::ParseMlirModuleString(kModule1, context));
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module2,
                       xla::ParseMlirModuleString(kModule2, context));
  ASSERT_OK_AND_ASSIGN(uint64_t fp1, FingerprintModuleOp(*module1));
  ASSERT_OK_AND_ASSIGN(uint64_t fp2, FingerprintModuleOp(*module2));
  EXPECT_EQ(fp1, fp2);
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
