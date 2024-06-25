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

#include "xla/pjrt/mlir_to_hlo.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "xla/test.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

// Portable artifacts are serialized using `serializePortableArtifact` will have
// a version tag with the target version, i.e. StableHLO_v1.0.0.
// Native serialization instead has an MLIR version tag.
MATCHER_P(IsVhloArtifact, version, "") {
  return ExplainMatchResult(HasSubstr(absl::StrCat("StableHLO_v", version)),
                            arg, result_listener);
}

TEST(MlirToHloTest, StablehloTest) {
  constexpr char kProgram[] =
      R"(
    func.func @add(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
      %cst = stablehlo.constant dense<1.0> : tensor<1x2xf32>
      %0 = stablehlo.add %arg0, %cst : tensor<1x2xf32>
      return %0 : tensor<1x2xf32>
    }
  )";
  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModuleString(kProgram, context));
  TF_ASSERT_OK_AND_ASSIGN(std::string blob, Serialize(*module, 47, "1.0.0"));

  // StableHLO uses VHLO for PJRT serialization.
  EXPECT_THAT(blob, IsVhloArtifact("1.0.0"));
}

TEST(MlirToHloTest, ChloTest) {
  constexpr char kProgram[] =
      R"(
    func.func @add(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
      %cst = stablehlo.constant dense<1.0> : tensor<1x2xf32>
      %0 = chlo.broadcast_add %arg0, %cst : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
      return %0 : tensor<1x2xf32>
    }
  )";
  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModuleString(kProgram, context));
  TF_ASSERT_OK_AND_ASSIGN(std::string blob, Serialize(*module, 47, "1.0.0"));

  // CHLO decomposes to StableHLO, so uses VHLO serialization.
  EXPECT_THAT(blob, IsVhloArtifact("1.0.0"));
}

TEST(MlirToHloTest, MhloTest) {
  constexpr char kProgram[] =
      R"(
    func.func @add(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
      %cst = mhlo.constant dense<1.0> : tensor<1x2xf32>
      %0 = mhlo.add %arg0, %cst : tensor<1x2xf32>
      return %0 : tensor<1x2xf32>
    }
  )";
  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModuleString(kProgram, context));
  TF_ASSERT_OK_AND_ASSIGN(std::string blob, Serialize(*module, 47, "1.0.0"));

  // MHLO and other dialects use native MLIR bytecode, not VHLO.
  EXPECT_THAT(blob, Not(IsVhloArtifact("1.0.0")));
}

}  // namespace
}  // namespace xla
