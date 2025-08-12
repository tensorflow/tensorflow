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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "stablehlo/api/PortableApi.h"
#include "xla/hlo/testlib/test.h"
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
  TF_ASSERT_OK_AND_ASSIGN(std::string blob, Serialize(*module, "1.0.0", 70));

  // StableHLO uses VHLO for PJRT serialization.
  EXPECT_THAT(blob, IsVhloArtifact("1.0.0"));
}

TEST(MlirToHloTest, StablehloPluginNewerThanFramework) {
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

  // Request version v100.99.88, newer than the framework version.
  // Serialize uses frameworks version when plugin requests a newer version.
  TF_ASSERT_OK_AND_ASSIGN(std::string blob,
                          Serialize(*module, "100.99.98", 70));
  EXPECT_THAT(blob, IsVhloArtifact(mlir::stablehlo::getCurrentVersion()));
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
  TF_ASSERT_OK_AND_ASSIGN(std::string blob, Serialize(*module, "1.0.0", 70));

  // CHLO decomposes to StableHLO, so uses VHLO serialization.
  EXPECT_THAT(blob, IsVhloArtifact("1.0.0"));
}

TEST(MlirToHloTest, ChloTanOpTest) {
  constexpr char kProgram[] =
      R"(
    func.func @add(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
      %0 = chlo.tan %arg0 : tensor<1x2xf32> -> tensor<1x2xf32>
      return %0 : tensor<1x2xf32>
    }
  )";
  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModuleString(kProgram, context));
  TF_ASSERT_OK_AND_ASSIGN(std::string blob, Serialize(*module, "1.0.0", 70));

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
  TF_ASSERT_OK_AND_ASSIGN(std::string blob, Serialize(*module, "1.0.0", 70));

  // MHLO and other dialects use native MLIR bytecode, not VHLO.
  EXPECT_THAT(blob, Not(IsVhloArtifact("1.0.0")));
}

TEST(MlirToHloTest, InvalidBytecodeTest) {
  // MLIR bytecode format has full compatibility.
  // Program using StableHLO v2.0.0 with op vhlo.constant_v99.
  // TODO: Once this file is exposed via the StableHLO repo, replace this
  // bytecode string with a read of the StableHLO file.
  unsigned char invalid_future_vhlo_mlirbc[] = {
      0x4d, 0x4c, 0xef, 0x52, 0x0d, 0x53, 0x74, 0x61, 0x62, 0x6c, 0x65, 0x48,
      0x4c, 0x4f, 0x5f, 0x76, 0x32, 0x2e, 0x30, 0x2e, 0x30, 0x00, 0x01, 0x19,
      0x05, 0x01, 0x05, 0x09, 0x01, 0x03, 0x0b, 0x03, 0x07, 0x0f, 0x13, 0x17,
      0x03, 0x2b, 0x15, 0x07, 0x01, 0x0b, 0x0b, 0x13, 0x13, 0x13, 0x13, 0x03,
      0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x1f, 0x03, 0x07, 0x0f, 0x13, 0x07, 0x02,
      0x53, 0x05, 0x0d, 0x17, 0x01, 0x03, 0x03, 0x17, 0x01, 0x05, 0x07, 0x17,
      0x01, 0x07, 0x15, 0x17, 0x01, 0x09, 0x0b, 0x03, 0x01, 0x23, 0x03, 0x1d,
      0x0f, 0x1d, 0x11, 0x1f, 0x01, 0x09, 0x00, 0x00, 0x80, 0x3f, 0x29, 0x01,
      0x05, 0x11, 0x01, 0x03, 0x01, 0x09, 0x04, 0x41, 0x05, 0x01, 0x50, 0x03,
      0x01, 0x07, 0x04, 0x31, 0x03, 0x01, 0x05, 0x03, 0x50, 0x05, 0x03, 0x07,
      0x04, 0x1d, 0x03, 0x03, 0x09, 0x05, 0x42, 0x07, 0x05, 0x03, 0x01, 0x07,
      0x04, 0x09, 0x03, 0x01, 0x06, 0x03, 0x01, 0x05, 0x01, 0x00, 0xad, 0x13,
      0x0f, 0x0b, 0x1b, 0x15, 0x1b, 0x11, 0x0f, 0x0b, 0x11, 0x62, 0x75, 0x69,
      0x6c, 0x74, 0x69, 0x6e, 0x00, 0x76, 0x68, 0x6c, 0x6f, 0x00, 0x6d, 0x6f,
      0x64, 0x75, 0x6c, 0x65, 0x00, 0x66, 0x75, 0x6e, 0x63, 0x5f, 0x76, 0x31,
      0x00, 0x63, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x5f, 0x76, 0x39,
      0x39, 0x00, 0x72, 0x65, 0x74, 0x75, 0x72, 0x6e, 0x5f, 0x76, 0x31, 0x00,
      0x2f, 0x74, 0x6d, 0x70, 0x2f, 0x74, 0x32, 0x2e, 0x6d, 0x6c, 0x69, 0x72,
      0x00, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x70, 0x75, 0x62, 0x6c, 0x69, 0x63,
      0x00, 0x08, 0x19, 0x07, 0x05, 0x01, 0x01, 0x0b, 0x0b, 0x0d, 0x0b, 0x0f,
      0x11, 0x03, 0x13};
  unsigned int invalid_future_vhlo_mlirbc_len = 243;

  std::string buffer(reinterpret_cast<char*>(invalid_future_vhlo_mlirbc),
                     invalid_future_vhlo_mlirbc_len);

  mlir::MLIRContext context;
  auto status = ParseMlirModuleString(buffer, context);
  ASSERT_FALSE(status.ok());
  // Check that the error message contains:
  //   - The name of the op that is not supported (vhlo.constant_v99)
  //   - The version that the StableHLO portable artifact was emit for (v2.0.0)
  //   - The current version of StableHLO (v1.X.Y)
  EXPECT_THAT(status.status().message(), HasSubstr("vhlo.constant_v99"));
  EXPECT_THAT(status.status().message(), HasSubstr("StableHLO_v2.0.0"));
  EXPECT_THAT(status.status().message(),
              HasSubstr(mlir::stablehlo::getCurrentVersion()));
}

}  // namespace
}  // namespace xla
