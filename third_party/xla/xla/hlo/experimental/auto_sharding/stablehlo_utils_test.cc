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
#include "xla/hlo/experimental/auto_sharding/stablehlo_utils.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace spmd {
namespace {

class StablehloUtilsTest : public HloHardwareIndependentTestBase {
 protected:
  void MatchHloModule(HloModule& module, absl::string_view pattern) {
    TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                            RunFileCheck(module.ToString(), pattern));
    EXPECT_TRUE(filecheck_result);
  }
  void ConvertShardyAndCompare(const std::string& shardy_mlir_string,
                               const std::string& expected_hlo_pattern) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::mhlo::MhloDialect, mlir::sdy::SdyDialect,
                        mlir::func::FuncDialect,
                        mlir::stablehlo::StablehloDialect>();

    mlir::OwningOpRef<mlir::ModuleOp> shardy_module =
        mlir::parseSourceString<mlir::ModuleOp>(shardy_mlir_string, &context);
    ASSERT_NE(shardy_module.get(), nullptr);

    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> converted_hlo_module,
                            ConvertShardyToHlo(shardy_module.get()));
    MatchHloModule(*converted_hlo_module, expected_hlo_pattern);
  }
};

constexpr absl::string_view kShardyTemplate = R"MLIR(
module @matmul attributes {
  mhlo.num_partitions = 8 : i32, mhlo.use_auto_spmd_partitioning = true} {
  sdy.mesh @mesh = <["x"=4, "y"=2]>
  func.func @main(
    %%arg0: tensor<400x400xf32>%s,
    %%arg1: tensor<400x400xf32>%s
  ) -> tensor<400x400xf32> {
    %%0 = stablehlo.dot %%arg0, %%arg1 {
      sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=400, j=400, k=400} reduction={k}>
    } : (tensor<400x400xf32>, tensor<400x400xf32>) -> tensor<400x400xf32>
    return %%0 : tensor<400x400xf32>
  }
}
)MLIR";

TEST_F(StablehloUtilsTest, ConvertShardyToHloNoSharding) {
  const std::string kShardyMlirString = absl::StrFormat(
      kShardyTemplate, R"( {sdy.sharding = #sdy.sharding<@mesh, [{?}, {?}]>})",
      R"( {sdy.sharding = #sdy.sharding<@mesh, [{?}, {?}]>})");

  const std::string kExpectedHloPattern = R"(
  CHECK: Arg_0.1 = f32[400,400]{1,0} parameter(0), sharding={replicated}
  CHECK: Arg_1.2 = f32[400,400]{1,0} parameter(1), sharding={replicated}
  )";

  ConvertShardyAndCompare(kShardyMlirString, kExpectedHloPattern);
}

TEST_F(StablehloUtilsTest, ConvertShardyToHlo1DSharding) {
  const std::string kShardyMlirString = absl::StrFormat(
      kShardyTemplate, R"( {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})",
      R"( {sdy.sharding = #sdy.sharding<@mesh, [{?}, {?}]>})");

  const std::string kExpectedHloPattern = R"(
  CHECK: %Arg_0.1 = f32[400,400]{1,0} parameter(0), sharding={devices=[4,1,2]<=[8] last_tile_dim_replicate}
  CHECK: %Arg_1.2 = f32[400,400]{1,0} parameter(1), sharding={replicated}
  )";

  ConvertShardyAndCompare(kShardyMlirString, kExpectedHloPattern);
}

TEST_F(StablehloUtilsTest, ConvertShardyToHlo2DSharding) {
  const std::string kShardyMlirString = absl::StrFormat(
      kShardyTemplate, R"( {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})",
      R"( {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})");

  const std::string kExpectedHloPattern = R"(
  CHECK: %Arg_0.1 = f32[400,400]{1,0} parameter(0), sharding={devices=[4,1,2]<=[8] last_tile_dim_replicate}
  CHECK:  %Arg_1.2 = f32[400,400]{1,0} parameter(1), sharding={devices=[1,2,4]<=[4,2]T(1,0) last_tile_dim_replicate}
  )";

  ConvertShardyAndCompare(kShardyMlirString, kExpectedHloPattern);
}

}  // namespace
}  // namespace spmd
}  // namespace xla
