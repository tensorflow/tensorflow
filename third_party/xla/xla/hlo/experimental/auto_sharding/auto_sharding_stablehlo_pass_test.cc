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

#include "xla/hlo/experimental/auto_sharding/auto_sharding_stablehlo_pass.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/hlo/testlib/filecheck.h"

namespace xla {
namespace spmd {
namespace {

class AutoShardingTest : public ::testing::Test {
 protected:
  AutoShardingTest() {
    RegisterDialectDependencies(registry_);
    context_.appendDialectRegistry(registry_);
  }

  void MatchSardyModule(mlir::ModuleOp module, absl::string_view pattern) {
    std::string module_string;
    llvm::raw_string_ostream os(module_string);
    module.print(os);
    absl::StatusOr<bool> filecheck_result =
        RunFileCheck(module_string, pattern);
    ASSERT_TRUE(filecheck_result.ok());
    EXPECT_TRUE(filecheck_result.value());
  }

  mlir::DialectRegistry registry_;
  mlir::MLIRContext context_;
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

TEST_F(AutoShardingTest, OpenDimensionsInputSharding) {
  const std::string kShardyMlirString = absl::StrFormat(
      kShardyTemplate, R"( {sdy.sharding = #sdy.sharding<@mesh, [{?}, {?}]>})",
      R"( {sdy.sharding = #sdy.sharding<@mesh, [{?}, {?}]>})");

  absl::string_view kExpectedTransformedShardyPattern = R"(
  CHECK: sdy.mesh @mesh_0 = <["x"=4, "y"=2]>
  CHECK: %arg0: tensor<400x400xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}]>}
  CHECK-SAME: %arg1: tensor<400x400xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}]>}
  CHECK-SAME: -> (tensor<400x400xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"x"}, {"y"}]>}
  CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"x"}, {}]>]>}
  CHECK: %1 = stablehlo.reshape %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {"y"}]>]>}
  CHECK: %2 = stablehlo.dot %0, %1, {{.*}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"x"}, {"y"}]>]>}
  )";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kShardyMlirString, &context_);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context_);
  AddAutoShardingToPipeline(pm);
  ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

  MatchSardyModule(module.get(), kExpectedTransformedShardyPattern);
}

TEST_F(AutoShardingTest, ClosedDimensionsInputSharding) {
  const std::string kShardyMlirString = absl::StrFormat(
      kShardyTemplate, R"( {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})",
      R"( {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})");

  absl::string_view kExpectedTransformedShardyPattern = R"(
  CHECK: sdy.mesh @mesh_0 = <["x"=4, "y"=2]>
  CHECK: %arg0: tensor<400x400xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}]>}
  CHECK-SAME: %arg1: tensor<400x400xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}]>}
  CHECK-SAME: -> (tensor<400x400xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"x"}, {"y"}]>}
  CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"x"}, {}]>]>}
  CHECK: %1 = stablehlo.reshape %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {"y"}]>]>}
  CHECK: %2 = stablehlo.dot %0, %1, {{.*}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"x"}, {"y"}]>]>}
  )";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kShardyMlirString, &context_);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context_);
  AddAutoShardingToPipeline(pm);
  ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

  MatchSardyModule(module.get(), kExpectedTransformedShardyPattern);
}

TEST_F(AutoShardingTest, HybridDimensionsInputSharding) {
  const std::string kShardyMlirString = absl::StrFormat(
      kShardyTemplate, R"( {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})",
      R"( {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})");

  absl::string_view kExpectedTransformedShardyPattern = R"(
  CHECK: sdy.mesh @mesh_0 = <["x"=4, "y"=2]>
  CHECK: %arg0: tensor<400x400xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"x"}, {}]>}
  CHECK-SAME: %arg1: tensor<400x400xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}]>}
  CHECK-SAME: -> (tensor<400x400xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"x"}, {"y"}]>}
  CHECK: %0 = stablehlo.reshape %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {"y"}]>]>}
  CHECK: %1 = stablehlo.dot %arg0, %0, {{.*}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"x"}, {"y"}]>]>}
  )";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kShardyMlirString, &context_);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context_);
  AddAutoShardingToPipeline(pm);
  ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

  MatchSardyModule(module.get(), kExpectedTransformedShardyPattern);
}

TEST_F(AutoShardingTest, FixedDimensionsInputSharding) {
  const std::string kShardyMlirString = absl::StrFormat(
      kShardyTemplate, R"( {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})",
      R"( {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})");

  absl::string_view kExpectedTransformedShardyPattern = R"(
  CHECK: sdy.mesh @mesh_0 = <["x"=4, "y"=2]>
  CHECK: %arg0: tensor<400x400xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"x"}, {}]>}
  CHECK-SAME: %arg1: tensor<400x400xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {"y"}]>}
  CHECK-SAME: -> (tensor<400x400xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"x"}, {"y"}]>}
  CHECK: %0 = stablehlo.dot %arg0, %arg1, {{.*}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"x"}, {"y"}]>]>}
  )";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kShardyMlirString, &context_);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context_);
  AddAutoShardingToPipeline(pm);
  ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

  MatchSardyModule(module.get(), kExpectedTransformedShardyPattern);
}
}  // namespace
}  // namespace spmd
}  // namespace xla
