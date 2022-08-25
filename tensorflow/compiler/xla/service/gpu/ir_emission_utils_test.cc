/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu {

TEST(IrEmissionUtilsTest, TestOperandPartitionNoAlias) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::lmhlo::LmhloDialect>();
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);

  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"(
    func.func @foo(%arg0 : memref<f32>, %arg1 : memref<f32>, %arg2 : memref<f32>) {
      "lmhlo.add" (%arg0, %arg1, %arg2) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator" () : () -> ()
    }
  )",
                                                        &context);
  mlir::func::FuncOp func =
      mlir::cast<mlir::func::FuncOp>(module->lookupSymbol("foo"));
  mlir::Operation* op = &func.getBody().front().front();
  EXPECT_EQ(2, PartitionLmhloOperandsAndOutputs(op));
}

TEST(IrEmissionUtilsTest, TestOperandPartitionWithAlias0) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::lmhlo::LmhloDialect>();
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);

  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"(
    func.func @foo(%arg0 : memref<f32>, %arg1 : memref<f32>, %arg2 : memref<f32>) {
      "lmhlo.add" (%arg0, %arg1, %arg0) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator" () : () -> ()
    }
  )",
                                                        &context);
  mlir::func::FuncOp func =
      mlir::cast<mlir::func::FuncOp>(module->lookupSymbol("foo"));
  mlir::Operation* op = &func.getBody().front().front();
  EXPECT_EQ(2, PartitionLmhloOperandsAndOutputs(op));
}

TEST(IrEmissionUtilsTest, TestOperandPartitionWithAlias1) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::lmhlo::LmhloDialect>();
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);

  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"(
    func.func @foo(%arg0 : memref<f32>, %arg1 : memref<f32>, %arg2 : memref<f32>) {
      "lmhlo.add" (%arg0, %arg1, %arg1) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator" () : () -> ()
    }
  )",
                                                        &context);
  mlir::func::FuncOp func =
      mlir::cast<mlir::func::FuncOp>(module->lookupSymbol("foo"));
  mlir::Operation* op = &func.getBody().front().front();
  EXPECT_EQ(2, PartitionLmhloOperandsAndOutputs(op));
}

class Match021Test : public HloTestBase {};

TEST_F(Match021Test, Simple) {
  using ::testing::Eq;

  constexpr const char* hlo = R"(
HloModule module

ENTRY main {
  p = f32[128,64]{1,0} parameter(0)
  ROOT out = f32[128,64]{0,1} copy(p)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  HloComputation* c = module->entry_computation();
  auto dims_and_params = Match021Transpose(c);
  EXPECT_THAT(dims_and_params->dims, Eq(Vector3{1, 64, 128}));
  EXPECT_THAT(dims_and_params->params, Eq(std::vector<int64_t>{0}));
}

TEST_F(Match021Test, DifferentTransposesFail) {
  using ::testing::Eq;

  constexpr const char* hlo = R"(
HloModule module

ENTRY main {
  p0 = f32[8,31,31,65]{3,2,1,0} parameter(0)
  p1 = f32[8,31,31,65]{1,2,3,0} parameter(1)
  ROOT out = f32[8,31,31,65]{2,1,3,0} copy(p0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  HloComputation* c = module->entry_computation();
  auto dims_and_params = Match021Transpose(c);
  EXPECT_THAT(dims_and_params, Eq(std::nullopt));
}

TEST_F(Match021Test, DimTooSmall) {
  using ::testing::Eq;

  constexpr const char* hlo = R"(
HloModule module

ENTRY main {
  p = f32[128,8]{1,0} parameter(0)
  ROOT out = f32[128,8]{0,1} copy(p)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  HloComputation* c = module->entry_computation();
  auto dims_and_params = Match021Transpose(c);
  EXPECT_THAT(dims_and_params, Eq(std::nullopt));
}

TEST_F(Match021Test, FilterFail) {
  using ::testing::Eq;

  constexpr const char* hlo = R"(
HloModule module

ENTRY main {
  p = f32[128,16]{1,0} parameter(0)
  b = f32[128,16]{1,0} bitcast(p)
  ROOT out = f32[128,16]{0,1} copy(b)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  HloComputation* c = module->entry_computation();
  auto dims_and_params = Match021Transpose(c);
  EXPECT_THAT(dims_and_params, Eq(std::nullopt));
}

}  // namespace gpu
}  // namespace xla
