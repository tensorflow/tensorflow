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
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace gpu {

class IrEmissionUtilsTest : public HloTestBase {};

TEST_F(IrEmissionUtilsTest, TestOperandPartitionNoAlias) {
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

TEST_F(IrEmissionUtilsTest, TestOperandPartitionWithAlias0) {
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

TEST_F(IrEmissionUtilsTest, TestOperandPartitionWithAlias1) {
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

TEST_F(IrEmissionUtilsTest, FindTiledLogicalTranspose) {
  const char* hlo = R"(
HloModule module

ENTRY entry {
  p = f32[32,48,64]{2,1,0} parameter(0)
  ROOT t = f32[64,32,48]{2,1,0} transpose(p), dimensions={2,0,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* tr = module->entry_computation()->root_instruction();
  EXPECT_EQ(FindTiledLogicalTranspose(*tr),
            std::make_optional(Vector3{1, 64, 1536}));
}

}  // namespace gpu
}  // namespace xla
