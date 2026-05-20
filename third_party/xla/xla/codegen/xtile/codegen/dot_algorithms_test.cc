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

#include "xla/codegen/xtile/codegen/dot_algorithms.h"

#include <memory>

#include <gtest/gtest.h>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/xtile_dialect.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla::xtile {
namespace {

class DotAlgorithmsTest : public HloHardwareIndependentTestBase {
 public:
  DotAlgorithmsTest() : b_(mlir::UnknownLoc::get(&ctx_), &ctx_) {
    ctx_.loadDialect<mlir::arith::ArithDialect,
                     mlir::stablehlo::StablehloDialect, xtile::XTileDialect>();
    module_ = xla::llvm_ir::CreateMlirModuleOp(b_.getLoc());
    b_.setInsertionPointToStart(module_->getBody());
  }

  mlir::MLIRContext ctx_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  mlir::ImplicitLocOpBuilder b_;
};

TEST_F(DotAlgorithmsTest, EmitSingleTileDotDefaultToBF16) {
  auto kHloText = R"hlo(
    HloModule test
    ENTRY test {
      p0 = f32[32,32] parameter(0)
      p1 = f32[32,32] parameter(1)
      ROOT dot = f32[32,32] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )hlo";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  auto* dot_instr = xla::Cast<HloDotInstruction>(
      module->entry_computation()->root_instruction());

  // Set the flag in debug options.
  DebugOptions debug_options = module->config().debug_options();
  debug_options.set_xla_gpu_default_to_alg_dot_bf16_bf16_f32(true);
  module->mutable_config().set_debug_options(debug_options);

  // Setup operands.
  auto type = mlir::RankedTensorType::get({32, 32}, b_.getF32Type());
  DotOperands operands;
  operands.lhs = b_.create<mlir::stablehlo::ConstantOp>(
      mlir::DenseElementsAttr::get(type, 1.0f));
  operands.rhs = b_.create<mlir::stablehlo::ConstantOp>(
      mlir::DenseElementsAttr::get(type, 1.0f));
  operands.accumulator = b_.create<mlir::stablehlo::ConstantOp>(
      mlir::DenseElementsAttr::get(type, 0.0f));

  TF_ASSERT_OK_AND_ASSIGN(mlir::Value result,
                          EmitSingleTileDot(b_, *dot_instr, operands));

  // The result should be an addition.
  auto add_op = mlir::dyn_cast<mlir::arith::AddFOp>(result.getDefiningOp());
  ASSERT_TRUE(add_op);

  // One of the operands of the addition should be the dot operation.
  auto dot_op = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(
      add_op.getRhs().getDefiningOp());
  ASSERT_TRUE(dot_op);

  // Verify that dot operands are BF16.
  EXPECT_TRUE(mlir::cast<mlir::ShapedType>(dot_op.getLhs().getType())
                  .getElementType()
                  .isBF16());
  EXPECT_TRUE(mlir::cast<mlir::ShapedType>(dot_op.getRhs().getType())
                  .getElementType()
                  .isBF16());
}

}  // namespace
}  // namespace xla::xtile
