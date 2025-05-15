/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/transforms/xla_legalize_targets.h"

#include <gtest/gtest.h>
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace hlo {
namespace {

mlir::DialectRegistry GetDefaultDialectRegistry() {
  mlir::DialectRegistry registry;

  registry.insert<arith::ArithDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<shape::ShapeDialect>();
  registry.insert<TF::TensorFlowDialect>();
  registry.insert<chlo::ChloDialect>();

  return registry;
}

class XlaLegalizeTargetsTest : public testing::Test {
 public:
  XlaLegalizeTargetsTest()
      : context_(GetDefaultDialectRegistry()),
        module_(mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_))),
        builder_(&module_->getBodyRegion()) {
    context_.loadAllAvailableDialects();
  }

 protected:
  mlir::MLIRContext context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  mlir::OpBuilder builder_;
};

TEST_F(XlaLegalizeTargetsTest, CreatesConversionTargets) {
  auto const_int = builder_.create<mlir::arith::ConstantIntOp>(
      builder_.getUnknownLoc(), /*value=*/10, builder_.getI32Type());

  ConversionTarget target =
      GetDefaultLegalConversionTargets(context_, /*legalize_chlo=*/false);
  EXPECT_TRUE(target.isLegal(const_int));
}

TEST_F(XlaLegalizeTargetsTest, AllowsCHLODialect) {
  auto const_int = builder_.create<chlo::ConstantOp>(
      builder_.getUnknownLoc(), builder_.getI32TensorAttr({42}));

  ConversionTarget target =
      GetDefaultLegalConversionTargets(context_, /*legalize_chlo=*/true);

  EXPECT_TRUE(target.isIllegal(const_int));
}

TEST_F(XlaLegalizeTargetsTest, DontAllowCHLODialect) {
  auto const_int = builder_.create<chlo::ConstantOp>(
      builder_.getUnknownLoc(), builder_.getI32TensorAttr({42}));

  ConversionTarget target =
      GetDefaultLegalConversionTargets(context_, /*legalize_chlo=*/false);
  EXPECT_TRUE(target.isLegal(const_int));
}

}  // namespace
}  // namespace hlo
}  // namespace mlir
