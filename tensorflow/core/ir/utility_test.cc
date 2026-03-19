/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/ir/utility.h"

#include <optional>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace tfg {
namespace {

TEST(DialectUtilityTest, TestLookupControlDependency) {
  MLIRContext context;
  context.getOrLoadDialect<tfg::TFGraphDialect>();
  const char *const code = R"mlir(
    tfg.func @test(%arg: tensor<i32> {tfg.name = "arg"}) -> (tensor<i32>) {
      %Copy, %ctl = Copy(%arg) : (tensor<i32>) -> (tensor<i32>)
      return(%Copy) : tensor<i32>
    }
  )mlir";

  OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  GraphFuncOp func = module->lookupSymbol<GraphFuncOp>("test");
  ASSERT_TRUE(func);
  auto ret_op = cast<ReturnOp>(func.getBody().front().getTerminator());

  Value copy = ret_op.getOperand(0);
  Value ctl = LookupControlDependency(copy);
  ASSERT_TRUE(ctl);
  OpResult ctl_result = mlir::dyn_cast<OpResult>(ctl);
  ASSERT_TRUE(ctl_result);
  EXPECT_EQ(ctl_result.getResultNumber(), 1);
  EXPECT_EQ(copy, ctl_result.getOwner()->getResult(0));
  EXPECT_EQ(ctl_result.getOwner()->getName().getStringRef(), "tfg.Copy");

  Value arg = ctl_result.getOwner()->getOperand(0);
  Value arg_ctl = LookupControlDependency(arg);
  ASSERT_TRUE(arg_ctl);
  BlockArgument ctl_arg = mlir::dyn_cast<BlockArgument>(arg_ctl);
  ASSERT_TRUE(ctl_arg);
  EXPECT_EQ(ctl_arg.getArgNumber(), 1);
  EXPECT_EQ(arg, ctl_arg.getOwner()->getArgument(0));
}

TEST(DialectUtilityTest, TestLookupDataValue) {
  MLIRContext context;
  context.getOrLoadDialect<tfg::TFGraphDialect>();
  const char *const code = R"mlir(
    tfg.func @test(%arg: tensor<i32> {tfg.name = "arg"}) -> (tensor<i32>) {
      %Produce, %ctl = Produce [%arg.ctl] : () -> (tensor<i32>)
      return(%arg) [%ctl] : tensor<i32>
    }
  )mlir";

  OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  GraphFuncOp func = module->lookupSymbol<GraphFuncOp>("test");
  ASSERT_TRUE(func);
  auto ret_op = cast<ReturnOp>(func.getBody().front().getTerminator());

  Value ctl = ret_op.getOperand(1);
  std::optional<Value> produce = LookupDataValue(ctl);
  ASSERT_TRUE(produce);
  OpResult produce_result = mlir::dyn_cast<OpResult>(*produce);
  ASSERT_TRUE(produce_result);
  ASSERT_EQ(produce_result.getResultNumber(), 0);
  ASSERT_EQ(produce_result.getOwner()->getName().getStringRef(), "tfg.Produce");
  ASSERT_EQ(produce_result.getOwner()->getResult(1), ctl);

  Value arg_ctl = produce_result.getOwner()->getOperand(0);
  std::optional<Value> arg = LookupDataValue(arg_ctl);
  ASSERT_TRUE(arg);
  BlockArgument arg_arg = mlir::dyn_cast<BlockArgument>(*arg);
  ASSERT_TRUE(arg_arg);
  ASSERT_EQ(arg_arg.getArgNumber(), 0);
  ASSERT_EQ(arg_arg.getOwner()->getArgument(1), arg_ctl);
}

TEST(DialectUtilityTest, TestLookupDataValueNoData) {
  MLIRContext context;
  context.getOrLoadDialect<tfg::TFGraphDialect>();
  const char *const code = R"mlir(
    tfg.func @test(%arg: tensor<i32> {tfg.name = "arg"}) -> (tensor<i32>) {
      %ctl = NoOp [%arg.ctl] : () -> ()
      return(%arg) [%ctl] : tensor<i32>
    }
  )mlir";

  OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  GraphFuncOp func = module->lookupSymbol<GraphFuncOp>("test");
  ASSERT_TRUE(func);
  auto ret_op = cast<ReturnOp>(func.getBody().front().getTerminator());

  Value ctl = ret_op.getOperand(1);
  std::optional<Value> no_data = LookupDataValue(ctl);
  ASSERT_FALSE(no_data);
}

}  // namespace
}  // namespace tfg
}  // namespace mlir
