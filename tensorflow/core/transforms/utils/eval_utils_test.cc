/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/transforms/utils/eval_utils.h"

#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace tfg {

TEST(EvalUtilsTest, EvaluateOperation) {
  const char *const code = R"mlir(
    tfg.func @test() -> (tensor<2x2xi32>) {
      %Const_0, %ctl_0 = Const name("c0") {dtype = i32, value = dense<1> : tensor<2x2xi32>} : () -> (tensor<2x2xi32>)
      %Const_1, %ctl_2 = Const name("c1") {dtype = i32, value = dense<2> : tensor<2x2xi32>} : () -> (tensor<2x2xi32>)
      %Add, %ctl_7 = Add(%Const_0, %Const_1) name("add") {T = i32} : (tensor<2x2xi32>, tensor<2x2xi32>) -> (tensor<2x2xi32>)
      return (%Const_1) : tensor<2x2xi32>
    }
  )mlir";

  MLIRContext context;
  context.getOrLoadDialect<tfg::TFGraphDialect>();
  OwningOpRef<ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  GraphFuncOp func = module->lookupSymbol<GraphFuncOp>("test");
  ASSERT_TRUE(func);

  auto iter = func.body().begin()->begin();
  Operation *const_0 = &*iter++;
  Operation *const_1 = &*iter++;
  Operation *add = &*iter++;

  auto cpu_device = std::make_unique<util::SimpleDevice>();
  auto resource_mgr = std::make_unique<tensorflow::ResourceMgr>();

  llvm::SmallVector<Attribute> result;

  EXPECT_TRUE(succeeded(util::EvaluateOperation(
      cpu_device.get(), resource_mgr.get(), const_0,
      {const_0->getAttrOfType<ElementsAttr>("value")}, result)));

  ASSERT_EQ(result.size(), 1);
  ASSERT_TRUE(result[0].isa<ElementsAttr>());
  EXPECT_EQ(result[0].cast<ElementsAttr>().getValues<int>()[0], 1);

  result.clear();

  EXPECT_TRUE(succeeded(util::EvaluateOperation(
      cpu_device.get(), resource_mgr.get(), const_1,
      {const_1->getAttrOfType<ElementsAttr>("value")}, result)));

  ASSERT_EQ(result.size(), 1);
  ASSERT_TRUE(result[0].isa<ElementsAttr>());
  EXPECT_EQ(result[0].cast<ElementsAttr>().getValues<int>()[0], 2);

  result.clear();

  EXPECT_TRUE(succeeded(
      util::EvaluateOperation(cpu_device.get(), resource_mgr.get(), add,
                              {const_0->getAttrOfType<ElementsAttr>("value"),
                               const_1->getAttrOfType<ElementsAttr>("value")},
                              result)));

  ASSERT_EQ(result.size(), 1);
  ASSERT_TRUE(result[0].isa<ElementsAttr>());
  EXPECT_EQ(result[0].cast<ElementsAttr>().getValues<int>()[0], 3);
}

}  // namespace tfg
}  // namespace mlir
