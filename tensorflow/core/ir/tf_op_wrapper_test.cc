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

#include "tensorflow/core/ir/tf_op_wrapper.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace tfg {
namespace {
TEST(TFOpWrapper, LLVMRTTI) {
  const char *const code = R"mlir(
    tfg.func @test() -> (tensor<i32>) {
      %A, %ctlA = A : () -> (tensor<i32>)
      return(%A) : tensor<i32>
    }
  )mlir";
  MLIRContext context;
  context.getOrLoadDialect<TFGraphDialect>();
  OwningOpRef<ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);

  Operation *module_op = module.get();
  EXPECT_FALSE(isa<TFOp>(module_op));
  EXPECT_FALSE(dyn_cast<TFOp>(module_op));

  module->walk([&](TFOp op) {
    EXPECT_TRUE(isa<TFOp>(op.getOperation()));
    EXPECT_TRUE(dyn_cast<TFOp>(op.getOperation()));
  });
}

TEST(TFOpWrapper, ControlOperands) {
  const char *const code = R"mlir(
    tfg.func @test(%a: tensor<i32> {tfg.name = "a"},
                   %b: tensor<i32> {tfg.name = "b"}) -> (tensor<i32>) {
      %A, %ctlA = A(%a, %b) [%a.ctl, %b.ctl] : (tensor<i32>, tensor<i32>)
                                               -> (tensor<i32>)
      return(%A) : tensor<i32>
    }
  )mlir";
  MLIRContext context;
  context.getOrLoadDialect<TFGraphDialect>();
  OwningOpRef<ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);

  TFOp a_op;
  module->walk([&](TFOp op) {
    if (op->getName().getStringRef() == "tfg.A") a_op = op;
  });
  ASSERT_TRUE(a_op);

  EXPECT_EQ(a_op.controlRet().getDefiningOp(), a_op.getOperation());

  OperandRange operands = a_op->getOperands();
  OperandRange data = a_op.getNonControlOperands();
  OperandRange ctls = a_op.getControlOperands();
  EXPECT_EQ(operands.size(), 4u);
  EXPECT_EQ(data.size(), 2u);
  EXPECT_EQ(ctls.size(), 2u);

  OperandRange::iterator ctl_it = llvm::find_if(operands, [](Value operand) {
    return mlir::isa<ControlType>(operand.getType());
  });
  EXPECT_NE(ctl_it, operands.end());
  EXPECT_EQ(data.end(), ctl_it);
  // OperandRange::iterator has a base and an index (not MutableArrayRef) so
  // iterators from different ranges cannot reliably be compared.
  EXPECT_EQ(*ctls.begin(), *ctl_it);
}

TEST(TFOpWrapper, AttributeGetterSetters) {
  MLIRContext context;
  auto *tfg_dialect = context.getOrLoadDialect<TFGraphDialect>();
  OperationState state(UnknownLoc::get(&context), "tfg.A");
  state.addTypes(tfg_dialect->getControlType());
  TFOp op = Operation::create(state);
  auto cleanup = llvm::make_scope_exit([&] { op->destroy(); });

  // name
  {
    EXPECT_FALSE(op.nameAttr());
    StringRef a_name = "a_name";
    op.setName(a_name);
    EXPECT_EQ(op.name(), a_name);
    StringRef another_name = "another_name";
    op.setName(StringAttr::get(&context, another_name));
    EXPECT_EQ(op.name(), another_name);
  }

  // requested device
  {
    StringRef a_device = "/some_device";
    EXPECT_FALSE(op.requestedDeviceAttr());
    op.setRequestedDevice(a_device);
    EXPECT_EQ(op.requestedDevice(), a_device);
    StringRef another_device = "/some_other_device";
    op.setRequestedDevice(StringAttr::get(&context, another_device));
    EXPECT_EQ(op.requestedDevice(), another_device);
  }

  // assigned device
  {
    StringRef a_device = "/some_assigned_device";
    EXPECT_FALSE(op.assignedDeviceAttr());
    op.setAssignedDevice(a_device);
    EXPECT_EQ(op.assignedDevice(), a_device);
    StringRef another_device = "/some_other_assigned_device";
    op.setAssignedDevice(StringAttr::get(&context, another_device));
    EXPECT_EQ(op.assignedDevice(), another_device);
  }

  // device
  {
    op->removeAttr(tfg_dialect->getAssignedDeviceAttrIdentifier());
    EXPECT_EQ(op.deviceAttr(), op.requestedDeviceAttr());
    StringRef device = "/an_assigned_device";
    op.setAssignedDevice(device);
    EXPECT_EQ(op.deviceAttr(), op.assignedDeviceAttr());
    EXPECT_EQ(op.device(), device);
    op->removeAttr(tfg_dialect->getAssignedDeviceAttrIdentifier());
    op->removeAttr(tfg_dialect->getDeviceAttrIdentifier());
    EXPECT_EQ(op.device(), "");
  }

  // tfg.tpu_replicate
  {
    auto tpu_replicate = StringAttr::get(op->getContext(), "a_tpu");
    op.setTpuReplicate(tpu_replicate);
    EXPECT_EQ(op.tpuReplicate(), tpu_replicate);
  }
}

TEST(TFOpWrapper, ValueControlRet) {
  const char *const code = R"mlir(
    tfg.func @test(%arg: tensor<i32> {tfg.name = "arg"}) -> (tensor<i32>) {
      %Const, %ctl = Const {dtype = i32, value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
      %Add, %ctl_2 = Add(%Const, %arg) [%ctl] {T = i32} : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
      return(%Add) : tensor<i32>
    }
  )mlir";

  MLIRContext context;
  context.getOrLoadDialect<TFGraphDialect>();
  OwningOpRef<ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  GraphFuncOp func = module->lookupSymbol<GraphFuncOp>("test");
  ASSERT_TRUE(func);

  auto iterator = func.getBody().begin()->begin();
  TFOp const_op = &(*iterator++);
  TFOp add_op = &(*iterator);

  OperandControlRetRange ret_range(add_op->getOperands());

  EXPECT_EQ(ret_range[0], const_op.controlRet());
  // The control token of an argument is the argument next to itself.
  EXPECT_EQ(ret_range[1], func.getBody().begin()->getArguments()[1]);
  // Value with ControlType will be the same.
  EXPECT_EQ(ret_range[2], const_op.controlRet());

  for (Value v : ret_range) EXPECT_TRUE(mlir::isa<ControlType>(v.getType()));
}

}  // namespace
}  // namespace tfg
}  // namespace mlir
