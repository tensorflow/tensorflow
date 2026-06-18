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

#include "tensorflow/compiler/mlir/tensorflow/utils/call_graph_util.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(CallGraphUtilTest, GetEntryFunctionAttributeNames) {
  auto attr_names = mlir::GetEntryFunctionAttributeNames();
  EXPECT_EQ(attr_names.size(), 2);
  EXPECT_EQ(attr_names[0], "tf.entry_function");
  EXPECT_EQ(attr_names[1],
            mlir::tf_saved_model::kTfSavedModelInitializerTypeAttr);
}

TEST(CallGraphUtilTest, GetEntryFunctions) {
  const char *const code = R"mlir(
func.func @entry_func_1(%arg0: tensor<i32>) -> tensor<i32> attributes {tf.entry_function = {}} {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {_xla_compile_device_type = "CPU", config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @entry_func_2(%arg0: tensor<i32>) -> tensor<i32> attributes {tf_saved_model.initializer_type = ""} {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {_xla_compile_device_type = "CPU", config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}
)mlir";
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::TF::TensorFlowDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  auto entry_funcs = GetEntryFunctions(*module);
  EXPECT_EQ(entry_funcs.size(), 2);
  EXPECT_EQ(entry_funcs[0].getSymName(), "entry_func_1");
  EXPECT_EQ(entry_funcs[1].getSymName(), "entry_func_2");
}

TEST(CallGraphUtilTest, GetCallees) {
  const char *const code = R"mlir(
func.func @entry_func(%arg0: tensor<i32>) -> tensor<i32> attributes {tf_saved_model.initializer_type = ""} {
  %0 = "tf.While"(%arg0) {cond = @while_cond_func, body = @while_body_func, is_stateless = true} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @while_cond_func(%arg0: tensor<i32>) -> tensor<i1> {
  %0 = "tf.Const"() {value = dense<0> : tensor<i1>} : () -> tensor<i1>
  func.return %0 : tensor<i1>
}

func.func @while_body_func(%arg0: tensor<i32>) -> (tensor<i32>) {
  %0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  func.return %0 : tensor<i32>
}


)mlir";
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::TF::TensorFlowDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  mlir::SymbolTable symtab(*module);
  llvm::SmallVector<mlir::func::FuncOp> callees;
  module->walk([&](mlir::SymbolUserOpInterface op) {
    auto result = GetCallees(op, symtab, callees).succeeded();
    ASSERT_TRUE(result);
    EXPECT_EQ(callees.size(), 2);
    EXPECT_EQ(callees[0].getSymName(), "while_body_func");
    EXPECT_EQ(callees[1].getSymName(), "while_cond_func");
  });
}

TEST(CallGraphUtilTest, GetFirstOpsOfType) {
  const char *const code = R"mlir(
func.func @entry_func(%arg0: tensor<i32>) -> tensor<i32> attributes {tf.entry_function = {}} {
  %0 = "tf.While"(%arg0) {cond = @while_cond_func, body = @while_body_func, is_stateless = true} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @while_cond_func(%arg0: tensor<i32>) -> tensor<i1> {
  %0 = "tf.Const"() {value = dense<0> : tensor<i1>} : () -> tensor<i1>
  func.return %0 : tensor<i1>
}

// CHECK-LABEL: func.func @while_body_func
func.func @while_body_func(%arg0: tensor<i32>) -> (tensor<i32>) {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @outer_stateful_pcall_func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @outer_stateful_pcall_func(%arg0: tensor<i32>) -> (tensor<i32>) {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {_xla_compile_device_type = "CPU", config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @inner_stateful_pcall_func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @inner_stateful_pcall_func(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {_xla_compile_device_type = "CPU", config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}
)mlir";
  auto has_compile_device_type = [](mlir::SymbolUserOpInterface op) {
    return op->hasAttr(tensorflow::kCompileDeviceTypeAttr);
  };
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::TF::TensorFlowDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  mlir::SymbolTable symtab(*module);
  llvm::SmallVector<mlir::func::FuncOp> entry_funcs =
      GetEntryFunctions(*module);
  EXPECT_EQ(entry_funcs.size(), 1);
  EXPECT_EQ(entry_funcs[0].getSymName(), "entry_func");
  llvm::SmallVector<mlir::SymbolUserOpInterface> outermost_pcall_ops;
  auto result =
      mlir::GetFirstOpsOfType<mlir::TF::StatefulPartitionedCallOp,
                              mlir::TF::PartitionedCallOp>(
          entry_funcs[0], symtab, has_compile_device_type, outermost_pcall_ops)
          .succeeded();
  ASSERT_TRUE(result);
  EXPECT_EQ(outermost_pcall_ops.size(), 1);
  auto func =
      llvm::dyn_cast<mlir::func::FuncOp>(outermost_pcall_ops[0]->getParentOp());
  ASSERT_TRUE(func);
  EXPECT_EQ(func.getSymName(), "outer_stateful_pcall_func");
}

TEST(CallGraphUtilTest, GetOpsOfTypeUntilMiss) {
  const char *const code = R"mlir(
func.func @entry_func(%arg0: tensor<i32>) -> tensor<i32> attributes {tf.entry_function = {}} {
  %0 = "tf.While"(%arg0) {cond = @while_cond_func, body = @while_body_func, is_stateless = true} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @while_cond_func(%arg0: tensor<i32>) -> tensor<i1> {
  %0 = "tf.Const"() {value = dense<0> : tensor<i1>} : () -> tensor<i1>
  func.return %0 : tensor<i1>
}

// CHECK-LABEL: func.func @while_body_func
func.func @while_body_func(%arg0: tensor<i32>) -> (tensor<i32>) {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @outer_stateful_pcall_func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @outer_stateful_pcall_func(%arg0: tensor<i32>) -> (tensor<i32>) {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @inner_stateful_pcall_func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @inner_stateful_pcall_func(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {_xla_compile_device_type = "CPU", config = "", config_proto = "", device = "/device:CPU:0", executor_type = "", f = @func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}
)mlir";
  auto has_no_compile_device_type = [](mlir::SymbolUserOpInterface op) {
    return !op->hasAttr(tensorflow::kCompileDeviceTypeAttr);
  };
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::TF::TensorFlowDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  mlir::SymbolTable symtab(*module);
  llvm::SmallVector<mlir::func::FuncOp> entry_funcs =
      GetEntryFunctions(*module);
  EXPECT_EQ(entry_funcs.size(), 1);
  EXPECT_EQ(entry_funcs[0].getSymName(), "entry_func");
  llvm::SmallVector<mlir::SymbolUserOpInterface> noinline_pcall_ops,
      outermost_pcall_ops;
  auto result =
      mlir::GetOpsOfTypeUntilMiss<mlir::TF::StatefulPartitionedCallOp,
                                  mlir::TF::PartitionedCallOp>(
          entry_funcs[0], symtab, has_no_compile_device_type,
          /*hits*/ noinline_pcall_ops, /*first_misses*/ outermost_pcall_ops)
          .succeeded();
  ASSERT_TRUE(result);
  EXPECT_EQ(noinline_pcall_ops.size(), 2);
  auto func =
      llvm::dyn_cast<mlir::func::FuncOp>(noinline_pcall_ops[0]->getParentOp());
  ASSERT_TRUE(func);
  EXPECT_EQ(func.getSymName(), "while_body_func");
  func =
      llvm::dyn_cast<mlir::func::FuncOp>(noinline_pcall_ops[1]->getParentOp());
  ASSERT_TRUE(func);
  EXPECT_EQ(func.getSymName(), "outer_stateful_pcall_func");

  EXPECT_EQ(outermost_pcall_ops.size(), 1);
  func =
      llvm::dyn_cast<mlir::func::FuncOp>(outermost_pcall_ops[0]->getParentOp());
  ASSERT_TRUE(func);
  EXPECT_EQ(func.getSymName(), "inner_stateful_pcall_func");
}

TEST(CallGraphUtilTest, SingleBlockEntryFunction) {
  const char *const code = R"mlir(
func.func @entry_func(%arg0: tensor<i32>) -> tensor<i32> attributes {tf.entry_function = {}} {
  func.return %arg0 : tensor<i32>
}
)mlir";

  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::TF::TensorFlowDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  llvm::errs() << "module:\n";
  ASSERT_TRUE(module);
  mlir::SymbolTable symtab(*module);
  llvm::SmallVector<mlir::func::FuncOp> entry_funcs =
      GetEntryFunctions(*module);
  EXPECT_EQ(entry_funcs.size(), 1);
  EXPECT_EQ(entry_funcs[0].getSymName(), "entry_func");
  EXPECT_TRUE(HasSingleBlock(entry_funcs[0]));
}

TEST(CallGraphUtilTest, MultipleBlocksEntryFunction) {
  const char *const code = R"mlir(
func.func @entry_func(%arg0: tensor<i32>) -> tensor<i32> attributes {tf.entry_function = {}} {
  cf.br ^bb1
^bb1:
  func.return %arg0 : tensor<i32>
}
)mlir";

  mlir::MLIRContext context;
  context.loadDialect<mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                      mlir::TF::TensorFlowDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  llvm::errs() << "module:\n";
  ASSERT_TRUE(module);
  mlir::SymbolTable symtab(*module);
  llvm::SmallVector<mlir::func::FuncOp> entry_funcs =
      GetEntryFunctions(*module);
  EXPECT_EQ(entry_funcs.size(), 1);
  EXPECT_EQ(entry_funcs[0].getSymName(), "entry_func");
  EXPECT_FALSE(HasSingleBlock(entry_funcs[0]));
}

}  // namespace
}  // namespace tensorflow
