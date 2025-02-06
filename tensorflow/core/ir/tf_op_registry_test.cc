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

#include "tensorflow/core/ir/tf_op_registry.h"

#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/interfaces.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace tfg {
namespace {

// Register the TFG dialect and register the op registry interface.
void PrepareContext(MLIRContext *context) {
  DialectRegistry registry;
  registry.insert<TFGraphDialect>();
  registry.addExtension(+[](mlir::MLIRContext *ctx, TFGraphDialect *dialect) {
    dialect->addInterfaces<TensorFlowOpRegistryInterface>();
  });
  context->appendDialectRegistry(registry);
}

TEST(TensorFlowOpRegistryInterface, TestIntrinsicOps) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  PrepareContext(&context);

  const char *const code = R"mlir(
    tfg.func @test(%arg: tensor<i32>) -> (tensor<i32>) {
      return(%arg) : tensor<i32>
    }
  )mlir";
  OwningOpRef<ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);

  auto func_op = cast<GraphFuncOp>(&module->front());
  auto ret_op = cast<ReturnOp>(func_op.getBody().front().getTerminator());
  EXPECT_FALSE(dyn_cast<TensorFlowRegistryInterface>(*func_op));
  EXPECT_FALSE(dyn_cast<TensorFlowRegistryInterface>(*ret_op));
}

TEST(TensorFlowOpRegistryInterface, TestStatelessTFOps) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  PrepareContext(&context);

  const char *const code = R"mlir(
    tfg.func @test(%lhs: tensor<i32>, %rhs: tensor<i32>) -> (tensor<i32>) {
      %Add, %ctl = Add(%lhs, %rhs) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
      return(%Add) : tensor<i32>
    }
  )mlir";
  OwningOpRef<ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);

  Operation *add =
      &cast<GraphFuncOp>(&module->front()).getBody().front().front();
  auto iface = dyn_cast<TensorFlowRegistryInterface>(add);
  ASSERT_TRUE(iface);
  EXPECT_FALSE(iface.isStateful());
}

TEST(TensorFlowOpRegistryInterface, TestStatelessAndStatefulRegionOps) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  PrepareContext(&context);

  const char *const code_template = R"mlir(
    tfg.func @test(%idx: tensor<i32>, %arg: tensor<i32>) -> (tensor<i32>) {{
      %Case, %ctl = {0}CaseRegion %idx {{
        yield(%arg) : tensor<i32>
      } : (tensor<i32>) -> (tensor<i32>)
      return(%Case) : tensor<i32>
    }
  )mlir";
  SmallVector<StringRef, 2> prefixes = {"", "Stateless"};
  SmallVector<bool, 2> expected = {true, false};
  for (auto it : llvm::zip(prefixes, expected)) {
    std::string code = llvm::formatv(code_template, std::get<0>(it)).str();
    OwningOpRef<ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(code, &context);
    ASSERT_TRUE(module);

    Operation *case_op =
        &cast<GraphFuncOp>(&module->front()).getBody().front().front();
    auto iface = dyn_cast<TensorFlowRegistryInterface>(case_op);
    ASSERT_TRUE(iface);
    EXPECT_EQ(iface.isStateful(), std::get<1>(it));
  }
}
}  // namespace
}  // namespace tfg
}  // namespace mlir
