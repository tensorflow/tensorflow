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
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace tf_saved_model {
namespace {

using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::SizeIs;

// Fixture for testing TfSavedModel functionalities. Initializes a
// `MLIRContext` by loading the `tf_saved_model` dialect.
class TfSavedModelTest : public ::testing::Test {
 protected:
  TfSavedModelTest() : ctx_() {
    ctx_.loadDialect<TensorFlowSavedModelDialect, func::FuncDialect>();
  }

  MLIRContext ctx_;
};

// Parses `module_op_str` and returns the resulting `ModuleOp`.
ModuleOp ParseModuleOp(const StringRef module_op_str, Block& block,
                       MLIRContext& ctx) {
  const LogicalResult parse_result =
      parseSourceString(module_op_str, &block, ParserConfig(&ctx));
  EXPECT_TRUE(succeeded(parse_result));

  return cast<ModuleOp>(block.front());
}

TEST_F(TfSavedModelTest,
       GetInitializerFunctionReturnsNullWhenNoSessionInitializerOp) {
  constexpr StringRef kModuleOpStr =
      R"mlir(module attributes {tf_saved_model.semantics} {})mlir";

  Block block;
  ModuleOp module_op = ParseModuleOp(kModuleOpStr, block, ctx_);

  func::FuncOp init_func_op = GetInitializerFunction(
      module_op, /*initializer_type=*/kTfSavedModelInitializerInitType);

  EXPECT_THAT(init_func_op, IsNull());
}

TEST_F(TfSavedModelTest,
       GetInitializerFunctionReturnsNullWhenInitializersEmpty) {
  constexpr StringRef kModuleOpStr = R"mlir(
    module attributes {tf_saved_model.semantics} {
      "tf_saved_model.session_initializer"() {initializers = []} : () -> ()
    }
  )mlir";

  Block block;
  ModuleOp module_op = ParseModuleOp(kModuleOpStr, block, ctx_);

  func::FuncOp init_func_op = GetInitializerFunction(
      module_op, /*initializer_type=*/kTfSavedModelInitializerInitType);

  EXPECT_THAT(init_func_op, IsNull());
}

TEST_F(TfSavedModelTest,
       GetInitializerFunctionReturnsFuncOpMatchingInitializerType) {
  constexpr StringRef kModuleOpStr = R"mlir(
    module attributes {tf_saved_model.semantics} {
      "tf_saved_model.session_initializer"() {initializers = [@init_func]} : () -> ()
      func.func @init_func() attributes {tf_saved_model.exported_names = ["init_func"], tf_saved_model.initializer_type = "init_op"} {
        func.return
      }
    }
  )mlir";

  Block block;
  ModuleOp module_op = ParseModuleOp(kModuleOpStr, block, ctx_);

  func::FuncOp init_func_op = GetInitializerFunction(
      module_op, /*initializer_type=*/kTfSavedModelInitializerInitType);

  EXPECT_THAT(init_func_op, NotNull());
  EXPECT_THAT(init_func_op.getSymName(), "init_func");
  EXPECT_THAT(
      init_func_op->getAttrOfType<StringAttr>(kTfSavedModelInitializerTypeAttr),
      kTfSavedModelInitializerInitType);
}

TEST_F(TfSavedModelTest, GetInitializerFunctionNoMatchingInitializerType) {
  constexpr StringRef kModuleOpStr = R"mlir(
    module attributes {tf_saved_model.semantics} {
      "tf_saved_model.session_initializer"() {initializers = [@init_func]} : () -> ()
      func.func @init_func() attributes {tf_saved_model.exported_names = ["init_func"], tf_saved_model.initializer_type = "restore_op"} {
        func.return
      }
    }
  )mlir";

  Block block;
  ModuleOp module_op = ParseModuleOp(kModuleOpStr, block, ctx_);

  func::FuncOp init_func_op = GetInitializerFunction(
      module_op, /*initializer_type=*/kTfSavedModelInitializerInitType);

  // No initializer function matches the initializer type.
  EXPECT_THAT(init_func_op, IsNull());
}

TEST_F(TfSavedModelTest, GetInitializerFunctionsEmptyWhenNoInitFunctions) {
  constexpr StringRef kModuleOpStr = R"mlir(
    module attributes {tf_saved_model.semantics} {
      "tf_saved_model.session_initializer"() {initializers = []} : () -> ()
    }
  )mlir";

  Block block;
  ModuleOp module_op = ParseModuleOp(kModuleOpStr, block, ctx_);

  SmallVector<func::FuncOp, 2> init_func_ops =
      GetInitializerFunctions(module_op);

  EXPECT_THAT(init_func_ops, IsEmpty());
}

TEST_F(TfSavedModelTest,
       GetInitializerFunctionsEmptyWhenNoSessionInitializerOp) {
  constexpr StringRef kModuleOpStr =
      R"mlir(module attributes {tf_saved_model.semantics} {})mlir";

  Block block;
  ModuleOp module_op = ParseModuleOp(kModuleOpStr, block, ctx_);

  SmallVector<func::FuncOp, 2> init_func_ops =
      GetInitializerFunctions(module_op);

  EXPECT_THAT(init_func_ops, IsEmpty());
}

TEST_F(TfSavedModelTest, GetInitializerFunctionsReturnsMultipleFuncOps) {
  constexpr StringRef kModuleOpStr = R"mlir(
    module attributes {tf_saved_model.semantics} {
      "tf_saved_model.session_initializer"() {initializers = [@init_func1, @init_func2]} : () -> ()

      func.func @init_func1() attributes {tf_saved_model.exported_names = ["init_func1"], tf_saved_model.initializer_type = "init_op"} {
        func.return
      }

      func.func @init_func2() attributes {tf_saved_model.exported_names = ["init_func2"], tf_saved_model.initializer_type = "restore_op"} {
        func.return
      }
    }
  )mlir";

  Block block;
  ModuleOp module_op = ParseModuleOp(kModuleOpStr, block, ctx_);

  SmallVector<func::FuncOp, 2> init_func_ops =
      GetInitializerFunctions(module_op);

  EXPECT_THAT(init_func_ops, SizeIs(2));
  EXPECT_THAT(init_func_ops[0].getSymName(), Eq("init_func1"));
  EXPECT_THAT(init_func_ops[1].getSymName(), Eq("init_func2"));
}

}  // namespace
}  // namespace tf_saved_model
}  // namespace mlir
