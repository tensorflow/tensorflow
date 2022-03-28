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

#include "tensorflow/core/ir/types/dialect.h"

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace tfg {
namespace {

TEST(TFTypesDialect, TestFuncAttrSubElement) {
  // Test that symbol references nested inside FuncAttr can be found and
  // replaced.
  const char *const code = R"mlir(
  "test.op"() {func = #tf_type.func<@foo, {bar = @foo}>} : () -> ()
)mlir";

  MLIRContext context;
  context.allowUnregisteredDialects();
  context.getOrLoadDialect<tf_type::TFTypeDialect>();
  OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  Operation &test_op = module->front();

  Builder b(&context);
  StringAttr baz = b.getStringAttr("baz");
  ASSERT_TRUE(succeeded(SymbolTable::replaceAllSymbolUses(
      b.getStringAttr("foo"), baz, &test_op)));

  auto func_attr = test_op.getAttr("func").dyn_cast<tf_type::FuncAttr>();
  ASSERT_TRUE(func_attr);
  auto sym_ref = FlatSymbolRefAttr::get(baz);
  EXPECT_TRUE(func_attr.getName() == sym_ref);
  EXPECT_TRUE(func_attr.getAttrs().get("bar") == sym_ref);
}

}  // namespace
}  // namespace tfg
}  // namespace mlir
