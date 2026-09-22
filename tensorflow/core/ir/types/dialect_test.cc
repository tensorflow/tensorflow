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

#include <cstdint>
#include <limits>

#include <gmock/gmock.h>
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
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
      b.getStringAttr("foo"), baz, test_op.getParentRegion())));

  auto func_attr = mlir::dyn_cast<tf_type::FuncAttr>(test_op.getAttr("func"));
  ASSERT_TRUE(func_attr);
  auto sym_ref = FlatSymbolRefAttr::get(baz);
  EXPECT_TRUE(func_attr.getName() == sym_ref);
  auto bar_ref = func_attr.getAttrs().get("bar");
  EXPECT_TRUE(bar_ref == sym_ref);
}

TEST(TFTypesDialect, ParsesDimensionListWithZero) {
  // Test that a dimension list with zero can be parsed.
  const char *const code = R"mlir(
  "test.op"() {shape = #tf_type.shape<0x128>} : () -> ()
)mlir";

  MLIRContext context;
  context.allowUnregisteredDialects();
  context.getOrLoadDialect<tf_type::TFTypeDialect>();
  OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  Operation &test_op = module->front();

  auto shape_attr =
      mlir::dyn_cast<tf_type::ShapeAttr>(test_op.getAttr("shape"));
  ASSERT_TRUE(shape_attr);
  EXPECT_THAT(shape_attr.getShape(), testing::ElementsAre(0, 128));
}

TEST(TFTypesDialect, ParsesDimensionListWithQuestionMark) {
  // Test that a dimension list with zero can be parsed.
  const char *const code = R"mlir(
  "test.op"() {shape = #tf_type.shape<0x?x2>} : () -> ()
)mlir";

  MLIRContext context;
  context.allowUnregisteredDialects();
  context.getOrLoadDialect<tf_type::TFTypeDialect>();
  OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  Operation &test_op = module->front();

  auto shape_attr =
      mlir::dyn_cast<tf_type::ShapeAttr>(test_op.getAttr("shape"));
  ASSERT_TRUE(shape_attr);
  EXPECT_THAT(shape_attr.getShape(),
              testing::ElementsAre(0, std::numeric_limits<int64_t>::min(), 2));
}

TEST(TFTypesDialect, ParsesDimensionListWithNegativeOne) {
  // Test that a dimension list with zero can be parsed.
  const char *const code = R"mlir(
  "test.op"() {shape = #tf_type.shape<0x-1x2>} : () -> ()
)mlir";

  MLIRContext context;
  context.allowUnregisteredDialects();
  context.getOrLoadDialect<tf_type::TFTypeDialect>();
  OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  Operation &test_op = module->front();

  auto shape_attr =
      mlir::dyn_cast<tf_type::ShapeAttr>(test_op.getAttr("shape"));
  ASSERT_TRUE(shape_attr);
  EXPECT_THAT(shape_attr.getShape(), testing::ElementsAre(0, -1, 2));
}

TEST(TFTypesDialect, TestFloat8RemoveRef) {
  MLIRContext context;
  context.getOrLoadDialect<tf_type::TFTypeDialect>();

  EXPECT_EQ(mlir::cast<tf_type::TensorFlowRefType>(
                tf_type::Float8E4M3FNRefType::get(&context))
                .RemoveRef(),
            Float8E4M3FNType::get(&context));
  EXPECT_EQ(mlir::cast<tf_type::TensorFlowRefType>(
                tf_type::Float8E5M2RefType::get(&context))
                .RemoveRef(),
            Float8E5M2Type::get(&context));
  EXPECT_EQ(mlir::cast<tf_type::TensorFlowRefType>(
                tf_type::Float8E4M3FNUZRefType::get(&context))
                .RemoveRef(),
            Float8E4M3FNUZType::get(&context));
  EXPECT_EQ(mlir::cast<tf_type::TensorFlowRefType>(
                tf_type::Float8E4M3B11FNUZRefType::get(&context))
                .RemoveRef(),
            Float8E4M3B11FNUZType::get(&context));
  EXPECT_EQ(mlir::cast<tf_type::TensorFlowRefType>(
                tf_type::Float8E5M2FNUZRefType::get(&context))
                .RemoveRef(),
            Float8E5M2FNUZType::get(&context));
  EXPECT_EQ(mlir::cast<tf_type::TensorFlowRefType>(
                tf_type::Float8E8M0FNURefType::get(&context))
                .RemoveRef(),
            Float8E8M0FNUType::get(&context));

  EXPECT_EQ(tf_type::DropRefType(tf_type::Float8E8M0FNURefType::get(&context)),
            Float8E8M0FNUType::get(&context));

  EXPECT_EQ(tf_type::TensorFlowRefType::get(Float8E8M0FNUType::get(&context)),
            tf_type::Float8E8M0FNURefType::get(&context));
}

}  // namespace
}  // namespace tfg
}  // namespace mlir
