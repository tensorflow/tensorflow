/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/spmd/shardy/utils.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "llvm/ADT/BitVector.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace xla {
namespace sdy {
namespace {

using ::absl_testing::IsOkAndHolds;

TEST(UtilsTest, DuplicateShardingsAtIndices) {
  std::string inputShardings =
      "#sdy.sharding_per_value<["
      "<@mesh, [{\"x\"}, {}]>, <@mesh, [{}, {\"y\"}]>, "
      "<@mesh, [{}, {}]>, <@mesh, [{\"x\"}, {\"y\"}]>]>";
  llvm::BitVector indicesToDuplicate(4);
  EXPECT_THAT(duplicateShardingsAtIndices(inputShardings, indicesToDuplicate),
              IsOkAndHolds(inputShardings));

  indicesToDuplicate.set(0);
  EXPECT_THAT(duplicateShardingsAtIndices(inputShardings, indicesToDuplicate),
              IsOkAndHolds("#sdy.sharding_per_value<["
                           "<@mesh, [{\"x\"}, {}]>, <@mesh, [{\"x\"}, {}]>, "
                           "<@mesh, [{}, {\"y\"}]>, <@mesh, [{}, {}]>, "
                           "<@mesh, [{\"x\"}, {\"y\"}]>]>"));

  indicesToDuplicate.reset();
  indicesToDuplicate.set(1);
  indicesToDuplicate.set(3);
  EXPECT_THAT(
      duplicateShardingsAtIndices(inputShardings, indicesToDuplicate),
      IsOkAndHolds(
          "#sdy.sharding_per_value<["
          "<@mesh, [{\"x\"}, {}]>, <@mesh, [{}, {\"y\"}]>, "
          "<@mesh, [{}, {\"y\"}]>, <@mesh, [{}, {}]>, "
          "<@mesh, [{\"x\"}, {\"y\"}]>, <@mesh, [{\"x\"}, {\"y\"}]>]>"));

  indicesToDuplicate.reset();
  indicesToDuplicate.set(1);
  indicesToDuplicate.set(2);
  EXPECT_THAT(duplicateShardingsAtIndices(inputShardings, indicesToDuplicate),
              IsOkAndHolds("#sdy.sharding_per_value<["
                           "<@mesh, [{\"x\"}, {}]>, <@mesh, [{}, {\"y\"}]>, "
                           "<@mesh, [{}, {\"y\"}]>, <@mesh, [{}, {}]>, "
                           "<@mesh, [{}, {}]>, <@mesh, [{\"x\"}, {\"y\"}]>]>"));
}

TEST(UtilsTest, SetAndRemoveFrontendAttributeOnOp) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::sdy::SdyDialect>();
  mlir::OpBuilder builder(&context);
  mlir::OperationState state(builder.getUnknownLoc(), "sdy.constant");
  state.addAttribute("value", builder.getI32IntegerAttr(42));
  mlir::Operation* op = mlir::Operation::create(state);

  EXPECT_FALSE(hasFrontendAttr(op, "foo"));
  setFrontendAttribute(op, "foo", builder.getStringAttr("bar"));
  EXPECT_TRUE(hasFrontendAttr(op, "foo"));
  EXPECT_EQ(getFrontendAttrs(op).getAs<mlir::StringAttr>("foo").getValue(),
            "bar");

  // Overwrite existing attribute.
  setFrontendAttribute(op, "foo", builder.getStringAttr("baz"));
  EXPECT_EQ(getFrontendAttrs(op).getAs<mlir::StringAttr>("foo").getValue(),
            "baz");

  // Add another attribute.
  setFrontendAttribute(op, "key2", builder.getStringAttr("val2"));
  EXPECT_TRUE(hasFrontendAttr(op, "key2"));

  // Remove one attribute.
  removeFrontendAttribute(op, "foo");
  EXPECT_FALSE(hasFrontendAttr(op, "foo"));
  EXPECT_TRUE(hasFrontendAttr(op, "key2"));

  // Remove remaining attribute; dictionary should be removed.
  removeFrontendAttribute(op, "key2");
  EXPECT_FALSE(hasFrontendAttr(op, "key2"));
  EXPECT_EQ(getFrontendAttrs(op), nullptr);

  // Removing a non-existent attribute is a no-op.
  removeFrontendAttribute(op, "non_existent");
  EXPECT_EQ(getFrontendAttrs(op), nullptr);

  op->destroy();
}

TEST(UtilsTest, SetAndRemoveFrontendAttributeOnFuncArg) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect>();
  mlir::OpBuilder builder(&context);
  mlir::FunctionType funcType =
      builder.getFunctionType({builder.getI32Type()}, {builder.getI32Type()});
  mlir::func::FuncOp funcOp = mlir::func::FuncOp::create(
      builder.getUnknownLoc(), "test_func", funcType);

  EXPECT_EQ(getFuncArgFrontendAttrs(funcOp, 0), nullptr);
  setFrontendAttribute(funcOp, "arg_key", builder.getStringAttr("arg_val"), 0);
  EXPECT_NE(getFuncArgFrontendAttrs(funcOp, 0), nullptr);
  EXPECT_EQ(getFuncArgFrontendAttrs(funcOp, 0)
                .getAs<mlir::StringAttr>("arg_key")
                .getValue(),
            "arg_val");

  removeFrontendAttribute(funcOp, "arg_key", 0);
  EXPECT_EQ(getFuncArgFrontendAttrs(funcOp, 0), nullptr);

  funcOp->destroy();
}

}  // namespace
}  // namespace sdy
}  // namespace xla
