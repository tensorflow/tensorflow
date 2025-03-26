/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/utils.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {
namespace {

TEST(UtilsTest, GetScalarConstOfType) {
  MLIRContext context;
  context.loadDialect<mlir::mhlo::MhloDialect>();
  OpBuilder builder(&context);
  Location loc = UnknownLoc::get(&context);
  Type ty = builder.getI32Type();
  mhlo::ConstantOp op = GetScalarConstOfType(ty, loc, 123, &builder);
  EXPECT_EQ(op.getValue().getValues<int32_t>()[0], 123);

  op->destroy();
}

TEST(UtilsTest, GetScalarNegZeroOfType) {
  MLIRContext context;
  context.loadDialect<mlir::mhlo::MhloDialect>();
  OpBuilder builder(&context);
  Location loc = UnknownLoc::get(&context);
  Type ty = builder.getF32Type();
  mhlo::ConstantOp op = GetScalarNegZeroOfType(ty, loc, &builder);
  EXPECT_EQ(op.getValue().getValues<float>()[0], -0.f);

  op->destroy();
}

TEST(UtilsTest, GetI64ElementsAttr) {
  MLIRContext context;
  context.loadDialect<mlir::mhlo::MhloDialect>();
  OpBuilder builder(&context);
  Location loc = UnknownLoc::get(&context);
  SmallVector<int64_t> values = {1, 2, 3};
  auto valuesAttr = builder.getI64ArrayAttr(values);
  DenseIntElementsAttr attr = GetI64ElementsAttr(valuesAttr);
  EXPECT_THAT(SmallVector<int64_t>(attr.getValues<int64_t>()),
              testing::ElementsAreArray(values));
}

TEST(UtilsTest, GetI64ElementsAttrBuilder) {
  MLIRContext context;
  context.loadDialect<mlir::mhlo::MhloDialect>();
  OpBuilder builder(&context);
  Location loc = UnknownLoc::get(&context);
  SmallVector<int64_t> values = {1, 2, 3};
  DenseIntElementsAttr attr = GetI64ElementsAttr(values, &builder);
  EXPECT_THAT(SmallVector<int64_t>(attr.getValues<int64_t>()),
              testing::ElementsAreArray(values));
}

}  // namespace

}  // namespace odml
}  // namespace mlir
