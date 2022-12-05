/*
 * Copyright 2022 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow/compiler/xla/mlir/runtime/transforms/calling_convention.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/runtime/ir/rt_ops.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace runtime {

using mlir::FunctionType;
using mlir::IntegerType;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::TypeRange;

namespace memref = mlir::memref;

TEST(CallingConventionTest, DefaultCallingConvention) {
  MLIRContext ctx;
  ctx.loadDialect<RuntimeDialect, memref::MemRefDialect>();

  CallingConvention calling_convention = DefaultCallingConvention();

  auto i32 = IntegerType::get(&ctx, 32);

  auto signature = FunctionType::get(&ctx, TypeRange(i32), TypeRange());
  auto converted = calling_convention(signature);

  EXPECT_EQ(converted.getNumInputs(), 2);
  EXPECT_TRUE(converted.getInput(0).isa<ExecutionContextType>());
  EXPECT_TRUE(converted.getInput(1).isa<IntegerType>());
}

TEST(CallingConventionTest, DefaultCallingConventionWithTypeConverter) {
  MLIRContext ctx;
  ctx.loadDialect<RuntimeDialect, memref::MemRefDialect>();

  mlir::TypeConverter type_converter;
  type_converter.addConversion(
      [](mlir::Type type) { return MemRefType::get({}, type); });

  CallingConvention calling_convention =
      DefaultCallingConvention(type_converter);

  auto i32 = IntegerType::get(&ctx, 32);

  auto signature = FunctionType::get(&ctx, TypeRange(i32), TypeRange());
  auto converted = calling_convention(signature);

  EXPECT_EQ(converted.getNumInputs(), 2);
  EXPECT_TRUE(converted.getInput(0).isa<ExecutionContextType>());
  EXPECT_TRUE(converted.getInput(1).isa<MemRefType>());
}

TEST(CallingConventionTest, ResultsToOutsCallingConvention) {
  MLIRContext ctx;
  ctx.loadDialect<RuntimeDialect, memref::MemRefDialect>();

  mlir::TypeConverter type_converter;
  type_converter.addConversion(
      [](mlir::Type type) { return MemRefType::get({}, type); });

  CallingConvention calling_convention =
      ResultsToOutsCallingConvention(type_converter);

  auto i32 = IntegerType::get(&ctx, 32);

  auto signature = FunctionType::get(&ctx, TypeRange(), TypeRange(i32));
  auto converted = calling_convention(signature);

  EXPECT_EQ(converted.getNumInputs(), 2);
  EXPECT_TRUE(converted.getInput(0).isa<ExecutionContextType>());
  EXPECT_TRUE(converted.getInput(1).isa<MemRefType>());

  EXPECT_EQ(converted.getNumResults(), 0);
}

}  // namespace runtime
}  // namespace xla
