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
#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/bfloat16_type.h"

#include <memory>

#include <gtest/gtest.h>
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/register_common_dialects.h"

namespace mlir::quant::stablehlo {
namespace {

std::unique_ptr<MLIRContext> CreateContext() {
  auto context = std::make_unique<MLIRContext>();
  DialectRegistry mlir_registry;
  RegisterCommonToolingDialects(mlir_registry);
  context->appendDialectRegistry(mlir_registry);
  return context;
}

TEST(IsLargeFloatTypeTest, scalars) {
  auto context = CreateContext();

  EXPECT_FALSE(IsLargeFloatType(Float8E4M3FNType::get(context.get())));
  EXPECT_FALSE(IsLargeFloatType(Float8E4M3FNUZType::get(context.get())));
  EXPECT_FALSE(IsLargeFloatType(Float8E4M3B11FNUZType::get(context.get())));
  EXPECT_FALSE(IsLargeFloatType(Float8E5M2FNUZType::get(context.get())));
  EXPECT_FALSE(IsLargeFloatType(Float8E5M2Type::get(context.get())));
  EXPECT_FALSE(IsLargeFloatType(Float16Type::get(context.get())));
  EXPECT_FALSE(IsLargeFloatType(BFloat16Type::get(context.get())));
  EXPECT_TRUE(IsLargeFloatType(Float32Type::get(context.get())));
  EXPECT_TRUE(IsLargeFloatType(Float64Type::get(context.get())));
  EXPECT_TRUE(IsLargeFloatType(Float80Type::get(context.get())));

  EXPECT_FALSE(IsLargeFloatType(IntegerType::get(context.get(), 8)));
  EXPECT_FALSE(IsLargeFloatType(IntegerType::get(context.get(), 16)));
  EXPECT_FALSE(IsLargeFloatType(IntegerType::get(context.get(), 32)));
}

TEST(IsLargeFloatTypeTest, tensors) {
  auto context = CreateContext();

  EXPECT_FALSE(IsLargeFloatType(
      RankedTensorType::get({2, 2}, Float8E4M3FNType::get(context.get()))));
  EXPECT_FALSE(IsLargeFloatType(
      RankedTensorType::get({2, 2}, Float16Type::get(context.get()))));
  EXPECT_FALSE(IsLargeFloatType(
      RankedTensorType::get({2, 2}, Float8E4M3FNUZType::get(context.get()))));
  EXPECT_FALSE(IsLargeFloatType(RankedTensorType::get(
      {2, 2}, Float8E4M3B11FNUZType::get(context.get()))));
  EXPECT_FALSE(IsLargeFloatType(
      RankedTensorType::get({2, 2}, Float8E5M2FNUZType::get(context.get()))));
  EXPECT_FALSE(IsLargeFloatType(
      RankedTensorType::get({2, 2}, Float8E5M2Type::get(context.get()))));
  EXPECT_FALSE(IsLargeFloatType(
      RankedTensorType::get({2, 2}, BFloat16Type::get(context.get()))));
  EXPECT_TRUE(IsLargeFloatType(
      RankedTensorType::get({2, 2}, Float32Type::get(context.get()))));
  EXPECT_TRUE(IsLargeFloatType(
      RankedTensorType::get({2, 2}, Float64Type::get(context.get()))));
  EXPECT_TRUE(IsLargeFloatType(
      RankedTensorType::get({2, 2}, Float80Type::get(context.get()))));

  EXPECT_FALSE(IsLargeFloatType(
      RankedTensorType::get({2, 2}, IntegerType::get(context.get(), 8))));
  EXPECT_FALSE(IsLargeFloatType(
      RankedTensorType::get({2, 2}, IntegerType::get(context.get(), 16))));
  EXPECT_FALSE(IsLargeFloatType(
      RankedTensorType::get({2, 2}, IntegerType::get(context.get(), 32))));
}

TEST(ToBfloat16TypeTest, scalars) {
  auto context = CreateContext();

  EXPECT_EQ(ToBfloat16Type(Float8E4M3FNType::get(context.get())),
            Float8E4M3FNType::get(context.get()));
  EXPECT_EQ(ToBfloat16Type(Float8E4M3FNUZType::get(context.get())),
            Float8E4M3FNUZType::get(context.get()));
  EXPECT_EQ(ToBfloat16Type(Float8E4M3B11FNUZType::get(context.get())),
            Float8E4M3B11FNUZType::get(context.get()));
  EXPECT_EQ(ToBfloat16Type(Float8E5M2FNUZType::get(context.get())),
            Float8E5M2FNUZType::get(context.get()));
  EXPECT_EQ(ToBfloat16Type(Float8E5M2Type::get(context.get())),
            Float8E5M2Type::get(context.get()));
  EXPECT_EQ(ToBfloat16Type(Float16Type::get(context.get())),
            Float16Type::get(context.get()));
  EXPECT_EQ(ToBfloat16Type(BFloat16Type::get(context.get())),
            BFloat16Type::get(context.get()));
  EXPECT_EQ(ToBfloat16Type(Float32Type::get(context.get())),
            BFloat16Type::get(context.get()));
  EXPECT_EQ(ToBfloat16Type(Float64Type::get(context.get())),
            BFloat16Type::get(context.get()));
  EXPECT_EQ(ToBfloat16Type(Float80Type::get(context.get())),
            BFloat16Type::get(context.get()));

  EXPECT_EQ(ToBfloat16Type(IntegerType::get(context.get(), 8)),
            IntegerType::get(context.get(), 8));
  EXPECT_EQ(ToBfloat16Type(IntegerType::get(context.get(), 16)),
            IntegerType::get(context.get(), 16));
  EXPECT_EQ(ToBfloat16Type(IntegerType::get(context.get(), 32)),
            IntegerType::get(context.get(), 32));
}

TEST(ToBfloat16TypeTest, tensors) {
  auto context = CreateContext();

  EXPECT_EQ(
      ToBfloat16Type(
          RankedTensorType::get({2, 2}, Float8E4M3FNType::get(context.get()))),
      RankedTensorType::get({2, 2}, Float8E4M3FNType::get(context.get())));
  EXPECT_EQ(
      ToBfloat16Type(RankedTensorType::get(
          {2, 2}, Float8E4M3FNUZType::get(context.get()))),
      RankedTensorType::get({2, 2}, Float8E4M3FNUZType::get(context.get())));
  EXPECT_EQ(
      ToBfloat16Type(RankedTensorType::get(
          {2, 2}, Float8E4M3B11FNUZType::get(context.get()))),
      RankedTensorType::get({2, 2}, Float8E4M3B11FNUZType::get(context.get())));
  EXPECT_EQ(
      ToBfloat16Type(RankedTensorType::get(
          {2, 2}, Float8E5M2FNUZType::get(context.get()))),
      RankedTensorType::get({2, 2}, Float8E5M2FNUZType::get(context.get())));
  EXPECT_EQ(ToBfloat16Type(RankedTensorType::get(
                {2, 2}, Float8E5M2Type::get(context.get()))),
            RankedTensorType::get({2, 2}, Float8E5M2Type::get(context.get())));
  EXPECT_EQ(ToBfloat16Type(
                RankedTensorType::get({2, 2}, Float16Type::get(context.get()))),
            RankedTensorType::get({2, 2}, Float16Type::get(context.get())));
  EXPECT_EQ(ToBfloat16Type(RankedTensorType::get(
                {2, 2}, BFloat16Type::get(context.get()))),
            RankedTensorType::get({2, 2}, BFloat16Type::get(context.get())));
  EXPECT_EQ(ToBfloat16Type(
                RankedTensorType::get({2, 2}, Float32Type::get(context.get()))),
            RankedTensorType::get({2, 2}, BFloat16Type::get(context.get())));
  EXPECT_EQ(ToBfloat16Type(
                RankedTensorType::get({2, 2}, Float64Type::get(context.get()))),
            RankedTensorType::get({2, 2}, BFloat16Type::get(context.get())));
  EXPECT_EQ(ToBfloat16Type(
                RankedTensorType::get({2, 2}, Float80Type::get(context.get()))),
            RankedTensorType::get({2, 2}, BFloat16Type::get(context.get())));

  EXPECT_EQ(ToBfloat16Type(RankedTensorType::get(
                {2, 2}, IntegerType::get(context.get(), 8))),
            RankedTensorType::get({2, 2}, IntegerType::get(context.get(), 8)));
  EXPECT_EQ(ToBfloat16Type(RankedTensorType::get(
                {2, 2}, IntegerType::get(context.get(), 16))),
            RankedTensorType::get({2, 2}, IntegerType::get(context.get(), 16)));
  EXPECT_EQ(ToBfloat16Type(RankedTensorType::get(
                {2, 2}, IntegerType::get(context.get(), 32))),
            RankedTensorType::get({2, 2}, IntegerType::get(context.get(), 32)));
}

}  // namespace
}  // namespace mlir::quant::stablehlo
