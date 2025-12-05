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

#include "xla/codegen/intrinsic/type.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/TypeSize.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {
namespace {

TEST(TypeTest, TypeToIrType) {
  llvm::LLVMContext context;
  EXPECT_EQ(Type::S(F32).to_ir_type(context), llvm::Type::getFloatTy(context));
  EXPECT_EQ(Type::V(F32, 4).to_ir_type(context),
            llvm::VectorType::get(llvm::Type::getFloatTy(context),
                                  llvm::ElementCount::getFixed(4)));
}

TEST(TypeTest, TypeFromIrType) {
  llvm::LLVMContext context;
  EXPECT_EQ(Type::TypeFromIrType(llvm::Type::getFloatTy(context)),
            Type::S(F32));
  EXPECT_EQ(
      Type::TypeFromIrType(llvm::VectorType::get(
          llvm::Type::getFloatTy(context), llvm::ElementCount::getFixed(4))),
      Type::V(F32, 4));
}

TEST(TypeTest, VerifySameWidth) {
  EXPECT_OK(Type::VerifySameWidth(Type::S(F32), Type::S(F32)));
  EXPECT_OK(Type::VerifySameWidth(Type::V(F32, 4), Type::V(F32, 4)));
  EXPECT_FALSE(Type::VerifySameWidth(Type::V(F32, 4), Type::V(F32, 8)).ok());
}

TEST(TypeTest, VerifySameWidthAndElementType) {
  EXPECT_OK(Type::VerifySameWidthAndElementType(Type::S(F32), Type::S(F32)));
  EXPECT_OK(
      Type::VerifySameWidthAndElementType(Type::V(F32, 4), Type::V(F32, 4)));
  EXPECT_FALSE(
      Type::VerifySameWidthAndElementType(Type::V(F32, 4), Type::V(F32, 8))
          .ok());
  EXPECT_FALSE(
      Type::VerifySameWidthAndElementType(Type::V(F32, 4), Type::V(BF16, 4))
          .ok());
}

TEST(TypeTest, FromName) {
  EXPECT_EQ(Type::FromName("f32"), Type::S(F32));
  EXPECT_EQ(Type::FromName("v4f32"), Type::V(F32, 4));
}

}  // namespace
}  // namespace xla::codegen::intrinsics
