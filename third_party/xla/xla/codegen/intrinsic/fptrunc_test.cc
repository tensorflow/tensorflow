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

#include "xla/codegen/intrinsic/fptrunc.h"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Target/TargetMachine.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/codegen/intrinsic/simple_jit_runner.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {
using ::xla::codegen::intrinsic::JitRunner;

TEST(FpTruncTest, SclarIninsic) {
  EXPECT_EQ(FpTrunc::Name(Type::S(F32), Type::S(BF16)),
            "xla.fptrunc.f32.to.bf16");
}

TEST(FpTruncTest, VectorIninsic) {
  EXPECT_EQ(FpTrunc::Name(Type::V(F32, 4), Type::V(BF16, 4)),
            "xla.fptrunc.v4f32.to.v4bf16");
}

// This function takes in an LLVM function that expects an fp argument and
// wraps it in a new function that takes in an integer argument of the same
// bit width as the fp argument and bitcasts it to the fp type.
// Because there's no great native C++ fp16 type, we need to pass integer
// values of the right bit width to the intrinsic. But because the calling
// convention looks for fp arguments in fp registers, the intrinsic won't see
// these arguments at all so we need to create a wrapper function that converts
// the integer arguments to fp for interop with C++ here in this test.
llvm::Function* CreateWrapperIntArgToFp(llvm::Function* func) {
  llvm::LLVMContext& context = func->getContext();
  llvm::IRBuilder<> builder(context);
  llvm::Value* wrapped_arg = func->getArg(0);
  llvm::Type* int_arg_type = llvm::IntegerType::get(
      context, wrapped_arg->getType()->getScalarSizeInBits());
  if (auto vec_type =
          llvm::dyn_cast<llvm::VectorType>(wrapped_arg->getType())) {
    int_arg_type =
        llvm::VectorType::get(int_arg_type, vec_type->getElementCount());
  }
  llvm::Function* wrapper = llvm::Function::Create(
      llvm::FunctionType::get(func->getReturnType(), {int_arg_type}, false),
      llvm::GlobalValue::ExternalLinkage, func->getName() + "_itofp",
      func->getParent());
  wrapper->copyAttributesFrom(func);
  llvm::BasicBlock* entry_bb =
      llvm::BasicBlock::Create(context, "entry", wrapper);
  builder.SetInsertPoint(entry_bb);
  llvm::Value* float_arg =
      builder.CreateBitCast(wrapper->getArg(0), wrapped_arg->getType());
  llvm::Value* ret = builder.CreateCall(func, {float_arg});
  builder.CreateRet(ret);
  return wrapper;
}

llvm::Function* CreateWrapperFpRetToInt(llvm::Function* func) {
  llvm::LLVMContext& context = func->getContext();
  llvm::IRBuilder<> builder(context);
  llvm::Type* ret_type = func->getReturnType();
  llvm::Type* int_ret_type =
      llvm::IntegerType::get(context, ret_type->getScalarSizeInBits());
  if (auto vec_type = llvm::dyn_cast<llvm::VectorType>(ret_type)) {
    int_ret_type =
        llvm::VectorType::get(int_ret_type, vec_type->getElementCount());
  }
  llvm::Function* wrapper = llvm::Function::Create(
      llvm::FunctionType::get(int_ret_type, {func->getArg(0)->getType()},
                              false),
      llvm::GlobalValue::ExternalLinkage, func->getName() + "_fptoi",
      func->getParent());
  wrapper->copyAttributesFrom(func);
  llvm::BasicBlock* entry_bb =
      llvm::BasicBlock::Create(context, "entry", wrapper);
  builder.SetInsertPoint(entry_bb);
  llvm::Value* ret = builder.CreateCall(func, {wrapper->getArg(0)});
  ret = builder.CreateBitCast(ret, int_ret_type);
  builder.CreateRet(ret);
  return wrapper;
}

JitRunner CreateJitRunner(Type from, Type to) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);

  llvm::Function* func =
      FpTrunc::CreateDefinition(module.get(), from, to).value();
  func->setLinkage(llvm::Function::ExternalLinkage);
  EXPECT_FALSE(llvm::verifyFunction(*func));

  llvm::Function* wrapper = CreateWrapperIntArgToFp(func);
  wrapper->setLinkage(llvm::Function::ExternalLinkage);
  EXPECT_FALSE(llvm::verifyFunction(*wrapper));

  llvm::Function* wrapper2 = CreateWrapperFpRetToInt(func);
  wrapper2->setLinkage(llvm::Function::ExternalLinkage);
  EXPECT_FALSE(llvm::verifyFunction(*wrapper2));

  return JitRunner(std::move(module), std::move(context));
}

TEST(FpTruncExecutionTest, F16ToF8e4m3fn) {
  JitRunner jit = CreateJitRunner(Type::S(F16), Type::S(F8E4M3FN));
  auto fptrunc = jit.GetScalarFn<int8_t(int16_t)>(
      FpTrunc::Name(Type::S(F16), Type::S(F8E4M3FN)) + "_itofp");
  EXPECT_EQ(fptrunc(0x7FFF), 0x7F);  // overflows
  EXPECT_EQ(fptrunc(0x0), 0x0);
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0b1100000000000000)),
            static_cast<int8_t>(0b11000000));  // -2.0
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0b0011100000000000)),
            0b00110000);  // 0.5
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0b1011100000000000)),
            static_cast<int8_t>(0b10110000));  // -0.5

  // Test denormals (exponent all 0s) round to 0 in fp8e4m3fn.
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0b0000000100000000)),
            static_cast<int8_t>(0b00000000));
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0b0000000010000000)),
            static_cast<int8_t>(0b00000000));
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0b0000000001000000)),
            static_cast<int8_t>(0b00000000));
}

TEST(FpTruncExecutionTest, F16ToF8e4m3fn_Vector4) {
  JitRunner jit = CreateJitRunner(Type::V(F16, 4), Type::V(F8E4M3FN, 4));
  auto fptrunc = jit.GetVectorizedFn<4, int8_t, int16_t>(
      FpTrunc::Name(Type::V(F16, 4), Type::V(F8E4M3FN, 4)) + "_itofp");
  std::array<int16_t, 4> vals = {0x7FFF, 0x0,
                                 static_cast<int16_t>(0b1100000000000000),
                                 static_cast<int16_t>(0b0011100000000000)};
  std::array<int8_t, 4> actuals = fptrunc(vals);
  EXPECT_EQ(actuals[0], 0x7F);
  EXPECT_EQ(actuals[1], 0x0);
  EXPECT_EQ(actuals[2], static_cast<int8_t>(0b11000000));
  EXPECT_EQ(actuals[3], static_cast<int8_t>(0b00110000));
}

TEST(FpTruncExecutionTest, F8e4m3fnToF16) {
  JitRunner jit = CreateJitRunner(Type::S(F8E4M3FN), Type::S(F16));
  auto fptrunc = jit.GetScalarFn<int16_t(int8_t)>(
      FpTrunc::Name(Type::S(F8E4M3FN), Type::S(F16)) + "_fptoi");
  EXPECT_EQ(fptrunc(static_cast<int8_t>(0b11000000)),
            static_cast<int16_t>(0b1100000000000000));  // -2.0
  EXPECT_EQ(fptrunc(static_cast<int8_t>(0b00110000)),
            static_cast<int16_t>(0b0011100000000000));  // 0.5
  EXPECT_EQ(fptrunc(static_cast<int8_t>(0b10110000)),
            static_cast<int16_t>(0b1011100000000000));  // -0.5
}

TEST(FpTruncExecutionTest, F8e4m3fnToF16_Vector4) {
  JitRunner jit = CreateJitRunner(Type::V(F8E4M3FN, 4), Type::V(F16, 4));
  auto fptrunc = jit.GetVectorizedFn<4, int16_t, int8_t>(
      FpTrunc::Name(Type::V(F8E4M3FN, 4), Type::V(F16, 4)) + "_fptoi");
  std::array<int8_t, 4> vals = {static_cast<int8_t>(0b11000000), 0x0,
                                static_cast<int8_t>(0b10110000),
                                static_cast<int8_t>(0b00110000)};
  std::array<int16_t, 4> actuals = fptrunc(vals);
  EXPECT_EQ(actuals[0], static_cast<int16_t>(0b1100000000000000));
  EXPECT_EQ(actuals[1], static_cast<int16_t>(0b0000000000000000));
  EXPECT_EQ(actuals[2], static_cast<int16_t>(0b1011100000000000));
  EXPECT_EQ(actuals[3], static_cast<int16_t>(0b0011100000000000));
}

}  // namespace xla::codegen::intrinsics
