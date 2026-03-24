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
#include <limits>
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
#include "xla/codegen/intrinsic/simple_jit_runner.h"
#include "xla/codegen/intrinsic/type.h"
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

TEST(FpTruncExecutionTest, F32ToF8e4m3fn) {
  JitRunner jit = CreateJitRunner(Type::S(F32), Type::S(F8E4M3FN));
  auto fptrunc = jit.GetScalarFn<int8_t(float)>(
      FpTrunc::Name(Type::S(F32), Type::S(F8E4M3FN)));
  EXPECT_EQ(fptrunc(0x7FFFFFFF), 0x7F);  // overflows
  EXPECT_EQ(fptrunc(0x0), 0x0);

  EXPECT_EQ(fptrunc(-2.0f), static_cast<int8_t>(0b11000000));
  EXPECT_EQ(fptrunc(0.5f), static_cast<int8_t>(0b00110000));
  EXPECT_EQ(fptrunc(-0.5f), static_cast<int8_t>(0b10110000));
  EXPECT_EQ(fptrunc(0.125f), static_cast<int8_t>(0b00100000));

  // Test denormals (exponent all 0s) round to 0 in fp8e4m3fn.
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::denorm_min()), 0);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::min()), 0);
  // largest f32 denormal:
  EXPECT_EQ(fptrunc(1.17549e-38f), 0);
}

TEST(FpTruncExecutionTest, F32ToF8e3m4) {
  JitRunner jit = CreateJitRunner(Type::S(F32), Type::S(F8E3M4));
  auto fptrunc = jit.GetScalarFn<int8_t(float)>(
      FpTrunc::Name(Type::S(F32), Type::S(F8E3M4)));
  EXPECT_EQ(fptrunc(0x0), 0x0);
  EXPECT_EQ(fptrunc(-2.0f), static_cast<int8_t>(0b11000000));
  EXPECT_EQ(fptrunc(0.5f), static_cast<int8_t>(0b00100000));
  EXPECT_EQ(fptrunc(-0.5f), static_cast<int8_t>(0b10100000));
  EXPECT_EQ(fptrunc(0.0156f), static_cast<int8_t>(0b00000001));

  // test underflow, denormals
  const float smallest_pos_subnormal_f8e3m4 = 0.015625;
  EXPECT_EQ(fptrunc(smallest_pos_subnormal_f8e3m4),
            static_cast<int8_t>(0b00000001));
  EXPECT_EQ(fptrunc(smallest_pos_subnormal_f8e3m4),
            static_cast<int8_t>(0b00000001));
  const float eps = std::numeric_limits<float>::epsilon();
  EXPECT_EQ(fptrunc(smallest_pos_subnormal_f8e3m4 / 2.0f - eps),
            static_cast<int8_t>(0b00000000));

  // test overflows and infinities
  const int8_t inf = static_cast<int8_t>(0b01110000);
  const int8_t neg_inf = static_cast<int8_t>(0b11110000);
  const float max_f8e3m4 = 15.5;
  EXPECT_EQ(fptrunc(max_f8e3m4 + 1.0f), inf);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::infinity()), inf);
  EXPECT_EQ(fptrunc(-std::numeric_limits<float>::infinity()), neg_inf);

  // test nan prop
  const int8_t nan = static_cast<int8_t>(0b01111000);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::quiet_NaN()), nan);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::signaling_NaN()), nan);
}
TEST(FpTruncExecutionTest, F32ToF8e4m3) {
  JitRunner jit = CreateJitRunner(Type::S(F32), Type::S(F8E4M3));
  auto fptrunc = jit.GetScalarFn<int8_t(float)>(
      FpTrunc::Name(Type::S(F32), Type::S(F8E4M3)));

  // Test basic values
  EXPECT_EQ(fptrunc(0.0f), 0x0);
  EXPECT_EQ(fptrunc(16.0f), static_cast<int8_t>(0b01011000));
  EXPECT_EQ(fptrunc(-16.0f), static_cast<int8_t>(0b11011000));
  EXPECT_EQ(fptrunc(0.5f), static_cast<int8_t>(0b00110000));
  EXPECT_EQ(fptrunc(-0.5f), static_cast<int8_t>(0b10110000));

  // Test underflow and subnormals
  // Smallest positive subnormal for f8e4m3 is 2^-9
  const float smallest_pos_subnormal_f8e4m3 = 0.001953125f;
  EXPECT_EQ(fptrunc(smallest_pos_subnormal_f8e4m3),
            static_cast<int8_t>(0b00000001));

  const float eps = std::numeric_limits<float>::epsilon();
  EXPECT_EQ(fptrunc(smallest_pos_subnormal_f8e4m3 / 2.0f - eps),
            static_cast<int8_t>(0b00000000));

  // Test overflows and infinities
  const int8_t pos_inf = static_cast<int8_t>(0b01111000);
  const int8_t neg_inf = static_cast<int8_t>(0b11111000);
  const float max_f8e4m3 = 448.0f;
  EXPECT_EQ(fptrunc(max_f8e4m3 + 1.0f), pos_inf);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::infinity()), pos_inf);
  EXPECT_EQ(fptrunc(-std::numeric_limits<float>::infinity()), neg_inf);

  // Test NaN propagation
  const int8_t nan = static_cast<int8_t>(0x7C);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::quiet_NaN()), nan);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::signaling_NaN()), nan);
}

TEST(FpTruncExecutionTest, F32ToF8e4m3fnuz) {
  JitRunner jit = CreateJitRunner(Type::S(F32), Type::S(F8E4M3FNUZ));
  auto fptrunc = jit.GetScalarFn<int8_t(float)>(
      FpTrunc::Name(Type::S(F32), Type::S(F8E4M3FNUZ)));

  EXPECT_EQ(fptrunc(0.0f), 0x0);
  EXPECT_EQ(fptrunc(16.0f), static_cast<int8_t>(0b01100000));
  EXPECT_EQ(fptrunc(-16.0f), static_cast<int8_t>(0b11100000));
  EXPECT_EQ(fptrunc(0.5f), static_cast<int8_t>(0b00111000));
  EXPECT_EQ(fptrunc(-0.5f), static_cast<int8_t>(0b10111000));

  // Test underflow and subnormals (FNUZ formats often support subnormals)
  const float smallest_pos_subnormal = 0.0009765625f;
  EXPECT_EQ(fptrunc(smallest_pos_subnormal), static_cast<int8_t>(0b00000001));
  EXPECT_EQ(fptrunc(smallest_pos_subnormal * 2.0f),
            static_cast<int8_t>(0b00000010));
  const float eps = std::numeric_limits<float>::epsilon();
  EXPECT_EQ(fptrunc(smallest_pos_subnormal / 2.0f - eps), 0x0);

  // Test overflows (clamps to nan, no infinity)
  const float max_val = 240.0f;
  EXPECT_EQ(fptrunc(max_val), static_cast<int8_t>(0b01111111));
  EXPECT_EQ(fptrunc(-max_val), static_cast<int8_t>(0b11111111));
  const int8_t nan = static_cast<int8_t>(0b10000000);
  EXPECT_EQ(fptrunc(max_val + 10.0f), nan);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::infinity()), nan);
  EXPECT_EQ(fptrunc(-(max_val + 10.0f)), nan);
  EXPECT_EQ(fptrunc(-std::numeric_limits<float>::infinity()), nan);

  // Test NaN propagation
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::quiet_NaN()), nan);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::signaling_NaN()), nan);
}

TEST(FpTruncExecutionTest, F32ToF8e4m3fnuz_Vector4) {
  JitRunner jit = CreateJitRunner(Type::V(F32, 4), Type::V(F8E4M3FNUZ, 4));
  auto fptrunc = jit.GetVectorizedFn<4, int8_t, float>(
      FpTrunc::Name(Type::V(F32, 4), Type::V(F8E4M3FNUZ, 4)));
  std::array<float, 4> vals = {500, 16.0f, -0.5f, -240.0f};
  std::array<int8_t, 4> actuals = fptrunc(vals);
  EXPECT_EQ(actuals[0], static_cast<int8_t>(0b10000000));
  EXPECT_EQ(actuals[1], static_cast<int8_t>(0b01100000));
  EXPECT_EQ(actuals[2], static_cast<int8_t>(0b10111000));
  EXPECT_EQ(actuals[3], static_cast<int8_t>(0b11111111));
}

TEST(FpTruncExecutionTest, F32ToF8e4m3fnuz_Vector8) {
  JitRunner jit = CreateJitRunner(Type::V(F32, 8), Type::V(F8E4M3FNUZ, 8));
  auto fptrunc = jit.GetVectorizedFn<8, int8_t, float>(
      FpTrunc::Name(Type::V(F32, 8), Type::V(F8E4M3FNUZ, 8)));
  std::array<float, 8> vals = {500,   16.0f,  -0.5f,  -240.0f,
                               -1.0f, -16.0f, 240.0f, 242.0f};
  std::array<int8_t, 8> actuals = fptrunc(vals);
  EXPECT_EQ(actuals[0], static_cast<int8_t>(0b10000000));
  EXPECT_EQ(actuals[1], static_cast<int8_t>(0b01100000));
  EXPECT_EQ(actuals[2], static_cast<int8_t>(0b10111000));
  EXPECT_EQ(actuals[3], static_cast<int8_t>(0b11111111));
  EXPECT_EQ(actuals[4], static_cast<int8_t>(0b11000000));
  EXPECT_EQ(actuals[5], static_cast<int8_t>(0b11100000));
  EXPECT_EQ(actuals[6], static_cast<int8_t>(0b01111111));

  // 242.0f rounds down to 240.0f:
  EXPECT_EQ(actuals[7], static_cast<int8_t>(0b01111111));
}

TEST(FpTruncExecutionTest, F32ToF8e4m3fn_Vector8) {
  JitRunner jit = CreateJitRunner(Type::V(F32, 8), Type::V(F8E4M3FN, 8));
  auto fptrunc = jit.GetVectorizedFn<8, int8_t, float>(
      FpTrunc::Name(Type::V(F32, 8), Type::V(F8E4M3FN, 8)));
  std::array<float, 8> vals = {500,   16.0f,  -0.5f, -240.0f,
                               -1.0f, -16.0f, 0.5f,  242.0f};
  std::array<int8_t, 8> actuals = fptrunc(vals);
  EXPECT_EQ(actuals[0], static_cast<int8_t>(0x7F));        // 500 -> NaN
  EXPECT_EQ(actuals[1], static_cast<int8_t>(0b01011000));  // 16.0f
  EXPECT_EQ(actuals[2], static_cast<int8_t>(0b10110000));  // -0.5f
  EXPECT_EQ(actuals[3], static_cast<int8_t>(0b11110111));  // -240.0f
  EXPECT_EQ(actuals[4], static_cast<int8_t>(0b10111000));  // -1.0f
  EXPECT_EQ(actuals[5], static_cast<int8_t>(0b11011000));  // -16.0f
  EXPECT_EQ(actuals[6], static_cast<int8_t>(0b00110000));  // 0.5f

  // 242.0f rounds down to 240.0f:
  EXPECT_EQ(actuals[7], static_cast<int8_t>(0b01110111));
}

TEST(FpTruncExecutionTest, F32ToF8e4m3b11fnuz) {
  JitRunner jit = CreateJitRunner(Type::S(F32), Type::S(F8E4M3B11FNUZ));
  auto fptrunc = jit.GetScalarFn<int8_t(float)>(
      FpTrunc::Name(Type::S(F32), Type::S(F8E4M3B11FNUZ)));

  // Test basic values (bias of 11 shifts the range)
  EXPECT_EQ(fptrunc(0.0f), 0x0);
  EXPECT_EQ(fptrunc(16.0f), static_cast<int8_t>(0b01111000));
  EXPECT_EQ(fptrunc(-16.0f), static_cast<int8_t>(0b11111000));
  EXPECT_EQ(fptrunc(0.5f), static_cast<int8_t>(0b01010000));
  EXPECT_EQ(fptrunc(-0.5f), static_cast<int8_t>(0b11010000));

  // Test underflow and subnormals
  // Smallest subnormal is 0.125 * 2^(1-11) = 2^-3 * 2^-10 = 2^-13
  constexpr float smallest_pos_subnormal = 0.0001220703125f;
  EXPECT_EQ(fptrunc(smallest_pos_subnormal), static_cast<int8_t>(0b00000001));
  EXPECT_EQ(fptrunc(smallest_pos_subnormal * 2.0f),
            static_cast<int8_t>(0b00000010));
  const float eps = std::numeric_limits<float>::epsilon();
  EXPECT_EQ(fptrunc(smallest_pos_subnormal / 2.0f - eps), 0x0);

  // Test overflows (clamps to nan, no infinity)
  // Max value is (1 + 7/8) * 2^(15-11) = 1.875 * 16 = 30
  const float max_val = 30.0f;
  EXPECT_EQ(fptrunc(max_val), static_cast<int8_t>(0b01111111));
  EXPECT_EQ(fptrunc(-max_val), static_cast<int8_t>(0b11111111));
  const int8_t nan = static_cast<int8_t>(0b10000000);
  EXPECT_EQ(fptrunc(max_val + 1.0f), nan);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::infinity()), nan);
  EXPECT_EQ(fptrunc(-(max_val + 1.0f)), nan);
  EXPECT_EQ(fptrunc(-std::numeric_limits<float>::infinity()), nan);
  EXPECT_EQ(fptrunc(166.0f), nan);

  // Test NaN propagation
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::quiet_NaN()), nan);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::signaling_NaN()), nan);
}

TEST(FpTruncExecutionTest, F32ToF8e5m2) {
  JitRunner jit = CreateJitRunner(Type::S(F32), Type::S(F8E5M2));
  auto fptrunc = jit.GetScalarFn<int8_t(float)>(
      FpTrunc::Name(Type::S(F32), Type::S(F8E5M2)));

  // Test basic values (bias of 15)
  EXPECT_EQ(fptrunc(0.0f), 0x0);
  EXPECT_EQ(fptrunc(16.0f), static_cast<int8_t>(0b01001100));
  EXPECT_EQ(fptrunc(-16.0f), static_cast<int8_t>(0b11001100));
  EXPECT_EQ(fptrunc(0.5f), static_cast<int8_t>(0b00111000));
  EXPECT_EQ(fptrunc(-0.5f), static_cast<int8_t>(0b10111000));

  // Test underflow and subnormals
  // Smallest subnormal is 0.25 * 2^(1-15) = 2^-2 * 2^-14 = 2^-16
  constexpr float smallest_pos_subnormal = 0.0000152587890625f;
  EXPECT_EQ(fptrunc(smallest_pos_subnormal), static_cast<int8_t>(0b00000001));
  EXPECT_EQ(fptrunc(smallest_pos_subnormal * 2.0f),
            static_cast<int8_t>(0b00000010));
  const float eps = std::numeric_limits<float>::epsilon();
  EXPECT_EQ(fptrunc(smallest_pos_subnormal / 2.0f - eps), 0x0);

  // Test overflows (goes to infinity)
  // Max value is (1 + 3/4) * 2^(30-15) = 1.75 * 2^15 = 57344
  const float max_val = 57344.0f;
  EXPECT_EQ(fptrunc(max_val), static_cast<int8_t>(0b01111011));
  EXPECT_EQ(fptrunc(-max_val), static_cast<int8_t>(0b11111011));
  const int8_t pos_inf = static_cast<int8_t>(0b01111100);
  const int8_t neg_inf = static_cast<int8_t>(0b11111100);
  EXPECT_EQ(fptrunc(max_val * 2.0f), pos_inf);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::infinity()), pos_inf);
  EXPECT_EQ(fptrunc(-max_val * 2.0f), neg_inf);
  EXPECT_EQ(fptrunc(-std::numeric_limits<float>::infinity()), neg_inf);

  // Test NaN propagation
  const int8_t nan = static_cast<int8_t>(0b01111110);  // Canonical qNaN
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::quiet_NaN()), nan);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::signaling_NaN()), nan);
}

TEST(FpTruncExecutionTest, F32ToF8e5m2fnuz) {
  JitRunner jit = CreateJitRunner(Type::S(F32), Type::S(F8E5M2FNUZ));
  auto fptrunc = jit.GetScalarFn<int8_t(float)>(
      FpTrunc::Name(Type::S(F32), Type::S(F8E5M2FNUZ)));

  // Test basic values (bias of 16 shifts the range)
  EXPECT_EQ(fptrunc(0.0f), 0x0);
  EXPECT_EQ(fptrunc(-0.0f), 0x0);
  EXPECT_EQ(fptrunc(16.0f), static_cast<int8_t>(0b01010000));
  EXPECT_EQ(fptrunc(-16.0f), static_cast<int8_t>(0b11010000));
  EXPECT_EQ(fptrunc(0.5f), static_cast<int8_t>(0b00111100));
  EXPECT_EQ(fptrunc(-0.5f), static_cast<int8_t>(0b10111100));

  // Test underflow and subnormals
  // Smallest subnormal is 0.25 * 2^(1-16) = 2^-2 * 2^-15 = 2^-17
  constexpr float smallest_pos_subnormal = 0.00000762939453125f;
  EXPECT_EQ(fptrunc(smallest_pos_subnormal), static_cast<int8_t>(0b00000001));
  EXPECT_EQ(fptrunc(smallest_pos_subnormal * 2.0f),
            static_cast<int8_t>(0b00000010));
  const float eps = std::numeric_limits<float>::epsilon();
  EXPECT_EQ(fptrunc(smallest_pos_subnormal / 2.0f - eps), 0x0);

  // Test overflows (clamps to nan, no infinity)
  // Max value is (1 + 3/4) * 2^(30-16) = 1.75 * 2^14 = 28672
  const float max_val = 57344.0f;
  EXPECT_EQ(fptrunc(max_val), static_cast<int8_t>(0b01111111));
  EXPECT_EQ(fptrunc(-max_val), static_cast<int8_t>(0b11111111));
  const int8_t nan = static_cast<int8_t>(0b10000000);
  EXPECT_EQ(fptrunc(max_val * 2.0f), nan);  // Overflow
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::infinity()), nan);
  EXPECT_EQ(fptrunc(-max_val * 2.0f), nan);  // Overflow
  EXPECT_EQ(fptrunc(-std::numeric_limits<float>::infinity()), nan);

  // Test NaN propagation
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::quiet_NaN()), nan);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::signaling_NaN()), nan);
}

TEST(FpTruncExecutionTest, F16ToF8e5m2fnuz) {
  JitRunner jit = CreateJitRunner(Type::S(F16), Type::S(F8E5M2FNUZ));
  auto fptrunc = jit.GetScalarFn<int8_t(int16_t)>(
      FpTrunc::Name(Type::S(F16), Type::S(F8E5M2FNUZ)) + "_itofp");

  EXPECT_EQ(fptrunc(static_cast<int16_t>(0x00)),
            static_cast<int8_t>(0b00000000));
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0b1100000000000000)),
            static_cast<int8_t>(0b11000100));  // -2.0
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0b0011100000000000)),
            0b00111100);  // 0.5
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0b1011100000000000)),
            static_cast<int8_t>(0b10111100));  // -0.5

  // Test overflow behavior (should be NaN for FNUZ)
  // Max finite F8E5M2FNUZ is 57344.0.
  // 47616.0 in F16 is 0x79D0. It rounds to 49152.0 (0x7E).
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0x79D0)),
            static_cast<int8_t>(0b01111110));  // 47616.0 -> 0x7E

  // Max finite F16 is 65504.0 (0x7BFF).
  // 65504.0 > 61440.0 (cutoff for overflow in F8E5M2FNUZ), so it should become
  // NaN.
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0x7BFF)),
            static_cast<int8_t>(0b10000000));  // 65504.0 -> NaN

  EXPECT_EQ(fptrunc(static_cast<int16_t>(0x7C00)),
            static_cast<int8_t>(0b10000000));  // Inf -> NaN
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0xFC00)),
            static_cast<int8_t>(0b10000000));  // -Inf -> NaN

  // Test NaN
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0x7E00)),
            static_cast<int8_t>(0b10000000));  // NaN -> NaN
}

TEST(FpTruncExecutionTest, F64ToF8e5m2) {
  JitRunner jit = CreateJitRunner(Type::S(F64), Type::S(F8E5M2));
  // JitRunner doesn't have a direct helper for F64 input to int8_t output
  // wrapped as int64->int8 But GetScalarFn is a template. We need to pass
  // double. However, CreateWrapperIntArgToFp creates an integer argument
  // wrapper. For F64, the wrapper will take int64_t and bitcast to double.
  auto fptrunc = jit.GetScalarFn<int8_t(int64_t)>(
      FpTrunc::Name(Type::S(F64), Type::S(F8E5M2)) + "_itofp");

  EXPECT_EQ(fptrunc(0), 0);
  // 1.0 in double: 0x3FF0000000000000
  // F8E5M2: 1 sign, 5 exp, 2 mantissa. Bias 15.
  // 1.0 = 1.0 * 2^0 -> Exp = 15 = 01111. Mantissa 00. Result 00111100 = 0x3C
  EXPECT_EQ(fptrunc(0x3FF0000000000000), 0x3C);

  // 1.5 in double: 0x3FF8000000000000
  // F8E5M2: 1.5 = 1.1 * 2^0 -> Exp 15. Mantissa 10. Result 00111110 = 0x3E
  EXPECT_EQ(fptrunc(0x3FF8000000000000), 0x3E);

  // Max finite F8E5M2: 57344.0
  // In double: 0x40EC000000000000
  // F8E5M2: 0x7B
  EXPECT_EQ(fptrunc(0x40EC000000000000), 0x7B);

  // 60000.0 in double: 0x40ED4C0000000000
  // 60000 < 61440 (overflow threshold), so it rounds to max finite (57344).
  EXPECT_EQ(fptrunc(0x40ED4C0000000000), 0x7B);

  // Overflow -> Inf
  // 62000.0 in double: 0x40EE460000000000
  // 62000 > 61440, so overflow.
  EXPECT_EQ(fptrunc(0x40EE460000000000), 0x7C);
}

TEST(FpTruncExecutionTest, F16ToF8e5m2) {
  JitRunner jit = CreateJitRunner(Type::S(F16), Type::S(F8E5M2));
  auto fptrunc = jit.GetScalarFn<int8_t(int16_t)>(
      FpTrunc::Name(Type::S(F16), Type::S(F8E5M2)) + "_itofp");

  // Bias is 15 for both.
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0x00)), 0x00);
  // 1.0 (0 01111 0000000000) -> 1.0 (0 01111 00)
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0x3C00)), 0x3C);
  // 1.5 (0 01111 1000000000) -> 1.5 (0 01111 10)
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0x3E00)), 0x3E);
}

}  // namespace xla::codegen::intrinsics
