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

#include "xla/codegen/math/simple_jit_runner.h"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/TypeSize.h"

namespace xla::codegen::math {
namespace {

TEST(SimpleJitRunnerTest, RunJitTest) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);

  llvm::IRBuilder<> builder(*context);
  llvm::FunctionType* func_type =
      llvm::FunctionType::get(llvm::Type::getDoubleTy(*context),
                              {llvm::Type::getDoubleTy(*context)}, false);
  llvm::Function* func = llvm::Function::Create(
      func_type, llvm::Function::ExternalLinkage, "test_func", *module);
  llvm::BasicBlock* entry_block =
      llvm::BasicBlock::Create(*context, "entry", func);
  builder.SetInsertPoint(entry_block);
  llvm::Value* arg = func->getArg(0);
  llvm::Value* result = builder.CreateFMul(
      arg, llvm::ConstantFP::get(llvm::Type::getDoubleTy(*context), 2.0));
  builder.CreateRet(result);

  JitRunner jit_runner(std::move(module), std::move(context));
  auto result_or_err =
      jit_runner.RunJitTest<double(double), double, double>("test_func", 3.0);
  EXPECT_DOUBLE_EQ(result_or_err.get(), 6.0);
}

TEST(SimpleJitRunnerTest, RunJitVectorized) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);

  llvm::IRBuilder<> builder(*context);
  constexpr int vector_size = 4;
  llvm::Type* f64_type = llvm::Type::getDoubleTy(*context);
  llvm::Type* vec_type = llvm::VectorType::get(
      f64_type, llvm::ElementCount::getFixed(vector_size));
  llvm::Type* int_vec_type =
      llvm::VectorType::get(llvm::Type::getInt64Ty(*context),
                            llvm::ElementCount::getFixed(vector_size));

  llvm::FunctionType* func_type =
      llvm::FunctionType::get(vec_type, {vec_type, int_vec_type}, false);
  llvm::Function* func = llvm::Function::Create(
      func_type, llvm::Function::ExternalLinkage, "test_vec_func", *module);
  llvm::BasicBlock* entry_block =
      llvm::BasicBlock::Create(*context, "entry", func);
  builder.SetInsertPoint(entry_block);
  llvm::Value* arg1 = func->getArg(0);
  llvm::Value* arg2 = func->getArg(1);
  llvm::Value* result =
      builder.CreateFMul(arg1, builder.CreateSIToFP(arg2, vec_type));
  builder.CreateRet(result);

  JitRunner jit_runner(std::move(module), std::move(context));
  std::array<double, vector_size> arg1_array = {1.0, 2.0, 3.0, 4.0};
  std::array<int64_t, vector_size> arg2_array = {2, 3, 4, 5};
  auto result_or_err = jit_runner.RunJitBinaryVectorized<vector_size>(
      "test_vec_func", arg1_array, arg2_array);
  std::array<double, vector_size> result_array = result_or_err.get();
  EXPECT_DOUBLE_EQ(result_array[0], 2.0);
  EXPECT_DOUBLE_EQ(result_array[1], 6.0);
  EXPECT_DOUBLE_EQ(result_array[2], 12.0);
  EXPECT_DOUBLE_EQ(result_array[3], 20.0);
}

TEST(SimpleJitRunnerTest, RunJitVectorizedF32) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);

  llvm::IRBuilder<> builder(*context);
  constexpr int vector_size = 4;
  llvm::Type* vec_type =
      llvm::VectorType::get(llvm::Type::getFloatTy(*context),
                            llvm::ElementCount::getFixed(vector_size));
  llvm::Type* int_vec_type =
      llvm::VectorType::get(llvm::Type::getInt64Ty(*context),
                            llvm::ElementCount::getFixed(vector_size));

  llvm::FunctionType* func_type =
      llvm::FunctionType::get(vec_type, {vec_type, int_vec_type}, false);
  llvm::Function* func = llvm::Function::Create(
      func_type, llvm::Function::ExternalLinkage, "test_vec_func", *module);
  llvm::BasicBlock* entry_block =
      llvm::BasicBlock::Create(*context, "entry", func);
  builder.SetInsertPoint(entry_block);
  llvm::Value* arg1 = func->getArg(0);
  llvm::Value* arg2 = func->getArg(1);
  llvm::Value* result =
      builder.CreateFMul(arg1, builder.CreateSIToFP(arg2, vec_type));
  builder.CreateRet(result);

  JitRunner jit_runner(std::move(module), std::move(context));
  std::array<float, vector_size> arg1_array = {1.0, 2.0, 3.0, 4.0};
  std::array<int64_t, vector_size> arg2_array = {2, 3, 4, 5};
  auto result_or_err = jit_runner.RunJitBinaryVectorized<vector_size>(
      "test_vec_func", arg1_array, arg2_array);
  std::array<float, vector_size> result_array = result_or_err.get();
  EXPECT_DOUBLE_EQ(result_array[0], 2.0);
  EXPECT_DOUBLE_EQ(result_array[1], 6.0);
  EXPECT_DOUBLE_EQ(result_array[2], 12.0);
  EXPECT_DOUBLE_EQ(result_array[3], 20.0);
}

}  // namespace
}  // namespace xla::codegen::math
