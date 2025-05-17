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

#include "xla/codegen/emitters/math/ldexp.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"

namespace xla::codegen::math {
namespace {

using ::testing::DoubleEq;
using ::testing::Eq;

// Does this already exist somewhere in XLA?
class JitRunner {
 public:
  explicit JitRunner(llvm::orc::ThreadSafeModule&& module) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    auto jit_or_err = llvm::orc::LLJITBuilder().create();
    if (!jit_or_err) {
      llvm::report_fatal_error(
          llvm::Twine(llvm::toString(jit_or_err.takeError())));
    }
    jit_ = std::move(jit_or_err.get());
    llvm::ExitOnError exit_on_err;
    exit_on_err(jit_->addIRModule(std::move(module)));
  }

  ~JitRunner() { llvm::llvm_shutdown(); }

  template <typename FnType, typename RetType, typename... ArgTypes>
  llvm::Expected<RetType> RunJitTest(const std::string& function_name,
                                     ArgTypes... args) {
    auto function_sym = jit_->lookup(function_name);
    if (!function_sym) {
      return function_sym.takeError();
    }

    auto* function_ptr = reinterpret_cast<FnType*>(function_sym->getValue());
    RetType result = (*function_ptr)(args...);
    return result;
  }
  template <int VectorSize>
  llvm::Expected<std::array<double, VectorSize>> RunJitVectorized(
      const std::string& original_function_name,
      const std::array<double, VectorSize>& arg1,
      const std::array<int64_t, VectorSize>& arg2) {
    std::string wrapper_function_key =
        original_function_name + "_wrapper_" + std::to_string(VectorSize);

    // Define the C++ function pointer type for the wrapper
    using WrapperFnPtrType = void (*)(double*, const double*, const int64_t*);

    // Check if the wrapper's function pointer is already cached
    auto it = wrapper_ptr_cache_.find(wrapper_function_key);
    WrapperFnPtrType wrapper_ptr = nullptr;

    if (it != wrapper_ptr_cache_.end()) {
      wrapper_ptr = reinterpret_cast<WrapperFnPtrType>(it->second);
    } else {
      // Wrapper not in cache, need to generate and add it
      llvm::Expected<void*> created_wrapper_ptr_or_err =
          CreateVectorWrapperFunction(original_function_name,
                                      wrapper_function_key, VectorSize);
      if (auto e = created_wrapper_ptr_or_err.takeError()) {
        return llvm::make_error<llvm::StringError>(
            llvm::errc::not_supported, llvm::toString(std::move(e)));
      }
      wrapper_ptr =
          reinterpret_cast<WrapperFnPtrType>(created_wrapper_ptr_or_err.get());
      wrapper_ptr_cache_[wrapper_function_key] =
          created_wrapper_ptr_or_err.get();  // Store raw address
    }

    alignas(32) std::array<double, VectorSize> result_array;
    (*wrapper_ptr)(result_array.data(), arg1.data(), arg2.data());
    return result_array;
  }

 private:
  std::unique_ptr<llvm::orc::LLJIT> jit_;
  // Cache for JITed wrapper function pointers
  // Key: unique function name (e.g., "xla.libm.ldexp.4xf64_wrapper_4")
  // Value: raw function pointer (void*)
  absl::flat_hash_map<std::string, void*> wrapper_ptr_cache_;

  // Private helper method to create and JIT the wrapper function
  llvm::Expected<void*> CreateVectorWrapperFunction(
      const std::string& original_function_name,
      const std::string& wrapper_function_name, int vector_size) {
    // Create a new LLVMContext and Module for the wrapper function.
    // LLJIT takes ownership of these when added.
    auto context_owner = std::make_unique<llvm::LLVMContext>();
    llvm::LLVMContext* context = context_owner.get();
    auto wrapper_module_owner = std::make_unique<llvm::Module>(
        wrapper_function_name + "_module", *context);
    llvm::Module* wrapper_module = wrapper_module_owner.get();

    llvm::IRBuilder<> builder(*context);

    // Original function's type: <VectorSize x double>(<VectorSize x double>,
    // ...)
    llvm::Type* f64_type = llvm::Type::getDoubleTy(*context);
    llvm::Type* vec_type = llvm::VectorType::get(
        f64_type, llvm::ElementCount::getFixed(vector_size));
    llvm::Type* int_vec_type =
        llvm::VectorType::get(llvm::Type::getInt64Ty(*context),
                              llvm::ElementCount::getFixed(vector_size));

    llvm::FunctionType* original_func_type =
        llvm::FunctionType::get(vec_type, {vec_type, int_vec_type}, false);

    // Declare the original function in the new wrapper module.
    // The JIT's symbol resolution will find the actual definition.
    llvm::Function* original_func = llvm::Function::Create(
        original_func_type, llvm::Function::ExternalLinkage,
        original_function_name, *wrapper_module);

    // Wrapper function's type: void (double*, double*, double*)
    llvm::Type* ptr_to_double_type = f64_type->getPointerTo();
    llvm::FunctionType* wrapper_llvm_func_type = llvm::FunctionType::get(
        builder.getVoidTy(),
        {ptr_to_double_type, ptr_to_double_type, ptr_to_double_type}, false);

    llvm::Function* wrapper_llvm_func = llvm::Function::Create(
        wrapper_llvm_func_type, llvm::Function::ExternalLinkage,
        wrapper_function_name, *wrapper_module);

    llvm::BasicBlock* entry_block =
        llvm::BasicBlock::Create(*context, "entry", wrapper_llvm_func);
    builder.SetInsertPoint(entry_block);

    // Get arguments from the wrapper function
    llvm::Function::arg_iterator arg_it = wrapper_llvm_func->arg_begin();
    llvm::Value* ret_ptr_arg = arg_it++;
    llvm::Value* a_ptr_arg = arg_it++;
    llvm::Value* exp_ptr_arg = arg_it++;

    // Perform bitcasts for memory access
    llvm::Value* a_vec_ptr =
        builder.CreateBitCast(a_ptr_arg, vec_type->getPointerTo());
    llvm::Value* exp_vec_ptr =
        builder.CreateBitCast(exp_ptr_arg, vec_type->getPointerTo());
    llvm::Value* ret_vec_ptr =
        builder.CreateBitCast(ret_ptr_arg, vec_type->getPointerTo());

    // Load the vector values from the pointers
    llvm::Align alignment(vector_size * sizeof(double));
    if (alignment.value() < 16) {
      alignment = llvm::Align(16);  // Min align for SIMD
    }

    llvm::LoadInst* a_vec = builder.CreateLoad(vec_type, a_vec_ptr);
    a_vec->setAlignment(alignment);

    llvm::LoadInst* exp_vec = builder.CreateLoad(int_vec_type, exp_vec_ptr);
    exp_vec->setAlignment(alignment);

    // Call the original vectorized function
    llvm::CallInst* result_vec =
        builder.CreateCall(original_func, {a_vec, exp_vec});

    // Store the result back to the output pointer
    llvm::StoreInst* store_result =
        builder.CreateStore(result_vec, ret_vec_ptr);
    store_result->setAlignment(alignment);

    builder.CreateRetVoid();  // Wrapper returns void

    // Verify the wrapper function
    std::string error_str;
    llvm::raw_string_ostream os(error_str);
    if (llvm::verifyFunction(*wrapper_llvm_func, &os)) {
      return llvm::make_error<llvm::StringError>(
          llvm::errc::invalid_argument,
          "Error in wrapper function IR: " + os.str());
    }

    // Add the wrapper module to the JIT.
    // LLJIT now owns wrapper_module_owner and context_owner.
    llvm::ExitOnError exit_on_err;
    exit_on_err(jit_->addIRModule(llvm::orc::ThreadSafeModule(
        std::move(wrapper_module_owner), std::move(context_owner))));

    // Look up the JITed function pointer and return it
    auto function_sym = jit_->lookup(wrapper_function_name);
    if (!function_sym) {
      return function_sym
          .takeError();  // Should not happen if addIRModule succeeded
    }
    return reinterpret_cast<void*>(function_sym->getValue());
  }
};

TEST(LdexpTest, EmitLdexpF64) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);
  llvm::IRBuilder<> builder(*context);
  llvm::Type* f64_type = llvm::Type::getDoubleTy(*context);

  llvm::Function* ldexp_func = EmitLdexpF64(module.get(), f64_type);
  EXPECT_FALSE(llvm::verifyFunction(*ldexp_func));

  // Test with various inputs
  double test_values[] = {1.0,
                          2.0,
                          0.5,
                          -1.0,
                          -2.0,
                          -0.5,
                          0.0,
                          2342093482.3,
                          std::numeric_limits<double>::min(),
                          std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::infinity(),
                          -std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::quiet_NaN()};
  int64_t exponents[] = {0, 1, -1, 10, -10, 50, -50, -700, 700};

  JitRunner runner(
      llvm::orc::ThreadSafeModule(std::move(module), std::move(context)));

  for (double a_val : test_values) {
    for (int64_t exp_val : exponents) {
      double expected = std::ldexp(a_val, exp_val);
      llvm::Expected<double> result_or_err =
          runner.RunJitTest<double(double, int64_t), double>(
              "xla.libm.ldexp.1xf64", a_val, exp_val);
      if (auto e = result_or_err.takeError()) {
        EXPECT_TRUE(false) << "Error: " << toString(std::move(e));
      }
      double result = result_or_err.get();

      if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(result));
      } else {
        EXPECT_THAT(result, DoubleEq(expected));
      }
    }
  }
}

TEST(LdexpTest, ClampsExponent) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);
  llvm::IRBuilder<> builder(*context);
  llvm::Type* f64_type = llvm::Type::getDoubleTy(*context);

  llvm::Function* ldexp_func = EmitLdexpF64(module.get(), f64_type);
  EXPECT_FALSE(llvm::verifyFunction(*ldexp_func));
  JitRunner runner(
      llvm::orc::ThreadSafeModule(std::move(module), std::move(context)));

  auto run = [&runner](double a, int64_t exp) {
    return runner
        .RunJitTest<double(double, int64_t), double>("xla.libm.ldexp.1xf64", a,
                                                     exp)
        .get();
  };
  EXPECT_THAT(run(2.0, 1e9), Eq(std::numeric_limits<double>::infinity()));
  EXPECT_THAT(run(std::numeric_limits<double>::min(), 2100),
              Eq(std::numeric_limits<double>::infinity()));
  EXPECT_THAT(run(std::numeric_limits<double>::max(), -2099), Eq(0.0));
}

TEST(LdexpTest, EmitLdexpF64_Vector4) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);
  llvm::IRBuilder<> builder(*context);
  llvm::Type* vec_type = llvm::VectorType::get(
      llvm::Type::getDoubleTy(*context), llvm::ElementCount::getFixed(4));

  llvm::Function* ldexp_func = EmitLdexpF64(module.get(), vec_type);
  EXPECT_FALSE(llvm::verifyFunction(*ldexp_func));
  JitRunner runner(
      llvm::orc::ThreadSafeModule(std::move(module), std::move(context)));

  // Test with various inputs
  std::vector<double> test_values = {1.0,
                                     2.0,
                                     0.5,
                                     -1.0,
                                     -2.0,
                                     -0.5,
                                     0.0,
                                     std::numeric_limits<double>::infinity(),
                                     -std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::quiet_NaN(),
                                     0,
                                     -23434};
  int64_t exponents[] = {0, 1, -1, 10, -10, 50, -50};
  using DoubleArray4 = std::array<double, 4>;

  for (int i = 0; i < test_values.size(); i += 4) {
    for (int64_t exp_val : exponents) {
      alignas(32) DoubleArray4 input_values;
      alignas(32) std::array<int64_t, 4> exp_val_vec = {exp_val, exp_val,
                                                        exp_val, exp_val};
      for (int j = 0; j < 4 && i + j < test_values.size(); ++j) {
        double a_val = test_values[i + j];
        input_values[j] = a_val;
      }

      llvm::Expected<DoubleArray4> result_or_err = runner.RunJitVectorized<4>(
          "xla.libm.ldexp.4xf64", input_values, exp_val_vec);
      if (auto e = result_or_err.takeError()) {
        EXPECT_TRUE(false) << "Error: " << toString(std::move(e));
      }

      alignas(32) DoubleArray4 actual_results = result_or_err.get();
      for (int j = 0; j < actual_results.size(); ++j) {
        double expected = std::ldexp(input_values[j], exp_val_vec[j]);
        if (std::isnan(expected)) {
          EXPECT_TRUE(std::isnan(actual_results[j]));
        } else {
          EXPECT_THAT(actual_results[j], DoubleEq(expected));
        }
      }
    }
  }
}
}  // namespace
}  // namespace xla::codegen::math
