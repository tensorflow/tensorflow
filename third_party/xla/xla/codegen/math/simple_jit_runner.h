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

#ifndef XLA_CODEGEN_MATH_SIMPLE_JIT_RUNNER_H_
#define XLA_CODEGEN_MATH_SIMPLE_JIT_RUNNER_H_

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/dynamic_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::math {

// Implementation note:
// - We could have used cpu/codegen/jit_compiler here, but we don't need the
//   ability to compile multiple modules in parallel, and we don't need to
//   support cross-module linking. For our use case it's actually slightly
//   simpler to use LLJIT.
// - The majority of complexity here lies in translating between C++ and LLVM
//   vector types (`CreateVectorWrapperFunction`). The particular JIT compiler
//   doesn't change that.
// - N.B. For kernels, Will's xla/backends/cpu/testlib/kernel_runner_test.cc
//   also exists, but these inlined math functions don't follow the kernel API.
class JitRunner {
 public:
  explicit JitRunner(std::unique_ptr<llvm::Module> module,
                     std::unique_ptr<llvm::LLVMContext> context);
  ~JitRunner();

  template <typename FnType, typename RetType, typename... ArgTypes>
  llvm::Expected<RetType> RunJitTest(const std::string& function_name,
                                     ArgTypes... args) {
    auto function_sym = jit_->lookup(function_name);
    if (!function_sym) {
      return function_sym.takeError();
    }

    auto* function_ptr =
        tsl::safe_reinterpret_cast<FnType*>(function_sym->getValue());
    RetType result = (*function_ptr)(args...);
    return result;
  }

  // Run a JITed function that takes a vector of arguments and returns a vector
  // of results.
  // The function is expected to have the following prototype:
  //   void func(Arg1Type* result, const Arg1Type* arg1, const Arg2Type* arg2)
  template <int VectorSize, typename Arg1Type, typename Arg2Type>
  llvm::Expected<std::array<Arg1Type, VectorSize>> RunJitBinaryVectorized(
      const std::string& original_function_name,
      const std::array<Arg1Type, VectorSize>& arg1,
      const std::array<Arg2Type, VectorSize>& arg2) {
    std::string wrapper_function_key =
        original_function_name + "_wrapper_" + std::to_string(VectorSize);

    // Define the C++ function pointer type for the wrapper
    using WrapperFnPtrType =
        void (*)(Arg1Type*, const Arg1Type*, const Arg2Type*);

    // Check if the wrapper's function pointer is already cached
    auto it = wrapper_ptr_cache_.find(wrapper_function_key);
    WrapperFnPtrType wrapper_ptr = nullptr;

    if (it != wrapper_ptr_cache_.end()) {
      wrapper_ptr = tsl::safe_reinterpret_cast<WrapperFnPtrType>(it->second);
    } else {
      // Wrapper not in cache, need to generate and add it
      llvm::Expected<void*> created_wrapper_ptr_or_err =
          CreateVectorWrapperFunction(
              original_function_name, VectorSize,
              primitive_util::NativeToPrimitiveType<Arg1Type>(),
              {primitive_util::NativeToPrimitiveType<Arg1Type>(),
               primitive_util::NativeToPrimitiveType<Arg2Type>()});
      if (auto e = created_wrapper_ptr_or_err.takeError()) {
        return llvm::make_error<llvm::StringError>(
            llvm::errc::not_supported, llvm::toString(std::move(e)));
      }
      wrapper_ptr = tsl::safe_reinterpret_cast<WrapperFnPtrType>(
          created_wrapper_ptr_or_err.get());
      wrapper_ptr_cache_[wrapper_function_key] =
          created_wrapper_ptr_or_err.get();  // Store raw address
    }

    alignas(32) std::array<Arg1Type, VectorSize> result_array;
    // Required to satisfy MSAN, which doesn't instrument the JITed code.
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(result_array.data(),
                                        result_array.size() * sizeof(Arg1Type));
    // Copy the arguments to make sure they are aligned. We could require
    // callers to pass aligned arrays, but the errors if they don't are hard to
    // debug, and most input arrays are likely to be small enough that a copy
    // during test is cheap.
    alignas(32) std::array<Arg1Type, VectorSize> arg1_aligned = arg1;
    alignas(32) std::array<Arg2Type, VectorSize> arg2_aligned = arg2;
    (*wrapper_ptr)(result_array.data(), arg1_aligned.data(),
                   arg2_aligned.data());
    return result_array;
  }

 private:
  std::unique_ptr<llvm::orc::LLJIT> jit_;
  // Cache for JITed wrapper function pointers
  // Key: unique function name (e.g., "xla.ldexp.4xf64_wrapper_4")
  // Value: raw function pointer (void*)
  absl::flat_hash_map<std::string, void*> wrapper_ptr_cache_;

  // Private helper method to create and JIT the wrapper function
  llvm::Expected<void*> CreateVectorWrapperFunction(
      const std::string& original_function_name, size_t vector_size,
      PrimitiveType ret_type, std::vector<PrimitiveType> arg_types);
};

}  // namespace xla::codegen::math

#endif  // XLA_CODEGEN_MATH_SIMPLE_JIT_RUNNER_H_
