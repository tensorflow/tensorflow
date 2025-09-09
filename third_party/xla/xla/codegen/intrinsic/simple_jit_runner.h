#include <cstdint>

#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "xla/primitive_util.h"
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

#ifndef XLA_CODEGEN_INTRINSIC_SIMPLE_JIT_RUNNER_H_
#define XLA_CODEGEN_INTRINSIC_SIMPLE_JIT_RUNNER_H_

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/dynamic_annotations.h"
#include "absl/log/log.h"
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
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsic {

namespace internal {
// Helper struct to enforce alignment on our stack-allocated arrays.
// To use with parameter packs and std::apply, we need to wrap each array in a
// struct that enforces the alignment.
template <typename T, size_t VectorSize>
struct alignas(32) AlignedArrayWrapper {
  std::array<T, VectorSize> array;
};
}  // namespace internal

// A simple JIT runner for testing and benchmarking LLVM IR functions.
// The JitRunner instance must survive any calls to the JITed functions.
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

  template <typename FnType>
  FnType* GetScalarFn(const std::string& function_name) {
    auto function_sym = jit_->lookup(function_name);
    if (!function_sym) {
      LOG(FATAL) << "Failed to lookup function: " << function_name << ": "
                 << llvm::toString(function_sym.takeError());
    }

    FnType* fn = reinterpret_cast<FnType*>(function_sym->getValue());
    return fn;
  }

  // Compiles a JITted function that operates on vectors.
  // The original function is expected to have a signature like:
  //   <VectorSize x RetType> func(<VectorSize x Arg1Type>, <VectorSize x
  //   Arg2Type>, ...)
  template <size_t VectorSize, typename RetType, typename... ArgTypes>
  std::function<std::array<RetType, VectorSize>(
      const std::array<ArgTypes, VectorSize>&...)>
  GetVectorizedFn(const std::string& original_function_name,
                  size_t iteration_count = 1) {
    // Define the C++ pointer type for the JIT-compiled wrapper function.
    using WrapperFnPtrType =
        void (*)(RetType*, size_t, size_t, const ArgTypes*...);

    // Create and compile the wrapper function that handles the vector ABI.
    llvm::Expected<void*> wrapper_ptr_or_err = CreateVectorWrapperWithLoop(
        original_function_name, VectorSize,
        primitive_util::NativeToPrimitiveType<RetType>(),
        {primitive_util::NativeToPrimitiveType<ArgTypes>()...});

    if (auto e = wrapper_ptr_or_err.takeError()) {
      LOG(FATAL) << "Failed to create wrapper for function '"
                 << original_function_name
                 << "': " << llvm::toString(std::move(e));
    }
    auto wrapper_ptr = reinterpret_cast<WrapperFnPtrType>(*wrapper_ptr_or_err);

    return [wrapper_ptr,
            iteration_count](const std::array<ArgTypes, VectorSize>&... args) {
      alignas(32) std::array<RetType, VectorSize> result_array;
      // MSAN doesn't instrument JIT-compiled code, so we need to annotate this.
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
          result_array.data(), result_array.size() * sizeof(RetType));

      // Create a tuple of these aligned wrapper structs, copying the arguments.
      // The compiler will lay out each element of the tuple on the stack with
      // 32-byte alignment.
      auto aligned_args_tuple = std::make_tuple(
          internal::AlignedArrayWrapper<ArgTypes, VectorSize>{args}...);
      std::apply(
          [&](const auto&... single_arg_wrapper) {
            wrapper_ptr(result_array.data(), iteration_count, VectorSize,
                        single_arg_wrapper.array.data()...);
          },
          aligned_args_tuple);

      return result_array;
    };
  }

 private:
  std::unique_ptr<llvm::orc::LLJIT> jit_;
  std::unique_ptr<llvm::orc::ThreadSafeContext> tsc_;
  std::unique_ptr<llvm::orc::RTDyldObjectLinkingLayer> object_layer_;
  llvm::JITEventListener* perf_listener_ = nullptr;

  llvm::Expected<void*> CreateVectorWrapperWithLoop(
      const std::string& original_function_name, size_t vector_size,
      PrimitiveType ret_type, std::vector<PrimitiveType> arg_types);
};

std::unique_ptr<llvm::TargetMachine> CreateHostTargetMachine();

// Creates a new LLVM function that wraps an existing function by
// unrolling calls in a sequence.
//
// This function takes an `original_func` and generates a new function with an
// identical signature. Instead of a loop, the new function's body consists of
// an explicitly unrolled sequence of `unroll_factor` calls to the original
// function. This avoids loop overhead and is suitable for small K.
// This primarily serves as a knob to attempt to reduce the dependence of very
// small kernels on memory bandwidth.
//
// `vector_size`: The size of the vectors being processed.
// Returns a pointer to the newly created unrolled wrapper function.
llvm::Function* CreateKTimesWrapper(llvm::Module* module,
                                    llvm::Function* original_func,
                                    int unroll_factor, size_t vector_size);

}  // namespace xla::codegen::intrinsic

#endif  // XLA_CODEGEN_INTRINSIC_SIMPLE_JIT_RUNNER_H_
